"""Activation patching with the direct action as readout."""

from __future__ import annotations

import argparse
import json
import random
import re
import time

import numpy as np

from cotstate.cf_action import flip_prompt
from cotstate.env import FILES
from cotstate.fixed_history2 import frame, split_prefix
from cotstate.patching import untouched
from cotstate.patching2 import cluster_ci, find_layers

GROUPS = ("src", "hist", "cur_obs", "frame", "last", "cont")
SPLIT = ("hist_near", "hist_open", "hist_f", "hist_other")
SPLIT2 = ("hist_g", "hist_gk", "hist_tmpl", "hist_rest")
TMPL = re.compile(r"<\|im_start\|>(?:user|assistant|system)?\n?|<\|im_end\|>\n?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--probes", required=True, help="json with test_episodes")
    ap.add_argument("--belief-dirs", default=None, help="patching2b *_dirs.npz")
    ap.add_argument("--n-test-pairs", type=int, default=120)
    ap.add_argument("--n-train-pairs", type=int, default=150)
    ap.add_argument("--min-sep", type=float, default=1.0)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--max-len", type=int, default=12000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--split-hist", action="store_true")
    ap.add_argument("--near", type=int, default=4)
    ap.add_argument("--split-hist2", action="store_true",
                    help="also split hist_other into mentions of OTHER files (hist_g), a size-matched "
                         "random subset of them (hist_gk), chat-template tokens (hist_tmpl) and the rest")
    ap.add_argument("--no-dirs", action="store_true")
    ap.add_argument("--compare", default=None, help="main-run json: check that the same pairs were drawn")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split_hist2:
        args.split_hist = True
    groups = GROUPS + (SPLIT if args.split_hist else ()) + (SPLIT2 if args.split_hist2 else ())
    if args.pilot:
        args.n_test_pairs, args.n_train_pairs = 4, 20

    import torch
    import cotstate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    te = set(json.load(open(args.probes))["test_episodes"])
    eps = [json.loads(l) for l in open(args.transcripts)]
    tr = set(range(len(eps))) - te
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    layers_mod = find_layers(model)
    nL = len(layers_mod)
    layers = args.layers or sorted(set(list(range(0, nL, 2)) + [nL - 1]))
    if args.pilot:
        layers = [nL // 2, (3 * nL) // 4, nL - 1]
    dir_layers = [] if args.no_dirs else [L for L in layers if L >= int(0.4 * nL)]
    bd = dict(np.load(args.belief_dirs)) if args.belief_dirs else {}
    print(f"[pa] {args.model}: {nL} layers; group layers {layers}; direction layers {dir_layers}; "
          f"belief directions loaded: {len(bd)}")

    st = {"mode": None, "L": None, "pos": None, "src": None, "Q": None, "cap": None}

    def make_hook(L):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if st["mode"] == "capture" and st["cap"] is not None and L in st["cap"]:
                st["cap"][L] = h[0].detach().clone()
                return out
            if st["mode"] == "patch" and st["L"] == L:
                h = h.clone()
                pos = st["pos"]
                if st["Q"] is None:
                    h[0, pos, :] = st["src"][pos, :].to(h.dtype)
                else:
                    diff = st["src"][pos, :].float() - h[0, pos, :].float()
                    h[0, pos, :] = (h[0, pos, :].float() + (diff @ st["Q"]) @ st["Q"].T).to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return out
        return hook

    for L in layers:
        layers_mod[L].register_forward_hook(make_hook(L))

    def readout(ids, n_cont):
        x = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            try:
                lg = model(input_ids=x, use_cache=False, logits_to_keep=n_cont + 1).logits[0, :-1]
            except TypeError:
                lg = model(input_ids=x, use_cache=False).logits[0, -(n_cont + 1):-1]
        lp = torch.log_softmax(lg.float(), -1)
        cont = ids[-n_cont:]
        return float(sum(lp[k, c] for k, c in enumerate(cont)))

    def build(ei, t, f):
        pr = flip_prompt(eps[ei]["turns"][t]["prompt"], f)
        if pr is None:
            return None, "flip"
        out = []
        for p in pr:
            fp = frame(p, "", "CMD:")
            ip = tok(fp, add_special_tokens=False)["input_ids"]
            ic = tok(" mv " + f, add_special_tokens=False)["input_ids"]
            out.append((fp, ip, ic))
        (fL, aL, c), (fU, aU, _) = out
        if len(aL) != len(aU):
            return None, "length"
        if len(aL) + len(c) > args.max_len:
            return None, "too_long"
        diff = [i for i in range(len(aL)) if aL[i] != aU[i]]
        if len(diff) != 1:
            return None, "diffpos"
        return (fU, aL + c, aU + c, len(c), diff[0], p), None

    f_name = [None]

    def groups_for(fprompt, prefix, ids_prompt, s, n_cont, obs):
        enc = tok(fprompt, add_special_tokens=False, return_offsets_mapping=True)
        assert enc["input_ids"] == ids_prompt
        offs = enc["offset_mapping"]
        frame_char = len(split_prefix(prefix)[0])
        obs_char = fprompt.rfind(obs) if obs else -1
        if obs_char < 0 or obs_char >= frame_char:
            obs_char = frame_char
        n = len(ids_prompt)
        G = {g: [] for g in GROUPS}
        G["_f"] = f_name[0]
        G["src"] = [s]
        for i in range(s + 1, n - 1):
            a = offs[i][0]
            G["hist" if a < obs_char else ("cur_obs" if a < frame_char else "frame")].append(i)
        G["last"] = [n - 1]
        G["cont"] = list(range(n, n + n_cont))
        if args.split_hist:
            f_ = G.pop("_f")
            src_char = offs[s][0]
            spans = [(m.start(), m.start() + len(f_)) for m in re.finditer(re.escape(f_), fprompt)
                     if src_char < m.start() < obs_char]
            open_end = fprompt.find("<|im_end|>", src_char)
            if open_end < 0 or open_end > obs_char:
                open_end = src_char
            spans_after = [sp for sp in spans if sp[0] >= open_end]
            G["hist_near"], G["hist_open"], G["hist_f"], G["hist_other"] = [], [], [], []
            for i in G["hist"]:
                a, b = offs[i]
                if i <= s + args.near:
                    G["hist_near"].append(i)
                elif a < open_end:
                    G["hist_open"].append(i)
                elif any(a < e and b > st_ for st_, e in spans):
                    G["hist_f"].append(i)
                else:
                    G["hist_other"].append(i)
            if args.split_hist2:
                gm = []
                for g_ in FILES:
                    if g_ == f_:
                        continue
                    for m in re.finditer(re.escape(g_), fprompt):
                        if open_end <= m.start() < obs_char:
                            gm.append((m.start(), m.start() + len(g_)))
                tm = [(m.start(), m.end()) for m in TMPL.finditer(fprompt, open_end, obs_char)]
                ov = lambda i, sp: any(offs[i][0] < e and offs[i][1] > st_ for st_, e in sp)
                G["hist_g"] = [i for i in G["hist_other"] if ov(i, gm)]
                G["hist_tmpl"] = [i for i in G["hist_other"] if i not in G["hist_g"] and ov(i, tm)]
                G["hist_rest"] = [i for i in G["hist_other"] if i not in G["hist_g"] and i not in G["hist_tmpl"]]
                k_f = len(spans_after)
                rng_ = random.Random(1000 * s + len(ids_prompt))
                pick = rng_.sample(gm, k_f) if (k_f and len(gm) >= k_f) else []
                G["hist_gk"] = [i for i in G["hist_g"] if ov(i, pick)]
        else:
            G.pop("_f", None)
        return G

    def cands(pool, seed):
        c = [(ei, t["turn"], f) for ei in sorted(pool) for t in eps[ei]["turns"]
             if t["turn"] >= 2 and "prompt" in t for f in FILES
             if int(t["state_before"][f"exists_{f}"]) == 1 and untouched(eps[ei], t["turn"], f)]
        random.Random(seed).shuffle(c)
        return c

    D = {L: [] for L in dir_layers}
    drop_tr = {}
    for (ei, t, f) in ([] if args.no_dirs else cands(tr, 5)):
        if len(D[dir_layers[0]]) >= args.n_train_pairs:
            break
        b, why = build(ei, t, f)
        if b is None:
            drop_tr[why] = drop_tr.get(why, 0) + 1
            continue
        _, idsL, idsU, nc, s, _ = b
        n_p = len(idsL) - nc
        st.update(mode="capture", cap={L: None for L in dir_layers})
        xL = readout(idsL, nc)
        cL = st["cap"]
        st.update(cap={L: None for L in dir_layers})
        xU = readout(idsU, nc)
        cU = st["cap"]
        st.update(mode=None, cap=None)
        if xU - xL < args.min_sep:
            drop_tr["sep"] = drop_tr.get("sep", 0) + 1
            continue
        for L in dir_layers:
            D[L].append((cL[L][n_p - 1] - cU[L][n_p - 1]).float().cpu().numpy())
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
    basis = {}
    rng = np.random.RandomState(7)
    for L in dir_layers:
        basis[(L, "dm_action")] = unit(np.mean(D[L], 0))
        for k_in, k_out in (("dm_pool", "belief_dm"), ("nat_dm_pool", "belief_nat")):
            key = f"L{L}|{k_in}"
            if key in bd:
                basis[(L, k_out)] = unit(np.asarray(bd[key]).reshape(-1))
        basis[(L, "rand_1")] = unit(rng.randn(len(basis[(L, "dm_action")])))
    cos = {L: {k: float(basis[(L, "dm_action")] @ basis[(L, k)]) for k in ("belief_dm", "belief_nat")
               if (L, k) in basis} for L in dir_layers}
    n_train = len(D[dir_layers[0]]) if dir_layers else 0
    print(f"[pa] {n_train} training pairs (dropped {drop_tr}); cos(dm_action, belief_dm) by layer: "
          + ", ".join(f"L{L} {c.get('belief_dm', float('nan')):+.2f}" for L, c in cos.items()))
    tb = {k: torch.tensor(v[:, None], dtype=torch.float32, device=args.device) for k, v in basis.items()}
    dnames = ["dm_action", "belief_dm", "belief_nat", "rand_1", "self"]

    pairs, dropped = [], {}
    for (ei, t, f) in cands(te, 11):
        if len(pairs) >= args.n_test_pairs:
            break
        b, why = build(ei, t, f)
        if b is None:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        fU, idsL, idsU, nc, s, prefU = b
        n_p = len(idsL) - nc
        f_name[0] = f
        G = groups_for(fU, prefU, idsU[:n_p], s, nc, eps[ei]["turns"][t]["obs"])
        st.update(mode="capture", cap={L: None for L in layers})
        xL = readout(idsL, nc)
        cL = st["cap"]
        st.update(cap={L: None for L in dir_layers})
        xU = readout(idsU, nc)
        cU = st["cap"]
        st.update(mode=None, cap=None)
        if xU - xL < args.min_sep:
            dropped["sep"] = dropped.get("sep", 0) + 1
            continue
        rec = {"episode": ei, "turn": t, "file": f, "x_locked": xL, "x_unlocked": xU,
               "group_sizes": {g: len(v) for g, v in G.items()}, "patch": {}, "dir": {}, "frac": {}}
        for L in layers:
            for g in groups:
                if G[g]:
                    st.update(mode="patch", L=L, pos=G[g], src=cL[L], Q=None)
                    rec["patch"][f"{L}|{g}"] = readout(idsU, nc)
        for L in ([] if args.no_dirs else dir_layers):
            diff = cL[L][n_p - 1].float() - cU[L][n_p - 1].float()
            for nm in dnames:
                Q = (diff / diff.norm()).unsqueeze(1) if nm == "self" else tb.get((L, nm))
                if Q is None:
                    continue
                st.update(mode="patch", L=L, pos=[n_p - 1], src=cL[L], Q=Q)
                rec["dir"][f"{L}|{nm}"] = readout(idsU, nc)
                pr_ = diff @ Q
                rec["frac"][f"{L}|{nm}"] = float(pr_ @ pr_) / max(float(diff @ diff), 1e-12)
        st.update(mode=None, Q=None, src=None)
        del cL, cU
        torch.cuda.empty_cache()
        pairs.append(rec)
        print(f"[pa] pair {len(pairs)}/{args.n_test_pairs} {f} ep{ei} t{t} sep {xU - xL:+.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    cl = np.array([p["episode"] for p in pairs])
    R_ = lambda p, x: (x - p["x_unlocked"]) / (p["x_locked"] - p["x_unlocked"])
    summ = {}
    for L in layers:
        for g in groups:
            k = f"{L}|{g}"
            m = np.array([k in p["patch"] for p in pairs])
            if m.any():
                R = np.array([R_(p, p["patch"][k]) for p in pairs if k in p["patch"]])
                lo, hi = cluster_ci(R, cl[m])
                summ[k] = {"mean": float(R.mean()), "lo": lo, "hi": hi, "n": int(m.sum())}
    for L in dir_layers:
        for nm in dnames:
            k = f"{L}|{nm}"
            m = np.array([k in p["dir"] for p in pairs])
            if m.any():
                R = np.array([R_(p, p["dir"][k]) for p in pairs if k in p["dir"]])
                fr = np.array([p["frac"][k] for p in pairs if k in p["dir"]])
                lo, hi = cluster_ci(R, cl[m])
                summ["dir:" + k] = {"mean": float(R.mean()), "lo": lo, "hi": hi, "frac_mean": float(fr.mean())}
    problems = []
    for L in dir_layers:
        a_, b_ = summ.get(f"{L}|last"), summ.get(f"dir:{L}|self")
        if a_ and b_ and abs(a_["mean"] - b_["mean"]) > 0.05:
            problems.append(f"L{L}: self {b_['mean']:.2f} != last {a_['mean']:.2f}")
    if args.split_hist:
        for p in pairs:
            gs_ = p["group_sizes"]
            if sum(gs_[g] for g in SPLIT) != gs_["hist"]:
                problems.append(f"split does not partition hist: ep{p['episode']} t{p['turn']}")
                break
            if args.split_hist2 and gs_["hist_g"] + gs_["hist_tmpl"] + gs_["hist_rest"] != gs_["hist_other"]:
                problems.append(f"split2 does not partition hist_other: ep{p['episode']} t{p['turn']}")
                break
    if args.split_hist2:
        both = [p for p in pairs if p["group_sizes"]["hist_f"] > 0 and p["group_sizes"]["hist_gk"] > 0]
        clb = np.array([p["episode"] for p in both])
        for L in layers:
            kf, kg = f"{L}|hist_f", f"{L}|hist_gk"
            if both and all(kf in p["patch"] and kg in p["patch"] for p in both):
                dv = np.array([R_(p, p["patch"][kf]) - R_(p, p["patch"][kg]) for p in both])
                lo, hi = cluster_ci(dv, clb)
                summ[f"diff:{L}|f_minus_gk"] = {"mean": float(dv.mean()), "lo": lo, "hi": hi, "n": len(both)}
    if args.compare:
        ref = {(q["episode"], q["turn"], q["file"]): q for q in json.load(open(args.compare))["pairs"]}
        bad = [p for p in pairs if (p["episode"], p["turn"], p["file"]) in ref
               and abs(ref[(p["episode"], p["turn"], p["file"])]["x_locked"] - p["x_locked"]) > 0.05]
        same = sum((p["episode"], p["turn"], p["file"]) in ref for p in pairs)
        print(f"[pa] compare: {same}/{len(pairs)} pairs also in {args.compare}; {len(bad)} with different x_locked")
        if same < 0.9 * len(pairs) or bad:
            problems.append(f"compare: only {same}/{len(pairs)} shared pairs or {len(bad)} readout mismatches")
    fin = summ.get(f"{nL - 1}|src")
    if fin and abs(fin["mean"]) > 0.1:
        problems.append(f"final-layer src patch R = {fin['mean']:.2f} (should be ~0: nothing downstream)")
    print("\n[pa] CHECKS PASSED" if not problems else "\n[pa] CHECK FAILURES:\n  " + "\n  ".join(problems))
    sep = np.array([p["x_unlocked"] - p["x_locked"] for p in pairs])
    print(f"[pa] {len(pairs)} pairs from {len(set(cl))} episodes; dropped {dropped}; "
          f"separation median {np.median(sep):.2f} logits")
    print(f"{'L':>3} " + " ".join(f"{g:>10}" for g in groups))
    for L in layers:
        print(f"{L:>3} " + " ".join(f"{summ[f'{L}|{g}']['mean']:+10.2f}" if f"{L}|{g}" in summ else f"{'-':>10}"
                                    for g in groups))
    if args.split_hist:
        mean_sz = {g: float(np.mean([p["group_sizes"][g] for p in pairs])) for g in SPLIT}
        n_f0 = sum(p["group_sizes"]["hist_f"] == 0 for p in pairs)
        print(f"[pa] mean sub-group sizes (tokens): {mean_sz}; pairs with no later mention of f: {n_f0}")
    if args.split_hist2:
        print("[pa] mean split2 sizes (tokens): "
              + str({g: round(float(np.mean([p['group_sizes'][g] for p in pairs])), 2) for g in SPLIT2}))
        print("[pa] hist_f - hist_gk (pairs with a later mention of f and a matched control):")
        for L in layers:
            k = f"diff:{L}|f_minus_gk"
            if k in summ:
                print(f"  L{L:>2} {summ[k]['mean']:+.3f} [{summ[k]['lo']:+.3f}, {summ[k]['hi']:+.3f}] n={summ[k]['n']}")
    print(f"\n{'L':>3} " + " ".join(f"{n:>11}" for n in dnames))
    for L in dir_layers:
        print(f"{L:>3} " + " ".join(f"{summ[f'dir:{L}|{n}']['mean']:+11.2f}" if f"dir:{L}|{n}" in summ
                                    else f"{'-':>11}" for n in dnames))
    json.dump({"schema": "patching_action_v1", "groups": list(groups), "split_hist": bool(args.split_hist), "split_hist2": bool(args.split_hist2),
               "near": args.near, "model": args.model, "layers": layers, "dir_layers": dir_layers,
               "n_pairs": len(pairs), "n_train_pairs": n_train, "dropped": dropped,
               "dropped_train": drop_tr, "cos_dm_action": cos, "pairs": pairs, "summary": summ,
               "checks": problems, "wall_seconds": time.time() - t0}, open(args.out, "w"))
    print(f"[pa] wrote {args.out}")


if __name__ == "__main__":
    main()
