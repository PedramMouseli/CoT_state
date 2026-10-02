"""Activation patching of the state readout and the direct action on identical pre-action prefixes."""

from __future__ import annotations

import argparse
import json
import random
import re
import time

import numpy as np

from cotstate.cf_action import cluster_ci, flip_prompt
from cotstate.fixed_history2 import QUESTION, STEM, frame, split_prefix
from cotstate.patching import LOCKWORD, TOUCH
from cotstate.patching2 import find_layers

LOGS = ("alpha.log", "beta.log", "gamma.log")
GROUPS_P = ("src", "hist", "cur_obs", "frame")


def untouched_pre(ep: dict, t: int, f: str) -> bool:
    """As cotstate.patching.untouched, but using only what precedes the action at turn t:
    observations through t, persisted replies and executed commands through t-1."""
    turns = ep["turns"]
    first = turns[0]["obs"].splitlines()
    texts = ["\n".join(l for l in first if not l.startswith(f + ":"))]
    texts += [x["obs"] for x in turns[1:t + 1]]
    texts += [x.get("persisted", "") or "" for x in turns[:t]]
    for txt in texts:
        for line in txt.splitlines():
            if f in line and LOCKWORD.search(line):
                return False
    for x in turns[:t]:
        parts = (x.get("cmd") or "").strip().split()
        if len(parts) >= 2 and parts[0].lower() in TOUCH and parts[1] == f:
            return False
    init = int(re.search(rf"^{re.escape(f)}: (locked|unlocked)\s*$", turns[0]["obs"], re.M).group(1) == "locked")
    return int(turns[t]["state_before"][f"locked_{f}"]) == init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--n-pairs", type=int, default=120)
    ap.add_argument("--min-sep", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.pilot:
        args.n_pairs = 4

    import torch
    import cotstate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    te = sorted(set(json.load(open(args.probes))["test_episodes"]))
    eps = [json.loads(l) for l in open(args.transcripts)]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    layers_mod = find_layers(model)
    nL = len(layers_mod)
    layers = sorted(set(args.layers))
    assert layers[-1] < nL
    ID_L = tok(" locked", add_special_tokens=False)["input_ids"][0]
    ID_U = tok(" unlocked", add_special_tokens=False)["input_ids"][0]

    st = {"mode": None, "L": None, "pos": None, "src": None, "cap": None}

    def make_hook(L):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if st["mode"] == "capture":
                st["cap"][L] = h[0].detach().clone()
                return out
            if st["mode"] == "patch" and st["L"] == L:
                h = h.clone()
                h[0, st["pos"], :] = st["src"][st["pos"], :].to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return out
        return hook

    for L in layers:
        layers_mod[L].register_forward_hook(make_hook(L))

    def run(ids, kind, n_cont):
        x = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            lg = model(input_ids=x, use_cache=False, logits_to_keep=n_cont + 1).logits[0]
        lp = torch.log_softmax(lg.float(), -1)
        if kind == "belief":
            return float(lp[-1, ID_L] - lp[-1, ID_U])
        cont = ids[-n_cont:]
        return float(sum(lp[k, c] for k, c in enumerate(cont)))

    def build(ei, t, f):
        pr = flip_prompt(eps[ei]["turns"][t]["prompt"], f)
        if pr is None:
            return None, "flip"
        P = [frame(p, "", "") for p in pr]
        ids_P = [tok(p, add_special_tokens=False)["input_ids"] for p in P]
        if len(ids_P[0]) != len(ids_P[1]):
            return None, "length"
        diff = [i for i in range(len(ids_P[0])) if ids_P[0][i] != ids_P[1][i]]
        if len(diff) != 1:
            return None, "diffpos"
        suf_a = tok("CMD:", add_special_tokens=False)["input_ids"]
        cont = tok(" mv " + f, add_special_tokens=False)["input_ids"]
        suf_b = tok(QUESTION.format(f=f) + "\n" + STEM.format(f=f), add_special_tokens=False)["input_ids"]
        if len(ids_P[0]) + len(suf_b) > args.max_len:
            return None, "too_long"
        joint_ok = (tok(P[1] + "CMD:", add_special_tokens=False)["input_ids"] == ids_P[1] + suf_a and
                    tok(P[1] + QUESTION.format(f=f) + "\n" + STEM.format(f=f),
                        add_special_tokens=False)["input_ids"] == ids_P[1] + suf_b)
        return dict(P=P, ids_P=ids_P, suf_a=suf_a, suf_b=suf_b, cont=cont, s=diff[0], joint_ok=joint_ok,
                    prefix_U=pr[1]), None

    def groups_P(b, obs):
        enc = tok(b["P"][1], add_special_tokens=False, return_offsets_mapping=True)
        assert enc["input_ids"] == b["ids_P"][1]
        offs = enc["offset_mapping"]
        frame_char = len(split_prefix(b["prefix_U"])[0])
        obs_char = b["P"][1].rfind(obs) if obs else -1
        if obs_char < 0 or obs_char >= frame_char:
            obs_char = frame_char
        G = {g: [] for g in GROUPS_P}
        G["src"] = [b["s"]]
        for i in range(b["s"] + 1, len(offs)):
            a = offs[i][0]
            G["hist" if a < obs_char else ("cur_obs" if a < frame_char else "frame")].append(i)
        return G

    cands = [(ei, t["turn"], f) for ei in te for t in eps[ei]["turns"]
             if t["turn"] >= 2 and "prompt" in t for f in LOGS
             if int(t["state_before"][f"exists_{f}"]) == 1 and untouched_pre(eps[ei], t["turn"], f)]
    n_cands = len(cands)
    random.Random(args.seed).shuffle(cands)
    print(f"[pm] {args.model}: {nL} layers, patch layers {layers}; {n_cands} pre-action candidates "
          f"from {len(te)} test episodes", flush=True)

    pairs, screened, dropped = [], [], {}
    for (ei, t, f) in cands:
        if len(pairs) >= args.n_pairs:
            break
        b, why = build(ei, t, f)
        if b is None:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        G = groups_P(b, eps[ei]["turns"][t]["obs"])
        nP = len(b["ids_P"][0])
        rd = {}
        for kind, suf, nc in (("action", b["suf_a"], len(b["cont"])), ("belief", b["suf_b"], 0)):
            idsL = b["ids_P"][0] + suf + (b["cont"] if kind == "action" else [])
            idsU = b["ids_P"][1] + suf + (b["cont"] if kind == "action" else [])
            st.update(mode="capture", cap={})
            xL = run(idsL, kind, nc)
            capL = st["cap"]
            st.update(mode=None, cap=None)
            xU = run(idsU, kind, nc)
            Gk = dict(G)
            Gk["suffix"] = list(range(nP, nP + len(suf) - 1))
            Gk["last"] = [nP + len(suf) - 1]
            if kind == "action":
                Gk["cont"] = list(range(nP + len(suf), nP + len(suf) + nc))
            rd[kind] = dict(idsU=idsU, xL=xL, xU=xU, cap=capL, G=Gk, nc=nc)
        sepA = rd["action"]["xU"] - rd["action"]["xL"]
        sepB = rd["belief"]["xL"] - rd["belief"]["xU"]
        screened.append({"episode": ei, "turn": t, "file": f, "sep_action": sepA, "sep_belief": sepB})
        if sepA < args.min_sep or sepB < args.min_sep:
            dropped["sep"] = dropped.get("sep", 0) + 1
            continue
        rec = {"episode": ei, "turn": t, "file": f, "joint_tokenisation_ok": b["joint_ok"],
               "n_prefix": nP, "group_sizes": {k: len(v) for k, v in rd["action"]["G"].items()},
               "belief_group_sizes": {k: len(v) for k, v in rd["belief"]["G"].items()}}
        for kind, r in rd.items():
            rec[kind] = {"x_locked": r["xL"], "x_unlocked": r["xU"], "patch": {}}
            for L in layers:
                for g, pos in r["G"].items():
                    if pos:
                        st.update(mode="patch", L=L, pos=pos, src=r["cap"][L])
                        rec[kind]["patch"][f"{L}|{g}"] = run(r["idsU"], kind, r["nc"])
            st.update(mode=None, src=None)
        del rd
        torch.cuda.empty_cache()
        pairs.append(rec)
        print(f"[pm] pair {len(pairs)}/{args.n_pairs} {f} ep{ei} t{t} sepA {sepA:+.2f} sepB {sepB:+.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    cl = np.array([p["episode"] for p in pairs])
    summ, problems = {}, []
    groups_all = GROUPS_P + ("suffix", "last", "cont")
    for kind in ("action", "belief"):
        for L in layers:
            for g in groups_all:
                k = f"{L}|{g}"
                vals = [(p[kind]["patch"][k] - p[kind]["x_unlocked"]) / (p[kind]["x_locked"] - p[kind]["x_unlocked"])
                        for p in pairs if k in p[kind]["patch"]]
                if vals:
                    m = np.array([k in p[kind]["patch"] for p in pairs])
                    lo, hi = cluster_ci(vals, cl[m])
                    summ[f"{kind}:{k}"] = {"mean": float(np.mean(vals)), "lo": lo, "hi": hi, "n": len(vals)}
        fin = summ.get(f"{kind}:{nL - 1}|src")
        if fin and abs(fin["mean"]) > 0.1:
            problems.append(f"{kind}: final-layer src R = {fin['mean']:.2f} (should be ~0)")
    for L in layers:
        k = f"{L}|hist"
        d = [(p["action"]["patch"][k] - p["action"]["x_unlocked"]) / (p["action"]["x_locked"] - p["action"]["x_unlocked"])
             - (p["belief"]["patch"][k] - p["belief"]["x_unlocked"]) / (p["belief"]["x_locked"] - p["belief"]["x_unlocked"])
             for p in pairs if k in p["action"]["patch"] and k in p["belief"]["patch"]]
        if d:
            lo, hi = cluster_ci(d, cl[: len(d)] if len(d) == len(pairs) else
                                np.array([p["episode"] for p in pairs if k in p["action"]["patch"]]))
            summ[f"diff_action_minus_belief:{k}"] = {"mean": float(np.mean(d)), "lo": lo, "hi": hi, "n": len(d)}
    n_joint_bad = sum(not p["joint_tokenisation_ok"] for p in pairs)
    if n_joint_bad:
        problems.append(f"{n_joint_bad} pairs where separate != joint suffix tokenisation (ids used: separate)")
    print("\n[pm] CHECKS PASSED" if not problems else "\n[pm] CHECK NOTES:\n  " + "\n  ".join(problems))
    print(f"[pm] {len(pairs)} pairs from {len(set(cl))} episodes; screened {len(screened)}; dropped {dropped}")
    for kind in ("action", "belief"):
        print(f"\n[{kind}] R by layer and group")
        print(f"{'L':>3} " + " ".join(f"{g:>8}" for g in groups_all))
        for L in layers:
            print(f"{L:>3} " + " ".join(f"{summ[f'{kind}:{L}|{g}']['mean']:+8.2f}" if f"{kind}:{L}|{g}" in summ
                                        else f"{'-':>8}" for g in groups_all))
    print("\n[pm] R(hist): action - belief, paired")
    for L in layers:
        s = summ.get(f"diff_action_minus_belief:{L}|hist")
        if s:
            print(f"  L{L:>2} {s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] n={s['n']}")
    json.dump({"schema": "patching_matched_v1", "model": args.model, "transcripts": args.transcripts,
               "mode": "immediate", "persist": "body", "format": "chat", "layers": layers,
               "min_sep": args.min_sep, "seed": args.seed, "n_candidates": n_cands,
               "eligibility": "pre-action (obs through t, replies/commands through t-1); log files only",
               "n_pairs": len(pairs), "dropped": dropped, "screened": screened, "pairs": pairs,
               "summary": summ, "checks": problems, "wall_seconds": time.time() - t0}, open(args.out, "w"))
    print(f"[pm] wrote {args.out}")


if __name__ == "__main__":
    main()
