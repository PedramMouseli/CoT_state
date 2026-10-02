"""Activation patching with the state readout: position groups and single-direction patches."""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np

from cotstate.env import FILES
from cotstate.intervene import QUESTION, build_msgs
from cotstate.patching import flip_opening, untouched
from cotstate.util import chat

STEM = "{f} is currently"


def find_layers(model):
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers",
                 "model.model.layers"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            if len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise SystemExit("[p2] cannot find the decoder layer list on this model")


def find_final_norm(model):
    for path in ("model.norm", "model.language_model.norm", "language_model.model.norm"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    return None


def groups_for(tok, prompt: str, ids: list[int], s: int, f: str, question: str) -> dict:
    """Token-position groups for one prompt. Uses character offsets, so it does
    not depend on the chat template's special tokens."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] != ids:
        raise SystemExit("[p2] offset tokenisation differs from the prompt ids")
    offs = enc["offset_mapping"]
    n = len(ids)
    q_char = prompt.rfind(question)
    if q_char < 0:
        raise SystemExit("[p2] question text not found in the prompt")
    spans, k = [], prompt.find(f, q_char)
    while k >= 0:
        spans.append((k, k + len(f)))
        k = prompt.find(f, k + 1)
    hist, query, qfile = [], [], []
    for i in range(s + 1, n - 1):
        a, b = offs[i]
        if a < q_char:
            hist.append(i)
        else:
            query.append(i)
            if any(a < e and b > st for st, e in spans):
                qfile.append(i)
    return {"src": [s], "hist": hist, "query": query, "qfile": qfile,
            "mid": hist + query, "last": [n - 1]}


def cluster_ci(vals: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0):
    u = np.unique(clusters)
    idx = {c: np.where(clusters == c)[0] for c in u}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(u, len(u), replace=True)
        sel = np.concatenate([idx[c] for c in pick])
        bs.append(vals[sel].mean())
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--probes", required=True,
                    help="probe json with test_episodes (probe_v4 for Qwen3-8B; probe_v3 works "
                         "too - then the v4 direction is skipped if scaler_scale is missing)")
    ap.add_argument("--n-test-pairs", type=int, default=120)
    ap.add_argument("--n-train-pairs", type=int, default=200)
    ap.add_argument("--site-probe-n", type=int, default=1600,
                    help="training prompts for the site probes (spread over the four files)")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="position-group layers; default 0,2,...,nL-1")
    ap.add_argument("--dir-layers", type=int, nargs="*", default=None,
                    help="layers for direction patching; default every group layer >= 40%% depth")
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--min-sep", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=12000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dir-only", action="store_true",
                    help="skip position-group patching (except the final token, needed for the "
                         "checks); for re-running only the direction analysis")
    ap.add_argument("--save-dirs", default=None,
                    help="npz path: save the per-layer directions (dm_pool, nat_dm_pool, site_pool, ...)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.pilot:
        args.n_test_pairs, args.n_train_pairs, args.site_probe_n = 4, 20, 80

    import torch
    import cotstate
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    P = json.load(open(args.probes))
    te = set(P["test_episodes"])
    eps = [json.loads(l) for l in open(args.transcripts)]
    tr = set(range(len(eps))) - te

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    layers_mod = find_layers(model)
    nL = len(layers_mod)
    layers = args.layers or sorted(set(list(range(0, nL, 2)) + [nL - 1]))
    if args.pilot:
        layers = [nL // 2, (3 * nL) // 4, nL - 1]
    dir_layers = args.dir_layers or [L for L in layers if L >= int(0.4 * nL)]
    if args.pilot:
        dir_layers = layers
    cap_layers = sorted(set(layers) | set(dir_layers))

    tl = tok(" locked", add_special_tokens=False)["input_ids"]
    tu = tok(" unlocked", add_special_tokens=False)["input_ids"]
    ID_L, ID_U = tl[0], tu[0]
    if ID_L == ID_U:
        raise SystemExit("[p2] ' locked' and ' unlocked' share a first token")
    print(f"[p2] {args.model}: {nL} layers; ' locked'->{tl} ' unlocked'->{tu}; "
          f"group layers {layers}; direction layers {dir_layers}")

    W_U = model.get_output_embeddings().weight
    unembed = (W_U[ID_L].detach().float() - W_U[ID_U].detach().float())
    unembed = (unembed / unembed.norm()).to(args.device)
    d_model = int(W_U.shape[1])

    st = {"mode": None, "L": None, "pos": None, "src": None, "Q": None,
          "cap": None, "cap_last": None}

    def make_hook(L):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if st["mode"] == "capture":
                if st["cap"] is not None and L in st["cap"]:
                    st["cap"][L] = h[0].detach().clone()
                if st["cap_last"] is not None and L in st["cap_last"]:
                    st["cap_last"][L] = h[0, -1].detach().float().cpu().numpy()
                return out
            if st["mode"] == "patch" and st["L"] == L:
                h = h.clone()
                pos = st["pos"]
                if st["Q"] is None:
                    h[0, pos, :] = st["src"][pos, :].to(h.dtype)
                else:
                    diff = st["src"][pos, :].float() - h[0, pos, :].float()
                    proj = (diff @ st["Q"]) @ st["Q"].T
                    h[0, pos, :] = (h[0, pos, :].float() + proj).to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return out
        return hook

    for L in cap_layers:
        layers_mod[L].register_forward_hook(make_hook(L))

    def fwd(ids):
        x = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            try:
                out = model(input_ids=x, use_cache=False, logits_to_keep=1)
            except TypeError:
                out = model(input_ids=x, use_cache=False)
        lp = torch.log_softmax(out.logits[0, -1].float(), -1)
        return float(lp[ID_L] - lp[ID_U])

    def prompt_of(ep, t, f):
        return chat(tok, build_msgs(ep, t, f)[0], thinking=False) + STEM.format(f=f)

    def make_pair(ei, t, f):
        ep = eps[ei]
        truth = int(ep["turns"][t]["state_before"][f"locked_{f}"])
        epL, epU = (ep, flip_opening(ep, f)) if truth == 1 else (flip_opening(ep, f), ep)
        pa, pb = prompt_of(epL, t, f), prompt_of(epU, t, f)
        a = tok(pa, add_special_tokens=False)["input_ids"]
        b = tok(pb, add_special_tokens=False)["input_ids"]
        if len(a) != len(b):
            return None, "length"
        if len(a) > args.max_len:
            return None, "too_long"
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) != 1:
            return None, "diffpos"
        return (truth, pa, pb, a, b, diff[0]), None

    def candidates(pool, seed):
        c = [(ei, t["turn"], f) for ei in sorted(pool) for t in eps[ei]["turns"]
             if t["turn"] >= 2 for f in FILES if untouched(eps[ei], t["turn"], f)]
        random.Random(seed).shuffle(c)
        return c

    print(f"[p2] building {args.n_train_pairs} TRAINING pairs (episodes disjoint from test)")
    D = {L: [] for L in dir_layers}
    D_file = []
    dropped_tr = {"length": 0, "diffpos": 0, "sep": 0, "too_long": 0}
    for (ei, t, f) in candidates(tr, 5):
        if len(D_file) >= args.n_train_pairs:
            break
        pr, why = make_pair(ei, t, f)
        if pr is None:
            dropped_tr[why] += 1
            continue
        _, _, _, a, b, _ = pr
        st.update(mode="capture", cap=None, cap_last={L: None for L in dir_layers})
        xL = fwd(a)
        hL = st["cap_last"]
        st.update(cap_last={L: None for L in dir_layers})
        xU = fwd(b)
        hU = st["cap_last"]
        st.update(mode=None, cap_last=None)
        if xL - xU < args.min_sep:
            dropped_tr["sep"] += 1
            continue
        for L in dir_layers:
            D[L].append(hL[L] - hU[L])
        D_file.append(f)
    D = {L: np.array(v, np.float64) for L, v in D.items()}
    D_file = np.array(D_file)
    print(f"[p2] {len(D_file)} training pairs; dropped {dropped_tr} ({time.time() - t0:.0f}s)")

    print(f"[p2] fitting site probes on {args.site_probe_n} training prompts")
    pool = [(ei, t["turn"], f) for ei in sorted(tr) for t in eps[ei]["turns"]
            if t["turn"] >= 2 for f in FILES]
    random.Random(9).shuffle(pool)
    X = {L: [] for L in dir_layers}
    y, yf, yep = [], [], []
    for k, (ei, t, f) in enumerate(pool):
        if len(y) >= args.site_probe_n:
            break
        ids = tok(prompt_of(eps[ei], t, f), add_special_tokens=False)["input_ids"]
        if len(ids) > args.max_len:
            continue
        st.update(mode="capture", cap=None, cap_last={L: None for L in dir_layers})
        fwd(ids)
        for L in dir_layers:
            X[L].append(st["cap_last"][L])
        st.update(mode=None, cap_last=None)
        y.append(int(eps[ei]["turns"][t]["state_before"][f"locked_{f}"]))
        yf.append(f)
        yep.append(ei)
        if len(y) % 400 == 0:
            print(f"[p2]   {len(y)}/{args.site_probe_n} ({time.time() - t0:.0f}s)", flush=True)
    y, yf, yep = np.array(y), np.array(yf), np.array(yep)
    u_eps = np.unique(yep)
    np.random.RandomState(3).shuffle(u_eps)
    va = np.isin(yep, u_eps[: max(1, len(u_eps) // 5)])

    def fit_probe(Xl, yy, vmask):
        mu, sd = Xl[~vmask].mean(0), Xl[~vmask].std(0) + 1e-6
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
        clf.fit((Xl[~vmask] - mu) / sd, yy[~vmask])
        auc = (roc_auc_score(yy[vmask], clf.decision_function((Xl[vmask] - mu) / sd))
               if len(np.unique(yy[vmask])) == 2 else float("nan"))
        v = clf.coef_[0] / sd
        return v / np.linalg.norm(v), float(auc)

    def unit(v):
        v = np.asarray(v, np.float64)
        return v / (np.linalg.norm(v) + 1e-12)

    def v4_normal(f, L):
        e = P.get("probes", {}).get(f"locked_{f}", {}).get("layers", {}).get(str(L))
        if not e or "scaler_scale" not in e:
            return None
        return unit(np.array(e["w"], np.float64) / np.array(e["scaler_scale"], np.float64))

    basis = {L: {} for L in dir_layers}
    diag = {L: {} for L in dir_layers}
    rng = np.random.RandomState(1234)
    for L in dir_layers:
        Xl = np.array(X[L], np.float64)
        v, auc = fit_probe(Xl, y, va)
        basis[L]["site_pool"] = v[:, None]
        diag[L]["site_pool_val_auroc"] = auc
        basis[L]["site_file"], basis[L]["dm_file"], basis[L]["v4"] = {}, {}, {}
        for f in FILES:
            m = yf == f
            if m.sum() >= 40 and len(np.unique(y[m])) == 2:
                v, auc = fit_probe(Xl[m], y[m], va[m])
                basis[L]["site_file"][f] = v[:, None]
                diag[L][f"site_file_val_auroc:{f}"] = auc
            mf = D_file == f
            if mf.sum() >= 5:
                basis[L]["dm_file"][f] = unit(D[L][mf].mean(0))[:, None]
            u4 = v4_normal(f, L)
            if u4 is not None:
                basis[L]["v4"][f] = u4[:, None]
        basis[L]["dm_pool"] = unit(D[L].mean(0))[:, None]
        trm = ~va
        basis[L]["nat_dm_pool"] = unit(Xl[trm & (y == 1)].mean(0) - Xl[trm & (y == 0)].mean(0))[:, None]
        basis[L]["nat_dm_file"] = {}
        for f in FILES:
            m1, m0 = trm & (yf == f) & (y == 1), trm & (yf == f) & (y == 0)
            if m1.sum() >= 10 and m0.sum() >= 10:
                basis[L]["nat_dm_file"][f] = unit(Xl[m1].mean(0) - Xl[m0].mean(0))[:, None]
        _, S, Vt = np.linalg.svd(D[L], full_matrices=False)
        diag[L]["train_diff_sv_share"] = (S ** 2 / (S ** 2).sum())[:40].tolist()
        for k in args.ks:
            if k <= Vt.shape[0]:
                basis[L][f"svd_{k}"] = Vt[:k].T
        basis[L]["unembed"] = unembed.cpu().numpy().astype(np.float64)[:, None]
        basis[L]["rand_1"] = unit(rng.randn(d_model))[:, None]
        kmax = max(args.ks)
        basis[L][f"rand_{kmax}"] = np.linalg.qr(rng.randn(d_model, kmax))[0]
        c = {}
        sp, dp = basis[L]["site_pool"][:, 0], basis[L]["dm_pool"][:, 0]
        c["site_pool.dm_pool"] = float(sp @ dp)
        c["site_pool.unembed"] = float(sp @ basis[L]["unembed"][:, 0])
        c["dm_pool.unembed"] = float(dp @ basis[L]["unembed"][:, 0])
        for f in FILES:
            if f in basis[L]["v4"] and f in basis[L]["site_file"]:
                c[f"v4.site_file:{f}"] = float(basis[L]["v4"][f][:, 0] @ basis[L]["site_file"][f][:, 0])
            if f in basis[L]["v4"]:
                c[f"v4.dm_pool:{f}"] = float(basis[L]["v4"][f][:, 0] @ dp)
            if f in basis[L]["site_file"]:
                c[f"site_file.dm_pool:{f}"] = float(basis[L]["site_file"][f][:, 0] @ dp)
        c["nat_dm_pool.dm_pool"] = float(basis[L]["nat_dm_pool"][:, 0] @ dp)
        c["nat_dm_pool.site_pool"] = float(basis[L]["nat_dm_pool"][:, 0] @ sp)
        diag[L]["cos"] = c
        dec = {}
        vm = va
        for name in ("site_pool", "dm_pool", "nat_dm_pool", "unembed", "rand_1"):
            if len(np.unique(y[vm])) == 2:
                dec[name] = float(roc_auc_score(y[vm], Xl[vm] @ basis[L][name][:, 0]))
        for name in ("site_file", "dm_file", "nat_dm_file", "v4"):
            vals = []
            for f in FILES:
                b = basis[L][name].get(f)
                m = vm & (yf == f)
                if b is not None and len(np.unique(y[m])) == 2:
                    vals.append(roc_auc_score(y[m], Xl[m] @ b[:, 0]))
            if vals:
                dec[name] = float(np.mean(vals))
        diag[L]["decode_auroc"] = dec
        print(f"[p2] L{L}: site_pool val AUROC {diag[L]['site_pool_val_auroc']:.3f}; "
              f"cos(site_pool, dm_pool) {c['site_pool.dm_pool']:+.2f}; "
              f"top-1/4 SV share {sum(diag[L]['train_diff_sv_share'][:1]):.2f}/"
              f"{sum(diag[L]['train_diff_sv_share'][:4]):.2f}")
    torch_basis = {}

    def Q_of(L, name, f):
        b = basis[L][name]
        if isinstance(b, dict):
            b = b.get(f)
        if b is None:
            return None
        key = (L, name, f if isinstance(basis[L][name], dict) else "")
        if key not in torch_basis:
            torch_basis[key] = torch.tensor(b, dtype=torch.float32, device=args.device)
        return torch_basis[key]

    if args.save_dirs:
        np.savez_compressed(args.save_dirs, **{f"L{L}|{nm}": basis[L][nm] for L in dir_layers
                                               for nm in ("dm_pool", "nat_dm_pool", "site_pool", "unembed")},
                            **{f"L{L}|{nm}|{f}": v for L in dir_layers
                               for nm in ("dm_file", "nat_dm_file", "site_file", "v4")
                               for f, v in basis[L][nm].items()})
        print(f"[p2] saved directions to {args.save_dirs}")

    dir_names = ["v4", "site_file", "site_pool", "nat_dm_file", "nat_dm_pool", "dm_file", "dm_pool"] + \
                [f"svd_{k}" for k in args.ks] + ["unembed", "rand_1", f"rand_{max(args.ks)}", "self"]

    pairs = []
    dropped = {"length": 0, "diffpos": 0, "sep": 0, "too_long": 0}
    for (ei, t, f) in candidates(te, 11):
        if len(pairs) >= args.n_test_pairs:
            break
        pr, why = make_pair(ei, t, f)
        if pr is None:
            dropped[why] += 1
            continue
        truth, pa, pb, a, b, s = pr
        n = len(a)
        G = groups_for(tok, pb, b, s, f, QUESTION.format(f=f))
        st.update(mode="capture", cap={L: None for L in cap_layers}, cap_last=None)
        xL = fwd(a)
        capL = st["cap"]
        st.update(cap={L: None for L in dir_layers})
        xU = fwd(b)
        capU = st["cap"]
        st.update(mode=None, cap=None)
        if xL - xU < args.min_sep:
            dropped["sep"] += 1
            continue
        rec = {"episode": ei, "turn": t, "file": f, "truth": truth, "n_tokens": n, "src_pos": s,
               "group_sizes": {g: len(v) for g, v in G.items()},
               "x_locked": xL, "x_unlocked": xU, "patch": {}, "dir": {}, "frac": {}}
        for L in layers:
            for g in ("src", "hist", "query", "qfile", "mid", "last"):
                if not G[g]:
                    continue
                if args.dir_only and (g != "last" or (L not in dir_layers and L != nL - 1)):
                    continue
                st.update(mode="patch", L=L, pos=G[g], src=capL[L], Q=None)
                rec["patch"][f"{L}|{g}"] = fwd(b)
        for L in dir_layers:
            diff = (capL[L][n - 1].float() - capU[L][n - 1].float())
            dn2 = float(diff @ diff)
            for name in dir_names:
                if name == "self":
                    Q = (diff / diff.norm()).unsqueeze(1)
                else:
                    Q = Q_of(L, name, f)
                if Q is None:
                    continue
                st.update(mode="patch", L=L, pos=[n - 1], src=capL[L], Q=Q)
                rec["dir"][f"{L}|{name}"] = fwd(b)
                pr_ = diff @ Q
                rec["frac"][f"{L}|{name}"] = float(pr_ @ pr_) / max(dn2, 1e-12)
        st.update(mode=None, Q=None, src=None)
        del capL, capU
        torch.cuda.empty_cache()
        pairs.append(rec)
        print(f"[p2] pair {len(pairs)}/{args.n_test_pairs} {f} ep{ei} t{t} "
              f"sep {xL - xU:+.2f} hist {len(G['hist'])} query {len(G['query'])} "
              f"qfile {len(G['qfile'])} ({time.time() - t0:.0f}s)", flush=True)

    cl = np.array([p["episode"] for p in pairs])
    summ = {}

    def R_of(p, x):
        return (x - p["x_unlocked"]) / (p["x_locked"] - p["x_unlocked"])

    for L in layers:
        for g in ("src", "hist", "query", "qfile", "mid", "last"):
            key = f"{L}|{g}"
            m = np.array([key in p["patch"] for p in pairs])
            if m.sum() == 0:
                continue
            R = np.array([R_of(p, p["patch"][key]) for p in pairs if key in p["patch"]])
            lo, hi = cluster_ci(R, cl[m])
            summ[key] = {"mean": float(R.mean()), "lo": lo, "hi": hi, "n": int(m.sum())}
    for L in dir_layers:
        for name in dir_names:
            key = f"{L}|{name}"
            m = np.array([key in p["dir"] for p in pairs])
            if m.sum() == 0:
                continue
            R = np.array([R_of(p, p["dir"][key]) for p in pairs if key in p["dir"]])
            fr = np.array([p["frac"][key] for p in pairs if key in p["dir"]])
            lo, hi = cluster_ci(R, cl[m])
            summ["dir:" + key] = {"mean": float(R.mean()), "lo": lo, "hi": hi, "n": int(m.sum()),
                                  "frac_mean": float(fr.mean())}

    problems = []
    for L in dir_layers:
        a_, b_ = summ.get(f"{L}|last"), summ.get(f"dir:{L}|self")
        if a_ and b_ and abs(a_["mean"] - b_["mean"]) > 0.05:
            problems.append(f"L{L}: self-direction R {b_['mean']:.2f} != full last R {a_['mean']:.2f}")
    fin = summ.get(f"{nL - 1}|last")
    if fin and fin["mean"] < 0.9:
        problems.append(f"final-layer last-position R = {fin['mean']:.2f} (< 0.9)")
    print("\n[p2] CHECKS PASSED" if not problems else "\n[p2] CHECK FAILURES:\n  " + "\n  ".join(problems))

    print(f"\n[p2] {len(pairs)} held-out pairs from {len(set(cl))} episodes; dropped {dropped}")
    hdr = ("src", "hist", "query", "qfile", "last")
    print(f"{'L':>3} " + " ".join(f"{g:>7}" for g in hdr))
    for L in layers:
        print(f"{L:>3} " + " ".join(f"{summ[f'{L}|{g}']['mean']:+7.2f}" if f"{L}|{g}" in summ
                                    else f"{'-':>7}" for g in hdr))
    print(f"\n{'L':>3} {'direction':>10} {'R':>6} {'95% CI':>16} {'frac':>6}")
    for L in dir_layers:
        for name in dir_names:
            k = f"dir:{L}|{name}"
            if k in summ:
                r = summ[k]
                print(f"{L:>3} {name:>10} {r['mean']:+6.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] "
                      f"{r['frac_mean']:6.3f}")

    json.dump({"schema": "patching_v2", "model": args.model, "layers": layers,
               "dir_layers": dir_layers, "ks": args.ks, "n_pairs": len(pairs),
               "n_train_pairs": int(len(D_file)), "site_probe_n": int(len(y)),
               "dropped": dropped, "dropped_train": dropped_tr, "diag": diag,
               "pairs": pairs, "summary": summ, "checks": problems,
               "wall_seconds": time.time() - t0},
              open(args.out, "w"))
    print(f"\n[p2] wrote {args.out}")


if __name__ == "__main__":
    main()
