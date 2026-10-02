"""Steering along probe directions with a random-direction null."""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np

from cotstate.env import FILES
from cotstate.intervene import build_msgs
from cotstate.util import chat

STEM = "{f} is currently"


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def select_items(eps, te, target, n_items):
    """Byte-for-byte the item selection of cotstate/intervene.py, so results line up."""
    items = []
    for ei, ep in enumerate(eps):
        if ei not in te:
            continue
        for t in ep["turns"]:
            if t["turn"] >= 2:
                items.append((ei, t["turn"], t["state_before"][target]))
    random.Random(0).shuffle(items)
    return items[:n_items]


def site_positions(tok, ids: list[int], f: str, site: str) -> list[int]:
    """Absolute token positions to steer.

    last  the final token of the prompt (the ' currently' of the stem).
    file  the tokens of the file name inside the appended stem, i.e. a position
          the readout must ATTEND to, not the position it reads from.
    """
    n = len(ids)
    if site == "last":
        return [n - 1]
    stem = STEM.format(f=f)
    k0 = None
    for k in range(1, min(40, n) + 1):
        if tok.decode(ids[n - k:]).lstrip().startswith(stem.split()[0]):
            if stem in tok.decode(ids[n - k:]):
                k0 = n - k
                break
    if k0 is None:
        raise SystemExit(f"[sn] could not locate the stem '{stem}' at the end of the prompt")
    pos = []
    for p in range(k0, n):
        pos.append(p)
        if f in tok.decode(ids[k0:p + 1]):
            break
    return pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--probes", required=True, help="probe_v4.json (must contain scaler_scale)")
    ap.add_argument("--acts", required=True, help="acts.npz, for the residual-norm scale")
    ap.add_argument("--target", default="locked_beta.log")
    ap.add_argument("--layers", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--alphas", type=float, nargs="+", default=[16, 32, 64])
    ap.add_argument("--n-random", type=int, default=24,
                    help="K random directions. The smallest attainable rank p is 1/(K+1); "
                         "K>=19 is needed to be able to reach p<=0.05")
    ap.add_argument("--random-both-signs", action="store_true",
                    help="also run -alpha for random directions (doubles their cost)")
    ap.add_argument("--site", choices=["last", "file"], default="last")
    ap.add_argument("--collateral", action="store_true")
    ap.add_argument("--fit-site-probe", type=int, default=0,
                    help="N training prompts to fit a probe at the steered site; 0 = off")
    ap.add_argument("--n-items", type=int, default=300)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--reference", default=None,
                    help="an intervene_*.json on the same target; clean readouts are compared")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import cotstate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    P = json.load(open(args.probes))
    te = set(P["test_episodes"])
    eps = [json.loads(l) for l in open(args.transcripts)]
    target = args.target
    f = target[len("locked_"):]
    others = [g for g in FILES if g != f]

    items = select_items(eps, te, target, args.pilot or args.n_items)
    truths = np.array([y for _, _, y in items])
    print(f"[sn] {len(items)} items from {len({e for e, _, _ in items})} episodes, "
          f"base rate {max(truths.mean(), 1 - truths.mean()):.3f}")

    Xacts = np.load(args.acts, allow_pickle=True)["X"]
    d_model = int(Xacts.shape[-1])
    resid_norm = {L: float(np.linalg.norm(Xacts[:, L, :].astype(np.float32), axis=1).mean())
                  for L in args.layers}
    del Xacts

    def normal(var, L):
        e = P["probes"][var]["layers"].get(str(L))
        if e is None or "scaler_scale" not in e:
            return None
        return unit(np.array(e["w"], np.float64) / np.array(e["scaler_scale"], np.float64))

    dirs = {L: {} for L in args.layers}
    for L in args.layers:
        u = normal(target, L)
        if u is None:
            raise SystemExit(f"[sn] {args.probes} has no scaler_scale for {target} at L{L}; "
                             "re-run the patched probe.py")
        dirs[L]["probe"] = u
        for g in others:
            ug = normal(f"locked_{g}", L)
            if ug is not None:
                dirs[L][f"other:{g}"] = ug
        for k in range(1, args.n_random + 1):
            dirs[L][f"random:{k}"] = unit(np.random.RandomState(1000 + k).randn(d_model))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    if "qwen3_5" in str(getattr(model.config, "model_type", "")).lower():
        raise SystemExit("[sn] Qwen3.5 linear-attention caches cannot be rewound; Qwen3-8B only")
    ID_L = tok(" locked", add_special_tokens=False)["input_ids"][0]
    ID_U = tok(" unlocked", add_special_tokens=False)["input_ids"][0]

    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    gain = model.model.norm.weight.detach().float().cpu().numpy()
    ud = W_U[ID_L] - W_U[ID_U]
    ud_raw, ud_gain = unit(ud), unit(ud * gain)
    unembed_cos = {L: {name: {"raw": float(v @ ud_raw), "with_gain": float(v @ ud_gain)}
                       for name, v in dirs[L].items() if not name.startswith("random:")}
                   for L in args.layers}
    for L in args.layers:
        print(f"[sn] L{L}: cos(probe, W_U[locked]-W_U[unlocked]) = "
              f"{unembed_cos[L]['probe']['raw']:+.3f} (raw), "
              f"{unembed_cos[L]['probe']['with_gain']:+.3f} (with final-norm gain)")

    state = {"L": None, "vec": None, "pos": [], "capture": None}

    def make_hook(L):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if state["capture"] is not None and L in state["capture"]:
                state["capture"][L] = h[0, state["pos"], :].detach().float().mean(0).cpu().numpy()
            if state["vec"] is not None and state["L"] == L:
                h = h.clone()
                h[0, state["pos"], :] += state["vec"].to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return out
        return hook

    for L in args.layers:
        model.model.layers[L].register_forward_hook(make_hook(L))

    def prompt_ids(ei, ti, g):
        p = chat(tok, build_msgs(eps[ei], ti, g)[0], thinking=False) + STEM.format(f=g)
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) > args.max_len:
            raise SystemExit(f"[sn] prompt {len(ids)} > --max-len")
        return ids

    def fwd(**kw):
        """Forward pass that only materialises the last position's logits.
        A full 8k x 152k logit tensor is ~2.4 GB in bf16 and is never needed."""
        try:
            return model(**kw, logits_to_keep=1)
        except TypeError:
            try:
                return model(**kw, num_logits_to_keep=1)
            except TypeError:
                return model(**kw)

    def readout(logits):
        lp = torch.log_softmax(logits.float(), -1)
        return float(lp[ID_L] - lp[ID_U])

    class Prefixed:
        """Run the prefix once, then any number of tail passes."""

        def __init__(self, ids, pos):
            self.ids, self.pos_abs = ids, pos
            self.p0 = min(pos)
            self.rel = [p - self.p0 for p in pos]
            x = torch.tensor([ids[: self.p0]], device=args.device)
            state["vec"], state["capture"] = None, None
            with torch.no_grad():
                out = fwd(input_ids=x, use_cache=True)
            self.cache = out.past_key_values
            if not hasattr(self.cache, "crop"):
                raise SystemExit("[sn] this transformers version returns a legacy tuple "
                                 "cache without .crop(); upgrade transformers (>=4.46)")
            self.tail = torch.tensor([ids[self.p0:]], device=args.device)

        def run(self, L=None, vec=None, capture=False):
            state["L"], state["pos"] = L, self.rel
            state["vec"] = None if vec is None else torch.tensor(vec, device=args.device)
            state["capture"] = ({l: None for l in args.layers} if capture else None)
            with torch.no_grad():
                out = fwd(input_ids=self.tail, past_key_values=self.cache, use_cache=True)
            self.cache.crop(self.p0)
            cap = state["capture"]
            state["vec"], state["capture"] = None, None
            return readout(out.logits[0, -1]), cap

    def full_clean(ids, pos, capture_layers=None):
        state["vec"] = None
        state["pos"] = pos
        state["capture"] = ({l: None for l in capture_layers} if capture_layers else None)
        with torch.no_grad():
            out = fwd(input_ids=torch.tensor([ids], device=args.device))
        cap = state["capture"]
        state["capture"] = None
        return readout(out.logits[0, -1]), cap

    site_probe = {}
    if args.fit_site_probe:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        pool = [(ei, t["turn"]) for ei, ep in enumerate(eps) if ei not in te
                for t in ep["turns"] if t["turn"] >= 2]
        random.Random(7).shuffle(pool)
        pool = pool[: args.fit_site_probe]
        print(f"[sn] fitting site probe on {len(pool)} TRAINING prompts at site={args.site}")
        feats = {L: [] for L in args.layers}
        ys, eps_of = [], []
        for k, (ei, ti) in enumerate(pool):
            ids = prompt_ids(ei, ti, f)
            _, cap = full_clean(ids, site_positions(tok, ids, f, args.site), args.layers)
            for L in args.layers:
                feats[L].append(cap[L])
            ys.append(eps[ei]["turns"][ti]["state_before"][target])
            eps_of.append(ei)
            if k % 200 == 0:
                print(f"[sn]   {k}/{len(pool)}  ({time.time() - t0:.0f}s)", flush=True)
        ys, eps_of = np.array(ys), np.array(eps_of)
        u_eps = np.unique(eps_of)
        np.random.RandomState(3).shuffle(u_eps)
        va = np.isin(eps_of, u_eps[: max(1, len(u_eps) // 5)])
        for L in args.layers:
            X = np.array(feats[L], np.float64)
            mu, sd = X[~va].mean(0), X[~va].std(0) + 1e-6
            clf = LogisticRegression(max_iter=3000, class_weight="balanced")
            clf.fit((X[~va] - mu) / sd, ys[~va])
            acc_va = balanced_accuracy_score(ys[va], clf.predict((X[va] - mu) / sd))
            w = clf.coef_[0]
            site_probe[L] = {"w": w.tolist(), "intercept": float(clf.intercept_[0]),
                             "scaler_mean": mu.tolist(), "scaler_scale": sd.tolist(),
                             "val_bal_acc": float(acc_va), "n_train": int((~va).sum()),
                             "n_val": int(va.sum())}
            dirs[L]["siteprobe"] = unit(w / sd)
            unembed_cos[L]["siteprobe"] = {"raw": float(dirs[L]["siteprobe"] @ ud_raw),
                                           "with_gain": float(dirs[L]["siteprobe"] @ ud_gain)}
            print(f"[sn] site probe L{L}: validation bal-acc {acc_va:.3f}; "
                  f"cos with v4 probe normal {float(dirs[L]['siteprobe'] @ dirs[L]['probe']):+.3f}")

    conds = []
    for L in args.layers:
        for dn in dirs[L]:
            both = dn in ("probe", "siteprobe") or dn.startswith("other:") or \
                (dn.startswith("random:") and args.random_both_signs)
            for a in args.alphas:
                conds.append((f"L{L}|{dn}|+{a:g}", L, dn, +a))
                if both:
                    conds.append((f"L{L}|{dn}|-{a:g}", L, dn, -a))
    print(f"[sn] {len(conds)} conditions per item, {len(args.layers)} layers")

    raw = {"clean": []}
    raw.update({c[0]: [] for c in conds})
    coll = {}
    site_feats = {L: [] for L in args.layers}
    site_norm = {L: [] for L in args.layers}
    checks = []
    for k, (ei, ti, _) in enumerate(items):
        ids = prompt_ids(ei, ti, f)
        pos = site_positions(tok, ids, f, args.site)
        pf = Prefixed(ids, pos)
        x0, cap = pf.run(capture=True)
        raw["clean"].append(x0)
        for L in args.layers:
            site_feats[L].append(cap[L])
            site_norm[L].append(float(np.linalg.norm(cap[L])))
        if k < 3:
            xf, _ = full_clean(ids, pos)
            checks.append(abs(xf - x0))
            if abs(xf - x0) > 0.5:
                raise SystemExit(f"[sn] cached clean {x0:+.4f} != full forward {xf:+.4f}; "
                                 "the KV-cache path is wrong, stop")
        for name, L, dn, a in conds:
            scale = a * resid_norm[L] / np.sqrt(d_model)
            x, _ = pf.run(L, dirs[L][dn] * scale)
            raw[name].append(x)
        if args.collateral:
            for g in others:
                ids_g = prompt_ids(ei, ti, g)
                pf_g = Prefixed(ids_g, site_positions(tok, ids_g, g, args.site))
                xg0, _ = pf_g.run()
                coll.setdefault(f"{g}|clean", []).append(xg0)
                for L in args.layers:
                    for dn in [d for d in ("probe", "siteprobe") if d in dirs[L]]:
                        for a in args.alphas:
                            scale = a * resid_norm[L] / np.sqrt(d_model)
                            xg, _ = pf_g.run(L, dirs[L][dn] * scale)
                            coll.setdefault(f"{g}|L{L}|{dn}|+{a:g}", []).append(xg)
                del pf_g
        del pf
        torch.cuda.empty_cache()
        if k % 10 == 0:
            el = time.time() - t0
            print(f"[sn] item {k + 1}/{len(items)}  {el:.0f}s  "
                  f"(~{el / (k + 1) * (len(items) - k - 1) / 60:.0f} min left)", flush=True)

    if site_probe:
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        for L in args.layers:
            sp = site_probe[L]
            X = (np.array(site_feats[L]) - np.array(sp["scaler_mean"])) / np.array(sp["scaler_scale"])
            s = X @ np.array(sp["w"]) + sp["intercept"]
            sp["test_bal_acc"] = float(balanced_accuracy_score(truths, (s > 0).astype(int)))
            sp["test_auroc"] = float(roc_auc_score(truths, s))
            print(f"[sn] site probe L{L}: TEST bal-acc {sp['test_bal_acc']:.3f}, "
                  f"AUROC {sp['test_auroc']:.3f} on the {len(truths)} evaluation items")

    ref_check = None
    if args.reference:
        R = json.load(open(args.reference))
        rc = np.array(R["raw"]["clean"][: len(items)], float)
        x0 = np.array(raw["clean"], float)
        same_truths = list(R["truths"][: len(items)]) == truths.tolist()
        ref_check = {"same_item_truths": bool(same_truths),
                     "corr": float(np.corrcoef(rc, x0)[0, 1]),
                     "max_abs_diff": float(np.abs(rc - x0).max()),
                     "mean_abs_diff": float(np.abs(rc - x0).mean())}
        print(f"[sn] reference check vs {args.reference}: {ref_check}")
        if not same_truths or ref_check["corr"] < 0.98:
            print("[sn] WARNING: items or clean readouts do not match the reference run. "
                  "Do not pool these results with it until this is explained.")

    out = {
        "schema": "steer_null_v1", "model": args.model, "target": target, "file": f,
        "site": args.site, "layers": args.layers, "alphas": args.alphas,
        "n_random": args.n_random, "d_model": d_model,
        "resid_norm_acts": resid_norm,
        "site_norm_measured": {L: float(np.mean(v)) for L, v in site_norm.items()},
        "norm_note": "perturbation norm = alpha * resid_norm_acts / sqrt(d_model). "
                     "Divide by site_norm_measured for the ratio at the steered token.",
        "items": [{"episode": int(e), "turn": int(t), "truth": int(y)} for e, t, y in items],
        "unembed_cos": unembed_cos, "site_probe": site_probe,
        "cached_vs_full_max_abs": float(max(checks)) if checks else None,
        "reference_check": ref_check,
        "raw": raw, "collateral": coll,
        "wall_seconds": time.time() - t0,
    }
    json.dump(out, open(args.out, "w"))
    print(f"\n[sn] wrote {args.out}  ({out['wall_seconds'] / 60:.1f} min). "
          f"Analyse with:  python -m cotstate.steer_null_stats {args.out}")


if __name__ == "__main__":
    main()
