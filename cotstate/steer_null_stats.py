"""Summary statistics for the steering experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re

import numpy as np

KEY = re.compile(r"^L(\d+)\|(.+)\|([+-])([\d.]+)$")


def cluster_ci(delta: np.ndarray, ep: np.ndarray, n_boot: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    u = np.unique(ep)
    idx_of = {e: np.where(ep == e)[0] for e in u}
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(u, len(u))
        bs.append(delta[np.concatenate([idx_of[e] for e in pick])].mean())
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    d = json.load(open(args.path))
    raw = {k: np.asarray(v, float) for k, v in d["raw"].items()}
    x0 = raw["clean"]
    thr = float(np.median(x0))
    y = np.array([it["truth"] for it in d["items"]])
    ep = np.array([it["episode"] for it in d["items"]])
    c0 = x0 <= thr
    coll = {k: np.asarray(v, float) for k, v in d.get("collateral", {}).items()}
    others = sorted({k.split("|")[0] for k in coll})

    cells = {}
    for k in raw:
        m = KEY.match(k)
        if not m:
            continue
        L, dn, sg, a = int(m.group(1)), m.group(2), m.group(3), float(m.group(4))
        cells.setdefault((L, a), {}).setdefault(dn, {})[sg] = raw[k]

    print(f"\n{'=' * 110}\nSTEERING WITH A NULL   {d['target']}   site={d['site']}   "
          f"K={d['n_random']} random directions   {len(y)} items / {len(set(ep.tolist()))} episodes\n"
          f"{'=' * 110}")
    if d.get("reference_check"):
        print(f"reference check: {d['reference_check']}")
    for L in d["layers"]:
        uc = d["unembed_cos"][str(L)] if str(L) in d["unembed_cos"] else d["unembed_cos"][L]
        sp = d.get("site_probe", {}).get(str(L)) or d.get("site_probe", {}).get(L)
        extra = (f"; site probe val {sp['val_bal_acc']:.3f}, test {sp.get('test_bal_acc', float('nan')):.3f}"
                 if sp else "")
        rn = d["resid_norm_acts"].get(str(L), d["resid_norm_acts"].get(L))
        sn = d["site_norm_measured"].get(str(L), d["site_norm_measured"].get(L))
        print(f"L{L}: |resid| at probe point {rn:.1f}, at steered token {sn:.1f}; "
              f"cos(probe, unembed diff) {uc['probe']['with_gain']:+.3f}{extra}")

    rows = []
    hdr = (f"\n{'L':>3} {'alpha':>5} {'ratio':>6} {'dir':<10} {'shift+':>8} {'95% CI (episodes)':>20} "
           f"{'null mean':>9} {'null max':>8} {'p_rank':>7} {'up':>6} {'null up':>8} "
           f"{'best other':>10} {'collat':>7} {'antisym':>8} {'AUROC+':>7}  verdict")
    print(hdr)
    for (L, a) in sorted(cells):
        c = cells[(L, a)]
        rnd = [dn for dn in c if dn.startswith("random:")]
        null_shift = np.array([(c[dn]["+"] - x0).mean() for dn in rnd])
        null_up = np.array([((c[dn]["+"] > thr) & c0).sum() / max(c0.sum(), 1) for dn in rnd])
        oth = [dn for dn in c if dn.startswith("other:")]
        best_other = max(((c[dn]["+"] - x0).mean() for dn in oth), default=float("nan"))
        rn = d["resid_norm_acts"].get(str(L), d["resid_norm_acts"].get(L))
        sn = d["site_norm_measured"].get(str(L), d["site_norm_measured"].get(L))
        ratio = a * rn / np.sqrt(d["d_model"]) / sn
        for dn in [x for x in ("probe", "siteprobe") if x in c]:
            xp = c[dn]["+"]
            delta = xp - x0
            sh = float(delta.mean())
            lo, hi = cluster_ci(delta, ep)
            p = (1 + int((null_shift >= sh).sum())) / (len(null_shift) + 1)
            up = float(((xp > thr) & c0).sum() / max(c0.sum(), 1))
            anti = float(((c[dn]["+"] - c[dn]["-"]) / 2).mean()) if "-" in c[dn] else float("nan")
            col = []
            for g in others:
                kk = f"{g}|L{L}|{dn}|+{a:g}"
                if kk in coll:
                    col.append((coll[kk] - coll[f"{g}|clean"]).mean())
            collat = float(np.mean(col)) if col else float("nan")
            ok1 = sh > 0
            ok2 = p <= 0.05
            ok3 = not np.isfinite(best_other) or sh > best_other
            ok4 = not np.isfinite(collat) or collat < 0.5 * sh
            if not ok1:
                verdict = "no push toward LOCKED"
            elif ok1 and ok2 and ok3 and ok4:
                verdict = "SPECIFIC LEVER"
            elif ok1 and ok2:
                verdict = "moves the answer, not file-specific" if not ok4 else \
                          "beats random, not other files"
            else:
                verdict = "within the random null"
            print(f"{L:>3} {a:>5g} {ratio:>6.2f} {dn:<10} {sh:>+8.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>20} "
                  f"{null_shift.mean():>+9.2f} {null_shift.max():>+8.2f} {p:>7.3f} {up:>6.3f} "
                  f"{np.mean(null_up):>8.3f} {best_other:>+10.2f} {collat:>+7.2f} {anti:>+8.2f} "
                  f"{auroc(y, xp):>7.3f}  {verdict}")
            rows.append({"layer": L, "alpha": a, "norm_ratio_at_site": round(float(ratio), 4),
                         "direction": dn, "shift_pos": sh, "ci_lo": lo, "ci_hi": hi,
                         "null_mean": float(null_shift.mean()), "null_sd": float(null_shift.std()),
                         "null_max": float(null_shift.max()), "p_rank": p,
                         "intended_up": up, "null_up_mean": float(np.mean(null_up)),
                         "best_other_shift": best_other, "collateral_shift": collat,
                         "antisym_signed": anti, "auroc_plus": auroc(y, xp),
                         "verdict": verdict})

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[sns] wrote {args.csv}")


if __name__ == "__main__":
    main()
