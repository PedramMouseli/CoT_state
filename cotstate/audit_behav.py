"""Violation rates per condition with episode-bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from cotstate.score_actions import ELIGIBILITY, score_episode


def cluster_bootstrap(per_ep: list[tuple[int, int]], n_boot: int = 10000,
                      seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for a ratio, resampling EPISODES with replacement.

    per_ep is [(violations, attempts), ...], one entry per episode.
    """
    if not per_ep:
        return float("nan"), float("nan")
    v = np.array([a for a, _ in per_ep], dtype=float)
    n = np.array([b for _, b in per_ep], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    num, den = v[idx].sum(1), n[idx].sum(1)
    ok = den > 0
    if not ok.any():
        return float("nan"), float("nan")
    r = num[ok] / den[ok]
    lo, hi = np.percentile(r, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_cluster_bootstrap(a: list[tuple[int, int]], b: list[tuple[int, int]],
                             n_boot: int = 10000, seed: int = 0,
                             alpha: float = 0.10) -> dict:
    """Difference of two rates, resampling the SAME episode indices in both.

    Use this only when the two conditions share episode seeds, which the
    generator guarantees within a model: episode i of think/body and episode i
    of think/full are the same world. Pairing removes the world-to-world
    variance, which is most of it.
    """
    m = min(len(a), len(b))
    if m == 0:
        return {}
    va = np.array([x for x, _ in a[:m]], float); na = np.array([y for _, y in a[:m]], float)
    vb = np.array([x for x, _ in b[:m]], float); nb = np.array([y for _, y in b[:m]], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    da = va[idx].sum(1); Da = na[idx].sum(1)
    db = vb[idx].sum(1); Db = nb[idx].sum(1)
    ok = (Da > 0) & (Db > 0)
    d = db[ok] / Db[ok] - da[ok] / Da[ok]
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = (vb.sum() / max(nb.sum(), 1)) - (va.sum() / max(na.sum(), 1))
    return {"diff_pp": 100 * float(point), "lo_pp": 100 * float(lo),
            "hi_pp": 100 * float(hi), "level": 1 - alpha, "n_episodes": m}


def analyse(path: str, eligibility: str) -> dict:
    eps = [json.loads(l) for l in open(path)]
    per_ep, n_turns, absent = [], 0, 0
    for ep in eps:
        rows = score_episode(ep, eligibility)
        n_turns += len(rows)
        absent += sum(r["absent"] for r in rows if r["attempt"])
        per_ep.append((sum(r["violation"] for r in rows),
                       sum(r["attempt"] for r in rows)))
    v = sum(a for a, _ in per_ep)
    n = sum(b for _, b in per_ep)
    lo, hi = cluster_bootstrap(per_ep)
    return {"episodes": len(eps), "turns": n_turns, "violations": v, "attempts": n,
            "rate": v / n if n else float("nan"),
            "cluster_lo": lo, "cluster_hi": hi,
            "attempts_per_episode": n / max(len(eps), 1),
            "absent_targets_among_attempts": absent, "per_episode": per_ep}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="name=dir (dir holds transcripts.jsonl)")
    ap.add_argument("--out", default="data/audit_behav.json")
    ap.add_argument("--csv", default="data/audit_behav.csv")
    ap.add_argument("--pairs", nargs="*", default=[],
                    help="a:b pairs to contrast with a paired episode bootstrap")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    res = {}
    for spec in args.specs:
        name, d = spec.split("=", 1)
        tp = os.path.join(d, "transcripts.jsonl")
        if not os.path.exists(tp):
            print(f"[ab] MISSING {tp}")
            continue
        res[name] = {"dir": d}
        for el in ELIGIBILITY:
            res[name][el] = analyse(tp, el)

    print(f"\n{'=' * 104}\nVIOLATION RATES  "
          f"(every turn scored from state_before; episode-clustered 95% CI)\n{'=' * 104}")
    hdr = ("{:<22}{:>5}{:>7} | {:>18}{:>8}{:>16}{:>8} | {:>16}{:>8}"
           .format("condition", "eps", "turns", "present viol/att", "rate",
                   "95% CI", "att/ep", "all viol/att", "rate"))
    print(hdr)
    for name, r in res.items():
        p, a = r["present"], r["all"]
        pf = "{}/{}".format(p["violations"], p["attempts"])
        af = "{}/{}".format(a["violations"], a["attempts"])
        ci = "[{:.1f}, {:.1f}]".format(100 * p["cluster_lo"], 100 * p["cluster_hi"])
        print("{:<22}{:>5}{:>7} | {:>18}{:>7.2f}%{:>16}{:>8.2f} | {:>16}{:>7.2f}%"
              .format(name, p["episodes"], p["turns"], pf, 100 * p["rate"], ci,
                      p["attempts_per_episode"], af, 100 * a["rate"]))

    contrasts = {}
    for pair in args.pairs:
        A, B = pair.split(":", 1)
        if A in res and B in res:
            for el in ELIGIBILITY:
                contrasts[f"{A}->{B}[{el}]"] = paired_cluster_bootstrap(
                    res[A][el]["per_episode"], res[B][el]["per_episode"],
                    n_boot=args.n_boot)
    if contrasts:
        print(f"\nPAIRED EPISODE-BOOTSTRAP CONTRASTS (90% CI on the rate difference, pp)")
        for k, c in contrasts.items():
            if c:
                print(f"  {k:<40} {c['diff_pp']:+7.3f} pp  "
                      f"[{c['lo_pp']:+.3f}, {c['hi_pp']:+.3f}]  over {c['n_episodes']} paired episodes")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    slim = {n: {el: {k: v for k, v in r[el].items() if k != "per_episode"}
                for el in ELIGIBILITY} | {"dir": r["dir"]} for n, r in res.items()}
    json.dump({"conditions": slim, "contrasts": contrasts}, open(args.out, "w"), indent=2)

    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "dir", "episodes", "turns",
                    "present_violations", "present_attempts", "present_rate_pct",
                    "present_ci_lo_pct", "present_ci_hi_pct", "attempts_per_episode",
                    "all_violations", "all_attempts", "all_rate_pct",
                    "absent_targets_among_all_attempts"])
        for n, r in res.items():
            p, a = r["present"], r["all"]
            w.writerow([n, r["dir"], p["episodes"], p["turns"],
                        p["violations"], p["attempts"], round(100 * p["rate"], 4),
                        round(100 * p["cluster_lo"], 4), round(100 * p["cluster_hi"], 4),
                        round(p["attempts_per_episode"], 3),
                        a["violations"], a["attempts"], round(100 * a["rate"], 4),
                        a["absent_targets_among_attempts"]])

    print(f"\n[ab] wrote {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
