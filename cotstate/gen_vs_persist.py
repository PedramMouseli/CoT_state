"""Generation and persistence contrasts, paired by episode seed, and their ratio."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from cotstate.score_actions import score_episode


def per_episode(path: str, eligibility: str) -> dict:
    out = {}
    for line in open(os.path.join(path, "transcripts.jsonl")):
        ep = json.loads(line)
        rows = score_episode(ep, eligibility)
        key = ep.get("seed", ep.get("episode"))
        out[key] = (sum(r["violation"] for r in rows), sum(r["attempt"] for r in rows))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terse", required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--body", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--eligibility", default="present", choices=["present", "all"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conds = {"terse": args.terse, "cmd": args.cmd, "full": args.full}
    if args.body:
        conds["body"] = args.body
    D = {k: per_episode(v, args.eligibility) for k, v in conds.items()}
    keys = sorted(set.intersection(*[set(d) for d in D.values()]))
    if len(keys) < min(len(d) for d in D.values()):
        print(f"[gp] WARNING: only {len(keys)} episode seeds are shared by all conditions; "
              "pairing is restricted to those")
    V = {k: np.array([D[k][e][0] for e in keys], float) for k in D}
    N = {k: np.array([D[k][e][1] for e in keys], float) for k in D}

    def rates(idx):
        return {k: V[k][idx].sum() / max(N[k][idx].sum(), 1) for k in D}

    full_idx = np.arange(len(keys))
    r = rates(full_idx)
    gen = r["terse"] - r["cmd"]
    per = r["cmd"] - r["full"]
    rng = np.random.default_rng(0)
    G, Pp, R = [], [], []
    for _ in range(args.n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        rb = rates(idx)
        g, p = rb["terse"] - rb["cmd"], rb["cmd"] - rb["full"]
        G.append(g); Pp.append(p); R.append(p / g if g > 0 else np.nan)
    G, Pp, R = map(np.array, (G, Pp, R))
    R = R[np.isfinite(R)]

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    res = {"label": args.label, "eligibility": args.eligibility, "n_paired_episodes": len(keys),
           "rates": {k: float(v) for k, v in r.items()},
           "counts": {k: [int(V[k].sum()), int(N[k].sum())] for k in D},
           "generation_effect_pp": 100 * gen, "generation_ci_pp": [100 * x for x in ci(G)],
           "persistence_effect_pp": 100 * per, "persistence_ci_pp": [100 * x for x in ci(Pp)],
           "ratio_persist_over_gen": per / gen if gen > 0 else None,
           "ratio_ci": ci(R), "ratio_upper95_one_sided": float(np.percentile(R, 95))}
    json.dump(res, open(args.out, "w"), indent=2)

    print(f"\n{'=' * 78}\nGENERATION vs PERSISTENCE   {args.label}   "
          f"({len(keys)} paired episodes, {args.eligibility} eligibility)\n{'=' * 78}")
    for k in D:
        print(f"  {k:<6} {int(V[k].sum()):>4}/{int(N[k].sum()):<5} = {100 * r[k]:6.2f}%")
    print(f"\n  generation  effect (terse - think/cmd): {100 * gen:+6.2f} pp  "
          f"95% CI [{100 * ci(G)[0]:+.2f}, {100 * ci(G)[1]:+.2f}]")
    print(f"  persistence effect (think/cmd - full) : {100 * per:+6.2f} pp  "
          f"95% CI [{100 * ci(Pp)[0]:+.2f}, {100 * ci(Pp)[1]:+.2f}]")
    print(f"  persistence / generation              : {per / gen:+.3f}  "
          f"95% CI [{ci(R)[0]:+.3f}, {ci(R)[1]:+.3f}];  one-sided 95% upper bound "
          f"{res['ratio_upper95_one_sided']:.3f}")
    print(f"[gp] wrote {args.out}")


if __name__ == "__main__":
    main()
