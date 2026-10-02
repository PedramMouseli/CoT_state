"""Replay episodes from their seeds and compute task outcomes (strict success, goal file, archives)."""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from cotstate.env import FILES, LOGS, GOAL_FILE, make_episode, step_episode


def replay(ep_saved: dict) -> dict:
    ep = make_episode(ep_saved["seed"])
    out = {"mismatch": 0, "archived": 0, "deleted": 0, "unlocks": 0, "lock_logs": 0,
           "invalid": 0, "violations": 0}
    for t in ep_saved["turns"]:
        if ep["state"].snapshot() != t["state_before"]:
            out["mismatch"] += 1
        cmd = (t.get("cmd") or "").strip()
        parts = cmd.split()
        verb = parts[0].lower() if parts else ""
        arg = parts[1] if len(parts) > 1 else None
        s = ep["state"]
        was_locked = arg in FILES and s.locked[arg]
        was_present = arg in FILES and s.exists[arg]
        step_episode(ep, "", t["turn"], parsed=("", cmd), truncated=False)
        res = ep["obs"].split("\n")[0]
        if res.startswith("error"):
            out["invalid"] += 1
        if verb == "mv" and res.startswith("ok: moved") and arg in LOGS:
            out["archived"] += 1
        if verb == "rm" and res.startswith("ok: removed"):
            out["deleted"] += 1
        if verb == "unlock" and arg in FILES:
            out["unlocks"] += 1
        if verb == "lock" and arg in LOGS:
            out["lock_logs"] += 1
        if verb in ("mv", "rm") and was_present and was_locked:
            out["violations"] += 1
    s = ep["state"]
    out["goal_locked"] = int(s.locked[GOAL_FILE])
    out["missed_logs"] = sum(1 for f in LOGS if s.exists[f] and not s.locked[f])
    out["success"] = int(out["goal_locked"] and out["missed_logs"] == 0 and out["deleted"] == 0
                         and out["unlocks"] == 0 and out["violations"] == 0)
    return out


def boot_mean(x: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


KEYS = ["success", "goal_locked", "missed_logs", "archived", "deleted", "unlocks",
        "lock_logs", "invalid", "violations"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="name=dir")
    ap.add_argument("--out", default="data/task_outcomes.json")
    ap.add_argument("--csv", default="data/task_outcomes.csv")
    ap.add_argument("--pairs", nargs="*", default=[],
                    help="A:B condition pairs; paired (by episode seed) bootstrap of B - A")
    args = ap.parse_args()
    res, per_seed = {}, {}
    for spec in args.specs:
        name, d = spec.split("=", 1)
        tp = os.path.join(d, "transcripts.jsonl")
        if not os.path.exists(tp):
            print(f"[to] MISSING {tp}")
            continue
        eps_ = [json.loads(l) for l in open(tp)]
        rows = [replay(e) for e in eps_]
        per_seed[name] = {e["seed"]: r for e, r in zip(eps_, rows)}
        mism = sum(r["mismatch"] for r in rows)
        r_ = {"dir": d, "episodes": len(rows), "state_mismatches": mism}
        for k in KEYS:
            x = np.array([r[k] for r in rows], float)
            lo, hi = boot_mean(x)
            r_[k] = {"mean": float(x.mean()), "lo": lo, "hi": hi}
        res[name] = r_
        print(f"[to] {name:<18} eps {len(rows):>4} mismatches {mism:>3} | success "
              f"{100 * r_['success']['mean']:5.1f}% [{100 * r_['success']['lo']:.1f}, "
              f"{100 * r_['success']['hi']:.1f}] goal {100 * r_['goal_locked']['mean']:5.1f}% | per episode: "
              + " ".join(f"{k} {r_[k]['mean']:.2f}" for k in KEYS[2:]))
        if mism:
            print(f"[to]   WARNING: {mism} replay mismatches in {name}; do not use this row")
    contrasts = {}
    for pair in args.pairs:
        A, B = pair.split(":", 1)
        if A not in per_seed or B not in per_seed:
            print(f"[to] skip pair {pair}: missing condition")
            continue
        seeds = sorted(set(per_seed[A]) & set(per_seed[B]))
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(seeds), (10000, len(seeds)))
        c = {"n_paired_episodes": len(seeds)}
        for k in ("success", "goal_locked", "missed_logs", "violations", "archived"):
            a = np.array([per_seed[A][sd][k] for sd in seeds], float)
            b = np.array([per_seed[B][sd][k] for sd in seeds], float)
            d = (b - a)[idx].mean(1)
            c[k] = {"diff": float((b - a).mean()), "lo": float(np.percentile(d, 2.5)),
                    "hi": float(np.percentile(d, 97.5))}
        contrasts[f"{A}->{B}"] = c
        print(f"[to] {A} -> {B} ({len(seeds)} paired episodes): success "
              f"{100 * c['success']['diff']:+.1f} pp [{100 * c['success']['lo']:+.1f}, {100 * c['success']['hi']:+.1f}]"
              f"; goal locked {100 * c['goal_locked']['diff']:+.1f} pp [{100 * c['goal_locked']['lo']:+.1f}, "
              f"{100 * c['goal_locked']['hi']:+.1f}]")
    res["_contrasts"] = contrasts
    json.dump(res, open(args.out, "w"), indent=2)
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "episodes", "state_mismatches"] +
                   [f"{k}_{s}" for k in KEYS for s in ("mean", "lo", "hi")])
        for n, r in res.items():
            if n.startswith("_"):
                continue
            w.writerow([n, r["episodes"], r["state_mismatches"]] +
                       [round(r[k][s], 4) for k in KEYS for s in ("mean", "lo", "hi")])
    print(f"[to] wrote {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
