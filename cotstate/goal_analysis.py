"""Goal-file analysis: episodes in which report.txt must be locked again, and paired contrasts."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from cotstate.env import GOAL_FILE, make_episode, step_episode


def analyse(ep_saved: dict) -> dict:
    ep = make_episode(ep_saved["seed"])
    unlocked_seen, locks, fix_turn, became = False, 0, None, None
    for t in ep_saved["turns"]:
        s = ep["state"]
        if not s.locked[GOAL_FILE]:
            unlocked_seen = True
            if became is None:
                became = t["turn"]
        cmd = (t.get("cmd") or "").strip().split()
        if len(cmd) >= 2 and cmd[0].lower() == "lock" and cmd[1] == GOAL_FILE:
            locks += 1
            if became is not None and fix_turn is None:
                fix_turn = t["turn"]
        step_episode(ep, "", t["turn"], parsed=("", " ".join(cmd)), truncated=False)
    final_locked = bool(ep["state"].locked[GOAL_FILE])
    needs = unlocked_seen or not final_locked
    return {"needs": int(needs), "goal_ok": int(final_locked), "locks": locks,
            "fix_turn": fix_turn, "became_unlocked_turn": became}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--out", default="data/goal_analysis.json")
    args = ap.parse_args()
    res, per = {}, {}
    for spec in args.specs:
        name, d = spec.split("=", 1)
        eps = [json.loads(l) for l in open(os.path.join(d, "transcripts.jsonl"))]
        rows = {e["seed"]: analyse(e) for e in eps}
        per[name] = rows
        R = list(rows.values())
        needs = np.array([r["needs"] for r in R])
        ok = np.array([r["goal_ok"] for r in R])
        fx = [r["fix_turn"] for r in R if r["fix_turn"] is not None]
        res[name] = {"episodes": len(R), "needs_attention": float(needs.mean()),
                     "goal_ok_given_needs": float(ok[needs == 1].mean()) if (needs == 1).any() else None,
                     "goal_ok_given_not": float(ok[needs == 0].mean()) if (needs == 0).any() else None,
                     "lock_goal_cmds": float(np.mean([r["locks"] for r in R])),
                     "first_fix_turn": float(np.mean(fx)) if fx else None}
        print(f"[ga] {name:<18} needs {100 * res[name]['needs_attention']:5.1f}%  goal ok | needs "
              f"{res[name]['goal_ok_given_needs']}  | not {res[name]['goal_ok_given_not']}  "
              f"lock-goal cmds/ep {res[name]['lock_goal_cmds']:.2f}  first fix turn {res[name]['first_fix_turn']}")
    contrasts = {}
    for pair in args.pairs:
        A, B = pair.split(":", 1)
        seeds = sorted(sd for sd in set(per[A]) & set(per[B]) if per[A][sd]["needs"] and per[B][sd]["needs"])
        a = np.array([per[A][sd]["goal_ok"] for sd in seeds], float)
        b = np.array([per[B][sd]["goal_ok"] for sd in seeds], float)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(seeds), (10000, len(seeds)))
        dd = (b - a)[idx].mean(1)
        contrasts[pair] = {"n": len(seeds), "diff": float((b - a).mean()),
                           "lo": float(np.percentile(dd, 2.5)), "hi": float(np.percentile(dd, 97.5))}
        print(f"[ga] {A} -> {B}: goal ok given attention needed in both: {100 * contrasts[pair]['diff']:+.1f} pp "
              f"[{100 * contrasts[pair]['lo']:+.1f}, {100 * contrasts[pair]['hi']:+.1f}] over {len(seeds)} episodes")
    json.dump({"conditions": res, "contrasts": contrasts}, open(args.out, "w"), indent=2)
    print(f"[ga] wrote {args.out}")


if __name__ == "__main__":
    main()
