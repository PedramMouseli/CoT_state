"""Rescore every saved rollout with the same scorer."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os

from cotstate.audit_behav import analyse
from cotstate.score_actions import ELIGIBILITY


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def describe(d: str) -> dict:
    out = {}
    for name in ("meta.json", "config.json", "args.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            try:
                m = json.load(open(p))
                for k in ("model", "mode", "persist", "fmt", "format", "n_episodes",
                          "max_tokens", "max_model_len", "temperature", "turns"):
                    if k in m:
                        out[k] = m[k]
            except Exception:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="data/rescore_all.json")
    ap.add_argument("--csv", default="data/rescore_all.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "**", "transcripts.jsonl"), recursive=True))
    print(f"[rs] {len(paths)} transcripts.jsonl files under {args.root}")
    rows = []
    for tp in paths:
        d = os.path.dirname(tp)
        try:
            first = json.loads(open(tp).readline())
            if not first.get("turns") or "state_before" not in first["turns"][0]:
                print(f"[rs] skip {d}: no state_before")
                continue
            max_turns = max(len(json.loads(l)["turns"]) for l in open(tp))
            r = {"dir": d, "sha256_16": sha16(tp), "max_turns": max_turns, **describe(d)}
            for el in ELIGIBILITY:
                a = analyse(tp, el)
                r[el] = {k: v for k, v in a.items() if k != "per_episode"}
            rows.append(r)
            p = r["present"]
            print(f"[rs] {d:<55} eps {p['episodes']:>4} maxT {max_turns:>2}  present "
                  f"{p['violations']:>4}/{p['attempts']:<5} {100 * p['rate']:6.2f}% "
                  f"[{100 * p['cluster_lo']:.1f}, {100 * p['cluster_hi']:.1f}]")
        except Exception as e:
            print(f"[rs] ERROR {d}: {e!r}")
            rows.append({"dir": d, "error": repr(e)})

    json.dump({"root": args.root, "runs": rows}, open(args.out, "w"), indent=2)
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dir", "sha256_16", "model", "max_turns", "episodes", "turns",
                    "present_violations", "present_attempts", "present_rate_pct",
                    "present_ci_lo_pct", "present_ci_hi_pct", "attempts_per_episode",
                    "all_violations", "all_attempts", "all_rate_pct"])
        for r in rows:
            if "error" in r:
                w.writerow([r["dir"], "ERROR"])
                continue
            p, a = r["present"], r["all"]
            w.writerow([r["dir"], r["sha256_16"], r.get("model", ""), r["max_turns"],
                        p["episodes"], p["turns"], p["violations"], p["attempts"],
                        round(100 * p["rate"], 3), round(100 * p["cluster_lo"], 3),
                        round(100 * p["cluster_hi"], 3), round(p["attempts_per_episode"], 3),
                        a["violations"], a["attempts"], round(100 * a["rate"], 3)])
    print(f"[rs] wrote {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
