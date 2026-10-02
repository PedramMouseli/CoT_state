"""Characters of reasoning generated per turn, per run."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="data/reasoning_stats.json")
    ap.add_argument("--csv", default="data/reasoning_stats.csv")
    args = ap.parse_args()
    rows = []
    for tp in sorted(glob.glob(os.path.join(args.root, "**", "transcripts.jsonl"), recursive=True)):
        th, pe, tr = [], [], []
        for line in open(tp):
            for t in json.loads(line).get("turns", []):
                th.append(len(t.get("think") or ""))
                pe.append(len(t.get("persisted") or ""))
                tr.append(bool(t.get("truncated", False)))
        if not th:
            continue
        th, pe = np.array(th), np.array(pe)
        r = {"dir": os.path.dirname(tp), "turns": int(len(th)),
             "think_chars_mean": float(th.mean()), "think_chars_median": float(np.median(th)),
             "share_think_lt20": float((th < 20).mean()), "share_truncated": float(np.mean(tr)),
             "persisted_chars_mean": float(pe.mean())}
        rows.append(r)
        print(f"[rs2] {r['dir']:<50} turns {r['turns']:>5}  think mean {r['think_chars_mean']:8.0f} "
              f"median {r['think_chars_median']:7.0f}  <20 chars {100 * r['share_think_lt20']:5.1f}%  "
              f"truncated {100 * r['share_truncated']:4.1f}%  persisted {r['persisted_chars_mean']:7.0f}")
    json.dump(rows, open(args.out, "w"), indent=2)
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[rs2] wrote {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
