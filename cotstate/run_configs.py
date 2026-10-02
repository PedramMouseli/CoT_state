"""Per-run configuration summary from the saved transcripts."""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--tokenize", action="store_true")
    ap.add_argument("--out", default="data/run_configs.json")
    ap.add_argument("--csv", default="data/run_configs.csv")
    args = ap.parse_args()
    toks = {}
    rows = []
    for tp in sorted(glob.glob(os.path.join(args.root, "**", "transcripts.jsonl"), recursive=True)):
        if "_unused_pilot" in tp:
            continue
        eps = [json.loads(l) for l in open(tp)]
        e0 = eps[0]
        turns = [t for e in eps for t in e.get("turns", [])]
        r = {"dir": os.path.dirname(tp), "sha256_16": sha16(tp), "model": e0.get("model"), "mode": e0.get("mode"),
             "persist": e0.get("persist"), "format": e0.get("format"), "goal_file": e0.get("goal_file"),
             "episodes": len(eps), "seed_min": min(e.get("seed", -1) for e in eps),
             "seed_max": max(e.get("seed", -1) for e in eps),
             "max_turns": max(len(e.get("turns", [])) for e in eps), "turns": len(turns),
             "share_truncated": sum(bool(t.get("truncated")) for t in turns) / max(len(turns), 1)}
        if args.tokenize and r["model"]:
            try:
                from transformers import AutoTokenizer
                if r["model"] not in toks:
                    toks[r["model"]] = AutoTokenizer.from_pretrained(r["model"])
                tok = toks[r["model"]]
                n = lambda s: len(tok(s or "", add_special_tokens=False)["input_ids"])
                tr = [n(t.get("raw")) for t in turns if t.get("truncated")]
                r["budget_est"] = collections.Counter(tr).most_common(1)[0][0] if tr else None
                r["max_reply"] = max(n(t.get("raw")) for t in turns)
                last = [e["turns"][-1] for e in eps if e.get("turns")]
                r["max_prompt"] = max(n(t.get("prompt")) for t in last)
                r["max_total"] = max(n(t.get("prompt")) + n(t.get("raw")) for t in last)
            except Exception as ex:
                r["tokenize_error"] = repr(ex)[:200]
        rows.append(r)
        print(f"[cfg] {r['dir']:45s} {str(r['model']):22s} {r['mode']}/{r['persist']}/{r['format']} "
              f"goal={r['goal_file']} eps={r['episodes']} seeds={r['seed_min']}-{r['seed_max']} T={r['max_turns']} "
              f"trunc={100 * r['share_truncated']:.1f}% budget~{r.get('budget_est')} max_total={r.get('max_total')}")
    json.dump(rows, open(args.out, "w"), indent=1)
    keys = sorted({k for r in rows for k in r})
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[cfg] wrote {args.out} and {args.csv} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
