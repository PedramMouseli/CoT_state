"""Violations by turn, by the history of the target file, per opportunity, and commands by verb."""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from cotstate.env import FILES
from cotstate.score_actions import parse_executed_command, score_command

TOGGLE = re.compile(r"TOGGLED the lock on (\S+)")
VERB_KEYS = ("ls", "cd", "stat", "lock", "unlock", "mv", "rm", "done", "other")


def boot_rate(v: np.ndarray, a: np.ndarray, n_boot: int = 4000, seed: int = 0):
    """Ratio sum(v)/sum(a) with an episode-bootstrap CI; v, a are per-episode arrays."""
    if a.sum() == 0:
        return {"violations": 0, "attempts": 0, "rate": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n_boot, len(v)))
    num, den = v[idx].sum(1), a[idx].sum(1)
    r = num[den > 0] / den[den > 0]
    return {"violations": int(v.sum()), "attempts": int(a.sum()), "rate": float(v.sum() / a.sum()),
            "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5))}


def _rename(b: dict) -> dict:
    """boot_rate's keys, renamed for opportunities: moves / opportunities."""
    return {"moves": b["violations"], "opportunities": b["attempts"], "rate": b["rate"], "lo": b["lo"], "hi": b["hi"]}


def analyse(d: str) -> dict:
    path = os.path.join(d, "transcripts.jsonl")
    eps = [json.loads(l) for l in open(path)]
    T = max(len(e["turns"]) for e in eps)
    E = len(eps)
    att = np.zeros((E, T))
    vio = np.zeros((E, T))
    cats = ("as_stated", "toggled", "own_change")
    catv = {c: np.zeros(E) for c in cats}
    cata = {c: np.zeros(E) for c in cats}
    verbs = {k: np.zeros(E) for k in VERB_KEYS}
    oppn = {(c, l): np.zeros(E) for c in cats for l in (0, 1)}
    oppm = {(c, l): np.zeros(E) for c in cats for l in (0, 1)}
    for i, ep in enumerate(eps):
        toggled, touched = set(), set()
        for t, turn in enumerate(ep["turns"]):
            for m in TOGGLE.finditer(turn.get("obs", "") or ""):
                toggled.add(m.group(1).strip(".,"))
            verb, arg = parse_executed_command(turn.get("cmd", "") or "")
            verbs[verb if verb in VERB_KEYS else "other"][i] += 1
            sb = turn["state_before"]
            for g in FILES:
                if int(sb[f"exists_{g}"]) != 1:
                    continue
                cg = "toggled" if g in toggled else ("own_change" if g in touched else "as_stated")
                lk = int(sb[f"locked_{g}"])
                oppn[(cg, lk)][i] += 1
                oppm[(cg, lk)][i] += int(verb in ("mv", "rm") and arg == g)
            s = score_command(verb, arg, turn["state_before"], "present")
            if s["attempt"]:
                att[i, t] += 1
                vio[i, t] += s["violation"]
                c = "toggled" if arg in toggled else ("own_change" if arg in touched else "as_stated")
                cata[c][i] += 1
                catv[c][i] += s["violation"]
            if verb in ("lock", "unlock") and arg in FILES:
                touched.add(arg)
    per_turn = []
    for t in range(T):
        r = boot_rate(vio[:, t], att[:, t], seed=t)
        r["turn"] = t
        per_turn.append(r)
    return {"dir": d, "episodes": E, "max_turns": T,
            "overall": boot_rate(vio.sum(1), att.sum(1)),
            "per_turn": per_turn,
            "by_history": {c: boot_rate(catv[c], cata[c], seed=11) for c in cats},
            "opportunities": {c: {("locked" if l else "unlocked"): _rename(boot_rate(oppm[(c, l)], oppn[(c, l)], seed=13))
                                  for l in (1, 0)} for c in cats},
            "verbs": {k: float(v.mean()) for k, v in verbs.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="name=dir")
    ap.add_argument("--out", default="data/supp_stats.json")
    args = ap.parse_args()
    out = {}
    for spec in args.specs:
        name, d = spec.split("=", 1)
        if not os.path.exists(os.path.join(d, "transcripts.jsonl")):
            print(f"[supp] {name}: {d}/transcripts.jsonl missing, skipped")
            continue
        r = analyse(d)
        out[name] = r
        o, b = r["overall"], r["by_history"]
        fmt = lambda x: "  n/a" if x["rate"] is None else f"{100 * x['rate']:5.1f}% ({x['violations']}/{x['attempts']})"
        print(f"[supp] {name:22s} eps {r['episodes']:4d} T {r['max_turns']:2d}  overall {fmt(o)}  "
              f"as_stated {fmt(b['as_stated'])}  toggled {fmt(b['toggled'])}  own {fmt(b['own_change'])}")
        op = r["opportunities"]
        fo = lambda x: "  n/a" if x["rate"] is None else f"{100 * x['rate']:5.1f}% ({x['moves']}/{x['opportunities']})"
        print(f"[supp]   P(mv|locked):   as_stated {fo(op['as_stated']['locked'])}  toggled {fo(op['toggled']['locked'])}")
        print(f"[supp]   P(mv|unlocked): as_stated {fo(op['as_stated']['unlocked'])}  toggled {fo(op['toggled']['unlocked'])}")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[supp] wrote {args.out}")


if __name__ == "__main__":
    main()
