"""Symbolic replay of the visible transcript."""

from __future__ import annotations

import argparse, json, re
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from cotstate.env import FILES, VARS_LOCKED
from cotstate.probe import last_assertion
from cotstate.textbase import persisted_reply

RE_INIT = {f: re.compile(rf"^{re.escape(f)}: (locked|unlocked)\s*$", re.M) for f in FILES}
RE_STAT = {f: re.compile(rf"^{re.escape(f)}: (locked|unlocked)\s*$", re.M) for f in FILES}
RE_TOGGLE = {f: re.compile(rf"TOGGLED the lock on {re.escape(f)}\b") for f in FILES}
RE_MV_FAIL = {f: re.compile(rf"error: {re.escape(f)} is locked") for f in FILES}
RE_MV_OK = {f: re.compile(rf"ok: (?:moved|removed) {re.escape(f)}\b") for f in FILES}
RE_CMD = re.compile(r"^\s*CMD:\s*(\S+)(?:\s+(\S+))?", re.M | re.I)


def _cmd_of(turn: dict) -> tuple[str, str | None]:
    """The agent's command as it appears in the VISIBLE persisted text."""
    ms = list(RE_CMD.finditer(persisted_reply(turn)))
    if not ms:
        return "", None
    m = ms[-1]
    return m.group(1).lower(), m.group(2)


def replay(ep: dict, upto: int, use_cmds: bool = True,
           window: int | None = None) -> dict[str, int | None]:
    """Reconstruct locked_* as it stands ENTERING turn `upto`, from text only.

    `window` is INITIAL STATE PLUS a window: the opening lock assignments are
    always read, and then only the last `window` turns of transitions. It is
    NOT a reader restricted to the last `window` turns. The boundary convention includes the
    observation at the start of the window.
    `use_cmds=False` blinds the reader to the agent's own lock/unlock lines.
    """
    turns = ep["turns"]
    state: dict[str, int | None] = {f: None for f in FILES}

    for f in FILES:
        m = RE_INIT[f].search(turns[0]["obs"])
        if m:
            state[f] = int(m.group(1) == "locked")

    start = 0 if window is None else max(0, upto - window)
    if window is not None and start > 0:
        pass

    for j in range(upto):
        t = turns[j]
        if j >= start:
            verb, arg = _cmd_of(t)
            if use_cmds and arg in FILES:
                if verb == "lock":
                    state[arg] = 1
                elif verb == "unlock":
                    state[arg] = 0
        nxt = turns[j + 1]["obs"] if j + 1 < len(turns) else None
        if nxt is None or j + 1 < start:
            continue
        for f in FILES:
            if RE_MV_FAIL[f].search(nxt):
                state[f] = 1
            elif RE_MV_OK[f].search(nxt):
                state[f] = 0
            m = RE_STAT[f].search(nxt)
            if m:
                state[f] = int(m.group(1) == "locked")
            if RE_TOGGLE[f].search(nxt):
                if state[f] is not None:
                    state[f] = 1 - state[f]
    return state


def evaluate(path: str, windows=(), verbose=True) -> dict:
    eps = [json.loads(l) for l in open(path)]
    gold = {v: [] for v in VARS_LOCKED}
    preds = {name: {v: [] for v in VARS_LOCKED}
             for name in ["oracle", "oracle_nocmd", "last_assertion"]
             + [f"oracle_w{w}" for w in windows]}
    from cotstate.textbase import all_histories

    for ep in eps:
        hists = all_histories(ep["turns"])
        for i, t in enumerate(ep["turns"]):
            full = replay(ep, i, use_cmds=True)
            nocmd = replay(ep, i, use_cmds=False)
            wins = {w: replay(ep, i, use_cmds=True, window=w) for w in windows}
            for v in VARS_LOCKED:
                f = v[len("locked_"):]
                y = t["state_before"][v]
                gold[v].append(y)
                preds["oracle"][v].append(1 if full[f] is None else full[f])
                preds["oracle_nocmd"][v].append(1 if nocmd[f] is None else nocmd[f])
                la, _ = last_assertion(hists[i], f)
                preds["last_assertion"][v].append(1 if la is None else la)
                for w in windows:
                    val = wins[w][f]
                    preds[f"oracle_w{w}"][v].append(1 if val is None else val)

    out = {"file": path, "n_turns": len(gold[VARS_LOCKED[0]]), "scores": {}}
    for name, pv in preds.items():
        out["scores"][name] = {
            v: float(balanced_accuracy_score(gold[v], pv[v])) for v in VARS_LOCKED}
        out["scores"][name]["mean"] = float(np.mean(
            [out["scores"][name][v] for v in VARS_LOCKED]))

    if verbose:
        names = list(preds)
        print(f"\n=== {path}  ({out['n_turns']} turn-variable pairs per reader)")
        print(f"{'reader':<16}" + "".join(f"{v.replace('locked_',''):>14}" for v in VARS_LOCKED) + f"{'mean':>10}")
        for name in names:
            s = out["scores"][name]
            print(f"{name:<16}" + "".join(f"{s[v]:>14.3f}" for v in VARS_LOCKED) + f"{s['mean']:>10.3f}")
        om = out["scores"]["oracle"]["mean"]
        if om < 0.999:
            print(f"\n  *** ORACLE IS {om:.4f}, NOT 1.000 ***\n"
                  "  The visible transcript does not determine state_before.\n"
                  "  Print the disagreeing turns:\n"
                  "      python -m cotstate.oracle --debug <file>")
        else:
            print("\n  Oracle = 1.000: the visible transcript fully determines the state.")
    return out


def debug(path: str, limit: int = 10):
    """Print turns where the oracle and ground truth disagree."""
    eps = [json.loads(l) for l in open(path)]
    n = 0
    for ei, ep in enumerate(eps):
        for i, t in enumerate(ep["turns"]):
            st = replay(ep, i)
            for v in VARS_LOCKED:
                f = v[len("locked_"):]
                pred = st[f]
                if pred is not None and pred != t["state_before"][v]:
                    print(f"\n--- episode {ei} turn {i}  {v}: oracle={pred} truth={t['state_before'][v]}")
                    for j in range(max(0, i - 3), i + 1):
                        print(f"  [t{j}] OBS: {ep['turns'][j]['obs'][:160]!r}")
                        print(f"        CMD: {ep['turns'][j]['cmd']!r}")
                    n += 1
                    if n >= limit:
                        return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--windows", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--out", default=None)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    if a.debug:
        debug(a.files[0])
    else:
        res = [evaluate(f, windows=a.windows) for f in a.files]
        if a.out:
            json.dump(res, open(a.out, "w"), indent=2)
            print(f"\n[oracle] wrote {a.out}")
