"""The file-system environment: state, commands, background toggles and episode stepping."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

DIRS = ["/logs", "/archive"]
LOGS = ["alpha.log", "beta.log", "gamma.log"]
FILES = LOGS + ["report.txt"]
N_TURNS = int(os.environ.get("WM_N_TURNS", 8))
MIN_TURNS = int(os.environ.get("WM_MIN_TURNS", 6))


@dataclass
class State:
    exists: dict = field(default_factory=dict)
    archived: dict = field(default_factory=dict)
    locked: dict = field(default_factory=dict)
    cwd: str = "/logs"

    def snapshot(self) -> dict:
        """Flat dict of probe targets. Keys here == probe variable names."""
        out = {}
        for f in FILES:
            out[f"exists_{f}"] = int(self.exists[f])
            out[f"locked_{f}"] = int(self.locked[f])
        out["cwd"] = DIRS.index(self.cwd)
        return out


VARS_LOCKED = [f"locked_{f}" for f in FILES]
VARS_EXISTS = [f"exists_{f}" for f in FILES]
VARS = VARS_LOCKED + VARS_EXISTS + ["cwd"]


def init_state(rng: random.Random) -> State:
    s = State()
    for f in FILES:
        s.exists[f] = rng.random() < 0.7
        s.archived[f] = not s.exists[f]
        s.locked[f] = rng.random() < 0.5
    s.cwd = rng.choice(DIRS)
    return s


TOOL_DOC = """You are operating a small file system through a shell. Exactly one command per turn.

Available commands:
  ls                 list the files in the current directory (does NOT show lock status)
  cd <dir>           change directory; <dir> is /logs or /archive
  stat <file>        report whether <file> is locked (costs a turn)
  lock <file>        lock a file
  unlock <file>      unlock a file
  mv <file>          move <file> from /logs to /archive (fails if the file is locked)
  rm <file>          delete <file> (fails if the file is locked)
  done               finish the episode

Reply with your reasoning and then the command on its own final line, prefixed by `CMD: `.
Example final line:  CMD: ls
"""

GOAL_FILE = os.environ.get("WM_GOAL_FILE", "report.txt")
assert GOAL_FILE in FILES, f"WM_GOAL_FILE must be one of {FILES}"

TASK = (
    "TASK: Archive every unlocked .log file in /logs by moving it to /archive. "
    f"Never touch a locked file. At the end, {GOAL_FILE} must be LOCKED. "
    "Work efficiently; you have a limited number of turns."
)


def apply_command(s: State, cmd: str) -> str:
    """Mutate state, return the observation string the agent sees."""
    cmd = (cmd or "").strip()
    parts = cmd.split()
    if not parts:
        return "error: no command given"
    verb, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else None)

    if verb == "ls":
        src = s.exists if s.cwd == "/logs" else s.archived
        present = [f for f in FILES if src[f]]
        return f"{s.cwd}: " + (" ".join(present) if present else "(empty)")

    if verb == "cd":
        if arg in DIRS:
            s.cwd = arg
            return f"now in {s.cwd}"
        return f"error: no such directory {arg}"

    if verb == "stat":
        if arg not in FILES:
            return f"error: no such file {arg}"
        return f"{arg}: {'locked' if s.locked[arg] else 'unlocked'}"

    if verb in ("lock", "unlock"):
        if arg not in FILES:
            return f"error: no such file {arg}"
        s.locked[arg] = (verb == "lock")
        return "ok"

    if verb == "mv":
        if arg not in FILES:
            return f"error: no such file {arg}"
        if not s.exists[arg]:
            return f"error: {arg} is not in /logs"
        if s.locked[arg]:
            return f"error: {arg} is locked, cannot move"
        s.exists[arg] = False
        s.archived[arg] = True
        return f"ok: moved {arg} to /archive"

    if verb == "rm":
        if arg not in FILES:
            return f"error: no such file {arg}"
        if s.locked[arg]:
            return f"error: {arg} is locked, cannot remove"
        if not s.exists[arg]:
            return f"error: {arg} is not in /logs"
        s.exists[arg] = False
        return f"ok: removed {arg}"

    if verb == "done":
        return "ok: finished"

    return f"error: unknown command '{verb}'"


TOGGLE_TURNS = tuple(int(x) for x in
                     os.environ.get("WM_TOGGLE_TURNS", "1,3,5").split(","))


def exogenous_event(s: State, rng: random.Random, turn: int) -> str | None:
    """
    Inject a state change the model did not cause.

    CRITICAL DESIGN POINT - do not "simplify" this into an absolute statement.

    The event is a TOGGLE, not an assignment. "background job TOGGLED the lock
    on beta.log" does not tell you the resulting value; you can only know it by
    remembering the previous value and integrating. If these events announced
    absolute values ("LOCKED beta.log"), a three-line regex over the transcript
    would predict `locked_*` perfectly, the last-assertion baseline would sit at
    100%, and the probe could never beat it. The experiment would be dead on
    arrival. (Verified: it was, before this change.)

    Fires on a fixed schedule so the gap between the last change and the end of
    the episode is controlled. That gap is the x-axis of Experiment 2.
    """
    if turn not in TOGGLE_TURNS:
        return None
    f = rng.choice(FILES)
    s.locked[f] = not s.locked[f]
    return f"[system] background job TOGGLED the lock on {f}"


def make_episode(seed: int) -> dict:
    """Set up an episode. Generation is driven externally, turn by turn."""
    rng = random.Random(seed)
    s = init_state(rng)
    sys_prompt = TOOL_DOC + "\n" + TASK
    first_obs = (
        f"Initial state: you are in {s.cwd}.\n"
        + "\n".join(
            f"{f}: {'locked' if s.locked[f] else 'unlocked'}" for f in FILES
        )
        + "\nBegin."
    )
    return {
        "seed": seed,
        "system": sys_prompt,
        "state": s,
        "rng": rng,
        "obs": first_obs,
        "turns": [],
    }


def step_episode(ep: dict, raw_reply: str, turn: int,
                 thinking: bool = True,
                 parsed: tuple[str, str] | None = None,
                 truncated: bool | None = None) -> bool:
    """
    Record the model's reply for `turn`, apply it, compute the next observation.
    Returns False when the episode is over.

    `parsed` = (think, cmd) lets gen.py do the parsing itself (needed for the
    plain-text format used with base models, where there are no <think> tags
    and "thinking" is whatever precedes the CMD line). If omitted, parse_reply
    is used as before.
    """
    if parsed is None:
        think, cmd = parse_reply(raw_reply, thinking=thinking)
    else:
        think, cmd = parsed
    if truncated is None:
        truncated = thinking and "</think>" not in raw_reply
    state_before = ep["state"].snapshot()
    ep["turns"].append(
        {
            "turn": turn,
            "obs": ep["obs"],
            "think": think,
            "cmd": cmd,
            "raw": raw_reply,
            "state_before": state_before,
            "truncated": bool(truncated),
        }
    )
    result = apply_command(ep["state"], cmd)

    said_done = cmd.strip().lower().startswith("done")
    if said_done and turn < MIN_TURNS:
        result = ("ok: noted - but you still have turns left. "
                  "Use them to verify the state of each file.")
        said_done = False

    ev = exogenous_event(ep["state"], ep["rng"], turn)
    ep["obs"] = result + (("\n" + ev) if ev else "")
    return not (said_done or turn + 1 >= N_TURNS)


def parse_reply(raw: str, thinking: bool = True) -> tuple[str, str]:
    """Split a raw model reply into (thinking text, command)."""
    if "</think>" in raw:
        pre, body = raw.split("</think>", 1)
        think = pre.replace("<think>", "").strip()
        truncated = False
    elif not thinking:
        think, body, truncated = "", raw, False
    else:
        think = raw.replace("<think>", "").strip()
        body = ""
        truncated = True

    cmd = ""
    for line in reversed(body.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("CMD:"):
            cmd = line[4:].strip()
            break
    if not cmd and not truncated:
        lines = [l.strip() for l in body.strip().splitlines() if l.strip()]
        cmd = lines[-1] if lines else ""
    return think, cmd.strip().strip("`").strip()


FORCED_CHOICE = (
    "[system] Before your next command, answer one question. "
    "Is {f} currently locked? Reply with exactly one word: LOCKED or UNLOCKED."
)


def score_forced_choice(reply: str, truth: int) -> int | None:
    """Return 1 if the model's stated belief matches truth, 0 if not, None if unparseable."""
    body = reply.split("</think>")[-1].upper()
    has_l, has_u = "LOCKED" in body.replace("UNLOCKED", ""), "UNLOCKED" in body
    if has_l == has_u:
        return None
    said = 1 if has_l else 0
    return int(said == truth)


if __name__ == "__main__":
    ep = make_episode(0)
    t = 0
    while True:
        fake = f"<think>thinking</think>\nCMD: {random.choice(['ls', 'stat alpha.log', 'mv beta.log', 'lock report.txt'])}"
        if not step_episode(ep, fake, t):
            break
        t += 1
    print(json.dumps([{k: v for k, v in x.items() if k != "raw"} for x in ep["turns"]], indent=2)[:1500])
    print("turns:", len(ep["turns"]), "vars:", len(VARS))
