"""The visible history of a turn, as the model saw it (used by the text baselines)."""

from __future__ import annotations


def persisted_reply(turn: dict) -> str:
    """Text of the assistant reply that later prompts contained."""
    if "persisted" in turn:
        return turn["persisted"]
    raw = turn.get("raw", "") or ""
    return raw.split("</think>")[-1].strip()


def visible_history(ep_turns: list[dict], i: int) -> list[str]:
    hist = []
    for j in range(i):
        t = ep_turns[j]
        hist.append(t["obs"] + "\n" + persisted_reply(t))
    hist.append(ep_turns[i]["obs"])
    return hist


def all_histories(ep_turns: list[dict]) -> list[list[str]]:
    return [visible_history(ep_turns, i) for i in range(len(ep_turns))]
