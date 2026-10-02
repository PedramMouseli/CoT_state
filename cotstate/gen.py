"""Generate agent rollouts with vLLM for one condition (mode, persistence, format)."""

from __future__ import annotations

import argparse, json, os, sys
from cotstate import env as E
from cotstate.util import chat, assert_thinking

MODES = ("think", "nothink", "terse")
PERSISTS = ("full", "body", "cmd")
FORMATS = ("chat", "plain")


def split_body(raw: str) -> str:
    """Text after the last </think>; the whole reply if there is no tag."""
    return raw.split("</think>")[-1].strip()


def find_cmd(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("CMD:"):
            return line[4:].strip().strip("`").strip()
    return ""


def parse_plain(raw: str) -> tuple[str, str, bool]:
    """Plain format: reasoning is whatever precedes the CMD line.
    Returns (think, cmd, truncated)."""
    lines = raw.strip().splitlines()
    for k in range(len(lines) - 1, -1, -1):
        if lines[k].strip().upper().startswith("CMD:"):
            cmd = lines[k].strip()[4:].strip().strip("`").strip()
            think = "\n".join(lines[:k]).strip()
            return think, cmd, False
    return raw.strip(), "", True


def persisted_text(think: str, body: str, cmd: str, persist: str,
                   mode: str, fmt: str) -> str:
    """What later prompts will contain for this turn. Always ends with the
    CMD line so the command itself is never lost."""
    cmd_line = "CMD: " + cmd
    if persist == "cmd" or mode == "terse":
        return cmd_line
    if fmt == "plain":
        return (think + "\n" if think else "") + cmd_line
    body_wo_cmd = "\n".join(l for l in body.splitlines()
                            if not l.strip().upper().startswith("CMD:")).strip()
    if persist == "body" or not think:
        return (body_wo_cmd + "\n" if body_wo_cmd else "") + cmd_line
    return ("[reasoning]\n" + think + "\n[/reasoning]\n"
            + (body_wo_cmd + "\n" if body_wo_cmd else "") + cmd_line)


PLAIN_DEMO_THINK = """Example of the expected format, from a different, unrelated session:

### Observation
Initial state: you are in /logs.
foo.log: unlocked
bar.log: locked
Begin.
### Agent
foo.log is unlocked, so it can be archived. bar.log is locked and must be left alone.
CMD: mv foo.log
### Observation
ok: moved foo.log to /archive
[system] background job TOGGLED the lock on bar.log
### Agent
bar.log was locked; the toggle means it is now unlocked, so it can be archived.
CMD: mv bar.log
"""

PLAIN_DEMO_TERSE = """Example of the expected format, from a different, unrelated session:

### Observation
Initial state: you are in /logs.
foo.log: unlocked
bar.log: locked
Begin.
### Agent
CMD: mv foo.log
### Observation
ok: moved foo.log to /archive
[system] background job TOGGLED the lock on bar.log
### Agent
CMD: mv bar.log
"""


def build_prompt_chat(tok, system, turns, obs, mode):
    msgs = [{"role": "system", "content": system}]
    for t in turns:
        msgs.append({"role": "user", "content": t["obs"]})
        msgs.append({"role": "assistant", "content": t["persisted"]})
    msgs.append({"role": "user", "content": obs})
    p = chat(tok, msgs, thinking=(mode == "think"))
    if mode == "terse":
        p += "CMD:"
    return p


def build_prompt_plain(system, turns, obs, mode):
    demo = PLAIN_DEMO_TERSE if mode == "terse" else PLAIN_DEMO_THINK
    parts = [system.rstrip(), "", demo.rstrip(), "", "Now the real session begins.", ""]
    for t in turns:
        parts += ["### Observation", t["obs"], "### Agent", t["persisted"]]
    parts += ["### Observation", obs, "### Agent"]
    p = "\n".join(parts) + "\n"
    if mode == "terse":
        p += "CMD:"
    return p


def build_prompt(tok, system, turns, obs, mode, fmt):
    if fmt == "chat":
        return build_prompt_chat(tok, system, turns, obs, mode)
    return build_prompt_plain(system, turns, obs, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--mode", choices=MODES, default="think")
    ap.add_argument("--persist", choices=PERSISTS, default="body")
    ap.add_argument("--format", choices=FORMATS, default="chat")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--attn-backend", default=None,
                    help="e.g. TRITON_ATTN if your CUDA/driver mismatch needs it "
                         "(sets VLLM_ATTENTION_BACKEND before vLLM is imported)")
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    if args.format == "plain" and args.mode == "nothink":
        sys.exit("[gen] nothink is a chat-template concept; use --mode think or terse with --format plain")
    if args.attn_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attn_backend

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.format == "chat" and not getattr(tok, "chat_template", None):
        sys.exit(f"[gen] {args.model} has no chat template - use --format plain")

    llm_kwargs = dict(model=args.model, tensor_parallel_size=args.tp,
                      gpu_memory_utilization=args.gpu_frac,
                      max_model_len=args.max_model_len, dtype="bfloat16")
    if args.attn_backend:
        llm_kwargs["attention_backend"] = args.attn_backend
        llm_kwargs["mm_encoder_attn_backend"] = "TORCH_SDPA"
    llm = LLM(**llm_kwargs)

    stop_main = ["\n### "] if args.format == "plain" else None
    if args.mode == "terse":
        sp_main = SamplingParams(temperature=args.temperature, top_p=0.9,
                                 max_tokens=12, stop=["\n"])
    else:
        sp_main = SamplingParams(temperature=args.temperature, top_p=0.9,
                                 max_tokens=args.max_tokens, stop=stop_main)
    sp_repair = SamplingParams(temperature=args.temperature, top_p=0.9,
                               max_tokens=12, stop=["\n"])

    eps = [E.make_episode(args.seed0 + i) for i in range(args.n)]
    live = list(range(len(eps)))
    n_repaired = 0

    for turn in range(E.N_TURNS):
        if not live:
            break
        prompts = [build_prompt(tok, eps[i]["system"], eps[i]["turns"],
                                eps[i]["obs"], args.mode, args.format)
                   for i in live]
        outs = llm.generate(prompts, sp_main)
        raws = [o.outputs[0].text for o in outs]
        if args.mode == "terse":
            raws = ["CMD:" + r for r in raws]

        if turn == 0 and args.format == "chat" and args.mode != "terse":
            assert_thinking(raws[0], expect=(args.mode == "think"))

        if args.mode == "think" and args.format == "chat":
            need = [k for k, r in enumerate(raws) if "</think>" not in r]
            suffix = "\n</think>\n\nCMD:"
        elif args.mode != "terse":
            need = [k for k, r in enumerate(raws) if not find_cmd(split_body(r))]
            suffix = "\nCMD:"
        else:
            need, suffix = [], ""
        truncated = [False] * len(raws)
        if need:
            fix = llm.generate([prompts[k] + raws[k] + suffix for k in need], sp_repair)
            for k, o in zip(need, fix):
                raws[k] = raws[k] + suffix + o.outputs[0].text
                truncated[k] = True
            n_repaired += len(need)

        still = []
        for j, (i, p, raw) in enumerate(zip(live, prompts, raws)):
            if args.format == "plain":
                think, cmd, _ = parse_plain(raw)
            else:
                think, cmd = E.parse_reply(raw, thinking=(args.mode == "think"))
            body = split_body(raw)
            alive = E.step_episode(eps[i], raw, turn, thinking=(args.mode == "think"),
                                   parsed=(think, cmd), truncated=truncated[j])
            t = eps[i]["turns"][-1]
            t["prompt"] = p
            t["persisted"] = persisted_text(think, body, cmd, args.persist,
                                            args.mode, args.format)
            if alive:
                still.append(i)
        live = still
        print(f"[gen] turn {turn} done, {len(live)} live, {len(need)} repaired this turn",
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        for ep in eps:
            fh.write(json.dumps({
                "seed": ep["seed"], "system": ep["system"],
                "model": args.model, "mode": args.mode, "persist": args.persist,
                "format": args.format, "goal_file": E.GOAL_FILE,
                "turns": ep["turns"],
            }) + "\n")
    n_turns = sum(len(e["turns"]) for e in eps)
    print(f"[gen] wrote {len(eps)} episodes / {n_turns} turns -> {args.out}"
          f"   (repaired {n_repaired} = {100*n_repaired/max(n_turns,1):.1f}%)")


if __name__ == "__main__":
    main()
