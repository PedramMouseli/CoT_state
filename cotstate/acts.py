"""Extract residual-stream activations at the final prompt token of saved rollouts."""

from __future__ import annotations

import argparse, json, os
import numpy as np
import torch

from cotstate.textbase import all_histories, persisted_reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcripts", default="data/transcripts.jsonl")
    ap.add_argument("--out", default="data/acts.npz")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-len", type=int, default=16384,
                    help="must be >= the longest prompt; full-persist runs "
                         "with long reasoning need this above 8192")
    args = ap.parse_args()

    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        print(f"[acts] {args.out} already exists, skipping")
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()

    layers = model.model.layers
    n_layers = len(layers)
    print(f"[acts] {n_layers} layers, d_model={model.config.hidden_size}")

    buf = {}

    def mk_hook(idx):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            buf[idx] = h[:, -1, :].detach().float().cpu()
        return hook

    handles = [l.register_forward_hook(mk_hook(i)) for i, l in enumerate(layers)]

    rows, meta = [], []
    with open(args.transcripts) as fh:
        eps = [json.loads(l) for l in fh]

    flat = []
    for ei, ep in enumerate(eps):
        hists = all_histories(ep["turns"])
        for ti, t in enumerate(ep["turns"]):
            if "prompt" not in t:
                continue
            flat.append((ei, ep["seed"], t, hists[ti]))
    cond = {k: eps[0].get(k) for k in ("model", "mode", "persist", "format", "goal_file")}
    print(f"[acts] condition: {cond}")

    print(f"[acts] {len(flat)} turns to encode")
    for b in range(0, len(flat), args.batch):
        chunk = flat[b:b + args.batch]
        enc = tok([c[2]["prompt"] for c in chunk], return_tensors="pt",
                  padding=True, truncation=False).to(args.device)
        if enc["input_ids"].shape[1] > args.max_len:
            raise SystemExit(f"[acts] prompt of {enc['input_ids'].shape[1]} tokens exceeds "
                             f"--max-len {args.max_len}; regenerate with a larger "
                             f"--max-model-len or raise --max-len")
        with torch.no_grad():
            model(**enc)
        stacked = torch.stack([buf[i] for i in range(n_layers)], dim=1)
        rows.append(stacked.half().numpy())
        for ei, seed, t, hist in chunk:
            meta.append({
                "episode": ei, "seed": seed, "turn": t["turn"],
                "state_before": t["state_before"],
                "think": t["think"], "cmd": t["cmd"], "obs": t["obs"],
                "persisted": persisted_reply(t),
                "hist": hist,
                "truncated": t.get("truncated", False),
                "prompt_chars": len(t["prompt"]),
            })
        if (b // args.batch) % 25 == 0:
            print(f"[acts] {b + len(chunk)}/{len(flat)}", flush=True)

    for h in handles:
        h.remove()

    X = np.concatenate(rows, axis=0)
    print("[acts] X", X.shape, X.dtype)
    np.savez_compressed(
        args.out, X=X, layers=np.arange(n_layers, dtype=np.int32),
        meta=np.array(json.dumps(meta), dtype=object),
        cond=np.array(json.dumps(cond), dtype=object),
    )
    print(f"[acts] wrote {args.out}")


if __name__ == "__main__":
    main()
