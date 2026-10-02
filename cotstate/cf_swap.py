"""Reasoning swap: each history of a counterfactual pair combined with the reasoning written for either version."""

from __future__ import annotations

import argparse
import json
import time

from cotstate.cf_action import flip_prompt
from cotstate.fixed_history2 import QUESTION, STEM, frame
from cotstate.score_actions import parse_command


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", required=True, help="cf_action_*.json with saved bodies")
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--model", default=None, help="default: the model recorded in --cf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import cotstate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    cf = json.load(open(args.cf))
    model_name = args.model or cf["model"]
    eps = [json.loads(l) for l in open(args.transcripts)]
    recs_in = cf["pairs"][:5] if args.pilot else cf["pairs"]

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(args.device).eval()
    ID_L = tok(" locked", add_special_tokens=False)["input_ids"][0]
    ID_U = tok(" unlocked", add_special_tokens=False)["input_ids"][0]
    stop_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
    extra = getattr(model.generation_config, "eos_token_id", None)
    stop_ids |= {extra} if isinstance(extra, int) else set(extra or [])

    def fwd_last(ids, keep=1):
        x = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            try:
                lg = model(input_ids=x, logits_to_keep=keep).logits[0]
            except TypeError:
                lg = model(input_ids=x).logits[0, -keep:]
        return torch.log_softmax(lg.float(), -1)

    def belief(prompt):
        lp = fwd_last(tok(prompt, add_special_tokens=False)["input_ids"])[-1]
        return float(lp[ID_L] - lp[ID_U])

    def lp_cont(prompt, cont):
        ip = tok(prompt, add_special_tokens=False)["input_ids"]
        ic = tok(cont, add_special_tokens=False)["input_ids"]
        lp = fwd_last(ip + ic, keep=len(ic) + 1)[:-1]
        return float(sum(lp[k, t] for k, t in enumerate(ic)))

    def act(prompt):
        enc = tok([prompt], return_tensors="pt", padding=True).to(args.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        new = g[0][enc["input_ids"].shape[1]:].tolist()
        n = next((k for k, x in enumerate(new) if x in stop_ids), len(new))
        return tok.decode(new[:n], skip_special_tokens=True)

    out, n_check_bad = [], 0
    for k, r in enumerate(recs_in):
        ei, t, f = r["episode"], r["turn"], r["file"]
        pL, pU = flip_prompt(eps[ei]["turns"][t]["prompt"], f)
        H = {"L": pL, "U": pU}
        B = {w: r["reasoning"][w]["body"] for w in "LU"}
        cells = {}
        for h in "LU":
            for b in "LU":
                pa = frame(H[h], B[b], "CMD:")
                pb = frame(H[h], B[b], QUESTION.format(f=f) + "\n" + STEM.format(f=f))
                txt = act(pa)
                verb, arg = parse_command("CMD: " + txt if "CMD:" not in txt else txt)
                cells[h + b] = {"lp_mv": lp_cont(pa, " mv " + f), "belief": belief(pb),
                                "action": txt.strip()[:120], "verb": verb, "arg": arg,
                                "mv_f": int(verb in ("mv", "rm") and arg == f)}
        chk = {w: {"d_lp_mv": cells[w + w]["lp_mv"] - r["reasoning"][w]["lp_mv"],
                   "same_action": cells[w + w]["mv_f"] == r["reasoning"][w]["mv_f"]} for w in "LU"}
        if any(abs(c["d_lp_mv"]) > 0.5 or not c["same_action"] for c in chk.values()):
            n_check_bad += 1
        out.append({"episode": ei, "turn": t, "file": f, "cells": cells, "check": chk,
                    "closed": {w: r["reasoning"][w].get("closed", True) for w in "LU"},
                    "body_chars": {w: len(B[w]) for w in "LU"}})
        c = cells
        print(f"[swap] {k + 1}/{len(recs_in)} {f} ep{ei} t{t}: mv_f LL {c['LL']['mv_f']} LU {c['LU']['mv_f']} "
              f"UL {c['UL']['mv_f']} UU {c['UU']['mv_f']}; lp_mv LL {c['LL']['lp_mv']:.1f} LU {c['LU']['lp_mv']:.1f} "
              f"UL {c['UL']['lp_mv']:.1f} UU {c['UU']['lp_mv']:.1f}; check {chk} ({time.time() - t0:.0f}s)", flush=True)

    print(f"[swap] {len(out)} pairs; diagonal cells differing from cf_action (|d lp| > 0.5 or other action): "
          f"{n_check_bad}")
    json.dump({"schema": "cf_swap_v1", "model": model_name, "cf": args.cf, "transcripts": args.transcripts,
               "mode": "think", "persist": "body", "format": "chat",
               "pairs": out, "n_check_bad": n_check_bad, "wall_seconds": time.time() - t0},
              open(args.out, "w"))
    print(f"[swap] wrote {args.out}")


if __name__ == "__main__":
    main()
