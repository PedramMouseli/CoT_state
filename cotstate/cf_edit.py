"""Targeted edit of explicit lock statements in the model's own reasoning, with a case-only control."""

from __future__ import annotations

import argparse
import json
import re
import time

from cotstate.cf_action import flip_prompt
from cotstate.fixed_history2 import QUESTION, STEM, frame
from cotstate.score_actions import parse_command

EXCL = re.compile(r"\bif\b|\bwhether\b|\?|\bmust\b|\bshould be\b|\bneeds? to be\b|\bmight\b|\bmay\b|"
                  r"\bcould\b|\bassum|\btoggl|\bwould\b|\bunless\b|\bin case\b|\bensure\b|\bmake sure\b", re.I)


def patterns(f: str):
    e = re.escape(f)
    return [re.compile(rf"{e}(?:\*\*|`)?\s*[:\-–]\s*(?:\*\*)?(?P<w>unlocked|locked)\b", re.I),
            re.compile(rf"{e}(?:\*\*|`)?\s+(?:is|was|remains|stays)\s+(?:still\s+|currently\s+|now\s+|also\s+)?"
                       rf"(?:\*\*)?(?P<w>unlocked|locked)\b", re.I),
            re.compile(rf"{e}(?:\*\*|`)?\s*\((?P<w>unlocked|locked)\)", re.I),
            re.compile(rf"{e}(?:\*\*|`)?\s+(?:as\s+)?(?P<w>unlocked|locked)\b", re.I)]


BOUND = re.compile(r"[.!?;](?=\s)|\n")


def sentence(body: str, i: int) -> str:
    starts = [m.end() for m in BOUND.finditer(body, 0, i)]
    m = BOUND.search(body, i)
    return body[(starts[-1] if starts else 0):(m.start() if m else len(body))]


def find_edits(body: str, f: str):
    spans = {}
    for pat in patterns(f):
        for m in pat.finditer(body):
            s, e = m.span("w")
            if EXCL.search(sentence(body, s)):
                continue
            spans[(s, e)] = m.group("w")
    return sorted(spans.items())


def apply(body: str, edits, mode: str) -> str:
    out, last = [], 0
    for (s, e), w in edits:
        if mode == "edit":
            new = "unlocked" if w.lower() == "locked" else "locked"
            new = new.upper() if w.isupper() else (new.capitalize() if w[0].isupper() else new)
        else:
            new = w.upper() if not w.isupper() else w.lower()
        out += [body[last:s], new]
        last = e
    return "".join(out + [body[last:]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dry", action="store_true", help="only list edits, no model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cf = json.load(open(args.cf))
    eps = [json.loads(l) for l in open(args.transcripts)]
    recs_in = [r for r in cf["pairs"] if r["file"] != "report.txt"]
    if args.pilot:
        recs_in = recs_in[:5]
    plan = []
    for r in recs_in:
        f = r["file"]
        per = {w: find_edits(r["reasoning"][w]["body"], f) for w in "LU"}
        plan.append((r, per))
    n_ed = {w: sum(bool(p[w]) for _, p in plan) for w in "LU"}
    print(f"[edit] {len(plan)} log-file pairs; traces with >= 1 editable assertion: L {n_ed['L']}, U {n_ed['U']}")
    if args.dry:
        json.dump([{"episode": r["episode"], "turn": r["turn"], "file": r["file"],
                    "edits": {w: [(s, e, wd, r["reasoning"][w]["body"][max(0, s - 80):e + 40]) for (s, e), wd in p[w]]
                              for w in "LU"}} for r, p in plan], open(args.out, "w"), indent=1)
        return

    import torch
    import cotstate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cf["model"], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cf["model"], torch_dtype=torch.bfloat16).to(args.device).eval()
    ID_L = tok(" locked", add_special_tokens=False)["input_ids"][0]
    ID_U = tok(" unlocked", add_special_tokens=False)["input_ids"][0]
    stop_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
    extra = getattr(model.generation_config, "eos_token_id", None)
    stop_ids |= {extra} if isinstance(extra, int) else set(extra or [])

    def fwd_last(ids, keep=1):
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids], device=args.device), logits_to_keep=keep).logits[0]
        return torch.log_softmax(lg.float(), -1)

    def cells(p, body, f):
        pa = frame(p, body, "CMD:")
        pb = frame(p, body, QUESTION.format(f=f) + "\n" + STEM.format(f=f))
        enc = tok([pa], return_tensors="pt").to(args.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        new = g[0][enc["input_ids"].shape[1]:].tolist()
        n = next((k for k, x in enumerate(new) if x in stop_ids), len(new))
        txt = tok.decode(new[:n], skip_special_tokens=True)
        verb, arg = parse_command("CMD: " + txt if "CMD:" not in txt else txt)
        ip = tok(pa, add_special_tokens=False)["input_ids"]
        ic = tok(" mv " + f, add_special_tokens=False)["input_ids"]
        lp = fwd_last(ip + ic, keep=len(ic) + 1)[:-1]
        lpb = fwd_last(tok(pb, add_special_tokens=False)["input_ids"])[-1]
        return {"lp_mv": float(sum(lp[k, t] for k, t in enumerate(ic))), "belief": float(lpb[ID_L] - lpb[ID_U]),
                "action": txt.strip()[:120], "verb": verb, "arg": arg, "mv_f": int(verb in ("mv", "rm") and arg == f)}

    out = []
    for k, (r, per) in enumerate(plan):
        ei, t, f = r["episode"], r["turn"], r["file"]
        pL, pU = flip_prompt(eps[ei]["turns"][t]["prompt"], f)
        rec = {"episode": ei, "turn": t, "file": f, "worlds": {}}
        for w, p in (("L", pL), ("U", pU)):
            b = r["reasoning"][w]["body"]
            if not per[w]:
                continue
            o = {"n_edits": len(per[w]), "edits": [(s, e, wd) for (s, e), wd in per[w]],
                 "orig": cells(p, b, f), "edit": cells(p, apply(b, per[w], "edit"), f),
                 "control": cells(p, apply(b, per[w], "control"), f)}
            o["check_orig"] = (abs(o["orig"]["lp_mv"] - r["reasoning"][w]["lp_mv"]) < 0.5 and
                               o["orig"]["mv_f"] == r["reasoning"][w]["mv_f"])
            rec["worlds"][w] = o
        out.append(rec)
        print(f"[edit] {k + 1}/{len(plan)} {f} ep{ei} t{t}: " + "; ".join(
            f"{w}: n={o['n_edits']} mv orig/edit/ctrl {o['orig']['mv_f']}/{o['edit']['mv_f']}/{o['control']['mv_f']} "
            f"ok={o['check_orig']}" for w, o in rec["worlds"].items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    bad = sum(not o["check_orig"] for r in out for o in r["worlds"].values())
    print(f"[edit] orig cells differing from cf_action: {bad}")
    json.dump({"schema": "cf_edit_v1", "model": cf["model"], "cf": args.cf, "mode": "think", "persist": "body",
               "format": "chat", "n_edited": n_ed, "pairs": out, "n_check_bad": bad,
               "wall_seconds": time.time() - t0}, open(args.out, "w"))
    print(f"[edit] wrote {args.out}")


if __name__ == "__main__":
    main()
