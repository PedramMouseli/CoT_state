"""Counterfactual pairs: state readout, action log-probability and greedy command, with and without reasoning."""

from __future__ import annotations

import argparse
import json
import random
import re
import time

import numpy as np

from cotstate.env import FILES
from cotstate.fixed_history2 import QUESTION, STEM, extract_body, frame, mcnemar_exact_p
from cotstate.patching import untouched
from cotstate.score_actions import parse_command


def flip_prompt(prompt: str, f: str) -> tuple[str, str] | None:
    """Return (locked_version, unlocked_version) of a saved prompt, or None if
    the opening line for f is not found exactly once."""
    pat = re.compile(rf"^({re.escape(f)}: )(locked|unlocked)([ \t]*)$", re.M)
    ms = list(pat.finditer(prompt))
    if len(ms) != 1:
        return None
    m = ms[0]
    mk = lambda v: prompt[:m.start()] + m.group(1) + v + m.group(3) + prompt[m.end():]
    return mk("locked"), mk("unlocked")


def stated_value(body: str, f: str):
    """Last lock value written right after a mention of f (descriptive only)."""
    ms = list(re.finditer(rf"{re.escape(f)}[^\n]{{0,40}}?\b(unlocked|locked)\b", body, re.I))
    if not ms:
        return None
    return int(ms[-1].group(1).lower() == "locked")


def cluster_ci(vals, clusters, n_boot=4000, seed=0):
    vals, clusters = np.asarray(vals, float), np.asarray(clusters)
    u = np.unique(clusters)
    idx = {c: np.where(clusters == c)[0] for c in u}
    rng = np.random.default_rng(seed)
    bs = [vals[np.concatenate([idx[c] for c in rng.choice(u, len(u))])].mean() for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarise(recs: list[dict]) -> dict:
    ep = [r["episode"] for r in recs]
    out = {"n_pairs": len(recs), "n_episodes": len(set(ep)), "arms": {}, "contrasts": {}}
    arms = [a for a in ("immediate", "reasoning") if all(a in r for r in recs)]
    per = {}
    for a in arms:
        bs = np.array([r[a]["L"]["belief"] - r[a]["U"]["belief"] for r in recs])
        ms = np.array([r[a]["U"]["lp_mv"] - r[a]["L"]["lp_mv"] for r in recs])
        mvL = np.array([r[a]["L"]["mv_f"] for r in recs], float)
        mvU = np.array([r[a]["U"]["mv_f"] for r in recs], float)
        per[a] = dict(bs=bs, ms=ms, mvL=mvL, mvU=mvU)
        o = {}
        for name, v in (("belief_sep", bs), ("mv_sep", ms), ("viol_locked", mvL),
                        ("mv_unlocked", mvU), ("disc", mvU - mvL)):
            lo, hi = cluster_ci(v, ep)
            o[name] = {"mean": float(v.mean()), "median": float(np.median(v)), "lo": lo, "hi": hi,
                       "share_pos": float((v > 0).mean())}
        if a == "reasoning":
            for w in ("L", "U"):
                sv = [r[a][w].get("stated") for r in recs]
                o[f"mentions_f_{w}"] = float(np.mean([r[a][w].get("mentions_f", False) for r in recs]))
                o[f"states_value_{w}"] = float(np.mean([s is not None for s in sv]))
                truth = 1 if w == "L" else 0
                ok = [s == truth for s in sv if s is not None]
                o[f"stated_correct_{w}"] = float(np.mean(ok)) if ok else None
                o[f"closed_{w}"] = float(np.mean([r[a][w].get("closed", True) for r in recs]))
        out["arms"][a] = o
    if len(arms) == 2:
        for name in ("mv_sep", "disc"):
            key = "ms" if name == "mv_sep" else None
            a_ = per["immediate"]["ms"] if key else per["immediate"]["mvU"] - per["immediate"]["mvL"]
            b_ = per["reasoning"]["ms"] if key else per["reasoning"]["mvU"] - per["reasoning"]["mvL"]
            lo, hi = cluster_ci(b_ - a_, ep)
            out["contrasts"][f"{name}: reasoning - immediate"] = {"diff": float((b_ - a_).mean()),
                                                                   "lo": lo, "hi": hi}
        vi, vr = per["immediate"]["mvL"], per["reasoning"]["mvL"]
        ao, bo = int(((vi == 1) & (vr == 0)).sum()), int(((vi == 0) & (vr == 1)).sum())
        out["contrasts"]["violations in locked world: immediate vs reasoning"] = {
            "immediate_only": ao, "reasoning_only": bo, "mcnemar_p": mcnemar_exact_p(ao, bo)}
    return out


def report(S: dict) -> None:
    print(f"\n[cf] {S['n_pairs']} pairs from {S['n_episodes']} episodes")
    print(f"{'arm':<10}{'belief_sep':>18}{'mv_sep (logits)':>22}{'P(mv f|U)':>12}{'P(mv f|L)':>12}{'disc':>20}")
    for a, o in S["arms"].items():
        print(f"{a:<10}{o['belief_sep']['median']:>+9.2f} med ({100 * o['belief_sep']['share_pos']:.0f}%>0)"
              f"{o['mv_sep']['mean']:>+10.2f} [{o['mv_sep']['lo']:+.2f},{o['mv_sep']['hi']:+.2f}]"
              f"{o['mv_unlocked']['mean']:>12.3f}{o['viol_locked']['mean']:>12.3f}"
              f"{o['disc']['mean']:>+9.3f} [{o['disc']['lo']:+.3f},{o['disc']['hi']:+.3f}]")
    for k, c in S["contrasts"].items():
        print(f"  {k}: {c}")
    r = S["arms"].get("reasoning")
    if r:
        print(f"  reasoning mentions f: L {r['mentions_f_L']:.2f} U {r['mentions_f_U']:.2f}; states a value: "
              f"L {r['states_value_L']:.2f} U {r['states_value_U']:.2f}; value correct: "
              f"L {r['stated_correct_L']} U {r['stated_correct_U']}; closed: L {r['closed_L']:.2f} U {r['closed_U']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--transcripts")
    ap.add_argument("--probes", help="json with test_episodes")
    ap.add_argument("--n-pairs", type=int, default=160)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.merge:
        recs, meta = [], None
        for p in args.merge:
            d = json.load(open(p))
            recs += d["pairs"]
            meta = meta or {k: d[k] for k in ("model", "transcripts", "max_new")}
        keys = [(r["episode"], r["turn"], r["file"]) for r in recs]
        assert len(keys) == len(set(keys)), "shards overlap"
        S = summarise(recs)
        report(S)
        json.dump({"schema": "cf_action_v1", **meta, "merged_from": args.merge, "pairs": recs,
                   "summary": S}, open(args.out, "w"))
        print(f"[cf] wrote {args.out}")
        return

    import torch
    import cotstate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    if args.pilot:
        args.n_pairs, args.max_new = 3, 300
    te = set(json.load(open(args.probes))["test_episodes"])
    eps = [json.loads(l) for l in open(args.transcripts)]

    cands = [(ei, t["turn"], f) for ei in sorted(te) for t in eps[ei]["turns"]
             if t["turn"] >= 2 and "prompt" in t for f in FILES
             if int(t["state_before"][f"exists_{f}"]) == 1 and untouched(eps[ei], t["turn"], f)]
    random.Random(23).shuffle(cands)
    pairs, dropped = [], 0
    for c in cands:
        if len(pairs) >= args.n_pairs:
            break
        ei, t, f = c
        pr = flip_prompt(eps[ei]["turns"][t]["prompt"], f)
        if pr is None:
            dropped += 1
            continue
        pairs.append((ei, t, f, pr))
    pairs = pairs[args.shard::args.nshards]
    print(f"[cf] {len(cands)} candidates; {dropped} dropped (opening line not unique); "
          f"shard {args.shard}/{args.nshards}: {len(pairs)} pairs")

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    ID_L = tok(" locked", add_special_tokens=False)["input_ids"][0]
    ID_U = tok(" unlocked", add_special_tokens=False)["input_ids"][0]
    stop_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
    extra = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(extra, int):
        stop_ids.add(extra)
    elif extra:
        stop_ids.update(extra)

    def gen(prompts, max_new):
        out = []
        for b in range(0, len(prompts), args.batch):
            enc = tok(prompts[b:b + args.batch], return_tensors="pt", padding=True).to(args.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            n_in = enc["input_ids"].shape[1]
            for j in range(g.shape[0]):
                new = g[j][n_in:].tolist()
                n = next((k for k, x in enumerate(new) if x in stop_ids), len(new))
                out.append(tok.decode(new[:n], skip_special_tokens=True))
            print(f"[cf]   generated {min(b + args.batch, len(prompts))}/{len(prompts)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        return out

    def fwd_last(ids, keep=1):
        x = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            try:
                lg = model(input_ids=x, logits_to_keep=keep).logits[0]
            except TypeError:
                lg = model(input_ids=x).logits[0, -keep:]
        return torch.log_softmax(lg.float(), -1)

    def belief(prompt):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > args.max_len:
            raise SystemExit("[cf] prompt longer than --max-len")
        lp = fwd_last(ids)[-1]
        return float(lp[ID_L] - lp[ID_U])

    def lp_cont(prompt, cont):
        ip = tok(prompt, add_special_tokens=False)["input_ids"]
        ic = tok(cont, add_special_tokens=False)["input_ids"]
        lp = fwd_last(ip + ic, keep=len(ic) + 1)[:-1]
        return float(sum(lp[k, t] for k, t in enumerate(ic)))

    def act(prompt):
        return gen([prompt], 12)[0]

    worlds = [(k, w, pr[0] if w == "L" else pr[1]) for k, (_, _, _, pr) in enumerate(pairs) for w in "LU"]
    print(f"[cf] generating reasoning for {len(worlds)} prompts (budget {args.max_new})")
    order = sorted(range(len(worlds)), key=lambda i: len(worlds[i][2]))
    texts = gen([worlds[i][2] for i in order], args.max_new)
    raw = {worlds[i][:2]: txt for i, txt in zip(order, texts)}

    recs = []
    for k, (ei, t, f, (pL, pU)) in enumerate(pairs):
        rec = {"episode": ei, "turn": t, "file": f,
               "state_before": eps[ei]["turns"][t]["state_before"], "immediate": {}, "reasoning": {}}
        for w, p in (("L", pL), ("U", pU)):
            body, closed = extract_body(p, raw[(k, w)])
            for arm, b in (("immediate", ""), ("reasoning", body)):
                pa = frame(p, b, "CMD:")
                pb = frame(p, b, QUESTION.format(f=f) + "\n" + STEM.format(f=f))
                txt = act(pa)
                verb, arg = parse_command("CMD: " + txt if "CMD:" not in txt else txt)
                r = {"belief": belief(pb), "lp_mv": lp_cont(pa, " mv " + f),
                     "action": txt.strip()[:120], "verb": verb, "arg": arg,
                     "mv_f": int(verb in ("mv", "rm") and arg == f)}
                if arm == "reasoning":
                    sv = stated_value(b, f)
                    r.update({"closed": bool(closed), "body_chars": len(b),
                              "mentions_f": f in b, "stated": sv, "body": b[:20000]})
                rec[arm][w] = r
        recs.append(rec)
        print(f"[cf] pair {k + 1}/{len(pairs)} {f} ep{ei} t{t}: belief_sep imm "
              f"{rec['immediate']['L']['belief'] - rec['immediate']['U']['belief']:+.1f}, "
              f"mv_sep imm {rec['immediate']['U']['lp_mv'] - rec['immediate']['L']['lp_mv']:+.2f} "
              f"rsn {rec['reasoning']['U']['lp_mv'] - rec['reasoning']['L']['lp_mv']:+.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    S = summarise(recs)
    report(S)
    json.dump({"schema": "cf_action_v1", "model": args.model, "transcripts": args.transcripts,
               "max_new": args.max_new, "shard": args.shard, "nshards": args.nshards,
               "pairs": recs, "summary": S, "wall_seconds": time.time() - t0}, open(args.out, "w"))
    print(f"[cf] wrote {args.out}")


if __name__ == "__main__":
    main()
