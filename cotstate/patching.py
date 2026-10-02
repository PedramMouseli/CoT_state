"""Counterfactual pairs and activation patching with the state readout (first version)."""

from __future__ import annotations

import argparse
import json
import random
import re
import time

import numpy as np

from cotstate.env import FILES
from cotstate.intervene import build_msgs
from cotstate.util import chat

STEM = "{f} is currently"
LOCKWORD = re.compile(r"lock", re.I)
TOUCH = ("mv", "rm", "stat", "lock", "unlock")


def untouched(ep: dict, t: int, f: str) -> bool:
    """True if nothing visible after the opening line says anything about f's lock."""
    turns = ep["turns"][: t + 1]
    first = turns[0]["obs"].splitlines()
    rest_obs = ["\n".join(l for l in first if not l.startswith(f + ":"))]
    rest_obs += [x["obs"] for x in turns[1:]]
    texts = rest_obs + [x.get("persisted", "") or "" for x in turns]
    for txt in texts:
        for line in txt.splitlines():
            if f in line and LOCKWORD.search(line):
                return False
    for x in turns:
        parts = (x.get("cmd") or "").strip().split()
        if len(parts) >= 2 and parts[0].lower() in TOUCH and parts[1] == f:
            return False
    init = int(re.search(rf"^{re.escape(f)}: (locked|unlocked)\s*$",
                         turns[0]["obs"], re.M).group(1) == "locked")
    return int(ep["turns"][t]["state_before"][f"locked_{f}"]) == init


def flip_opening(ep: dict, f: str) -> dict:
    """A copy of the episode whose opening observation states the opposite lock for f."""
    ep2 = json.loads(json.dumps(ep))
    o = ep2["turns"][0]["obs"]
    o2, n = re.subn(rf"^({re.escape(f)}: )(locked|unlocked)(\s*)$",
                    lambda m: m.group(1) + ("unlocked" if m.group(2) == "locked" else "locked")
                    + m.group(3), o, flags=re.M)
    assert n == 1
    ep2["turns"][0]["obs"] = o2
    return ep2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--probes", required=True, help="probe_v4.json, for test episodes and u")
    ap.add_argument("--n-pairs", type=int, default=80)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="default: 0,2,...,34,35")
    ap.add_argument("--min-sep", type=float, default=1.0,
                    help="drop pairs whose clean readouts differ by less than this (logits); "
                         "restoration is undefined when the two worlds give the same answer")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import cotstate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    P = json.load(open(args.probes))
    te = set(P["test_episodes"])
    eps = [json.loads(l) for l in open(args.transcripts)]

    cands = [(ei, t["turn"], f) for ei, ep in enumerate(eps) if ei in te
             for t in ep["turns"] if t["turn"] >= 2 for f in FILES
             if untouched(ep, t["turn"], f)]
    random.Random(11).shuffle(cands)
    print(f"[pt] {len(cands)} candidate (episode, turn, file) triples pass the untouched filter")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    nL = len(model.model.layers)
    layers = args.layers or sorted(set(list(range(0, nL, 2)) + [nL - 1]))
    ID_L = tok(" locked", add_special_tokens=False)["input_ids"][0]
    ID_U = tok(" unlocked", add_special_tokens=False)["input_ids"][0]

    def unit_normal(f, L):
        e = P["probes"].get(f"locked_{f}", {}).get("layers", {}).get(str(L))
        if not e or "scaler_scale" not in e:
            return None
        v = np.array(e["w"], np.float64) / np.array(e["scaler_scale"], np.float64)
        return torch.tensor(v / np.linalg.norm(v), dtype=torch.float32, device=args.device)

    st = {"mode": None, "L": None, "pos": None, "src": None, "u": None, "cap": None}

    def make_hook(L):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if st["mode"] == "capture" and L in st["cap"]:
                st["cap"][L] = h[0].detach().clone()
            elif st["mode"] == "patch" and st["L"] == L:
                h = h.clone()
                if st["u"] is None:
                    h[0, st["pos"], :] = st["src"][st["pos"], :].to(h.dtype)
                else:
                    diff = (st["src"][st["pos"], :].float() - h[0, st["pos"], :].float())
                    proj = (diff @ st["u"]).unsqueeze(-1) * st["u"]
                    h[0, st["pos"], :] = (h[0, st["pos"], :].float() + proj).to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return out
        return hook

    for L in layers:
        model.model.layers[L].register_forward_hook(make_hook(L))

    def fwd(ids):
        x = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            try:
                out = model(input_ids=x, logits_to_keep=1)
            except TypeError:
                out = model(input_ids=x)
        lp = torch.log_softmax(out.logits[0, -1].float(), -1)
        return float(lp[ID_L] - lp[ID_U])

    def ids_of(ep, t, f):
        p = chat(tok, build_msgs(ep, t, f)[0], thinking=False) + STEM.format(f=f)
        return tok(p, add_special_tokens=False)["input_ids"]

    pairs, dropped = [], {"length": 0, "diffpos": 0, "sep": 0}
    for (ei, t, f) in cands:
        if len(pairs) >= args.n_pairs:
            break
        ep = eps[ei]
        truth = int(ep["turns"][t]["state_before"][f"locked_{f}"])
        epL, epU = (ep, flip_opening(ep, f)) if truth == 1 else (flip_opening(ep, f), ep)
        a, b = ids_of(epL, t, f), ids_of(epU, t, f)
        if len(a) != len(b):
            dropped["length"] += 1
            continue
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) != 1:
            dropped["diffpos"] += 1
            continue
        s = diff[0]
        n = len(a)

        st["mode"], st["cap"] = "capture", {L: None for L in layers}
        xL = fwd(a)
        capL = st["cap"]
        st["mode"], st["cap"] = None, None
        xU = fwd(b)
        if xL - xU < args.min_sep:
            dropped["sep"] += 1
            continue

        groups = {"src": [s], "mid": list(range(s + 1, n - 1)), "last": [n - 1]}
        rec = {"episode": ei, "turn": t, "file": f, "truth": truth, "n_tokens": n,
               "src_pos": s, "x_locked": xL, "x_unlocked": xU, "patch": {}, "dironly": {}}
        for L in layers:
            for gname, pos in groups.items():
                st.update(mode="patch", L=L, pos=pos, src=capL[L], u=None)
                x = fwd(b)
                rec["patch"][f"{L}|{gname}"] = x
            u = unit_normal(f, L)
            if u is not None:
                st.update(mode="patch", L=L, pos=[n - 1], src=capL[L], u=u)
                rec["dironly"][f"{L}|last"] = fwd(b)
            st.update(mode=None, u=None, src=None)
        del capL
        torch.cuda.empty_cache()
        pairs.append(rec)
        el = time.time() - t0
        print(f"[pt] pair {len(pairs)}/{args.n_pairs}  {f} ep{ei} t{t}  "
              f"x_locked {xL:+.2f}  x_unlocked {xU:+.2f}  ({el:.0f}s)", flush=True)

    sep = np.array([p["x_locked"] - p["x_unlocked"] for p in pairs])
    print(f"\n[pt] {len(pairs)} pairs kept; dropped {dropped}; "
          f"textual control: x_locked - x_unlocked mean {sep.mean():+.2f}, "
          f"{(sep > 0).mean():.0%} positive")
    rng = np.random.default_rng(0)
    summary = {}
    print(f"\n{'layer':>5} " + " ".join(f"{g:>20}" for g in ("src", "mid", "last", "dir-only@last")))
    for L in layers:
        row = []
        for g in ("src", "mid", "last", "dironly"):
            R = []
            for p in pairs:
                key = f"{L}|last" if g == "dironly" else f"{L}|{g}"
                src = p["dironly"] if g == "dironly" else p["patch"]
                if key in src:
                    R.append((src[key] - p["x_unlocked"]) / (p["x_locked"] - p["x_unlocked"]))
            if R:
                R = np.array(R)
                bs = [R[rng.integers(0, len(R), len(R))].mean() for _ in range(2000)]
                summary[f"{L}|{g}"] = {"mean": float(R.mean()),
                                       "lo": float(np.percentile(bs, 2.5)),
                                       "hi": float(np.percentile(bs, 97.5)), "n": int(len(R))}
                row.append(f"{R.mean():+.2f} [{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]")
            else:
                row.append(f"{'-':>20}")
        print(f"{L:>5} " + " ".join(f"{c:>20}" for c in row))

    json.dump({"schema": "patching_v1", "model": args.model, "layers": layers,
               "n_pairs": len(pairs), "dropped": dropped, "pairs": pairs,
               "restoration": summary, "wall_seconds": time.time() - t0},
              open(args.out, "w"))
    print(f"\n[pt] wrote {args.out}")


if __name__ == "__main__":
    main()
