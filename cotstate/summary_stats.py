"""Episode-clustered statistics for the fixed-history, counterfactual, swap and edit experiments, and task-success variants."""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

B = 10000


def ep_boot(vals, clusters, seed=0, n=B):
    vals, clusters = np.asarray(vals, float), np.asarray(clusters)
    u = np.unique(clusters)
    idx = [np.where(clusters == c)[0] for c in u]
    sums = np.array([vals[i].sum() for i in idx])
    cnts = np.array([len(i) for i in idx])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(u), (n, len(u)))
    m = sums[draw].sum(1) / cnts[draw].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def signflip_p(vals, clusters, seed=0, n_mc=200000):
    """Two-sided episode-level sign-flip test of mean(vals) = 0."""
    vals, clusters = np.asarray(vals, float), np.asarray(clusters)
    d = np.array([vals[clusters == c].sum() for c in np.unique(clusters)])
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    T = abs(d.sum())
    if len(d) <= 20:
        tot = [abs(np.dot(s, d)) for s in itertools.product((-1, 1), repeat=len(d))]
        return float(np.mean(np.array(tot) >= T - 1e-12))
    rng = np.random.default_rng(seed)
    s = rng.choice((-1, 1), (n_mc, len(d)))
    return float(np.mean(np.abs(s @ d) >= T - 1e-12))


def contrast(a, b, ep):
    d = np.asarray(b, float) - np.asarray(a, float)
    lo, hi = ep_boot(d, ep)
    return {"n": len(d), "n_episodes": int(len(set(ep))), "a": int(np.sum(a)), "b": int(np.sum(b)),
            "diff_per_100": 100 * float(d.mean()), "lo": 100 * lo, "hi": 100 * hi,
            "signflip_p": signflip_p(d, ep), "a_only": int(np.sum((np.asarray(a) == 1) & (np.asarray(b) == 0))),
            "b_only": int(np.sum((np.asarray(a) == 0) & (np.asarray(b) == 1)))}


def fixed(model_dir):
    v3 = json.load(open(f"{model_dir}/fixed_history_v3.json"))["items"]
    ct = json.load(open(f"{model_dir}/fixed_history_ctrl.json"))["items"]
    out = {}
    ep = [x["episode"] for x in v3]
    V = lambda items, arm: [x[f"{arm}_action"]["violation"] for x in items]
    for arm in ("reasoning", "statetable"):
        out[f"v3: immediate->{arm}"] = contrast(V(v3, "immediate"), V(v3, arm), ep)
    epc = [x["episode"] for x in ct]
    for arm in ("filler", "foreign", "reasoning"):
        out[f"ctrl: immediate->{arm}"] = contrast(V(ct, "immediate"), V(ct, arm), epc)
    out["ctrl: filler->reasoning"] = contrast(V(ct, "filler"), V(ct, "reasoning"), epc)
    A = lambda items, arm: int(sum(x[f"{arm}_action"]["attempt"] for x in items))
    out["attempts"] = {**{f"v3:{a}": A(v3, a) for a in ("immediate", "reasoning", "statetable")},
                       **{f"ctrl:{a}": A(ct, a) for a in ("immediate", "filler", "foreign", "reasoning")}}
    return out


def cf(path):
    P = json.load(open(path))["pairs"]
    out = {}
    for name, sel in (("all", lambda r: True), ("logs", lambda r: r["file"] != "report.txt"),
                      ("goal_file", lambda r: r["file"] == "report.txt")):
        R = [r for r in P if sel(r)]
        ep = [r["episode"] for r in R]
        o = {"n_pairs": len(R), "n_episodes": len(set(ep))}
        for arm in ("immediate", "reasoning"):
            for w in "LU":
                o[f"{arm}_mv_{w}"] = int(sum(r[arm][w]["mv_f"] for r in R))
        o["locked_moves_immediate_vs_reasoning"] = contrast([r["immediate"]["L"]["mv_f"] for r in R],
                                                            [r["reasoning"]["L"]["mv_f"] for r in R], ep)
        o["unlocked_moves_immediate_vs_reasoning"] = contrast([r["immediate"]["U"]["mv_f"] for r in R],
                                                              [r["reasoning"]["U"]["mv_f"] for r in R], ep)
        out[name] = o
    return out


def swap(path):
    D = json.load(open(path))
    P = D["pairs"]
    out = {"n_check_bad": D["n_check_bad"]}
    for name, sel in (("logs", lambda r: r["file"] != "report.txt"),
                      ("goal_file", lambda r: r["file"] == "report.txt"),
                      ("logs_closed", lambda r: r["file"] != "report.txt" and all(r["closed"].values()))):
        R = [r for r in P if sel(r)]
        if not R:
            continue
        ep = [r["episode"] for r in R]
        o = {"n_pairs": len(R), "n_episodes": len(set(ep)), "cells": {}}
        for c in ("LL", "LU", "UL", "UU"):
            mv = [r["cells"][c]["mv_f"] for r in R]
            lp = [r["cells"][c]["lp_mv"] for r in R]
            bl = [r["cells"][c]["belief"] for r in R]
            lo, hi = ep_boot(mv, ep)
            o["cells"][c] = {"mv": int(sum(mv)), "mv_rate": float(np.mean(mv)), "mv_lo": lo, "mv_hi": hi,
                             "lp_mv_mean": float(np.mean(lp)), "belief_mean": float(np.mean(bl)),
                             "belief_locked_share": float(np.mean(np.array(bl) > 0))}
        mv = lambda c: [r["cells"][c]["mv_f"] for r in R]
        lpd = lambda c1, c2: [r["cells"][c2]["lp_mv"] - r["cells"][c1]["lp_mv"] for r in R]
        o["body_effect_in_locked_history (LL->LU)"] = contrast(mv("LL"), mv("LU"), ep)
        o["body_effect_in_unlocked_history (UL->UU)"] = contrast(mv("UL"), mv("UU"), ep)
        o["history_effect_with_locked_body (LL->UL)"] = contrast(mv("LL"), mv("UL"), ep)
        o["history_effect_with_unlocked_body (LU->UU)"] = contrast(mv("LU"), mv("UU"), ep)
        for k, (c1, c2) in {"lp_body_effect_L": ("LL", "LU"), "lp_body_effect_U": ("UL", "UU"),
                            "lp_history_effect_bodyL": ("LL", "UL"), "lp_history_effect_bodyU": ("LU", "UU")}.items():
            d = lpd(c1, c2)
            lo, hi = ep_boot(d, ep)
            o[k] = {"mean": float(np.mean(d)), "lo": lo, "hi": hi}
        for k, (c1, c2) in {"belief_body_effect_L": ("LU", "LL"), "belief_history_effect_bodyU": ("UU", "LU")}.items():
            d = [r["cells"][c2]["belief"] - r["cells"][c1]["belief"] for r in R]
            lo, hi = ep_boot(d, ep)
            o[k] = {"mean": float(np.mean(d)), "lo": lo, "hi": hi}
        out[name] = o
    return out


def edit(path):
    D = json.load(open(path))
    out = {"n_check_bad": D["n_check_bad"], "n_edited": D["n_edited"]}
    for w in "LU":
        R = [r for r in D["pairs"] if w in r["worlds"]]
        ep = [r["episode"] for r in R]
        g = lambda arm, k: [r["worlds"][w][arm][k] for r in R]
        o = {"n_pairs": len(R), "n_episodes": len(set(ep)),
             "edits_per_trace": float(np.mean([r["worlds"][w]["n_edits"] for r in R]))}
        for arm in ("orig", "control", "edit"):
            o[arm] = {"mv": int(sum(g(arm, "mv_f"))), "lp_mv": float(np.mean(g(arm, "lp_mv"))),
                      "belief": float(np.mean(g(arm, "belief")))}
        o["moves control->edit"] = contrast(g("control", "mv_f"), g("edit", "mv_f"), ep)
        o["moves orig->control"] = contrast(g("orig", "mv_f"), g("control", "mv_f"), ep)
        for k in ("lp_mv", "belief"):
            for a, b in (("control", "edit"), ("orig", "control")):
                d = np.array(g(b, k)) - np.array(g(a, k))
                lo, hi = ep_boot(d, ep)
                o[f"{k} {a}->{b}"] = {"mean": float(d.mean()), "lo": lo, "hi": hi}
        out[w] = o
    return out


def success(specs):
    from cotstate.task_outcomes import replay
    import cotstate.env as env
    res, rows = {}, {}
    for name, d in specs.items():
        eps = [json.loads(l) for l in open(os.path.join(d, "transcripts.jsonl"))]
        r = []
        for e in eps:
            o = replay(e)
            any_rm = any(((t.get("cmd") or "").strip().split() or [""])[0].lower() == "rm" for t in e["turns"])
            o["success_no_lock_log"] = int(o["success"] and o["lock_logs"] == 0)
            o["success_no_rm_any"] = int(o["success"] and not any_rm)
            o["seed"] = e["seed"]
            r.append(o)
        rows[name] = {x["seed"]: x for x in r}
        res[name] = {k: int(sum(x[k] for x in r)) for k in ("success", "success_no_lock_log", "success_no_rm_any")}
        res[name]["episodes"] = len(r)
        res[name]["mismatch"] = int(sum(x["mismatch"] for x in r))
    con = {}
    for a, b in (("q3c-terse", "q3c-think-cmd"), ("q3c-think-cmd", "q3c-think-full"),
                 ("q35-terse", "q35-think-cmd"), ("q35-think-cmd", "q35-think-full")):
        if a in rows and b in rows:
            seeds = sorted(set(rows[a]) & set(rows[b]))
            for k in ("success", "success_no_lock_log"):
                dv = np.array([rows[b][s][k] - rows[a][s][k] for s in seeds], float)
                lo, hi = ep_boot(dv, seeds)
                con[f"{a}->{b}:{k}"] = {"diff_pct": 100 * dv.mean(), "lo": 100 * lo, "hi": 100 * hi}
    return {"counts": res, "contrasts": con}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/summary_stats.json")
    args = ap.parse_args()
    out = {"fixed": {"q3": fixed("data/q3_8b_clean"), "q35": fixed("data/q35_9b")},
           "cf": {"q3": cf("data/cf_action_q3.json"), "q35": cf("data/cf_action_q35.json")}}
    out["swap"] = {m: swap(f"data/cf_swap_{m}.json") for m in ("q3", "q35") if os.path.exists(f"data/cf_swap_{m}.json")}
    out["edit"] = {m: edit(f"data/cf_edit_{m}.json") for m in ("q3", "q35") if os.path.exists(f"data/cf_edit_{m}.json")}
    out["matched"] = {m: {k: json.load(open(f"data/patching_matched_{m}.json"))[k]
                          for k in ("n_pairs", "n_candidates", "dropped", "summary", "checks")}
                      for m in ("q3", "q35") if os.path.exists(f"data/patching_matched_{m}.json")}
    specs = {f"q3c-{c}": f"data/q3_8b_clean/{d}_chat" for c, d in
             (("terse", "terse_cmd"), ("think-cmd", "think_cmd"), ("think-body", "think_body"), ("think-full", "think_full"))}
    specs.update({f"q35-{c}": f"data/q35_9b/{d}_chat" for c, d in
                  (("terse", "terse_cmd"), ("think-cmd", "think_cmd"), ("think-body", "think_body"), ("think-full", "think_full"))})
    out["success"] = success(specs)
    json.dump(out, open(args.out, "w"), indent=1)

    for m, F in out["fixed"].items():
        print(f"\n[fixed {m}] attempts {F['attempts']}")
        for k, c in F.items():
            if k != "attempts":
                print(f"  {k:<28} {c['a']:>3}->{c['b']:<3} {c['diff_per_100']:+6.2f} [{c['lo']:+6.2f},{c['hi']:+6.2f}] "
                      f"p_signflip={c['signflip_p']:.4g}  ({c['n_episodes']} episodes)")
    for m, C in out["cf"].items():
        for s, o in C.items():
            c = o["locked_moves_immediate_vs_reasoning"]
            print(f"[cf {m} {s}] n={o['n_pairs']} ({o['n_episodes']} eps) imm L/U {o['immediate_mv_L']}/{o['immediate_mv_U']} "
                  f"rsn L/U {o['reasoning_mv_L']}/{o['reasoning_mv_U']}; locked moves p_signflip={c['signflip_p']:.4g}")
    for m, S in out["swap"].items():
        print(f"\n[swap {m}] diagonal check failures: {S['n_check_bad']}")
        for s, o in S.items():
            if s == "n_check_bad":
                continue
            print(f"  {s}: n={o['n_pairs']} ({o['n_episodes']} eps); mv counts " +
                  " ".join(f"{c}={v['mv']}" for c, v in o["cells"].items()) +
                  "; lp_mv " + " ".join(f"{c}={v['lp_mv_mean']:.2f}" for c, v in o["cells"].items()) +
                  "; belief>0 " + " ".join(f"{c}={v['belief_locked_share']:.2f}" for c, v in o["cells"].items()))
            for k in [k for k in o if "effect" in k and isinstance(o[k], dict) and "diff_per_100" in o[k]]:
                c = o[k]
                print(f"    {k:<44} {c['a']:>3}->{c['b']:<3} {c['diff_per_100']:+6.1f} [{c['lo']:+6.1f},{c['hi']:+6.1f}] p={c['signflip_p']:.3g}")
            for k in [k for k in o if k.startswith(("lp_", "belief_")) and isinstance(o[k], dict)]:
                print(f"    {k:<44} {o[k]['mean']:+.2f} [{o[k]['lo']:+.2f},{o[k]['hi']:+.2f}]")
    for m, E in out["edit"].items():
        print(f"\n[edit {m}] orig mismatches {E['n_check_bad']}; traces edited {E['n_edited']}")
        for w in "LU":
            o = E[w]
            print(f"  world {w}: n={o['n_pairs']} ({o['n_episodes']} eps), edits/trace {o['edits_per_trace']:.1f}; moves orig/control/edit "
                  f"{o['orig']['mv']}/{o['control']['mv']}/{o['edit']['mv']}; lp {o['orig']['lp_mv']:.2f}/{o['control']['lp_mv']:.2f}/{o['edit']['lp_mv']:.2f}; "
                  f"s_f {o['orig']['belief']:.2f}/{o['control']['belief']:.2f}/{o['edit']['belief']:.2f}")
            for k in [k for k in o if "->" in k]:
                c = o[k]
                if "diff_per_100" in c:
                    print(f"    {k:<28} {c['diff_per_100']:+6.1f} [{c['lo']:+6.1f},{c['hi']:+6.1f}] p={c['signflip_p']:.3g}")
                else:
                    print(f"    {k:<28} {c['mean']:+6.2f} [{c['lo']:+6.2f},{c['hi']:+6.2f}]")
    print("\n[success]", json.dumps(out["success"]["counts"]))
    for k, c in out["success"]["contrasts"].items():
        print(f"  {k:<50} {c['diff_pct']:+.1f} [{c['lo']:+.1f},{c['hi']:+.1f}]")


if __name__ == "__main__":
    main()
