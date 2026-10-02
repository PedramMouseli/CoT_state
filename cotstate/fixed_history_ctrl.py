"""Fixed-history content controls: filler and other-history reasoning arms."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from cotstate.env import FILES
from cotstate.fixed_history2 import (QUESTION, STEM, balanced_accuracy, cluster_boot_auroc,
                               cluster_boot_rate, frame, mcnemar_exact_p)
from cotstate.score_actions import parse_command, score_command

ARMS = ["immediate", "reasoning", "filler", "foreign"]


def donors(item_eps: list[int], seed: int = 0) -> list[int]:
    """For each item, the index of another item from a DIFFERENT episode.
    A fixed derangement: shift by half the list in a seeded order, then repair."""
    n = len(item_eps)
    order = np.random.RandomState(seed).permutation(n)
    d = [0] * n
    for pos, k in enumerate(order):
        j = order[(pos + n // 2) % n]
        step = 1
        while item_eps[j] == item_eps[k]:
            if step > n:
                raise SystemExit("[fc] no donor from a different episode")
            j = order[(pos + n // 2 + step) % n]
            step += 1
        d[k] = int(j)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--v3", required=True, help="fixed_history_v3.json with saved reasoning bodies")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eligibility", default="present", choices=["present", "all"])
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    import torch
    import cotstate
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    V = json.load(open(args.v3))
    eps = [json.loads(l) for l in open(args.transcripts)]
    all_items = V["items"]
    dn_all = donors([it["episode"] for it in all_items])
    items = all_items[: (args.pilot or None)]
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    ID_L = tok(" locked", add_special_tokens=False)["input_ids"][0]
    ID_U = tok(" unlocked", add_special_tokens=False)["input_ids"][0]

    ep_of = np.array([it["episode"] for it in items])
    is_calib = np.array([bool(it["is_calibration"]) for it in items])
    own = [it["reasoning_body"] for it in items]
    ntok = [len(tok(b, add_special_tokens=False)["input_ids"]) for b in own]
    dn = dn_all[: len(items)]
    all_own = [it["reasoning_body"] for it in all_items]
    bodies = {"immediate": [""] * len(items), "reasoning": own,
              "filler": [" ".join(["."] * n) for n in ntok],
              "foreign": [all_own[j] for j in dn]}
    fill_tok = [len(tok(b, add_special_tokens=False)["input_ids"]) for b in bodies["filler"]]
    print(f"[fc] {len(items)} histories, {len(set(ep_of.tolist()))} episodes; own reasoning median "
          f"{int(np.median(ntok))} tokens; filler median {int(np.median(fill_tok))} tokens; "
          f"foreign median {int(np.median([len(tok(b, add_special_tokens=False)['input_ids']) for b in bodies['foreign']]))} tokens")

    def prefix(it):
        return eps[it["episode"]]["turns"][it["turn"]]["prompt"]

    def sb(it):
        return eps[it["episode"]]["turns"][it["turn"]]["state_before"]

    if args.pilot:
        for it_k, it in enumerate(items):
            for arm in ARMS:
                p = frame(prefix(it), bodies[arm][it_k], "CMD:")
                print(f"\n----- item {it_k} arm {arm}: {len(tok(p)['input_ids'])} tokens; tail:\n"
                      f"{p[-400:]}")
        print("\nCheck: every arm ends '</think>\\n\\nCMD:'; filler is dots; foreign is another "
              "history's reasoning. Then re-run without --pilot.")
        return

    def batches(prompts):
        order = np.argsort([len(p) for p in prompts])
        for b in range(0, len(order), args.batch):
            yield order[b:b + args.batch]

    def run_logit(prompts):
        out = np.zeros(len(prompts))
        for idx in batches(prompts):
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(args.device)
            if enc["input_ids"].shape[1] > args.max_len:
                raise SystemExit("[fc] prompt longer than --max-len; lower --batch, do not truncate")
            with torch.no_grad():
                try:
                    lg = model(**enc, logits_to_keep=1).logits[:, -1, :]
                except TypeError:
                    lg = model(**enc).logits[:, -1, :]
            lp = torch.log_softmax(lg.float(), -1)
            out[idx] = (lp[:, ID_L] - lp[:, ID_U]).cpu().numpy()
        return out.tolist()

    def run_gen(prompts, max_new=12):
        out = [None] * len(prompts)
        for idx in batches(prompts):
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(args.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            n_in = enc["input_ids"].shape[1]
            for r, i in enumerate(idx):
                out[i] = tok.decode(g[r][n_in:], skip_special_tokens=True)
        return out

    raw, acts = {}, {}
    for arm in ARMS:
        raw[arm] = {}
        for f in FILES:
            raw[arm][f] = run_logit([frame(prefix(it), bodies[arm][k],
                                           QUESTION.format(f=f) + "\n" + STEM.format(f=f))
                                     for k, it in enumerate(items)])
        texts = run_gen([frame(prefix(it), bodies[arm][k], "CMD:") for k, it in enumerate(items)])
        acts[arm] = []
        for it, txt in zip(items, texts):
            verb, a = parse_command("CMD: " + txt if "CMD:" not in txt else txt)
            sc = score_command(verb, a, sb(it), args.eligibility)
            sc["text"] = txt.strip()[:200]
            acts[arm].append(sc)
        print(f"[fc] arm {arm} done ({time.time() - t0:.0f}s)", flush=True)

    check = {}
    for arm in ("immediate", "reasoning"):
        a = np.concatenate([raw[arm][f] for f in FILES])
        b = np.concatenate([[it[f"{arm}_logit_{f}"] for it in items] for f in FILES])
        agree = np.mean([acts[arm][k]["verb"] == it[f"{arm}_action"]["verb"] and
                         acts[arm][k]["arg"] == it[f"{arm}_action"]["arg"]
                         for k, it in enumerate(items)])
        check[arm] = {"logit_corr": float(np.corrcoef(a, b)[0, 1]),
                      "logit_max_abs_diff": float(np.abs(a - b).max()),
                      "action_agreement": float(agree)}
        print(f"[fc] check vs v3, {arm}: logit corr {check[arm]['logit_corr']:.4f}, "
              f"max |diff| {check[arm]['logit_max_abs_diff']:.2f}; same command on "
              f"{100 * agree:.1f}% of histories")

    y = np.concatenate([[sb(it)[f"locked_{f}"] for it in items] for f in FILES]).astype(int)
    ep_rep = np.concatenate([ep_of] * len(FILES))
    calib_rep = np.concatenate([is_calib] * len(FILES))
    S = {arm: np.concatenate([raw[arm][f] for f in FILES]) for arm in ARMS}
    belief = {"per_arm": {}, "contrasts": {}}
    for arm in ARMS:
        s = S[arm]
        cand = np.unique(s[calib_rep])
        thr = max(((balanced_accuracy(y[calib_rep], (s[calib_rep] > t).astype(int)), t) for t in cand))[1]
        belief["per_arm"][arm] = {"auroc_all": float(roc_auc_score(y, s)),
                                  "bal_acc_eval_half": balanced_accuracy(
                                      y[~calib_rep], (s[~calib_rep] > thr).astype(int))}
    pairs = [("immediate", "reasoning"), ("immediate", "filler"), ("immediate", "foreign"),
             ("filler", "reasoning"), ("foreign", "reasoning")]
    for a_, b_ in pairs:
        belief["contrasts"][f"{a_}->{b_}"] = cluster_boot_auroc(y, S[a_], S[b_], ep_rep)
    action = {"eligibility": args.eligibility, "per_arm": {}, "contrasts": {}}
    for arm in ARMS:
        v = np.array([a["violation"] for a in acts[arm]], float)
        n = np.array([a["attempt"] for a in acts[arm]])
        lo, hi = cluster_boot_rate(v, ep_of)
        verbs = {}
        for a in acts[arm]:
            verbs[a["verb"] or "?"] = verbs.get(a["verb"] or "?", 0) + 1
        action["per_arm"][arm] = {"attempts": int(n.sum()), "violations": int(v.sum()),
                                  "rate_per_attempt": float(v.sum() / n.sum()) if n.sum() else None,
                                  "violations_per_item": float(v.mean()),
                                  "violations_per_item_ci": [lo, hi], "verbs": verbs}
    for a_, b_ in pairs:
        va = np.array([a["violation"] for a in acts[a_]])
        vb = np.array([a["violation"] for a in acts[b_]])
        ao, bo = int(((va == 1) & (vb == 0)).sum()), int(((va == 0) & (vb == 1)).sum())
        action["contrasts"][f"{a_}->{b_}"] = {"a_only": ao, "b_only": bo,
                                              "mcnemar_p_items": mcnemar_exact_p(ao, bo)}

    print(f"\n{'arm':<11}{'AUROC':>7}{'bal-acc':>9}{'attempts':>10}{'viol':>6}{'viol/100 hist':>15}  verbs")
    for arm in ARMS:
        b, a = belief["per_arm"][arm], action["per_arm"][arm]
        print(f"{arm:<11}{b['auroc_all']:>7.3f}{b['bal_acc_eval_half']:>9.3f}{a['attempts']:>10}"
              f"{a['violations']:>6}{100 * a['violations_per_item']:>15.2f}  {a['verbs']}")
    for k, c in belief["contrasts"].items():
        print(f"belief {k:<22} {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]")
    for k, c in action["contrasts"].items():
        print(f"action {k:<22} {c['a_only']} vs {c['b_only']}, p = {c['mcnemar_p_items']:.3g}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"schema": "fixed_history_ctrl_v1", "model": args.model, "v3": args.v3,
               "n_items": len(items), "own_tokens": ntok, "filler_tokens": fill_tok, "donor": dn,
               "check_vs_v3": check, "belief": belief, "action": action,
               "items": [{"i": k, "episode": it["episode"], "turn": it["turn"],
                          **{f"{arm}_action": acts[arm][k] for arm in ARMS},
                          **{f"{arm}_logit_{f}": raw[arm][f][k] for arm in ARMS for f in FILES}}
                         for k, it in enumerate(items)],
               "wall_seconds": time.time() - t0}, open(args.out, "w"))
    print(f"\n[fc] wrote {args.out}")


if __name__ == "__main__":
    main()
