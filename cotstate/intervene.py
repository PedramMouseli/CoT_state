"""Text corruption of lock statements and the state readout before and after."""

from __future__ import annotations

import argparse, json, random, re
import numpy as np
from cotstate.env import FILES
from cotstate.probe import LOCK_RE
from cotstate.util import chat
from cotstate.stats import mcnemar_exact, wilson

QUESTION = ("Is {f} currently locked? Ignore the task for one turn and answer "
            "with exactly one word: LOCKED or UNLOCKED.")


def parse_answer(text: str):
    body = text.split("</think>")[-1].upper()
    u = "UNLOCKED" in body
    l = "LOCKED" in body.replace("UNLOCKED", "")
    if u == l:
        return None
    return 1 if l else 0


def _flip(seg: str) -> str:
    if re.search(r"\bunlocked\b", seg, re.I):
        return re.sub(r"\bunlocked\b", "locked", seg, flags=re.I)
    return re.sub(r"\blocked\b", "unlocked", seg, flags=re.I)


def corrupt_text(s: str, f: str, mode: str = "last") -> tuple[str, int]:
    """Flip lock assertions about f. Returns (new_s, n_flipped)."""
    matches = list(LOCK_RE[f].finditer(s))
    if not matches:
        return s, 0
    if mode == "last":
        matches = matches[-1:]
    out, prev, n = [], 0, 0
    for m in matches:
        out.append(s[prev:m.start()])
        out.append(_flip(m.group(0)))
        prev = m.end()
        n += 1
    out.append(s[prev:])
    return "".join(out), n


def build_msgs(ep, upto_turn, f, corrupt=False, mode="last"):
    msgs = [{"role": "system", "content": ep["system"]}]
    for t in ep["turns"][:upto_turn + 1]:
        msgs.append({"role": "user", "content": t["obs"]})
        msgs.append({"role": "assistant",
                     "content": t.get("persisted", t["raw"].split("</think>")[-1].strip())})
    n_flipped = 0
    if corrupt:
        if mode == "all":
            for i in range(1, len(msgs)):
                new, n = corrupt_text(msgs[i]["content"], f, "all")
                if n:
                    msgs[i] = {**msgs[i], "content": new}
                    n_flipped += n
        else:
            for i in range(len(msgs) - 1, 0, -1):
                new, n = corrupt_text(msgs[i]["content"], f, "last")
                if n:
                    msgs[i] = {**msgs[i], "content": new}
                    n_flipped = n
                    break
    msgs.append({"role": "user", "content": QUESTION.format(f=f)})
    return msgs, n_flipped


def pick_target(P, how: str) -> str:
    lock = {k: v for k, v in P["probes"].items() if k.startswith("locked_")}
    if how in lock:
        return how
    if how == "auto-strongest":
        return max(lock, key=lambda k: lock[k]["best_probe"])
    return max(lock, key=lambda k: lock[k].get("last_assertion", 0)
               - lock[k]["best_probe"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcripts", default="data/transcripts.jsonl")
    ap.add_argument("--probes", default="data/probe_results.json")
    ap.add_argument("--acts", default="data/acts.npz")
    ap.add_argument("--target", default="auto-story",
                    help="a locked_* name, 'auto-story' (max regex-probe margin), "
                         "or 'auto-strongest' (max probe)")
    ap.add_argument("--layer", type=int, default=None,
                    help="override the probe's best layer, so the steering null "
                         "can be checked at several depths rather than at the one "
                         "layer validation selection happened to pick")
    ap.add_argument("--corrupt-mode", choices=["last", "all"], default="all")
    ap.add_argument("--steer-at", choices=["all", "last"], default="all")
    ap.add_argument("--n-items", type=int, default=300)
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0, 4, 8, 16, 32, 64])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--probe-target", choices=["truth", "model"], default="truth",
                    help="truth: the direction that decodes GROUND-TRUTH state "
                         "model: the direction that decodes the model's own clean "
                         "answer, fitted on a disjoint pool of held-out turns.")
    ap.add_argument("--pool", type=int, default=600,
                    help="held-out turns used to fit the model-answer probe")
    ap.add_argument("--direction", choices=["normal", "raw_w", "diffmean"],
                    default="normal",
                    help="normal = w/scale, the decision normal in RAW space "
                         "(correct); raw_w = adding the "
                         "standardised-space coefficients straight into raw "
                         "activations; diffmean = difference of class means")
    ap.add_argument("--steer-vec", choices=["probe", "diffmean"], default="probe",
                    help="probe: the logistic-regression weight vector (trained on "
                         "z-scored activations, applied in raw space). "
                         "diffmean: mean(locked) - mean(unlocked) of raw activations at "
                         "the probe layer, the standard steering direction.")
    ap.add_argument("--readout", choices=["logit", "generate"], default="logit",
                    help="logit: compare next-token log-probs of ' locked' vs "
                         "' unlocked' after an answer stem (one forward pass, "
                         "always parseable, no answer bias). generate: an "
                         "8-token free-text answer.")
    ap.add_argument("--think-readout", action="store_true",
                    help="let the model think before answering (default: no)")
    args = ap.parse_args()

    import torch
    import cotstate
    from transformers import AutoTokenizer, AutoModelForCausalLM

    P = json.load(open(args.probes))
    eps = [json.loads(l) for l in open(args.transcripts)]
    te = set(P["test_episodes"])
    Xacts = np.load(args.acts, allow_pickle=True)["X"].astype(np.float32)

    target = pick_target(P, args.target)
    tag = args.tag or target.replace("locked_", "").replace(".", "_")
    out_path = f"data/intervene_{tag}.json"
    f = target[len("locked_"):]
    meta_fit = json.loads(np.load(args.acts, allow_pickle=True)["meta"].item())
    is_train = np.array([m["episode"] not in te for m in meta_fit])
    lock = P["probes"][target]
    li = lock["best_layer"] if args.layer is None else args.layer
    if str(li) not in lock["layers"]:
        raise SystemExit(f"[iv] layer {li} was not swept in {args.probes}; "
                         f"available: {sorted(int(k) for k in lock['layers'])}")
    w = np.array(lock["layers"][str(li)]["w"], dtype=np.float32)
    if args.direction == "normal":
        if "scaler_scale" not in lock["layers"][str(li)]:
            raise SystemExit(f"[iv] probe does not contain scaler_scale at L{li}; "
                             "regenerate with patched probe.py or pass --direction raw_w")
        scale = np.array(lock["layers"][str(li)]["scaler_scale"], dtype=np.float32)
        w = w / scale
    if args.steer_vec == "diffmean":
        meta = json.loads(np.load(args.acts, allow_pickle=True)["meta"].item())
        yv = np.array([m["state_before"][target] for m in meta])
        Xl = Xacts[:, li, :][is_train]
        yv = yv[is_train]
        w = Xl[yv == 1].mean(0) - Xl[yv == 0].mean(0)
        print(f"[iv] steering vector = difference of means at L{li}, |d|={np.linalg.norm(w):.1f}")
    w /= np.linalg.norm(w) + 1e-9
    resid_norm = float(np.linalg.norm(Xacts[:, li, :], axis=1).mean())
    print(f"[iv] target={target} (probe {lock['best_probe']:.3f} vs regex "
          f"{lock.get('last_assertion', float('nan')):.3f})  layer={li}  "
          f"|resid|~{resid_norm:.1f}  corrupt={args.corrupt_mode}  steer_at={args.steer_at}  "
          f"direction={args.direction}")

    rng = np.random.RandomState(0)
    w_rand = rng.randn(len(w)).astype(np.float32)
    w_rand /= np.linalg.norm(w_rand)
    others = [k for k in P["probes"]
              if k.startswith("locked_") and k != target
              and str(li) in P["probes"][k]["layers"]]
    w_other = None
    if others:
        k2 = max(others, key=lambda k: P["probes"][k]["best_probe"])
        w_other = np.array(P["probes"][k2]["layers"][str(li)]["w"], dtype=np.float32)
        if args.direction == "normal" and "scaler_scale" in P["probes"][k2]["layers"][str(li)]:
            scale_other = np.array(P["probes"][k2]["layers"][str(li)]["scaler_scale"], dtype=np.float32)
            w_other = w_other / scale_other
        if args.steer_vec == "diffmean":
            y2 = np.array([m["state_before"][k2] for m in meta])
            w_other = Xl[y2 == 1].mean(0) - Xl[y2 == 0].mean(0)
        w_other /= np.linalg.norm(w_other) + 1e-9
        print(f"[iv] other-probe control = {k2}")

    items = []
    for ei, ep in enumerate(eps):
        if ei not in te:
            continue
        for t in ep["turns"]:
            if t["turn"] >= 2:
                items.append((ei, t["turn"], t["state_before"][target]))
    random.Random(0).shuffle(items)
    items = items[:args.n_items]
    truths = np.array([t for _, _, t in items])
    print(f"[iv] {len(items)} items, base rate {max(truths.mean(), 1-truths.mean()):.3f}")

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    steer = {"vec": None, "mode": args.steer_at}

    def hook(_m, _i, out):
        if steer["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        v = steer["vec"].to(h.dtype).to(h.device)
        if steer["mode"] == "last":
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + v
        else:
            h = h + v
        return (h,) + out[1:] if isinstance(out, tuple) else h

    model.model.layers[li].register_forward_hook(hook)

    STEM = "{f} is currently"

    def _first_tok(word):
        ids = tok(word, add_special_tokens=False)["input_ids"]
        return ids[0]
    if args.readout == "logit":
        ID_L, ID_U = _first_tok(" locked"), _first_tok(" unlocked")
        if ID_L == ID_U:
            raise SystemExit("[iv] ' locked' and ' unlocked' share a first token; "
                             "pick other answer words")
        print(f"[iv] logit readout: ' locked'->{ID_L} ({tok.decode([ID_L])!r})  "
              f"' unlocked'->{ID_U} ({tok.decode([ID_U])!r})")

    def run(msg_list):
        """Returns per-item raw readouts: str (generate) or float diff (logit)."""
        outs = []
        for b in range(0, len(msg_list), args.batch):
            chunk = msg_list[b:b + args.batch]
            prompts = [chat(tok, m, thinking=args.think_readout) for m in chunk]
            if args.readout == "logit":
                prompts = [pr + STEM.format(f=f) for pr in prompts]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=False).to(args.device)
            if enc["input_ids"].shape[1] > args.max_len:
                raise SystemExit(f"[iv] prompt {enc['input_ids'].shape[1]} tokens > --max-len; "
                                 "truncating would cut the question off")
            with torch.no_grad():
                if args.readout == "logit":
                    lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), -1)
                    outs += (lp[:, ID_L] - lp[:, ID_U]).cpu().tolist()
                else:
                    g = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                    for j in range(len(chunk)):
                        outs.append(tok.decode(g[j][enc["input_ids"].shape[1]:],
                                               skip_special_tokens=True))
        return outs

    thr = {"v": 0.0}

    def to_pred(raw_list):
        if args.readout == "logit":
            return np.array([1 if d > thr["v"] else 0 for d in raw_list])
        return np.array([parse_answer(o) if parse_answer(o) is not None else -1
                         for o in raw_list])

    def auroc(raw_list, mask):
        if args.readout != "logit":
            return None
        from sklearn.metrics import roc_auc_score
        dd = np.array(raw_list)[mask]; tt = truths[mask]
        return float(roc_auc_score(tt, dd)) if len(set(tt)) > 1 else None

    preds_by_cond, raw_by_cond, auc_by_cond = {}, {}, {}
    out_extra = {"probe_target": args.probe_target, "steer_vec": args.steer_vec}

    if args.probe_target == "model":
        if args.readout != "logit":
            raise SystemExit("[iv] --probe-target model requires --readout logit")
        meta_all = json.loads(np.load(args.acts, allow_pickle=True)["meta"].item())
        idx_of = {(m["episode"], m["turn"]): i for i, m in enumerate(meta_all)}
        n_turns_max = 1 + max(m["turn"] for m in meta_all)
        used = {(ei, tn) for ei, tn, _ in items}
        used_eps = {ei for ei, _, _ in items}
        pool = [(ei, tn) for ei in sorted(te) for tn in range(2, n_turns_max)
                if (ei, tn) in idx_of and ei not in used_eps]
        random.Random(1).shuffle(pool)
        pool = pool[:args.pool]
        if len(pool) < 100:
            raise SystemExit(f"[iv] only {len(pool)} disjoint pool turns available; "
                             "lower --n-items or raise --test-frac when probing")
        print(f"[iv] fitting the model-answer probe on {len(pool)} disjoint turns")
        steer["vec"] = None
        pool_raw = np.array(run([build_msgs(eps[ei], tn, f)[0] for ei, tn in pool]),
                            dtype=np.float32)
        y_model = (pool_raw > np.median(pool_raw)).astype(int)
        y_true_pool = np.array([eps[ei]["turns"][tn]["state_before"][target]
                                for ei, tn in pool])
        agree = float((y_model == y_true_pool).mean())
        Xp = Xacts[[idx_of[k] for k in pool], li, :]
        mu_p, sd_p = Xp.mean(0), Xp.std(0) + 1e-6
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        clf_m = LogisticRegression(max_iter=2000, class_weight="balanced")
        Zp = (Xp - mu_p) / sd_p
        _ep = np.array([ei for ei, _ in pool])
        _u = np.unique(_ep); np.random.RandomState(7).shuffle(_u)
        _hold = set(_u[: max(1, len(_u) // 3)].tolist())
        _m = np.array([e in _hold for e in _ep])
        hold_acc = None
        if _m.sum() >= 20 and len(set(y_model[~_m])) > 1 and len(set(y_model[_m])) > 1:
            _c = LogisticRegression(max_iter=2000, class_weight="balanced")
            _c.fit(Zp[~_m], y_model[~_m])
            hold_acc = float(balanced_accuracy_score(y_model[_m], _c.predict(Zp[_m])))
            print(f"[iv] model-answer probe held-out bal-acc {hold_acc:.3f} "
                  f"(fit on {int((~_m).sum())} turns, scored on {int(_m.sum())})")
        clf_m.fit(Zp, y_model)
        fit_acc = float(balanced_accuracy_score(y_model, clf_m.predict(Zp)))
        w = clf_m.coef_[0].astype(np.float32)
        if args.direction == "normal":
            w = w / sd_p.astype(np.float32)
        w /= np.linalg.norm(w) + 1e-9
        print(f"[iv] model-answer probe: in-sample bal-acc {fit_acc:.3f}; the model's "
              f"clean answer agrees with ground truth on {agree:.3f} of the pool, so "
              f"this is not the truth direction under another name")
        out_extra.update({"pool_n": len(pool), "model_truth_agreement": agree,
                          "fit_bal_acc": fit_acc,
                          "holdout_bal_acc": hold_acc})

    msgs = [build_msgs(eps[ei], tn, f)[0] for ei, tn, _ in items]
    steer["vec"] = None
    raw_clean = run(msgs)
    if args.readout == "logit":
        thr["v"] = float(np.median(raw_clean))
        print(f"[iv] clean logit-diff median {thr['v']:+.3f} used as threshold; "
              f"clean AUROC {auroc(raw_clean, np.ones(len(items), bool)):.3f}")
    clean = to_pred(raw_clean)
    preds_by_cond["clean"] = clean.tolist(); raw_by_cond["clean"] = list(raw_clean)
    valid = clean >= 0
    clean_ok = (clean == truths) & valid
    auc_by_cond["clean"] = auroc(raw_clean, valid)
    print(f"[iv] clean accuracy {clean_ok.sum()}/{valid.sum()} = "
          f"{clean_ok.sum()/max(valid.sum(),1):.4f}  (unparseable {int((~valid).sum())})"
          f"  says-locked {float(np.mean(clean[valid])):.2f} vs truth {truths.mean():.2f}")

    pairs = [build_msgs(eps[ei], tn, f, corrupt=True, mode=args.corrupt_mode)
             for ei, tn, _ in items]
    n_flip = np.array([n for _, n in pairs])
    corr = np.full(len(items), -1)
    raw_corr = [None] * len(items)
    did = n_flip > 0
    if did.any():
        got = run([pairs[i][0] for i in np.where(did)[0]])
        gp = to_pred(got)
        for i, o, pp in zip(np.where(did)[0], got, gp):
            corr[i] = pp; raw_corr[i] = o
    preds_by_cond["token_corrupt"] = corr.tolist(); raw_by_cond["token_corrupt"] = raw_corr
    auc_by_cond["token_corrupt"] = auroc([x if x is not None else 0.0 for x in raw_corr], did)
    print(f"[iv] corruption applied to {int(did.sum())}/{len(items)} items, "
          f"mean {n_flip[did].mean():.2f} visible assertions flipped" if did.any() else
          "[iv] corruption applied to 0 items")

    d = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    vecs = [("probe", w), ("random", w_rand)]
    if w_other is not None:
        vecs.append(("other_probe", w_other))
    for name, vec in vecs:
        for a in args.alphas:
            for sign in ([1] if a == 0 else [1, -1]):
                key = f"steer_{name}_a{sign*a:+g}"
                steer["vec"] = (None if a == 0 else
                                torch.tensor(sign * a * resid_norm / np.sqrt(d) * vec))
                raw = run(msgs)
                p = to_pred(raw)
                preds_by_cond[key] = p.tolist(); raw_by_cond[key] = list(raw)
                v = p >= 0
                auc_by_cond[key] = auroc(raw, v)
                acc = ((p == truths) & v).sum() / max(v.sum(), 1)
                print(f"[iv] {key:26s} acc={acc:.4f} unparseable={int((~v).sum())}"
                      + (f" AUROC={auc_by_cond[key]:.3f}" if auc_by_cond[key] is not None else ""),
                      flush=True)
                if a == 0:
                    break
        steer["vec"] = None

    summary = {}
    for key, plist in preds_by_cond.items():
        if key == "clean":
            continue
        p = np.array(plist)
        both = valid & (p >= 0)
        a_ok = (clean == truths) & both
        b_ok = (p == truths) & both
        mc = mcnemar_exact(a_ok[both], b_ok[both])
        broke = int(np.sum(a_ok & ~b_ok))
        n_was_right = int(np.sum(a_ok))
        rate, lo, hi = wilson(broke, n_was_right)
        summary[key] = {
            "n_paired": int(both.sum()),
            "acc_clean": float((a_ok & both).sum() / max(both.sum(), 1)),
            "acc_intervened": float((b_ok & both).sum() / max(both.sum(), 1)),
            "delta_acc": float(((b_ok & both).sum() - (a_ok & both).sum())
                               / max(both.sum(), 1)),
            "broke_correct_beliefs": broke,
            "of_n_correct": n_was_right,
            "conditional_flip_rate": rate,
            "flip_ci": [lo, hi],
            "mcnemar": mc,
        }

    out = {
        "target": target, "file": f, "layer": li,
        "corrupt_mode": args.corrupt_mode, "steer_at": args.steer_at,
        "n_items": len(items),
        "resid_norm": resid_norm,
        "d_model": int(Xacts.shape[-1]),
        "direction": args.direction,
        "norm_ratio_per_alpha": 1.0 / float(np.sqrt(Xacts.shape[-1])),
        "norm_ratio_note": "||perturbation||/||residual|| = alpha/sqrt(d_model); "
                           "alpha=16 is a quarter of the residual norm, "
                           "alpha=32 a half, alpha=64 the whole of it",
        "truths": truths.tolist(),
        "readout": args.readout,
        "threshold": thr["v"],
        **out_extra,
        "preds": preds_by_cond,
        "raw": raw_by_cond,
        "auroc": auc_by_cond,
        "summary": summary,
    }
    json.dump(out, open(out_path, "w"), indent=2)

    print(f"\n{'condition':<26}{'acc':>8}{'Δacc':>8}{'broke':>8}"
          f"{'flip rate (95% CI)':>26}{'McNemar p':>12}{'AUROC':>8}")
    for k, s in summary.items():
        ci = (f"{100*s['conditional_flip_rate']:.1f}% "
              f"[{100*s['flip_ci'][0]:.1f},{100*s['flip_ci'][1]:.1f}]")
        au = auc_by_cond.get(k)
        print(f"{k:<26}{s['acc_intervened']:>8.4f}{s['delta_acc']:>+8.4f}"
              f"{s['broke_correct_beliefs']:>4}/{s['of_n_correct']:<3}"
              f"{ci:>26}{s['mcnemar']['p']:>12.3g}"
              + (f"{au:>8.3f}" if au is not None else ""))
    if auc_by_cond.get("clean") is not None:
        print(f"{'clean':<26}{'':>8}{'':>8}{'':>8}{'':>26}{'':>12}{auc_by_cond['clean']:>8.3f}")
    print(f"\n[iv] wrote {out_path}")


if __name__ == "__main__":
    main()
