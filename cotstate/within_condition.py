"""Probes trained within each rollout condition."""

from __future__ import annotations

import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from cotstate.env import VARS_LOCKED, VARS_EXISTS
from cotstate.probe import last_assertion, build_text_features
from cotstate.stats import boot_ci

CONDS = [("thinking", "data/acts.npz"),
         ("nothink", "data/acts_nothink.npz"),
         ("terse", "data/acts_terse.npz")]


def run_condition(name, path, layers, test_frac=0.3, val_frac=0.2):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    meta = json.loads(d["meta"].item())
    ep_id = np.array([m["episode"] for m in meta])
    uniq = np.unique(ep_id)
    rng = np.random.RandomState(0)
    rng.shuffle(uniq)
    te_eps = set(uniq[: max(1, int(len(uniq) * test_frac))].tolist())
    is_te = np.array([e in te_eps for e in ep_id])
    _rest = [e for e in uniq if e not in te_eps]
    _va = set(_rest[: int(len(_rest) * val_frac)])
    is_va = np.array([e in _va for e in ep_id])
    is_tr = ~is_te & ~is_va
    _, hists = build_text_features(meta)
    print(f"\n=== {name}: {len(meta)} turns, {len(uniq)} episodes "
          f"({is_te.sum()} test) ===")

    out = {}
    for var in VARS_LOCKED + VARS_EXISTS:
        y = np.array([m["state_before"][var] for m in meta])
        if min(y.mean(), 1 - y.mean()) < 0.08:
            continue
        best = None
        oracle_auc = 0.0
        for li in layers:
            if li >= X.shape[1]:
                continue
            Xl = X[:, li, :]
            mu, sd = Xl[is_tr].mean(0), Xl[is_tr].std(0) + 1e-6
            Z = (Xl - mu) / sd
            clf = LogisticRegression(max_iter=2000, C=1.0,
                                     class_weight="balanced")
            clf.fit(Z[is_tr], y[is_tr])
            auc = roc_auc_score(y[is_te], clf.decision_function(Z[is_te]))
            acc = balanced_accuracy_score(y[is_te], clf.predict(Z[is_te]))
            sel = (roc_auc_score(y[is_va], clf.decision_function(Z[is_va]))
                   if is_va.sum() and len(set(y[is_va])) > 1 else auc)
            oracle_auc = max(oracle_auc, float(auc))
            if best is None or sel > best["_sel"]:
                best = {"layer": li, "auroc": float(auc), "bal_acc": float(acc),
                        "_sel": float(sel), "_Z": Z, "_clf": clf}
        if best is None:
            continue
        _, lo, hi = boot_ci(y[is_te],
                            best["_clf"].decision_function(best["_Z"][is_te]),
                            roc_auc_score, n_boot=800, groups=ep_id[is_te])
        entry = {k: v for k, v in best.items() if not k.startswith("_")}
        entry.update({"val_gap": float(oracle_auc - best["auroc"]),
                      "auroc_ci": [lo, hi],
                      "base_rate": float(max(y.mean(), 1 - y.mean())),
                      "kind": "locked" if var in VARS_LOCKED else "exists"})
        if var in VARS_LOCKED:
            fn = var[len("locked_"):]
            b = entry["base_rate"]
            r = np.array([(lambda v: v if v is not None else int(b >= 0.5))(
                last_assertion(hists[i], fn)[0]) for i in np.where(is_te)[0]])
            entry["regex"] = float(balanced_accuracy_score(y[is_te], r))
        out[var] = entry
        print(f"  {var:<20} L{entry['layer']:<3} AUROC={entry['auroc']:.3f} "
              f"[{lo:.3f},{hi:.3f}]  bal-acc={entry['bal_acc']:.3f}"
              + (f"  regex={entry['regex']:.3f}" if "regex" in entry else ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*",
                    default=[1, 2, 4, 8, 16, 20, 22, 26, 30])
    ap.add_argument("--out", default="data/within_condition.json")
    ap.add_argument("--conds", nargs="*", default=None,
                    help="name=path/to/acts.npz ... (default: three conditions)")
    args = ap.parse_args()

    conds = CONDS if not args.conds else [tuple(c.split("=", 1)) for c in args.conds]
    import os
    res = {}
    for name, path in conds:
        if not os.path.exists(path):
            print(f"[wc] missing {path} - skipping {name}")
            continue
        res[name] = run_condition(name, path, args.layers)

    json.dump(res, open(args.out, "w"), indent=2)

    print("\n=== THE COMPARISON THAT MATTERS (mean AUROC within condition) ===")
    print(f"{'condition':<12}{'exists_* (surface)':>22}{'locked_* (memory)':>22}{'gap':>8}")
    for name, r in res.items():
        e = np.mean([v["auroc"] for v in r.values() if v["kind"] == "exists"])
        l = np.mean([v["auroc"] for v in r.values() if v["kind"] == "locked"])
        print(f"{name:<12}{e:>22.3f}{l:>22.3f}{e-l:>8.3f}")
    print(f"\n[wc] wrote {args.out}")


if __name__ == "__main__":
    main()
