"""Paired comparison of selected-layer probes with the TF-IDF baseline (episode bootstrap)."""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from cotstate.probe import VARS_LOCKED, build_text_features, load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    P = json.load(open(args.probes))["probes"]
    X, meta = load(args.acts)
    texts, _ = build_text_features(meta)
    eps = sorted({m["episode"] for m in meta})
    rng = np.random.RandomState(0)
    rng.shuffle(eps)
    n_te = max(1, int(len(eps) * args.test_frac))
    te_eps = set(eps[:n_te])
    rest = eps[n_te:]
    va_eps = set(rest[:int(len(rest) * args.val_frac)])
    ep_arr = np.array([m["episode"] for m in meta])
    is_te = np.isin(ep_arr, list(te_eps))
    is_tr = ~is_te & ~np.isin(ep_arr, list(va_eps))

    preds, per_var = {}, {}
    for var in [v for v in VARS_LOCKED if v in P]:
        y = np.array([m["state_before"][var] for m in meta])
        li = P[var]["best_layer"]
        Xl = X[:, li, :]
        mu, sd = Xl[is_tr].mean(0), Xl[is_tr].std(0) + 1e-6
        Z = (Xl - mu) / sd
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Z[is_tr], y[is_tr])
        vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
        Ttr = vec.fit_transform([t for t, m in zip(texts, is_tr) if m])
        Tte = vec.transform([t for t, m in zip(texts, is_te) if m])
        tf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Ttr, y[is_tr])
        preds[var] = (y[is_te], clf.predict(Z[is_te]), tf.predict(Tte))
        per_var[var] = {"layer": li, "probe": float(balanced_accuracy_score(y[is_te], preds[var][1])),
                        "tfidf": float(balanced_accuracy_score(y[is_te], preds[var][2]))}

    te_ep = ep_arr[is_te]
    uniq = np.unique(te_ep)
    idx = {e: np.where(te_ep == e)[0] for e in uniq}

    def stat(sel):
        d = [balanced_accuracy_score(g[sel], p[sel]) - balanced_accuracy_score(g[sel], t[sel])
             for g, p, t in preds.values()]
        return float(np.mean(d))

    full = stat(np.arange(len(te_ep)))
    rb = np.random.default_rng(0)
    boots = [stat(np.concatenate([idx[e] for e in rb.choice(uniq, len(uniq))])) for _ in range(args.n_boot)]
    out = {"per_variable": per_var, "mean_probe": float(np.mean([v["probe"] for v in per_var.values()])),
           "mean_tfidf": float(np.mean([v["tfidf"] for v in per_var.values()])),
           "diff": full, "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
           "n_test_episodes": int(len(uniq)), "n_test_turns": int(is_te.sum()), "n_boot": args.n_boot}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
