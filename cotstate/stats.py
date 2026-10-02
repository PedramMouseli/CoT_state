"""Statistical helpers."""

from __future__ import annotations

from math import comb, erf, log, sqrt
import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Use this instead of the normal approximation: with k=3, n=282 the normal
    interval runs below zero, which is not a thing a proportion can do.
    Returns (point, lo, hi).
    """
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact two-sided McNemar on paired correctness vectors.

    a_only : a right, b wrong.  b_only : b right, a wrong.
    Only the discordant pairs carry information; the concordant ones cancel.
    """
    a_correct = np.asarray(a_correct, dtype=bool)
    b_correct = np.asarray(b_correct, dtype=bool)
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "n_discordant": 0, "p": 1.0}
    k = min(a_only, b_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"a_only": a_only, "b_only": b_only, "n_discordant": n, "p": p}


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a,b],[c,d]]. No scipy required."""
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c

    def hyper(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    obs = hyper(a)
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    tol = obs * (1 + 1e-7)
    return min(1.0, sum(hyper(x) for x in range(lo, hi + 1) if hyper(x) <= tol))


def rate_ratio(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Rate ratio with a log-normal 95% CI. k1/n1 is the numerator group."""
    if k1 == 0 or k2 == 0:
        rr = float("inf") if k2 == 0 else 0.0
        return {"rr": rr, "lo": float("nan"), "hi": float("nan")}
    rr = (k1 / n1) / (k2 / n2)
    se = sqrt(1 / k1 - 1 / n1 + 1 / k2 - 1 / n2)
    return {"rr": rr, "lo": rr * np.exp(-1.96 * se), "hi": rr * np.exp(1.96 * se)}


def boot_ci(y_true, y_pred, metric, n_boot: int = 2000, seed: int = 0,
            groups=None) -> tuple[float, float, float]:
    """Bootstrap CI for any metric(y_true, y_pred).

    Pass `groups` (episode ids) to resample EPISODES rather than turns - turns
    inside an episode are correlated, so resampling them independently gives a
    CI that is far too narrow.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    point = metric(y_true, y_pred)
    rng = np.random.RandomState(seed)
    vals = []
    if groups is None:
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            if len(set(y_true[idx])) < 2:
                continue
            vals.append(metric(y_true[idx], y_pred[idx]))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        index_of = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n_boot):
            picked = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([index_of[g] for g in picked])
            if len(set(y_true[idx])) < 2:
                continue
            vals.append(metric(y_true[idx], y_pred[idx]))
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fmt_ci(point: float, lo: float, hi: float, pct: bool = False) -> str:
    s = 100 if pct else 1
    u = "%" if pct else ""
    return f"{point*s:.3g}{u} [{lo*s:.3g}, {hi*s:.3g}]"
