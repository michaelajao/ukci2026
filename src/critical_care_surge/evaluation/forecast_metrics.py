"""Forecast metrics for the UKCI 2026 paper.

Implements the full metric set named in `02_METHODOLOGY.md` §4.1:

Standard accuracy:
    MAE       Mean Absolute Error
    RMSE      Root Mean Squared Error
    sMAPE     symmetric Mean Absolute Percentage Error
    MASE      Mean Absolute Scaled Error (vs naive 1-step)

Calibration:
    WIS       Weighted Interval Score (Bracher et al. 2021)

Decision-relevant:
    underestimation_rate   fraction of timesteps where y_hat < y
    expected_shortage      sum of max(0, y - y_hat)
    peak_error             |max(y) - max(y_hat)|, averaged over regions
    peak_timing_error      |argmax(y) - argmax(y_hat)| in days, averaged

All functions accept either NumPy arrays or pandas Series. Inputs are
flattened over the first axis (time × regions × horizons collapsed to one
sample dimension) unless otherwise stated. NaNs are masked before computation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Core point-forecast metrics
# ---------------------------------------------------------------------------

def _flatten(*arrs: np.ndarray) -> tuple[np.ndarray, ...]:
    """Cast to float, ravel, and broadcast a shared NaN mask."""
    flat = [np.asarray(a, dtype=float).ravel() for a in arrs]
    if len({len(a) for a in flat}) != 1:
        raise ValueError(f"shape mismatch: {[a.shape for a in arrs]}")
    mask = np.ones_like(flat[0], dtype=bool)
    for a in flat:
        mask &= np.isfinite(a)
    return tuple(a[mask] for a in flat)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _flatten(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if yt.size else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _flatten(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if yt.size else float("nan")


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE in (0, 2). Returns 0 where both true and pred are zero."""
    yt, yp = _flatten(y_true, y_pred)
    if not yt.size:
        return float("nan")
    denom = np.abs(yt) + np.abs(yp)
    safe = denom > eps
    return float(np.mean(np.where(safe, 2.0 * np.abs(yt - yp) / np.maximum(denom, eps), 0.0)))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Scales test-set MAE by in-sample MAE of a seasonal-naive forecaster on
    ``y_train`` (Hyndman & Koehler 2006). ``seasonality=1`` corresponds to
    the random-walk baseline (most common; matches `02_METHODOLOGY.md` §4.1).
    Use ``seasonality=7`` for weekly seasonality.
    """
    yt, yp = _flatten(y_true, y_pred)
    if not yt.size:
        return float("nan")
    train = np.asarray(y_train, dtype=float).ravel()
    train = train[np.isfinite(train)]
    if train.size <= seasonality:
        return float("nan")
    naive_diff = np.abs(train[seasonality:] - train[:-seasonality])
    scale = float(np.mean(naive_diff))
    if scale < 1e-12:
        return float("nan")
    return float(np.mean(np.abs(yt - yp))) / scale


# ---------------------------------------------------------------------------
# Calibration: Weighted Interval Score
# ---------------------------------------------------------------------------

def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Per-observation interval score for a (1 - alpha) prediction interval.

    Following Gneiting & Raftery (2007):
        IS = (u - l) + (2/alpha) * (l - y) * 1{y < l} + (2/alpha) * (y - u) * 1{y > u}

    Returns a NumPy array of shape ``y_true.shape`` (no aggregation).
    """
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    width = hi - lo
    below = np.where(yt < lo, lo - yt, 0.0)
    above = np.where(yt > hi, yt - hi, 0.0)
    return width + (2.0 / alpha) * (below + above)


def wis(
    y_true: np.ndarray,
    median: np.ndarray,
    quantile_levels: Sequence[float],
    quantile_predictions: np.ndarray,
) -> float:
    """Weighted Interval Score (Bracher et al. 2021, *PLoS Comput Biol*).

    Args:
        y_true: ground-truth values, shape ``(N,)`` or any broadcastable shape.
        median: median forecast, same shape as ``y_true``.
        quantile_levels: sorted list of K quantile levels in (0, 0.5), e.g.
            ``[0.05, 0.25]`` corresponding to alpha = 0.10 and 0.50 intervals.
        quantile_predictions: quantile forecasts of shape ``(2K, ...)`` ordered
            ``[q_lo_1, q_lo_2, ..., q_lo_K, q_hi_K, q_hi_{K-1}, ..., q_hi_1]``
            i.e. low quantiles ascending, then high quantiles descending so the
            pairs ``(q_lo_k, q_hi_k)`` correspond to alpha_k = 2 * q_lo_k.
            Equivalently we accept shape ``(K, 2, ...)`` with ``[k, 0]`` lower
            and ``[k, 1]`` upper.

    Returns:
        Mean WIS across all observations.
    """
    yt = np.asarray(y_true, dtype=float)
    med = np.asarray(median, dtype=float)
    qp = np.asarray(quantile_predictions, dtype=float)

    levels = np.asarray(quantile_levels, dtype=float)
    K = len(levels)
    if K == 0:
        # Degenerate case: WIS reduces to absolute error.
        return float(np.mean(np.abs(yt - med)))

    # Reshape quantile predictions to (K, 2, ...).
    if qp.ndim == yt.ndim and qp.shape[0] == 2 * K:
        # Concatenated form: low ascending, high descending.
        lows = qp[:K]
        highs = qp[K:][::-1]   # reverse to get high_1, high_2, ..., high_K
    elif qp.shape[:2] == (K, 2):
        lows = qp[:, 0]
        highs = qp[:, 1]
    else:
        raise ValueError(
            f"quantile_predictions has shape {qp.shape}; "
            f"expected (2K, ...) with K={K} or (K, 2, ...)."
        )

    alphas = 2.0 * levels  # interval levels (1 - alpha) coverage
    # Mean over per-observation WIS:
    # WIS = (1 / (K + 0.5)) * (0.5 * |y - median|
    #         + sum_k (alpha_k / 2) * IS_alpha_k)
    is_terms = np.zeros_like(yt, dtype=float)
    for k in range(K):
        is_k = interval_score(yt, lows[k], highs[k], alphas[k])
        is_terms = is_terms + (alphas[k] / 2.0) * is_k
    per_obs = (0.5 * np.abs(yt - med) + is_terms) / (K + 0.5)
    return float(np.mean(per_obs))


# ---------------------------------------------------------------------------
# Decision-relevant metrics
# ---------------------------------------------------------------------------

def underestimation_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of observations where the forecast underestimates truth."""
    yt, yp = _flatten(y_true, y_pred)
    if not yt.size:
        return float("nan")
    return float(np.mean(yp < yt))


def expected_shortage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sum of max(0, y - y_hat) across all observations.

    Operationally: total under-provisioned demand if the forecast were used
    directly to provision capacity.
    """
    yt, yp = _flatten(y_true, y_pred)
    if not yt.size:
        return 0.0
    return float(np.sum(np.maximum(0.0, yt - yp)))


def peak_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """|max(y) - max(y_hat)| averaged over the leading axis (regions).

    Inputs ``y_true`` and ``y_pred`` should be 2D ``(T, R)``. If 1D, treats
    the whole series as a single region.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.ndim == 1:
        return float(np.abs(np.nanmax(yt) - np.nanmax(yp)))
    if yt.ndim != 2:
        raise ValueError(f"peak_error expects 1D or 2D input, got {yt.ndim}D")
    diffs = np.abs(np.nanmax(yt, axis=0) - np.nanmax(yp, axis=0))
    return float(np.mean(diffs))


def peak_timing_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """|argmax(y) - argmax(y_hat)| in days, averaged over regions.

    Inputs as for :func:`peak_error`.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.ndim == 1:
        return float(abs(int(np.nanargmax(yt)) - int(np.nanargmax(yp))))
    if yt.ndim != 2:
        raise ValueError(f"peak_timing_error expects 1D or 2D input, got {yt.ndim}D")
    diffs = np.abs(np.nanargmax(yt, axis=0) - np.nanargmax(yp, axis=0))
    return float(np.mean(diffs))


# ---------------------------------------------------------------------------
# Convenience: full report
# ---------------------------------------------------------------------------

def all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    seasonality: int = 1,
) -> dict[str, float]:
    """Compute the standard accuracy + decision-relevant metric panel.

    Args:
        y_true: ground truth, shape ``(T, R)`` or ``(N,)``.
        y_pred: matching point forecast.
        y_train: training series for MASE scaling. Optional.
        seasonality: seasonality for MASE (1 = random-walk; 7 = weekly).

    Returns a dict mapping metric name → float.
    """
    out: dict[str, float] = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "underestimation_rate": underestimation_rate(y_true, y_pred),
        "expected_shortage": expected_shortage(y_true, y_pred),
    }
    if y_train is not None:
        out["mase"] = mase(y_true, y_pred, y_train, seasonality=seasonality)
    if np.asarray(y_true).ndim == 2:
        out["peak_error"] = peak_error(y_true, y_pred)
        out["peak_timing_error"] = peak_timing_error(y_true, y_pred)
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    T, R = 100, 7
    y = rng.gamma(2.0, 30.0, size=(T, R))
    y_hat = y + rng.normal(0, 5.0, size=y.shape)

    print("=== point-forecast metrics ===")
    print(f"MAE                 = {mae(y, y_hat):.3f}")
    print(f"RMSE                = {rmse(y, y_hat):.3f}")
    print(f"sMAPE               = {smape(y, y_hat):.3f}")
    print(f"MASE (seasonal=1)   = {mase(y[80:], y_hat[80:], y[:80]):.3f}")

    print("\n=== decision-relevant metrics ===")
    print(f"underestimation     = {underestimation_rate(y, y_hat):.3f}")
    print(f"expected_shortage   = {expected_shortage(y, y_hat):.1f}")
    print(f"peak_error          = {peak_error(y, y_hat):.3f}")
    print(f"peak_timing_error   = {peak_timing_error(y, y_hat):.3f}")

    print("\n=== WIS calibration ===")
    levels = [0.05, 0.25]   # 90% and 50% prediction intervals
    qp = np.stack([
        np.stack([y - 20, y + 20], axis=0),    # 90% band [q05, q95]
        np.stack([y - 8,  y + 8],  axis=0),    # 50% band [q25, q75]
    ], axis=0)  # shape (K=2, 2, T, R)
    print(f"WIS                 = {wis(y, y_hat, levels, qp):.3f}")
