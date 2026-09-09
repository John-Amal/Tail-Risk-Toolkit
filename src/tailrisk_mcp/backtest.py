"""Backtests for Value-at-Risk forecasts.

A VaR model at confidence ``q`` should be breached on a fraction ``1 - q`` of
days, and breaches should be independent over time. Clustered breaches indicate
the model reacts too slowly to volatility even when the average breach rate
looks correct. These are the standard tests supervisors expect in a model
validation report.

References
----------
Kupiec (1995), Techniques for verifying the accuracy of risk measurement models.
Christoffersen (1998), Evaluating interval forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal, Sequence

import numpy as np
from scipy import stats

from .evt import EVTError, Tail, fit_gpd, orient_series, value_at_risk_evt, value_at_risk_gaussian

Method = Literal["evt", "historical", "gaussian"]


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a rolling-window VaR backtest.

    Attributes:
        method: Estimator used to produce each day's forecast.
        confidence: VaR confidence level.
        window: Number of trailing observations used to fit each forecast.
        n_forecasts: Number of out-of-sample forecasts evaluated.
        n_breaches: Number of realised losses exceeding the forecast.
        breach_rate: ``n_breaches / n_forecasts``.
        expected_breach_rate: ``1 - confidence``.
        kupiec_statistic: Unconditional coverage likelihood-ratio statistic.
        kupiec_p_value: Chi-squared(1) p-value for correct breach frequency.
        christoffersen_statistic: Independence likelihood-ratio statistic.
        christoffersen_p_value: Chi-squared(1) p-value for breach independence.
        conditional_coverage_statistic: Joint statistic (Kupiec + independence).
        conditional_coverage_p_value: Chi-squared(2) p-value for the joint test.
        verdict: Plain-language reading of the joint test at the 5% level.
    """

    method: str
    confidence: float
    window: int
    n_forecasts: int
    n_breaches: int
    breach_rate: float
    expected_breach_rate: float
    kupiec_statistic: float
    kupiec_p_value: float
    christoffersen_statistic: float
    christoffersen_p_value: float
    conditional_coverage_statistic: float
    conditional_coverage_p_value: float
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-serialisable dictionary."""
        return asdict(self)


def kupiec_pof(n_breaches: int, n_forecasts: int, confidence: float) -> tuple[float, float]:
    """Run the Kupiec proportion-of-failures test for unconditional coverage.

    Args:
        n_breaches: Observed number of VaR breaches.
        n_forecasts: Number of forecasts evaluated.
        confidence: VaR confidence level.

    Returns:
        ``(statistic, p_value)``. A small p-value rejects the hypothesis that the
        breach frequency matches ``1 - confidence``.
    """
    p = 1.0 - float(confidence)
    n, x = int(n_forecasts), int(n_breaches)
    if n == 0:
        return float("nan"), float("nan")
    if x == 0:
        stat = -2.0 * n * np.log(1.0 - p)
    elif x == n:
        stat = -2.0 * n * np.log(p)
    else:
        pi = x / n
        log_null = (n - x) * np.log(1.0 - p) + x * np.log(p)
        log_alt = (n - x) * np.log(1.0 - pi) + x * np.log(pi)
        stat = -2.0 * (log_null - log_alt)
    stat = float(max(stat, 0.0))
    return stat, float(stats.chi2.sf(stat, df=1))


def christoffersen_independence(breaches: Sequence[bool]) -> tuple[float, float]:
    """Test whether VaR breaches are independent across consecutive periods.

    Args:
        breaches: Boolean breach indicators in chronological order.

    Returns:
        ``(statistic, p_value)``. A small p-value indicates breach clustering.
        Returns ``(0.0, 1.0)`` when a transition count is empty, which makes the
        statistic degenerate rather than significant.
    """
    b = np.asarray(breaches, dtype=bool)
    if b.size < 2:
        return float("nan"), float("nan")

    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))

    if (n00 + n01) == 0 or (n10 + n11) == 0 or (n01 + n11) == 0:
        return 0.0, 1.0

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _safe_log(value: float) -> float:
        return float(np.log(value)) if value > 0 else 0.0

    log_null = (n00 + n10) * _safe_log(1.0 - pi) + (n01 + n11) * _safe_log(pi)
    log_alt = (
        n00 * _safe_log(1.0 - pi01)
        + n01 * _safe_log(pi01)
        + n10 * _safe_log(1.0 - pi11)
        + n11 * _safe_log(pi11)
    )
    stat = float(max(-2.0 * (log_null - log_alt), 0.0))
    return stat, float(stats.chi2.sf(stat, df=1))


def _forecast_var(window_values: np.ndarray, confidence: float, method: Method) -> float:
    """Produce a one-step-ahead VaR forecast from a trailing window."""
    if method == "historical":
        return float(np.quantile(window_values, confidence))
    if method == "gaussian":
        return value_at_risk_gaussian(window_values, confidence, tail="upper")
    fit = fit_gpd(window_values, threshold_quantile=0.90, tail="upper")
    return value_at_risk_evt(fit, confidence)


def backtest_var(
    values: Sequence[float],
    confidence: float = 0.99,
    window: int = 500,
    method: Method = "evt",
    tail: Tail = "upper",
) -> BacktestResult:
    """Run a rolling-window, out-of-sample VaR backtest.

    Each forecast is fitted on the trailing ``window`` observations and compared
    with the next realised value, so no future information enters a forecast.

    Args:
        values: Raw observations in chronological order.
        confidence: VaR confidence level, e.g. ``0.99``.
        window: Trailing observations used per forecast.
        method: ``"evt"``, ``"historical"`` or ``"gaussian"``.
        tail: Which tail is adverse.

    Returns:
        A :class:`BacktestResult` with breach counts and test statistics.

    Raises:
        EVTError: If the series is too short for the requested window.
    """
    series = orient_series(values, tail)
    if series.size <= window:
        raise EVTError(
            f"Series has {series.size} observations but the window is {window}; "
            "at least one more observation than the window is required. Shorten "
            "the window or supply a longer series."
        )

    forecasts: list[float] = []
    breaches: list[bool] = []
    for end in range(window, series.size):
        var = _forecast_var(series[end - window : end], confidence, method)
        forecasts.append(var)
        breaches.append(bool(series[end] > var))

    n = len(forecasts)
    x = int(np.sum(breaches))
    kupiec_stat, kupiec_p = kupiec_pof(x, n, confidence)
    ind_stat, ind_p = christoffersen_independence(breaches)

    cc_stat = float(kupiec_stat + ind_stat)
    cc_p = float(stats.chi2.sf(cc_stat, df=2))

    if cc_p >= 0.05:
        verdict = "Model passes conditional coverage at the 5% level."
    elif kupiec_p < 0.05 and ind_p >= 0.05:
        verdict = "Breach frequency is off target; breaches are not clustered."
    elif kupiec_p >= 0.05 and ind_p < 0.05:
        verdict = "Breach frequency is on target but breaches cluster in time."
    else:
        verdict = "Breach frequency is off target and breaches cluster in time."

    return BacktestResult(
        method=method,
        confidence=float(confidence),
        window=int(window),
        n_forecasts=n,
        n_breaches=x,
        breach_rate=x / n if n else float("nan"),
        expected_breach_rate=1.0 - float(confidence),
        kupiec_statistic=kupiec_stat,
        kupiec_p_value=kupiec_p,
        christoffersen_statistic=ind_stat,
        christoffersen_p_value=ind_p,
        conditional_coverage_statistic=cc_stat,
        conditional_coverage_p_value=cc_p,
        verdict=verdict,
    )
