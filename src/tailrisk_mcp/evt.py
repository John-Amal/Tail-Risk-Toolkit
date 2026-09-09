"""Extreme value methods for tail risk estimation.

The peaks-over-threshold (POT) approach models exceedances above a high
threshold ``u`` with a Generalised Pareto Distribution (GPD). This is the
same estimator used for hydrological return levels and for financial
Value-at-Risk in the far tail; only the units and the reporting convention
differ.

Sign convention
---------------
All functions in this module operate on a *loss-oriented* series: large
positive values are the adverse outcomes whose tail we care about. Callers
holding returns (where large negative values are adverse) should negate the
series first, or use ``orient_series``.

References
----------
Coles (2001), *An Introduction to Statistical Modeling of Extreme Values*.
McNeil & Frey (2000), Estimation of tail-related risk measures for
heteroscedastic financial time series.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal, Sequence

import numpy as np
from scipy import stats

Tail = Literal["upper", "lower"]

MIN_EXCEEDANCES = 30
"""Below this many threshold exceedances, GPD parameter estimates are unstable."""


class EVTError(ValueError):
    """Raised when a series or threshold cannot support a GPD fit."""


def orient_series(values: Sequence[float], tail: Tail = "upper") -> np.ndarray:
    """Return a clean, loss-oriented 1-D array.

    Args:
        values: Raw observations. NaN and infinite entries are dropped.
        tail: ``"upper"`` if large positive values are the adverse tail (losses),
            ``"lower"`` if large negative values are (returns). A ``"lower"``
            series is negated so downstream code always sees an upper tail.

    Returns:
        A finite float64 array oriented so that the tail of interest is the
        upper tail.

    Raises:
        EVTError: If fewer than two finite observations remain.
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise EVTError(
            f"Need at least 2 finite observations, got {arr.size}. "
            "Check the series for NaNs or an empty payload."
        )
    return -arr if tail == "lower" else arr


@dataclass(frozen=True)
class GPDFit:
    """Fitted generalised Pareto model for threshold exceedances.

    Attributes:
        threshold: The threshold ``u`` above which exceedances were modelled.
        shape: GPD shape parameter ``xi``. Positive means a heavy, unbounded
            tail; negative means a finite upper endpoint.
        scale: GPD scale parameter ``sigma``, in the units of the input series.
        n_observations: Size of the cleaned input series.
        n_exceedances: Number of observations strictly above the threshold.
        exceedance_rate: ``n_exceedances / n_observations``, the empirical
            probability of exceeding the threshold.
        log_likelihood: Maximised GPD log-likelihood of the exceedances.
    """

    threshold: float
    shape: float
    scale: float
    n_observations: int
    n_exceedances: int
    exceedance_rate: float
    log_likelihood: float

    def as_dict(self) -> dict[str, Any]:
        """Return the fit as a JSON-serialisable dictionary."""
        return asdict(self)


def choose_threshold(series: np.ndarray, quantile: float = 0.90) -> float:
    """Return the empirical quantile used as the default POT threshold.

    Args:
        series: Loss-oriented observations.
        quantile: Probability level in ``(0, 1)``.

    Returns:
        The threshold value.
    """
    return float(np.quantile(series, quantile))


def fit_gpd(
    values: Sequence[float],
    threshold: float | None = None,
    threshold_quantile: float = 0.90,
    tail: Tail = "upper",
) -> GPDFit:
    """Fit a GPD to threshold exceedances by maximum likelihood.

    Args:
        values: Raw observations.
        threshold: Explicit threshold. If ``None``, the empirical
            ``threshold_quantile`` of the oriented series is used.
        threshold_quantile: Quantile used when ``threshold`` is not given.
        tail: Which tail is adverse. See :func:`orient_series`.

    Returns:
        The fitted :class:`GPDFit`.

    Raises:
        EVTError: If the threshold leaves too few exceedances to fit.
    """
    series = orient_series(values, tail)
    u = choose_threshold(series, threshold_quantile) if threshold is None else float(threshold)

    exceedances = series[series > u] - u
    n_exc = int(exceedances.size)
    if n_exc < MIN_EXCEEDANCES:
        raise EVTError(
            f"Threshold {u:.6g} leaves only {n_exc} exceedances "
            f"(minimum {MIN_EXCEEDANCES}). Lower the threshold, or pass a "
            "smaller threshold_quantile."
        )

    shape, _, scale = stats.genpareto.fit(exceedances, floc=0.0)
    log_lik = float(np.sum(stats.genpareto.logpdf(exceedances, shape, loc=0.0, scale=scale)))

    return GPDFit(
        threshold=u,
        shape=float(shape),
        scale=float(scale),
        n_observations=int(series.size),
        n_exceedances=n_exc,
        exceedance_rate=n_exc / series.size,
        log_likelihood=log_lik,
    )


def _gpd_quantile(fit: GPDFit, exceedance_probability: float) -> float:
    """Invert the POT tail model for a given unconditional exceedance probability.

    Solves ``P(X > z) = p`` where the tail is
    ``P(X > z) = zeta_u * [1 + xi (z - u) / sigma] ** (-1 / xi)``.

    Args:
        fit: A fitted GPD model.
        exceedance_probability: Target probability ``p``, strictly between 0 and
            the fitted exceedance rate.

    Returns:
        The quantile ``z``.

    Raises:
        EVTError: If ``p`` is not below the exceedance rate, which would require
            extrapolating below the threshold where the GPD does not apply.
        """
    p = float(exceedance_probability)
    if not 0.0 < p < fit.exceedance_rate:
        raise EVTError(
            f"Exceedance probability {p:.6g} must lie strictly between 0 and the "
            f"fitted exceedance rate {fit.exceedance_rate:.6g}. The GPD only "
            "describes the tail above the threshold; use the empirical quantile "
            "for less extreme levels, or refit at a lower threshold."
        )
    ratio = fit.exceedance_rate / p
    if abs(fit.shape) < 1e-8:
        return fit.threshold + fit.scale * np.log(ratio)
    return fit.threshold + (fit.scale / fit.shape) * (ratio**fit.shape - 1.0)


def value_at_risk_evt(fit: GPDFit, confidence: float) -> float:
    """Return the EVT Value-at-Risk at a given confidence level.

    Args:
        fit: A fitted GPD model.
        confidence: Confidence level such as ``0.99``.

    Returns:
        The loss quantile that is exceeded with probability ``1 - confidence``.
    """
    return _gpd_quantile(fit, 1.0 - float(confidence))


def expected_shortfall_evt(fit: GPDFit, confidence: float) -> float:
    """Return the EVT Expected Shortfall (mean loss beyond VaR).

    Args:
        fit: A fitted GPD model.
        confidence: Confidence level such as ``0.99``.

    Returns:
        The conditional mean loss given that VaR is exceeded.

    Raises:
        EVTError: If the fitted shape is at or above 1, where the tail has no
            finite mean and ES is undefined.
    """
    if fit.shape >= 1.0:
        raise EVTError(
            f"Expected Shortfall is undefined for shape xi = {fit.shape:.4f} >= 1 "
            "(infinite-mean tail). Report VaR only, or revisit the threshold."
        )
    var = value_at_risk_evt(fit, confidence)
    return (var + fit.scale - fit.shape * fit.threshold) / (1.0 - fit.shape)


def value_at_risk_historical(values: Sequence[float], confidence: float, tail: Tail = "upper") -> float:
    """Return the empirical (historical simulation) VaR."""
    series = orient_series(values, tail)
    return float(np.quantile(series, float(confidence)))


def expected_shortfall_historical(values: Sequence[float], confidence: float, tail: Tail = "upper") -> float:
    """Return the empirical mean loss beyond the historical VaR.

    Raises:
        EVTError: If no observation exceeds the empirical VaR, which happens when
            the sample is too small for the requested confidence level.
    """
    series = orient_series(values, tail)
    var = float(np.quantile(series, float(confidence)))
    beyond = series[series > var]
    if beyond.size == 0:
        raise EVTError(
            f"No observations exceed the {confidence:.4g} empirical quantile in a "
            f"sample of {series.size}. Historical ES needs a longer sample at this "
            "confidence level; the EVT estimate extrapolates instead."
        )
    return float(beyond.mean())


def value_at_risk_gaussian(values: Sequence[float], confidence: float, tail: Tail = "upper") -> float:
    """Return the Gaussian (variance-covariance) VaR.

    Included as a deliberately naive baseline: comparing it with the EVT
    estimate shows how much tail risk a normality assumption discards.
    """
    series = orient_series(values, tail)
    return float(series.mean() + series.std(ddof=1) * stats.norm.ppf(float(confidence)))


def return_level(fit: GPDFit, return_period: float) -> float:
    """Return the level expected to be exceeded once every ``return_period`` observations.

    This is the hydrological framing of the same POT model that produces VaR:
    a 100-year return level and a 99% VaR on annual data are the same quantile.

    Args:
        fit: A fitted GPD model.
        return_period: Return period in units of the observation frequency
            (days for a daily series, years for annual maxima).

    Returns:
        The return level.
    """
    if return_period <= 1.0:
        raise EVTError(
            f"Return period must exceed 1 observation, got {return_period:.6g}."
        )
    return _gpd_quantile(fit, 1.0 / float(return_period))


def threshold_stability(
    values: Sequence[float],
    quantiles: Sequence[float] | None = None,
    tail: Tail = "upper",
) -> list[dict[str, Any]]:
    """Refit the GPD across candidate thresholds to check parameter stability.

    Threshold choice is the main subjective step in a POT analysis. If the shape
    parameter is roughly constant across a range of thresholds, the fit is in the
    asymptotic regime and the choice is defensible.

    Args:
        values: Raw observations.
        quantiles: Candidate threshold quantiles. Defaults to 0.85 through 0.99.
        tail: Which tail is adverse.

    Returns:
        One record per threshold that supported a fit, each containing the
        quantile, threshold, shape, scale, modified scale
        (``sigma - xi * u``, which is threshold-invariant under a correct model),
        and exceedance count. Thresholds with too few exceedances are skipped.
    """
    if quantiles is None:
        quantiles = [0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 0.99]

    records: list[dict[str, Any]] = []
    for q in quantiles:
        try:
            fit = fit_gpd(values, threshold_quantile=float(q), tail=tail)
        except EVTError:
            continue
        records.append(
            {
                "quantile": float(q),
                "threshold": fit.threshold,
                "shape": fit.shape,
                "scale": fit.scale,
                "modified_scale": fit.scale - fit.shape * fit.threshold,
                "n_exceedances": fit.n_exceedances,
            }
        )

    if not records:
        raise EVTError(
            "No candidate threshold left enough exceedances to fit. The series is "
            f"likely too short (needs roughly {MIN_EXCEEDANCES * 10} observations)."
        )
    return records
