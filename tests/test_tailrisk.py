"""Tests for the tail-risk estimators.

The statistical tests use simulated data from distributions whose tail
behaviour is known analytically, so the assertions check that the estimators
recover the truth rather than merely that they run.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from tailrisk_mcp.backtest import backtest_var, christoffersen_independence, kupiec_pof
from tailrisk_mcp.evt import (
    EVTError,
    expected_shortfall_evt,
    fit_gpd,
    return_level,
    threshold_stability,
    value_at_risk_evt,
)


@pytest.fixture(scope="module")
def pareto_sample() -> np.ndarray:
    """Heavy-tailed sample with known tail index (Pareto alpha=3 implies xi=1/3)."""
    rng = np.random.default_rng(20260907)
    return rng.pareto(3.0, size=20_000) + 1.0


@pytest.fixture(scope="module")
def student_t_losses() -> np.ndarray:
    """Student-t losses: heavy tailed with an analytic quantile to compare against."""
    rng = np.random.default_rng(11)
    return stats.t.rvs(df=4, size=30_000, random_state=rng)


def test_gpd_recovers_known_shape(pareto_sample: np.ndarray) -> None:
    """A Pareto(alpha=3) tail should give a GPD shape near 1/3."""
    fit = fit_gpd(pareto_sample, threshold_quantile=0.95)
    assert fit.shape == pytest.approx(1 / 3, abs=0.06)
    assert fit.n_exceedances == pytest.approx(1000, rel=0.05)
    assert fit.exceedance_rate == pytest.approx(0.05, abs=0.005)


def test_evt_var_matches_analytic_quantile(student_t_losses: np.ndarray) -> None:
    """EVT VaR should land close to the true t-distribution quantile."""
    fit = fit_gpd(student_t_losses, threshold_quantile=0.95)
    estimated = value_at_risk_evt(fit, 0.995)
    truth = float(stats.t.ppf(0.995, df=4))
    assert estimated == pytest.approx(truth, rel=0.10)


def test_evt_var_increases_with_confidence(student_t_losses: np.ndarray) -> None:
    """VaR must be monotone in the confidence level."""
    fit = fit_gpd(student_t_losses, threshold_quantile=0.95)
    levels = [value_at_risk_evt(fit, c) for c in (0.99, 0.995, 0.999)]
    assert levels == sorted(levels)


def test_expected_shortfall_exceeds_var(student_t_losses: np.ndarray) -> None:
    """ES is a mean beyond a quantile, so it must exceed VaR."""
    fit = fit_gpd(student_t_losses, threshold_quantile=0.95)
    assert expected_shortfall_evt(fit, 0.99) > value_at_risk_evt(fit, 0.99)


def test_return_level_equals_equivalent_var(student_t_losses: np.ndarray) -> None:
    """A 1-in-N return level is the same quantile as VaR at confidence 1 - 1/N."""
    fit = fit_gpd(student_t_losses, threshold_quantile=0.95)
    assert return_level(fit, 200.0) == pytest.approx(value_at_risk_evt(fit, 0.995), rel=1e-9)


def test_lower_tail_orientation_is_symmetric(student_t_losses: np.ndarray) -> None:
    """Negating a series and flipping the tail must give the same answer."""
    upper = fit_gpd(student_t_losses, threshold_quantile=0.95, tail="upper")
    lower = fit_gpd(-student_t_losses, threshold_quantile=0.95, tail="lower")
    assert lower.shape == pytest.approx(upper.shape, rel=1e-9)
    assert lower.threshold == pytest.approx(upper.threshold, rel=1e-9)


def test_gaussian_var_understates_heavy_tail(student_t_losses: np.ndarray) -> None:
    """The point of the tool: a normal assumption misses far-tail risk."""
    from tailrisk_mcp.evt import value_at_risk_gaussian

    fit = fit_gpd(student_t_losses, threshold_quantile=0.95)
    assert value_at_risk_evt(fit, 0.999) > value_at_risk_gaussian(student_t_losses, 0.999)


def test_threshold_stability_reports_each_candidate(student_t_losses: np.ndarray) -> None:
    """The scan should return one usable record per candidate quantile."""
    scan = threshold_stability(student_t_losses, quantiles=[0.90, 0.95, 0.98])
    assert [record["quantile"] for record in scan] == [0.90, 0.95, 0.98]
    shapes = [record["shape"] for record in scan]
    assert max(shapes) - min(shapes) < 0.15


def test_kupiec_accepts_correct_breach_rate() -> None:
    """Ten breaches in 1000 forecasts is exactly the 99% expectation."""
    stat, p_value = kupiec_pof(n_breaches=10, n_forecasts=1000, confidence=0.99)
    assert stat == pytest.approx(0.0, abs=1e-9)
    assert p_value > 0.99


def test_kupiec_rejects_excessive_breaches() -> None:
    """Five times the expected breach count should be rejected."""
    _, p_value = kupiec_pof(n_breaches=50, n_forecasts=1000, confidence=0.99)
    assert p_value < 0.01


def test_christoffersen_flags_clustered_breaches() -> None:
    """Breaches arriving in one consecutive block are not independent."""
    breaches = [False] * 180 + [True] * 20
    _, p_value = christoffersen_independence(breaches)
    assert p_value < 0.01


def test_christoffersen_accepts_scattered_breaches() -> None:
    """Evenly spaced breaches should not look clustered."""
    breaches = [i % 20 == 0 for i in range(400)]
    _, p_value = christoffersen_independence(breaches)
    assert p_value > 0.05


def test_backtest_is_out_of_sample_and_well_calibrated() -> None:
    """On i.i.d. data the EVT model should breach at roughly the nominal rate.

    Coverage is asserted across several seeds rather than one: with about 20
    breaches in a run, the Christoffersen independence statistic is noisy enough
    that a single seed will occasionally reject on correctly specified data, so
    a one-seed assertion would be testing the random draw rather than the model.
    """
    rates, coverage_p_values = [], []
    for seed in (7, 12, 42, 101):
        series = stats.t.rvs(df=5, size=3000, random_state=np.random.default_rng(seed))
        result = backtest_var(series, confidence=0.99, window=750, method="evt")
        assert result.n_forecasts == 2250
        rates.append(result.breach_rate)
        coverage_p_values.append(result.kupiec_p_value)

    assert float(np.mean(rates)) == pytest.approx(0.01, abs=0.005)
    assert min(coverage_p_values) > 0.05


def test_gaussian_backtest_breaches_more_than_evt() -> None:
    """A normal model on heavy-tailed data should breach more often than EVT."""
    series = stats.t.rvs(df=4, size=3000, random_state=np.random.default_rng(3))
    evt = backtest_var(series, confidence=0.99, window=750, method="evt")
    gaussian = backtest_var(series, confidence=0.99, window=750, method="gaussian")
    assert gaussian.n_breaches > evt.n_breaches


def test_short_series_raises_actionable_error() -> None:
    """Too few exceedances must fail loudly rather than return a silent fit."""
    with pytest.raises(EVTError, match="exceedances"):
        fit_gpd(np.arange(50.0), threshold_quantile=0.95)


def test_quantile_below_threshold_is_rejected(student_t_losses: np.ndarray) -> None:
    """The GPD does not describe the body of the distribution."""
    fit = fit_gpd(student_t_losses, threshold_quantile=0.95)
    with pytest.raises(EVTError, match="exceedance rate"):
        value_at_risk_evt(fit, 0.50)


def test_window_longer_than_series_is_rejected() -> None:
    """A backtest needs more observations than its window."""
    with pytest.raises(EVTError, match="window"):
        backtest_var(np.random.default_rng(1).normal(size=200), window=500)
