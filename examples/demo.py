"""Reproduce the worked example in the README.

Compares EVT, historical and Gaussian tail estimates on heavy-tailed data
where the true quantile is known analytically.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from tailrisk_mcp.backtest import backtest_var
from tailrisk_mcp.evt import (
    expected_shortfall_evt,
    expected_shortfall_historical,
    fit_gpd,
    value_at_risk_evt,
    value_at_risk_gaussian,
    value_at_risk_historical,
)

CONFIDENCE = 0.999
DEGREES_OF_FREEDOM = 4


def main() -> None:
    """Print the tail comparison and a backtest summary."""
    losses = stats.t.rvs(df=DEGREES_OF_FREEDOM, size=5000, random_state=np.random.default_rng(5))
    truth = float(stats.t.ppf(CONFIDENCE, df=DEGREES_OF_FREEDOM))

    fit = fit_gpd(losses, threshold_quantile=0.95)
    print(f"GPD shape {fit.shape:.3f} on {fit.n_exceedances} exceedances above {fit.threshold:.3f}\n")

    print(f"{'method':<12}{'VaR':>10}{'ES':>10}")
    print(f"{'evt':<12}{value_at_risk_evt(fit, CONFIDENCE):>10.2f}{expected_shortfall_evt(fit, CONFIDENCE):>10.2f}")
    print(
        f"{'historical':<12}{value_at_risk_historical(losses, CONFIDENCE):>10.2f}"
        f"{expected_shortfall_historical(losses, CONFIDENCE):>10.2f}"
    )
    print(f"{'gaussian':<12}{value_at_risk_gaussian(losses, CONFIDENCE):>10.2f}{'-':>10}")
    print(f"{'truth':<12}{truth:>10.2f}{'-':>10}\n")

    result = backtest_var(losses[:3000], confidence=0.99, window=750, method="evt")
    print(
        f"Backtest: {result.n_breaches} breaches in {result.n_forecasts} forecasts "
        f"(expected {result.expected_breach_rate:.1%}, observed {result.breach_rate:.1%})"
    )
    print(result.verdict)


if __name__ == "__main__":
    main()
