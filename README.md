# tailrisk-mcp

An MCP server that gives an LLM agent a set of extreme value statistics tools for tail risk: peaks-over-threshold GPD fitting, Value-at-Risk and Expected Shortfall, return levels, threshold-stability diagnostics, and out-of-sample VaR backtesting.

## Why

Extreme value theory is the same machinery whether the tail is a flood or a portfolio loss. A 100-year return level and a 99% VaR on annual data are the same quantile of the same fitted distribution. This server exposes that machinery as agent-callable tools, so a model can be asked "how heavy is this tail, and does the risk model hold up out of sample?" and answer it with a real estimator rather than a plausible-sounding number.

The tools are deliberately opinionated about statistical practice:

- Threshold choice is the main judgement call in a POT analysis, so `tailrisk_threshold_stability` exists to make an agent check it rather than accept a single fit.
- `tailrisk_var_es` returns EVT, historical and Gaussian estimates side by side, because the gap between them is the argument for using a tail model at all.
- Backtests are strictly out of sample: every forecast is fitted on trailing data only.
- Errors explain what to do next ("lower the threshold", "the sample is too short at this confidence level") instead of failing silently or returning a fit that should not be trusted.

## Tools

| Tool | Purpose |
| --- | --- |
| `tailrisk_fit_gpd` | Fit a generalised Pareto distribution to threshold exceedances |
| `tailrisk_var_es` | VaR and Expected Shortfall by EVT, historical simulation and Gaussian baseline |
| `tailrisk_return_level` | Level exceeded once per return period, with the equivalent VaR confidence |
| `tailrisk_backtest_var` | Rolling-window backtest with Kupiec, Christoffersen and conditional coverage tests |
| `tailrisk_threshold_stability` | Refit across candidate thresholds to check the fit is in the asymptotic regime |
| `tailrisk_load_csv_series` | Read a numeric column from a local CSV so a series can be analysed |

All tools are read-only and side-effect free.

## Install

```bash
git clone https://github.com/John-Amal/tailrisk-mcp.git
cd tailrisk-mcp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Requires Python 3.10+ and MCP SDK 2.x.

## Connect to an MCP client

Add to your client's server configuration (for Claude Desktop, `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "tailrisk": {
      "command": "/absolute/path/to/tailrisk-mcp/.venv/bin/python",
      "args": ["-m", "tailrisk_mcp.server"]
    }
  }
}
```

The server speaks stdio. Use `mcp.run(transport="streamable-http")` in `server.py` for a remote deployment.

## Worked example

On 5,000 draws from a Student-t with 4 degrees of freedom, asking for the 99.9% loss quantile:

| Method | VaR (99.9%) | Expected Shortfall |
| --- | --- | --- |
| EVT (POT, threshold at the 95th percentile) | 7.52 | 9.97 |
| Historical simulation | 8.51 | 10.02 |
| Gaussian | 4.41 | not defined here |

The true value is 7.17. The Gaussian figure understates it by 39%, which is the practical point: at the 99.9% level a normality assumption discards most of the risk. Historical simulation happens to land close here but cannot go beyond the largest observed loss at all, so it stops working exactly where the question gets interesting.

Reproduce with:

```bash
python examples/demo.py
```

## Methods

Exceedances above a threshold `u` are modelled with a GPD by maximum likelihood, giving shape `xi` and scale `sigma`. The tail is

```
P(X > z) = zeta_u * [1 + xi (z - u) / sigma] ** (-1 / xi)
```

where `zeta_u` is the empirical exceedance rate. Inverting it gives both VaR (set the exceedance probability to `1 - q`) and return levels (set it to `1 / T`). Expected Shortfall follows in closed form as `(VaR + sigma - xi*u) / (1 - xi)` and is reported as undefined when `xi >= 1`, since the tail then has no finite mean.

Backtesting uses the Kupiec proportion-of-failures test for breach frequency and the Christoffersen test for breach independence, combined into a conditional coverage statistic.

References: Coles (2001); McNeil and Frey (2000); Kupiec (1995); Christoffersen (1998).

## Testing

`pytest` runs 17 tests. The statistical ones simulate from distributions with known tail behaviour and check the estimators recover it: a Pareto(3) sample should return a shape near 1/3, and EVT VaR on Student-t data should match the analytic quantile within 10%. The calibration test evaluates coverage across several random seeds, because with around 20 breaches per run the independence statistic is noisy enough that a single seed will occasionally reject correctly specified data.

## Limitations

- Observations are assumed independent and identically distributed. Real financial returns are volatility-clustered, so a production implementation would filter with a GARCH model first and fit the GPD to the standardised residuals, following McNeil and Frey. This server fits the raw series.
- Parameter uncertainty is not propagated: no confidence intervals on VaR, ES or return levels. Profile likelihood or a bootstrap would be the next addition.
- No multivariate or dependence modelling, so nothing here addresses portfolio aggregation across risk factors.

## Licence

MIT.
