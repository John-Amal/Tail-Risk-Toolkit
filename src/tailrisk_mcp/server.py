"""MCP server exposing extreme value tail-risk analytics as agent tools.

Run locally over stdio::

    python -m tailrisk_mcp.server

Every tool is read-only and side-effect free: the series is supplied inline or
read from a local CSV, and nothing is written back.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .backtest import backtest_var
from .evt import (
    EVTError,
    expected_shortfall_evt,
    expected_shortfall_historical,
    fit_gpd,
    return_level,
    threshold_stability,
    value_at_risk_evt,
    value_at_risk_gaussian,
    value_at_risk_historical,
)

mcp = MCPServer("tailrisk_mcp", version=__version__)

MAX_SERIES_LENGTH = 100_000


def _read_only(title: str) -> ToolAnnotations:
    """Annotate a tool as read-only, non-destructive and idempotent."""
    return ToolAnnotations(
        title=title,
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

TailField = Annotated[
    Literal["upper", "lower"],
    Field(
        default="upper",
        description=(
            "Which tail is adverse. Use 'upper' for a loss series where large "
            "positive values are bad, 'lower' for a return series where large "
            "negative values are bad."
        ),
    ),
]


class SeriesInput(BaseModel):
    """Common input carrying the observation series."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    values: list[float] = Field(
        ...,
        description="Observations in chronological order, e.g. [0.4, -1.2, 3.8]",
        min_length=2,
        max_length=MAX_SERIES_LENGTH,
    )
    tail: TailField


class FitInput(SeriesInput):
    """Input for a peaks-over-threshold GPD fit."""

    threshold: float | None = Field(
        default=None,
        description="Explicit POT threshold. Omit to use threshold_quantile instead.",
    )
    threshold_quantile: float = Field(
        default=0.90,
        description="Empirical quantile used as the threshold when none is given, e.g. 0.95",
        gt=0.5,
        lt=1.0,
    )


class RiskMeasureInput(FitInput):
    """Input for Value-at-Risk and Expected Shortfall estimation."""

    confidence: float = Field(
        default=0.99,
        description="Confidence level for VaR and ES, e.g. 0.99 for a 1-in-100 loss",
        gt=0.5,
        lt=1.0,
    )


class ReturnLevelInput(FitInput):
    """Input for return-level estimation."""

    return_period: float = Field(
        ...,
        description=(
            "Return period in units of the observation frequency: 100 means the "
            "level exceeded once per 100 observations."
        ),
        gt=1.0,
    )


class BacktestInput(SeriesInput):
    """Input for a rolling-window VaR backtest."""

    confidence: float = Field(default=0.99, description="VaR confidence level", gt=0.5, lt=1.0)
    window: int = Field(
        default=500,
        description="Trailing observations used to fit each one-step-ahead forecast",
        ge=100,
        le=10_000,
    )
    method: Literal["evt", "historical", "gaussian"] = Field(
        default="evt",
        description="Estimator to backtest. Compare 'evt' against 'gaussian' to show tail-model value.",
    )


class ThresholdStabilityInput(SeriesInput):
    """Input for a threshold-stability scan."""

    quantiles: list[float] | None = Field(
        default=None,
        description="Candidate threshold quantiles, e.g. [0.9, 0.95, 0.99]. Omit for a default sweep.",
        max_length=25,
    )


class LoadCsvInput(BaseModel):
    """Input for reading a numeric column out of a local CSV file."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    path: str = Field(..., description="Path to a local CSV file with a header row", min_length=1)
    column: str = Field(..., description="Name of the numeric column to extract", min_length=1)
    max_rows: int = Field(
        default=20_000,
        description="Stop after this many parsed values",
        ge=2,
        le=MAX_SERIES_LENGTH,
    )


def _respond(payload: dict[str, Any]) -> str:
    """Serialise a successful tool payload."""
    return json.dumps(payload, indent=2, default=float)


def _fail(error: Exception) -> str:
    """Serialise an error with guidance the agent can act on."""
    if isinstance(error, EVTError):
        return json.dumps({"error": "invalid_input", "message": str(error)}, indent=2)
    if isinstance(error, FileNotFoundError):
        return json.dumps(
            {"error": "not_found", "message": f"{error}. Check the path is absolute and readable."},
            indent=2,
        )
    return json.dumps(
        {"error": type(error).__name__, "message": str(error) or "Unexpected failure."}, indent=2
    )


@mcp.tool(name="tailrisk_fit_gpd", annotations=_read_only("Fit GPD to tail exceedances"))
async def tailrisk_fit_gpd(params: FitInput) -> str:
    """Fit a generalised Pareto distribution to threshold exceedances.

    Use this first when asked how heavy a tail is. The shape parameter is the
    headline number: above zero means a heavy tail with no upper bound, near
    zero means exponential decay, below zero means a finite worst case.

    Returns:
        str: JSON with keys ``threshold``, ``shape``, ``scale``,
        ``n_observations``, ``n_exceedances``, ``exceedance_rate``,
        ``log_likelihood`` and an ``interpretation`` string.
    """
    try:
        fit = fit_gpd(params.values, params.threshold, params.threshold_quantile, params.tail)
    except (EVTError, ValueError) as exc:
        return _fail(exc)

    if fit.shape > 0.05:
        reading = "Heavy tail: extremes are unbounded and a Gaussian model will understate them."
    elif fit.shape < -0.05:
        reading = "Bounded tail: the model implies a finite worst case."
    else:
        reading = "Near-exponential tail: decay is close to the Gumbel case."

    return _respond({**fit.as_dict(), "interpretation": reading})


@mcp.tool(name="tailrisk_var_es", annotations=_read_only("Value-at-Risk and Expected Shortfall"))
async def tailrisk_var_es(params: RiskMeasureInput) -> str:
    """Estimate VaR and Expected Shortfall by three methods for comparison.

    The EVT estimate extrapolates beyond the observed sample, the historical
    estimate cannot, and the Gaussian estimate is a deliberately naive baseline.
    A large gap between the EVT and Gaussian figures is the quantitative case
    for using a tail model at all.

    Returns:
        str: JSON with an ``evt`` object (``var``, ``es``, ``shape``,
        ``threshold``), ``historical`` and ``gaussian`` objects, plus
        ``evt_vs_gaussian_ratio`` and the ``confidence`` level. Individual
        methods report a ``note`` instead of a number when undefined.
    """
    result: dict[str, Any] = {"confidence": params.confidence, "tail": params.tail}

    try:
        fit = fit_gpd(params.values, params.threshold, params.threshold_quantile, params.tail)
        evt_var = value_at_risk_evt(fit, params.confidence)
        evt_block: dict[str, Any] = {
            "var": evt_var,
            "shape": fit.shape,
            "threshold": fit.threshold,
            "n_exceedances": fit.n_exceedances,
        }
        try:
            evt_block["es"] = expected_shortfall_evt(fit, params.confidence)
        except EVTError as exc:
            evt_block["es"] = None
            evt_block["note"] = str(exc)
        result["evt"] = evt_block
    except (EVTError, ValueError) as exc:
        return _fail(exc)

    hist: dict[str, Any] = {"var": value_at_risk_historical(params.values, params.confidence, params.tail)}
    try:
        hist["es"] = expected_shortfall_historical(params.values, params.confidence, params.tail)
    except EVTError as exc:
        hist["es"] = None
        hist["note"] = str(exc)
    result["historical"] = hist

    gaussian_var = value_at_risk_gaussian(params.values, params.confidence, params.tail)
    result["gaussian"] = {"var": gaussian_var}
    result["evt_vs_gaussian_ratio"] = evt_var / gaussian_var if gaussian_var else None

    return _respond(result)


@mcp.tool(name="tailrisk_return_level", annotations=_read_only("Return level for a return period"))
async def tailrisk_return_level(params: ReturnLevelInput) -> str:
    """Estimate the level exceeded once per return period.

    This is the same POT quantile as VaR in a different vocabulary: a 100-period
    return level equals the 99% VaR on data at that frequency. Use it when the
    question is phrased as a 1-in-N event rather than a confidence level.

    Returns:
        str: JSON with ``return_period``, ``return_level``,
        ``equivalent_var_confidence``, ``threshold`` and ``shape``.
    """
    try:
        fit = fit_gpd(params.values, params.threshold, params.threshold_quantile, params.tail)
        level = return_level(fit, params.return_period)
    except (EVTError, ValueError) as exc:
        return _fail(exc)

    return _respond(
        {
            "return_period": params.return_period,
            "return_level": level,
            "equivalent_var_confidence": 1.0 - 1.0 / params.return_period,
            "threshold": fit.threshold,
            "shape": fit.shape,
        }
    )


@mcp.tool(name="tailrisk_backtest_var", annotations=_read_only("Backtest a VaR model"))
async def tailrisk_backtest_var(params: BacktestInput) -> str:
    """Backtest a VaR model out of sample with Kupiec and Christoffersen tests.

    Each forecast uses only trailing data, so the result is a genuine
    out-of-sample assessment of whether breaches occur at the right rate and
    without clustering. This is the evidence a model validation report needs.

    Returns:
        str: JSON with ``n_forecasts``, ``n_breaches``, ``breach_rate``,
        ``expected_breach_rate``, the Kupiec, Christoffersen and conditional
        coverage statistics with p-values, and a plain-language ``verdict``.
    """
    try:
        result = backtest_var(params.values, params.confidence, params.window, params.method, params.tail)
    except (EVTError, ValueError) as exc:
        return _fail(exc)
    return _respond(result.as_dict())


@mcp.tool(name="tailrisk_threshold_stability", annotations=_read_only("Check POT threshold stability"))
async def tailrisk_threshold_stability(params: ThresholdStabilityInput) -> str:
    """Refit across candidate thresholds to test whether the tail fit is stable.

    Threshold choice is the main judgement call in a POT analysis. Call this
    before trusting a single fit: if the shape parameter drifts steadily with
    the threshold, the estimate is not yet in the asymptotic regime.

    Returns:
        str: JSON with a ``scan`` list (one record per threshold, each with
        ``quantile``, ``threshold``, ``shape``, ``scale``, ``modified_scale``,
        ``n_exceedances``), the ``shape_range`` across the scan, and a
        ``stability`` verdict.
    """
    try:
        scan = threshold_stability(params.values, params.quantiles, params.tail)
    except (EVTError, ValueError) as exc:
        return _fail(exc)

    shapes = [record["shape"] for record in scan]
    spread = max(shapes) - min(shapes)
    stability = (
        "Shape is stable across thresholds; the fit is defensible."
        if spread < 0.15
        else "Shape drifts with the threshold; treat the extrapolation with caution."
    )
    return _respond(
        {
            "scan": scan,
            "shape_range": {"min": min(shapes), "max": max(shapes), "spread": spread},
            "stability": stability,
        }
    )


@mcp.tool(name="tailrisk_load_csv_series", annotations=_read_only("Load a series from CSV"))
async def tailrisk_load_csv_series(params: LoadCsvInput) -> str:
    """Read one numeric column from a local CSV so a series can be analysed.

    Call this before the analysis tools when the data lives in a file rather
    than in the conversation. Non-numeric and empty cells are skipped and
    counted so data quality is visible.

    Returns:
        str: JSON with ``values`` (the parsed numbers), ``count``,
        ``skipped_rows`` and ``truncated``.
    """
    path = Path(params.path).expanduser()
    values: list[float] = []
    skipped = 0
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or params.column not in reader.fieldnames:
                return json.dumps(
                    {
                        "error": "invalid_input",
                        "message": (
                            f"Column '{params.column}' not found. Available columns: "
                            f"{reader.fieldnames}"
                        ),
                    },
                    indent=2,
                )
            for row in reader:
                if len(values) >= params.max_rows:
                    break
                try:
                    values.append(float(row[params.column]))
                except (TypeError, ValueError):
                    skipped += 1
    except (OSError, UnicodeDecodeError) as exc:
        return _fail(exc)

    if len(values) < 2:
        return json.dumps(
            {
                "error": "invalid_input",
                "message": (
                    f"Only {len(values)} numeric values parsed from '{params.column}' "
                    f"({skipped} rows skipped). Check the column and delimiter."
                ),
            },
            indent=2,
        )

    return _respond(
        {
            "values": values,
            "count": len(values),
            "skipped_rows": skipped,
            "truncated": len(values) >= params.max_rows,
        }
    )


def main() -> None:
    """Run the server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
