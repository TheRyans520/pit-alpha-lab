"""Semantic quality gates for canonical point-in-time panels."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


class DataQualityError(RuntimeError):
    """Raised when an error-level semantic data check fails."""


def audit_panel_quality(
    panel: pd.DataFrame,
    feature_columns: Iterable[str],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate data semantics without modifying the input panel."""

    features = list(feature_columns)
    thresholds = config["data_quality"]
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, severity: str, observed: Any, expected: str, detail: Any = None) -> None:
        rows.append(
            {
                "check": name,
                "status": "pass" if passed else ("warning" if severity == "warning" else "fail"),
                "severity": severity,
                "observed": observed,
                "expected": expected,
                "detail": "" if detail is None else json.dumps(detail, sort_keys=True),
            }
        )

    duplicate_keys = int(panel.duplicated(["datetime", "instrument"]).sum())
    add("unique_instrument_date_key", duplicate_keys == 0, "error", duplicate_keys, "0 duplicate rows")
    add(
        "minimum_row_count",
        len(panel) >= int(thresholds["minimum_rows"]),
        "error",
        int(len(panel)),
        f">= {int(thresholds['minimum_rows'])}",
    )
    daily_members = panel.groupby("datetime", sort=False)["instrument"].nunique()
    observed_min = int(daily_members.min())
    observed_max = int(daily_members.max())
    add(
        "daily_universe_size",
        observed_min >= int(thresholds["minimum_daily_members"])
        and observed_max <= int(thresholds["maximum_daily_members"]),
        "error",
        f"{observed_min}..{observed_max}",
        f"{int(thresholds['minimum_daily_members'])}..{int(thresholds['maximum_daily_members'])}",
    )
    sorted_keys = panel[["datetime", "instrument"]].reset_index(drop=True)
    expected_order = sorted_keys.sort_values(["datetime", "instrument"], kind="stable").reset_index(drop=True)
    add("canonical_sort_order", sorted_keys.equals(expected_order), "error", sorted_keys.equals(expected_order), "true")

    coverage = {column: float(np.isfinite(panel[column].to_numpy(dtype=float)).mean()) for column in features}
    lowest_features = sorted(coverage.items(), key=lambda item: item[1])[:10]
    minimum_feature_coverage = min(coverage.values()) if coverage else 0.0
    add(
        "minimum_feature_coverage",
        minimum_feature_coverage >= float(thresholds["minimum_feature_coverage"]),
        "error",
        minimum_feature_coverage,
        f">= {float(thresholds['minimum_feature_coverage'])}",
        {name: value for name, value in lowest_features},
    )
    all_missing = [name for name, value in coverage.items() if value == 0.0]
    add("no_all_missing_features", not all_missing, "error", len(all_missing), "0", all_missing)

    label_coverage = float(np.isfinite(panel["label"].to_numpy(dtype=float)).mean())
    return_coverage = float(np.isfinite(panel["realized_return_1d"].to_numpy(dtype=float)).mean())
    add(
        "label_coverage",
        label_coverage >= float(thresholds["minimum_label_coverage"]),
        "error",
        label_coverage,
        f">= {float(thresholds['minimum_label_coverage'])}",
    )
    add(
        "realized_return_coverage",
        return_coverage >= float(thresholds["minimum_realized_return_coverage"]),
        "error",
        return_coverage,
        f">= {float(thresholds['minimum_realized_return_coverage'])}",
    )
    finite_returns = panel.loc[np.isfinite(panel["realized_return_1d"]), "realized_return_1d"].abs()
    maximum_return = float(finite_returns.max()) if not finite_returns.empty else np.nan
    add(
        "maximum_absolute_realized_return",
        bool(np.isfinite(maximum_return) and maximum_return <= float(thresholds["maximum_absolute_realized_return"])),
        "error",
        maximum_return,
        f"<= {float(thresholds['maximum_absolute_realized_return'])}",
    )

    raw_columns = ["raw_open", "raw_high", "raw_low", "raw_close", "raw_volume", "raw_factor"]
    if all(column in panel for column in raw_columns):
        prices = panel[["raw_open", "raw_high", "raw_low", "raw_close"]]
        finite_prices = np.isfinite(prices)
        complete_prices = finite_prices.all(axis=1)
        price_coverage = float(complete_prices.mean())
        nonpositive = int(((prices <= 0.0) & finite_prices).sum().sum())
        ordering_violations = int(
            (
                complete_prices
                & (
                    (panel["raw_high"] < panel[["raw_open", "raw_close"]].max(axis=1))
                    | (panel["raw_low"] > panel[["raw_open", "raw_close"]].min(axis=1))
                    | (panel["raw_high"] < panel["raw_low"])
                )
            ).sum()
        )
        negative_volume = int(((panel["raw_volume"] < 0.0) & np.isfinite(panel["raw_volume"])).sum())
        nonpositive_factor = int(((panel["raw_factor"] <= 0.0) & np.isfinite(panel["raw_factor"])).sum())
        add(
            "ohlc_complete_coverage",
            price_coverage >= float(thresholds["minimum_ohlc_coverage"]),
            "error",
            price_coverage,
            f">= {float(thresholds['minimum_ohlc_coverage'])}",
        )
        add("positive_observed_prices", nonpositive == 0, "error", nonpositive, "0")
        add("ohlc_ordering", ordering_violations == 0, "error", ordering_violations, "0")
        add("nonnegative_observed_volume", negative_volume == 0, "error", negative_volume, "0")
        add("positive_observed_adjustment_factor", nonpositive_factor == 0, "error", nonpositive_factor, "0")
    else:
        rows.append(
            {
                "check": "raw_ohlcv_semantics",
                "status": "skipped",
                "severity": "info",
                "observed": "not available",
                "expected": "raw fields are optional for synthetic fixtures",
                "detail": "",
            }
        )

    checks = pd.DataFrame(rows)
    failed_errors = checks[(checks["severity"] == "error") & (checks["status"] == "fail")]
    warnings = checks[checks["status"] == "warning"]
    summary = {
        "status": "fail" if not failed_errors.empty else ("pass_with_warnings" if not warnings.empty else "pass"),
        "checks": int(len(checks)),
        "passed": int((checks["status"] == "pass").sum()),
        "warnings": int(len(warnings)),
        "failed_errors": int(len(failed_errors)),
        "minimum_feature_coverage": minimum_feature_coverage,
        "lowest_coverage_features": {name: value for name, value in lowest_features},
        "label_coverage": label_coverage,
        "realized_return_coverage": return_coverage,
    }
    return checks, summary


def require_quality_pass(summary: Mapping[str, Any]) -> None:
    if summary["status"] == "fail":
        raise DataQualityError(f"semantic data quality gate failed with {summary['failed_errors']} error checks")
