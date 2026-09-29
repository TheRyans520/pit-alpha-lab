"""Small dependency-free Markdown report renderer."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    def render(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def render_experiment_report(
    *,
    manifest: Mapping[str, Any],
    data_summary: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    data_quality_checks: pd.DataFrame,
    data_quality_summary: Mapping[str, Any],
    fold_summary: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    quantile_summary: pd.DataFrame,
    quantile_statistics: Mapping[str, Any],
    decay_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    coverage: Mapping[str, Any],
    portfolio_summary: pd.DataFrame,
    break_even_summary: pd.DataFrame,
    primary_cost_bps: int,
) -> str:
    overall_prediction = prediction_summary[prediction_summary["window"] == "ALL"]
    primary_portfolio = portfolio_summary[
        (portfolio_summary["window"] == "ALL") & (portfolio_summary["cost_bps"] == primary_cost_bps)
    ]
    synthetic = data_provenance["adapter"] == "deterministic_synthetic_fixture"
    scope = (
        "This is a deterministic software and methodology fixture. It validates point-in-time membership, annual walk-forward fitting, score-to-holdings translation, turnover, one-way costs and artifact lineage. It is not evidence of a real tradable alpha."
        if synthetic
        else "This run uses a checksum-verified, frozen Qlib parquet export with historical interval membership. Results describe this specific public snapshot and protocol; they are not a live-performance claim."
    )
    limitations = (
        "Synthetic business-day data omit exchange holidays, suspensions, limit moves, queue position and market impact. Break-even cost is not a liquidity or market-impact model. The fixture exists to make the pipeline testable in public CI; empirical claims require the frozen provider adapter and separate data audit."
        if synthetic
        else "The public daily dataset can contain survivorship, delisting, suspension and corporate-action limitations that are not equivalent to institutional point-in-time feeds. Daily bars cannot establish queue position or intraday execution, and break-even cost is not a calibrated capacity model."
    )
    lines = [
        f"# PIT Alpha Lab {data_provenance['snapshot_id']} experiment report",
        "",
        f"**Run:** `{manifest['run_id']}`  ",
        f"**Config SHA-256:** `{manifest['content_id']}`  ",
        "**Status:** completed",
        "",
        "## Scope",
        "",
        scope,
        "",
        "## Data",
        "",
        f"- Rows: {data_summary['rows']:,}",
        f"- Dates: {data_summary['dates']:,}",
        f"- Instruments observed: {data_summary['instruments']}",
        f"- Eligible names per date: {data_summary['daily_members_min']} to {data_summary['daily_members_max']}",
        f"- Adapter: `{data_provenance['adapter']}`",
        f"- Feature count: {data_provenance['feature_count']}",
        "",
        "## Semantic data-quality gate",
        "",
        f"Overall status: **{data_quality_summary['status']}**; {data_quality_summary['passed']} checks passed, {data_quality_summary['failed_errors']} error-level checks failed.",
        "",
        _table(data_quality_checks),
        "",
        "## Walk-forward folds",
        "",
        _table(fold_summary),
        "",
        "## Prediction metrics",
        "",
        _table(overall_prediction),
        "",
        "Year-by-year prediction metrics are stored in `prediction_summary.csv`.",
        "",
        "## Signal diagnostics",
        "",
        _table(quantile_summary),
        "",
        f"Quantile monotonicity (Spearman): {quantile_statistics['quantile_monotonicity_spearman']:.4f}; top-minus-bottom mean daily five-day return: {quantile_statistics['top_minus_bottom_mean_daily_return']:.4%}.",
        "",
        "### RankIC decay by forward horizon",
        "",
        _table(decay_summary),
        "",
        f"Score coverage is {coverage['score_coverage']:.2%}; label coverage is {coverage['label_coverage']:.2%}. Terminal unavailable labels are retained in predictions but excluded from metric denominators.",
        "",
        "### Moving-block uncertainty",
        "",
        _table(bootstrap_summary),
        "",
        f"## Portfolio metrics at {primary_cost_bps} bps one-way cost",
        "",
        _table(primary_portfolio),
        "",
        "The complete cost frontier and annual values are stored in `portfolio_summary.csv`.",
        "",
        "### Transaction-cost break-even",
        "",
        _table(break_even_summary),
        "",
        "Break-even is the hypothetical constant one-way cost that reduces terminal cumulative return to zero. It is a sensitivity statistic, not a capacity estimate.",
        "",
        "## Timing and accounting contract",
        "",
        "A score formed on decision date `t` is applied to the next observed daily open-to-open return. The portfolio rebalances on the first available date of each ISO week. The buffered strategy retains existing names through the configured retention rank before filling vacancies from the highest scores. Turnover is the sum of absolute risky-weight changes, initial entry is charged, and missing held-name returns remain explicit as forced cash weight.",
        "",
        "## Limitations",
        "",
        limitations,
        "",
    ]
    return "\n".join(lines)
