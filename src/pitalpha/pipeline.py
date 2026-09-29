"""Deterministic end-to-end experiment pipeline."""

from __future__ import annotations

import copy
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pitalpha.artifacts import (
    build_run_manifest,
    json_ready,
    sha256_file,
    write_frame_atomic,
    write_json_atomic,
    write_text_atomic,
)
from pitalpha.backtest import (
    build_buffered_ranked_portfolio,
    build_equal_weight_benchmark,
    build_ranked_portfolio,
)
from pitalpha.config import load_config
from pitalpha.data import (
    FEATURE_COLUMNS,
    audit_panel_quality,
    generate_synthetic_panel,
    load_audited_qlib_panel,
    require_quality_pass,
)
from pitalpha.environment import repository_root
from pitalpha.metrics import breakeven_cost_bps, portfolio_summary
from pitalpha.models import resolve_model, run_model_adapter
from pitalpha.reporting import render_experiment_report
from pitalpha.signals import (
    coverage_diagnostics,
    daily_information_coefficients,
    quantile_return_diagnostics,
    signal_decay_diagnostics,
    summarize_information_coefficients,
)
from pitalpha.splits import annual_walk_forward_folds
from pitalpha.statistics import moving_block_mean_interval


def _fold_frame(folds) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": fold.name,
                "train_start": fold.train_start.date().isoformat(),
                "train_end": fold.train_end.date().isoformat(),
                "test_start": fold.test_start.date().isoformat(),
                "test_end": fold.test_end.date().isoformat(),
                "embargo_trading_days": fold.embargo_trading_days,
            }
            for fold in folds
        ]
    )


def _apply_model_overrides(
    config: dict[str, Any],
    *,
    seed: int | None = None,
    disable_market_context: bool = False,
    ranking_weight: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an auditable neural-model variant and its explicit override map."""

    resolved = copy.deepcopy(config)
    overrides: dict[str, Any] = {}
    params = resolved["model"]["params"]
    if seed is not None:
        if seed < 0:
            raise ValueError("model seed override must be non-negative")
        if "seed" not in params:
            raise ValueError("model seed override requires model.params.seed")
        params["seed"] = seed
        resolved["experiment"]["model_seed_override"] = seed
        overrides["model.params.seed"] = seed
    if disable_market_context:
        if "use_market_context" not in params:
            raise ValueError("market-context ablation requires model.params.use_market_context")
        params["use_market_context"] = False
        overrides["model.params.use_market_context"] = False
    if ranking_weight is not None:
        if ranking_weight < 0:
            raise ValueError("ranking weight override must be non-negative")
        if "ranking_weight" not in params:
            raise ValueError("ranking ablation requires model.params.ranking_weight")
        params["ranking_weight"] = ranking_weight
        overrides["model.params.ranking_weight"] = ranking_weight

    base_name = re.sub(r"-seed\d+$", "", str(resolved["experiment"]["name"]))
    tags: list[str] = []
    if disable_market_context:
        tags.append("no-context")
    if ranking_weight is not None:
        tags.append("no-rank" if ranking_weight == 0 else f"rank-{ranking_weight:g}".replace(".", "p"))
    model_seed = int(params["seed"])
    suffix = "-".join([*tags, f"seed{model_seed}"])
    resolved["experiment"]["name"] = f"{base_name}-{suffix}"
    if tags:
        resolved["experiment"]["ablation_overrides"] = overrides
    root = Path(str(resolved["artifacts"]["root"]))
    root_name = re.sub(r"-seed\d+$", "", root.name)
    resolved["artifacts"]["root"] = (root.parent / f"{root_name}-{suffix}").as_posix()
    return resolved, overrides


def _apply_model_seed_override(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return an auditable config variant that changes model randomness only."""

    resolved, _ = _apply_model_overrides(config, seed=seed)
    return resolved


def run_experiment(
    config_path: str | Path,
    output_root: str | Path | None = None,
    *,
    model_seed_override: int | None = None,
    disable_market_context: bool = False,
    ranking_weight_override: float | None = None,
    allow_external_model: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    override_map: dict[str, Any] = {}
    if model_seed_override is not None or disable_market_context or ranking_weight_override is not None:
        config, override_map = _apply_model_overrides(
            config,
            seed=model_seed_override,
            disable_market_context=disable_market_context,
            ranking_weight=ranking_weight_override,
        )
    model_name = str(config["model"]["name"])
    model_adapter = resolve_model(model_name, allow_external=allow_external_model)

    started = datetime.now(timezone.utc)
    manifest = build_run_manifest(
        config,
        config_path,
        created_at=started,
        code_revision=os.environ.get("GITHUB_SHA"),
    )
    if override_map:
        manifest["config"]["overrides"] = override_map
    base = Path(output_root) if output_root is not None else repository_root() / config["artifacts"]["root"]
    run_directory = base / manifest["run_id"]
    if run_directory.exists():
        raise RuntimeError(f"run directory already exists: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=False)
    write_json_atomic(run_directory / "run_manifest.json", json_ready(manifest))

    if config["data"]["source"] == "synthetic":
        panel = generate_synthetic_panel(config)
        feature_columns = tuple(FEATURE_COLUMNS)
        data_provenance = {
            "adapter": "deterministic_synthetic_fixture",
            "snapshot_id": config["data"]["snapshot_id"],
            "feature_count": len(feature_columns),
        }
    else:
        panel, feature_columns, data_provenance = load_audited_qlib_panel(
            config, repository_root=repository_root()
        )
    quality_checks, quality_summary = audit_panel_quality(panel, feature_columns, config)
    require_quality_pass(quality_summary)
    folds = annual_walk_forward_folds(panel["datetime"], config)
    prediction_parts = []
    model_runs = []
    for fold in folds:
        if model_adapter.descriptor.input_kind == "temporal":
            history_mask = (panel["datetime"] >= fold.train_start) & (
                panel["datetime"] < fold.test_start
            )
            train = panel.loc[
                history_mask,
                ["datetime", "instrument", "label", *feature_columns],
            ].copy()
            context_only = ~fold.training_mask(train["datetime"])
            train.loc[context_only, "label"] = float("nan")
        else:
            train = panel.loc[
                fold.training_mask(panel["datetime"]),
                ["datetime", "label", *feature_columns],
            ]
        test = panel[fold.test_mask(panel["datetime"])].copy()
        scores, metadata = run_model_adapter(
            model_adapter,
            train,
            test,
            feature_columns,
            config["model"]["params"],
        )
        test["score"] = scores
        test["fold"] = fold.name
        forward_columns = [
            column
            for column in test.columns
            if column.startswith("forward_return_") and column.endswith("d")
        ]
        prediction_parts.append(
            test[["datetime", "instrument", "fold", "score", "label", "realized_return_1d", *forward_columns]]
        )
        model_runs.append({"fold": fold.name, **metadata})
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["datetime", "instrument"], kind="stable"
    )

    prediction_evaluation = predictions[predictions["label"].notna()].copy()
    portfolio_evaluation = predictions[predictions["realized_return_1d"].notna()].copy()
    daily_ic = daily_information_coefficients(prediction_evaluation)
    prediction_summary = summarize_information_coefficients(daily_ic)
    evaluation_config = config["evaluation"]
    quantile_summary, quantile_statistics = quantile_return_diagnostics(
        prediction_evaluation, int(evaluation_config["quantiles"])
    )
    decay_summary = signal_decay_diagnostics(
        predictions, [int(value) for value in evaluation_config["signal_decay_horizons"]]
    )
    coverage = coverage_diagnostics(predictions)
    bootstrap_config = evaluation_config["bootstrap"]
    bootstrap_rows = []
    for index, metric in enumerate(["ic", "rank_ic"]):
        interval = moving_block_mean_interval(
            daily_ic[metric],
            block_length=int(bootstrap_config["block_length_dates"]),
            replications=int(bootstrap_config["replications"]),
            seed=int(config["experiment"]["seed"]) + index,
        )
        bootstrap_rows.append({"metric": metric, **interval})
    bootstrap_summary = pd.DataFrame(bootstrap_rows)
    costs = [int(value) for value in config["portfolio"]["one_way_cost_bps"]]
    top_k = int(config["portfolio"]["top_k"])
    retention_rank = int(config["portfolio"]["retention_rank"])
    ranked_daily, ranked_holdings = build_ranked_portfolio(portfolio_evaluation, top_k=top_k, cost_bps=costs)
    buffered_daily, buffered_holdings = build_buffered_ranked_portfolio(
        portfolio_evaluation, top_k=top_k, retention_rank=retention_rank, cost_bps=costs
    )
    benchmark_daily, benchmark_holdings = build_equal_weight_benchmark(portfolio_evaluation, cost_bps=costs)
    ledgers = {
        f"{model_name}_top_k": ranked_daily,
        f"{model_name}_top_k_buffered": buffered_daily,
        "eligible_equal_weight": benchmark_daily,
    }
    combined_portfolio_summary = pd.concat(
        [portfolio_summary(daily, strategy=strategy, cost_bps=costs) for strategy, daily in ledgers.items()],
        ignore_index=True,
    )
    break_even_summary = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "breakeven_one_way_cost_bps": breakeven_cost_bps(daily),
                "definition": "cost where terminal cumulative return equals zero",
            }
            for strategy, daily in ledgers.items()
        ]
    )
    folds_frame = _fold_frame(folds)

    daily_counts = panel.groupby("datetime")["instrument"].nunique()
    data_summary = {
        "rows": int(len(panel)),
        "dates": int(panel["datetime"].nunique()),
        "instruments": int(panel["instrument"].nunique()),
        "daily_members_min": int(daily_counts.min()),
        "daily_members_max": int(daily_counts.max()),
    }
    primary_cost = int(config["portfolio"]["primary_cost_bps"])
    metrics = json_ready(
        {
            "schema_version": 1,
            "data": data_summary,
            "data_provenance": data_provenance,
            "data_quality": quality_summary,
            "prediction": prediction_summary.to_dict(orient="records"),
            "quantiles": quantile_summary.to_dict(orient="records"),
            "quantile_statistics": quantile_statistics,
            "signal_decay": decay_summary.to_dict(orient="records"),
            "coverage": coverage,
            "bootstrap_uncertainty": bootstrap_summary.to_dict(orient="records"),
            "portfolio": combined_portfolio_summary.to_dict(orient="records"),
            "cost_break_even": break_even_summary.to_dict(orient="records"),
            "model_runs": model_runs,
        }
    )

    written: list[Path] = []
    written.append(write_json_atomic(run_directory / "config.resolved.json", json_ready(config)))
    written.append(write_frame_atomic(quality_checks, run_directory / "data_quality_checks.csv"))
    written.append(write_frame_atomic(predictions, run_directory / "predictions.parquet"))
    holding_exports = []
    for strategy, frame in {
        f"{model_name}_top_k": ranked_holdings,
        f"{model_name}_top_k_buffered": buffered_holdings,
        "eligible_equal_weight": benchmark_holdings,
    }.items():
        export = frame.copy()
        export.insert(1, "strategy", strategy)
        holding_exports.append(export)
    holdings = pd.concat(holding_exports, ignore_index=True)
    written.append(write_frame_atomic(holdings, run_directory / "holdings.parquet"))
    return_exports = []
    for strategy, frame in ledgers.items():
        export = frame.copy()
        export.insert(1, "strategy", strategy)
        return_exports.append(export)
    daily_returns = pd.concat(return_exports, ignore_index=True)
    written.append(write_frame_atomic(daily_returns, run_directory / "daily_returns.csv"))
    written.append(write_frame_atomic(daily_ic, run_directory / "prediction_daily_ic.csv"))
    written.append(write_frame_atomic(prediction_summary, run_directory / "prediction_summary.csv"))
    written.append(write_frame_atomic(quantile_summary, run_directory / "signal_quantiles.csv"))
    written.append(write_frame_atomic(decay_summary, run_directory / "signal_decay.csv"))
    written.append(write_frame_atomic(bootstrap_summary, run_directory / "bootstrap_uncertainty.csv"))
    written.append(write_frame_atomic(combined_portfolio_summary, run_directory / "portfolio_summary.csv"))
    written.append(write_frame_atomic(break_even_summary, run_directory / "cost_break_even.csv"))
    written.append(write_json_atomic(run_directory / "metrics.json", metrics))

    completed_manifest = dict(manifest)
    completed_manifest["status"] = "completed"
    completed_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    completed_manifest["folds"] = json_ready(folds_frame.to_dict(orient="records"))
    completed_manifest["data_provenance"] = json_ready(data_provenance)
    completed_manifest["data_quality"] = json_ready(quality_summary)
    report = render_experiment_report(
        manifest=completed_manifest,
        data_summary=data_summary,
        data_provenance=data_provenance,
        data_quality_checks=quality_checks,
        data_quality_summary=quality_summary,
        fold_summary=folds_frame,
        prediction_summary=prediction_summary,
        quantile_summary=quantile_summary,
        quantile_statistics=quantile_statistics,
        decay_summary=decay_summary,
        bootstrap_summary=bootstrap_summary,
        coverage=coverage,
        portfolio_summary=combined_portfolio_summary,
        break_even_summary=break_even_summary,
        primary_cost_bps=primary_cost,
    )
    written.append(write_text_atomic(report, run_directory / "report.md"))
    completed_manifest["artifacts"] = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in written
    ]
    manifest_path = write_json_atomic(run_directory / "run_manifest.json", json_ready(completed_manifest))
    return {
        "run_id": completed_manifest["run_id"],
        "run_directory": str(run_directory.resolve()),
        "manifest": str(manifest_path.resolve()),
        "report": str((run_directory / "report.md").resolve()),
        "primary_cost_bps": primary_cost,
    }
