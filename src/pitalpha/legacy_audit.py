"""Read-only diagnosis of saved predictions and historical cost-overlay artifacts.

This verifies file consistency and saved-return arithmetic. It cannot refit a
model, certify raw market data, or create a corrected return for an unvalued holding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitalpha.artifacts import json_ready, sha256_file
from pitalpha.backtest import build_ranked_portfolio
from pitalpha.pipeline import portfolio_evaluation_window


def audit_saved_run(directory: Path) -> dict:
    directory = directory.resolve()
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("audit requires a completed run")
    entries = manifest.get("artifacts", [])
    names = [entry["path"] for entry in entries]
    required = {"config.resolved.json", "predictions.parquet", "holdings.parquet",
                "daily_returns.csv", "portfolio_summary.csv"}
    if len(set(names)) != len(names) or not required.issubset(names):
        raise ValueError("manifest has duplicate or missing required artifacts")
    for entry in entries:
        path = (directory / entry["path"]).resolve()
        if path.parent != directory or path.name != entry["path"]:
            raise ValueError("artifact path must be a direct child of the run directory")
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {path.name}")
    config = json.loads((directory / "config.resolved.json").read_text(encoding="utf-8"))
    if config["portfolio"]["rebalance"] not in {"weekly", "first_available_trading_day_each_week"}:
        raise ValueError("saved-score diagnostic currently supports weekly portfolios only")
    model = str(config["model"]["name"])
    strategy = f"{model}_top_k"
    predictions = pd.read_parquet(directory / "predictions.parquet")
    predictions["datetime"] = pd.to_datetime(predictions["datetime"])
    if (predictions.empty or predictions[["datetime", "instrument"]].isna().any().any()
            or predictions.duplicated(["datetime", "instrument"]).any()):
        raise ValueError("predictions are empty or have duplicate keys")
    if not np.isfinite(predictions["score"]).all():
        raise ValueError("saved scores must be finite")
    daily = pd.read_csv(directory / "daily_returns.csv", parse_dates=["datetime"])
    if daily.duplicated(["strategy", "datetime"]).any():
        raise ValueError("daily returns have duplicate strategy-date keys")
    summaries = pd.read_csv(directory / "portfolio_summary.csv", dtype={"window": str})
    checks = 0
    for row in summaries.itertuples():
        window = daily[daily["strategy"] == row.strategy].sort_values("datetime")
        if row.window != "ALL":
            window = window[window["datetime"].dt.year == int(row.window)]
        values = window[f"net_return_{row.cost_bps}bps"].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all() or (values <= -1.).any():
            raise ValueError("saved returns must be finite, nonempty and solvent")
        wealth = np.r_[1., np.cumprod(1. + values)]
        deviation = values.std(ddof=1) if len(values) > 1 else np.nan
        calculated = {"days": len(values), "annualized_return": wealth[-1] ** (252. / len(values)) - 1.,
                      "sharpe": values.mean() / deviation * np.sqrt(252.) if deviation > 0 else np.nan,
                      "maximum_drawdown": (wealth / np.maximum.accumulate(wealth) - 1.).min()}
        if any(not np.isclose(getattr(row, key), value, rtol=1e-9, atol=1e-11, equal_nan=True)
               for key, value in calculated.items()):
            raise ValueError(f"saved summary arithmetic mismatch: {row.strategy}/{row.window}/{row.cost_bps}")
        checks += 1
    primary = summaries[(summaries["strategy"] == strategy) & (summaries["window"] == "ALL") &
                        (summaries["cost_bps"] == config["portfolio"]["primary_cost_bps"])]
    if len(primary) != 1:
        raise ValueError("one primary summary row is required")
    window, policy = portfolio_evaluation_window(predictions, predictions["datetime"], source=config["data"]["source"])
    top_k = int(config["portfolio"]["top_k"])
    differences = []
    previous_week = None
    for date, frame in window.groupby("datetime", sort=True):
        week = tuple(date.isocalendar()[:2])
        if week == previous_week:
            continue
        previous_week = week
        frame = frame.sort_values("instrument", kind="stable")
        exante = set(frame.nlargest(top_k, "score", keep="first")["instrument"])
        legacy = set(frame[np.isfinite(frame["realized_return_1d"])].nlargest(top_k, "score", keep="first")["instrument"])
        if exante != legacy:
            differences.append({"date": date.date().isoformat(), "excluded_high_score_names": sorted(exante - legacy),
                                "substituted_names": sorted(legacy - exante)})
    try:
        build_ranked_portfolio(window, top_k=top_k, cost_bps=[config["portfolio"]["primary_cost_bps"]],
                               universe_exit_policy=policy["universe_exit_policy"])
        replay = {"status": "passed_saved_score_accounting_only"}
    except ValueError as exc:
        replay = {"status": "blocked_by_valuation_policy", "first_error": str(exc)}
    return json_ready({
        "schema_version": 1, "run_id": manifest["run_id"], "model": model,
        "manifest_sha256": sha256_file(directory / "run_manifest.json"), "artifact_hashes_verified": len(entries),
        "prediction_sha256": sha256_file(directory / "predictions.parquet"),
        "prediction_rows": len(predictions), "dates": predictions["datetime"].nunique(),
        "summary_rows_reconciled": checks,
        "legacy_primary_metrics": primary[["annualized_return", "sharpe", "maximum_drawdown"]].iloc[0].to_dict(),
        "calendar_policy": policy,
        "missing_returns_after_calendar_crop": int((~np.isfinite(window["realized_return_1d"])).sum()),
        "weekly_selection_differences": differences, "strict_replay": replay,
        "corrected_market_performance": "not_established",
        "limitations": ["saved scores only; models and raw outcomes not independently reproduced",
                        "selection comparison covers unbuffered weekly ranks only, not all path-dependent effects",
                        "calendar crop uses saved prediction dates, not an independently verified exchange calendar"],
    })
