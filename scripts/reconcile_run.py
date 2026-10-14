"""Independent arithmetic check of v2 position ledgers and their published metrics.

Does not call the portfolio engine or its metric functions. SHA256 establishes
file consistency, not economic validity or authentic market-data provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def close(actual, expected, description):
    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-11, equal_nan=True):
        raise ValueError(f"reconciliation failed: {description}")


def reconcile(directory: Path) -> dict:
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("only completed runs can be reconciled")
    if manifest.get("accounting_policy", {}).get("holdings_schema") != "daily_positions_and_exits_v2":
        raise ValueError("requires daily_positions_and_exits_v2; legacy runs are not revalidated")
    artifacts = manifest.get("artifacts", [])
    required = {"holdings.parquet", "predictions.parquet", "daily_returns.csv", "portfolio_summary.csv",
                "prediction_daily_ic.csv", "config.resolved.json"}
    if not required.issubset({entry["path"] for entry in artifacts}):
        raise ValueError("manifest is missing required artifact checksums")
    for artifact in artifacts:
        path = (directory / artifact["path"]).resolve()
        if path.parent != directory.resolve():
            raise ValueError("artifact must be a direct child of the run directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError(f"checksum mismatch: {path.name}")

    positions = pd.read_parquet(directory / "holdings.parquet")
    predictions = pd.read_parquet(directory / "predictions.parquet")
    daily = pd.read_csv(directory / "daily_returns.csv", parse_dates=["datetime"])
    summary = pd.read_csv(directory / "portfolio_summary.csv", dtype={"window": str})
    config = json.loads((directory / "config.resolved.json").read_text())
    joined = positions.merge(predictions[["datetime", "instrument", "realized_return_1d"]],
                             on=["datetime", "instrument"], how="left", validate="many_to_one")
    held = joined["target_weight"] > 0
    if not np.isfinite(joined.loc[held, "realized_return_1d"]).all():
        raise ValueError("held outcome is missing")
    close(joined.loc[held, "asset_return"], joined.loc[held, "realized_return_1d"], "held outcomes")
    close(positions["gross_contribution"], positions["target_weight"] * positions["asset_return"], "asset P&L")
    close(positions["turnover_contribution"], abs(positions["target_weight"] - positions["previous_weight"]), "trades")
    grouped = positions.groupby(["strategy", "datetime"])
    sums = grouped[["gross_contribution", "turnover_contribution", "target_weight"]].sum()
    indexed = daily.set_index(["strategy", "datetime"])
    sums = sums.reindex(indexed.index, fill_value=0.)
    close(sums["gross_contribution"], indexed["gross_return"], "daily gross return")
    close(sums["turnover_contribution"], indexed["turnover"], "daily turnover")
    close(sums["target_weight"], indexed["invested_weight"], "exposure")
    close(indexed["cash_weight"] + indexed["invested_weight"], 1., "cash plus holdings")

    # Verify carried capital, including explicit zero-weight exit rows.
    for strategy, days in daily.groupby("strategy"):
        prior = {}
        for row in days.sort_values("datetime").itertuples():
            try:
                frame = grouped.get_group((strategy, row.datetime)).set_index("instrument")
            except KeyError:
                if prior:
                    raise ValueError("positions disappeared from the ledger")
                continue
            if not set(prior).issubset(frame.index):
                raise ValueError("prior holdings disappeared without an exit row")
            close(frame["previous_weight"], [prior.get(name, 0.) for name in frame.index], "weight drift")
            prior = {name: weight for name, weight in
                     (frame["target_weight"] * (1 + frame["asset_return"]) / (1 + row.gross_return)).items()
                     if weight > 0}
        for cost in config["portfolio"]["one_way_cost_bps"]:
            close(days[f"net_return_{cost}bps"], days["gross_return"] - days["turnover"] * cost / 10000, "cost overlay")
        for item in summary[summary["strategy"] == strategy].itertuples():
            selected = days if item.window == "ALL" else days[days["datetime"].dt.year == int(item.window)]
            r = selected[f"net_return_{item.cost_bps}bps"].to_numpy()
            wealth = np.r_[1., np.cumprod(1 + r)]
            close(item.days, len(r), "day count")
            close(item.annualized_return, wealth[-1] ** (252 / len(r)) - 1, "annualized return")
            deviation = np.std(r, ddof=1) if len(r) > 1 else np.nan
            close(item.sharpe, np.mean(r) / deviation * np.sqrt(252) if deviation > 0 else np.nan, "Sharpe")
            close(item.maximum_drawdown, np.min(wealth / np.maximum.accumulate(wealth) - 1), "drawdown")
            close(item.total_turnover, selected["turnover"].sum(), "total turnover")

    expected_ic = pd.read_csv(directory / "prediction_daily_ic.csv", parse_dates=["datetime"])
    for row in expected_ic.itertuples():
        frame = predictions[predictions["datetime"] == row.datetime][["score", "label"]].dropna()
        close(row.ic, frame["score"].corr(frame["label"]), "IC")
        close(row.rank_ic, frame["score"].rank().corr(frame["label"].rank()), "Rank IC")
    return {"status": "pass", "run_id": manifest["run_id"], "artifact_hashes": len(artifacts),
            "prediction_rows": len(predictions), "position_rows": len(positions), "daily_rows": len(daily),
            "scope": "software arithmetic only; not independent market-data or execution validation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.run_directory), indent=2))
