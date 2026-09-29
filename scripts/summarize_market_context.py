"""Aggregate completed CSI300 market-context seeds after verifying every artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitalpha.artifacts import sha256_file, write_frame_atomic, write_json_atomic, write_text_atomic
from pitalpha.environment import repository_root


EXPECTED_SEEDS = (7, 17, 29, 41, 53)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verified(directory: Path, manifest: dict) -> bool:
    return all(
        (directory / artifact["path"]).is_file()
        and sha256_file(directory / artifact["path"]) == artifact["sha256"]
        for artifact in manifest["artifacts"]
    )


def collect(root: Path) -> pd.DataFrame:
    latest: dict[int, tuple[str, dict]] = {}
    for manifest_path in root.glob(
        "artifacts/csi300-pit-market-context-seed*/*/run_manifest.json"
    ):
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "completed":
            continue
        metrics = _read_json(manifest_path.parent / "metrics.json")
        model_runs = metrics.get("model_runs", [])
        if not model_runs:
            continue
        seeds = {int(item["seed"]) for item in model_runs}
        if len(seeds) != 1:
            raise RuntimeError(f"fold seeds disagree in {manifest_path.parent}")
        seed = seeds.pop()
        completed = str(manifest["completed_at_utc"])
        if seed not in latest or completed > latest[seed][0]:
            latest[seed] = (completed, {"manifest": manifest, "metrics": metrics, "directory": manifest_path.parent})

    if tuple(sorted(latest)) != EXPECTED_SEEDS:
        raise RuntimeError(f"expected seeds {EXPECTED_SEEDS}, found {tuple(sorted(latest))}")

    rows = []
    for seed in EXPECTED_SEEDS:
        payload = latest[seed][1]
        manifest = payload["manifest"]
        metrics = payload["metrics"]
        directory = payload["directory"]
        if manifest.get("data_quality", {}).get("status") != "pass":
            raise RuntimeError(f"data quality did not pass for seed {seed}")
        print(f"verifying seed={seed} artifacts={len(manifest['artifacts'])}", flush=True)
        if not _verified(directory, manifest):
            raise RuntimeError(f"artifact hash verification failed for seed {seed}")
        print(f"verified seed={seed}", flush=True)
        prediction = next(item for item in metrics["prediction"] if item["window"] == "ALL")
        confidence = {item["metric"]: item for item in metrics["bootstrap_uncertainty"]}
        portfolio = {
            item["strategy"]: item
            for item in metrics["portfolio"]
            if item["window"] == "ALL" and int(item["cost_bps"]) == 10
        }
        plain = portfolio["market_context_top_k"]
        buffered = portfolio["market_context_top_k_buffered"]
        break_even = {
            item["strategy"]: item["breakeven_one_way_cost_bps"]
            for item in metrics["cost_break_even"]
        }
        model_runs = metrics["model_runs"]
        rows.append(
            {
                "model_seed": seed,
                "run_id": manifest["run_id"],
                "content_id": manifest["content_id"],
                "artifacts_verified": True,
                "data_quality_status": "pass",
                "parameter_count": int(model_runs[0]["parameter_count"]),
                "mean_selected_epoch": float(np.mean([item["best_epoch"] for item in model_runs])),
                "ic": prediction["ic"],
                "ic_ci_low": confidence["ic"]["ci_low"],
                "ic_ci_high": confidence["ic"]["ci_high"],
                "rank_ic": prediction["rank_ic"],
                "rank_ic_ci_low": confidence["rank_ic"]["ci_low"],
                "rank_ic_ci_high": confidence["rank_ic"]["ci_high"],
                "annualized_return_10bps": plain["annualized_return"],
                "sharpe_10bps": plain["sharpe"],
                "maximum_drawdown_10bps": plain["maximum_drawdown"],
                "mean_turnover": plain["mean_turnover"],
                "breakeven_cost_bps": break_even["market_context_top_k"],
                "buffered_annualized_return_10bps": buffered["annualized_return"],
                "buffered_sharpe_10bps": buffered["sharpe"],
                "buffered_maximum_drawdown_10bps": buffered["maximum_drawdown"],
                "buffered_mean_turnover": buffered["mean_turnover"],
                "buffered_breakeven_cost_bps": break_even["market_context_top_k_buffered"],
                "run_directory": directory.relative_to(root).as_posix(),
            }
        )
    return pd.DataFrame(rows)


def _stat(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _latest_completed(root: Path, artifact_group: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for path in (root / "artifacts" / artifact_group).glob("*/run_manifest.json"):
        manifest = _read_json(path)
        if manifest.get("status") == "completed":
            candidates.append((str(manifest["completed_at_utc"]), path.parent))
    if not candidates:
        raise RuntimeError(f"no completed run found for {artifact_group}")
    return max(candidates, key=lambda item: item[0])[1]


def collect_ablations(root: Path) -> pd.DataFrame:
    specifications = [
        ("ridge", "csi300-pit-ridge", "ridge_top_k", False, 0.0),
        ("mlp", "csi300-pit-mlp-seed17", "mlp_top_k", False, 0.0),
        (
            "context_without_ranking",
            "csi300-pit-market-context-no-rank-seed17",
            "market_context_top_k",
            True,
            0.0,
        ),
        (
            "ranking_without_context",
            "csi300-pit-market-context-no-context-seed17",
            "market_context_top_k",
            False,
            0.1,
        ),
        (
            "context_plus_ranking",
            "csi300-pit-market-context-seed17",
            "market_context_top_k",
            True,
            0.1,
        ),
    ]
    rows = []
    for variant, artifact_group, strategy, uses_context, ranking_weight in specifications:
        directory = _latest_completed(root, artifact_group)
        manifest = _read_json(directory / "run_manifest.json")
        metrics = _read_json(directory / "metrics.json")
        if manifest.get("data_quality", {}).get("status") != "pass" or not _verified(directory, manifest):
            raise RuntimeError(f"audit failed for ablation variant {variant}")
        prediction = next(item for item in metrics["prediction"] if item["window"] == "ALL")
        portfolios = {
            item["strategy"]: item
            for item in metrics["portfolio"]
            if item["window"] == "ALL" and int(item["cost_bps"]) == 10
        }
        plain = portfolios[strategy]
        buffered = portfolios[f"{strategy}_buffered"]
        model_run = metrics["model_runs"][0]
        rows.append(
            {
                "variant": variant,
                "market_context": uses_context,
                "ranking_weight": ranking_weight,
                "model_seed": model_run.get("seed"),
                "parameter_count": model_run.get("parameter_count"),
                "ic": prediction["ic"],
                "rank_ic": prediction["rank_ic"],
                "annualized_return_10bps": plain["annualized_return"],
                "sharpe_10bps": plain["sharpe"],
                "maximum_drawdown_10bps": plain["maximum_drawdown"],
                "mean_turnover": plain["mean_turnover"],
                "buffered_annualized_return_10bps": buffered["annualized_return"],
                "buffered_sharpe_10bps": buffered["sharpe"],
                "buffered_mean_turnover": buffered["mean_turnover"],
                "run_id": manifest["run_id"],
                "content_id": manifest["content_id"],
                "artifacts_verified": True,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    root = repository_root()
    output = root / "case_studies" / "csi300_alpha" / "results"
    frame = collect(root)
    csv_path = output / "market_context_multiseed_10bps.csv"
    write_frame_atomic(frame, csv_path)
    summary = {
        "schema_version": 1,
        "protocol": "same frozen CSI300 snapshot and folds; model randomness only",
        "cost_bps": 10,
        "seeds": frame["model_seed"].astype(int).tolist(),
        "seed_count": int(len(frame)),
        "all_artifacts_verified": bool(frame["artifacts_verified"].all()),
        "all_data_quality_passed": bool(frame["data_quality_status"].eq("pass").all()),
        "metrics": {
            column: _stat(frame[column])
            for column in [
                "ic",
                "rank_ic",
                "annualized_return_10bps",
                "sharpe_10bps",
                "maximum_drawdown_10bps",
                "buffered_annualized_return_10bps",
                "buffered_sharpe_10bps",
            ]
        },
        "fractions": {
            "positive_rank_ic": float((frame["rank_ic"] > 0).mean()),
            "positive_net_annualized_return": float((frame["annualized_return_10bps"] > 0).mean()),
            "positive_buffered_net_annualized_return": float(
                (frame["buffered_annualized_return_10bps"] > 0).mean()
            ),
        },
        "decision": "reject_for_promotion",
        "reason": "Net performance is negative for every unbuffered seed and unstable after buffering.",
        "source_csv": csv_path.relative_to(root).as_posix(),
    }
    json_path = write_json_atomic(output / "market_context_multiseed_summary.json", summary)
    metrics = summary["metrics"]
    report = f"""# Market-context model: five-seed result

The model is **not promoted**. All five runs use the frozen CSI300 snapshot, identical walk-forward
folds and identical portfolio/cost rules; only neural initialization and training order change.

| Metric | Mean | Seed range |
|---|---:|---:|
| Rank IC | {metrics['rank_ic']['mean']:.4f} | {metrics['rank_ic']['minimum']:.4f} to {metrics['rank_ic']['maximum']:.4f} |
| Annualized return, 10 bps | {metrics['annualized_return_10bps']['mean']:.2%} | {metrics['annualized_return_10bps']['minimum']:.2%} to {metrics['annualized_return_10bps']['maximum']:.2%} |
| Sharpe, 10 bps | {metrics['sharpe_10bps']['mean']:.2f} | {metrics['sharpe_10bps']['minimum']:.2f} to {metrics['sharpe_10bps']['maximum']:.2f} |
| Buffered annualized return, 10 bps | {metrics['buffered_annualized_return_10bps']['mean']:.2%} | {metrics['buffered_annualized_return_10bps']['minimum']:.2%} to {metrics['buffered_annualized_return_10bps']['maximum']:.2%} |

Rank IC is positive in all seeds, but unbuffered net annualized return is negative in all seeds.
Only {summary['fractions']['positive_buffered_net_annualized_return']:.0%} of buffered seeds are positive,
and the positive case is economically small. This is evidence that the current same-date context and
ranking objective do not overcome turnover and temporal-regime limitations. The result remains
exploratory because the 2021-2025 test period had already been observed before this architecture.

Every included run passed data quality and artifact checksum verification. See
`results/market_context_multiseed_10bps.csv` and `results/market_context_multiseed_summary.json`.
"""
    report_path = write_text_atomic(
        report, root / "case_studies" / "csi300_alpha" / "market_context_multiseed.md"
    )

    ablations = collect_ablations(root)
    ablation_csv = write_frame_atomic(
        ablations, output / "market_context_ablation_10bps.csv"
    )
    indexed = ablations.set_index("variant")
    ablation_summary = {
        "schema_version": 1,
        "cost_bps": 10,
        "seed": 17,
        "all_artifacts_verified": bool(ablations["artifacts_verified"].all()),
        "observations": {
            "ranking_effect_on_rank_ic_without_context": float(
                indexed.loc["ranking_without_context", "rank_ic"]
                - indexed.loc["mlp", "rank_ic"]
            ),
            "context_effect_on_rank_ic_with_ranking": float(
                indexed.loc["context_plus_ranking", "rank_ic"]
                - indexed.loc["ranking_without_context", "rank_ic"]
            ),
            "ridge_minus_best_neural_net_annualized_return": float(
                indexed.loc["ridge", "annualized_return_10bps"]
                - ablations.loc[
                    ablations["variant"] != "ridge", "annualized_return_10bps"
                ].max()
            ),
        },
        "decision": "retain_ridge_reject_current_context_architecture",
        "caveat": "Exploratory comparison after the test period had already been observed.",
        "source_csv": ablation_csv.relative_to(root).as_posix(),
    }
    ablation_json = write_json_atomic(
        output / "market_context_ablation_summary.json", ablation_summary
    )
    ablation_report = f"""# Market-context ablation

This seed-17 comparison is explanatory, not a new confirmatory test. Every row uses the same frozen
CSI300 data, walk-forward folds, Top-30 portfolio and 10 bps one-way cost.

| Variant | Rank IC | Annualized return | Sharpe | Buffered return |
|---|---:|---:|---:|---:|
| Ridge | {indexed.loc['ridge', 'rank_ic']:.4f} | {indexed.loc['ridge', 'annualized_return_10bps']:.2%} | {indexed.loc['ridge', 'sharpe_10bps']:.2f} | {indexed.loc['ridge', 'buffered_annualized_return_10bps']:.2%} |
| MLP | {indexed.loc['mlp', 'rank_ic']:.4f} | {indexed.loc['mlp', 'annualized_return_10bps']:.2%} | {indexed.loc['mlp', 'sharpe_10bps']:.2f} | {indexed.loc['mlp', 'buffered_annualized_return_10bps']:.2%} |
| Context, no ranking loss | {indexed.loc['context_without_ranking', 'rank_ic']:.4f} | {indexed.loc['context_without_ranking', 'annualized_return_10bps']:.2%} | {indexed.loc['context_without_ranking', 'sharpe_10bps']:.2f} | {indexed.loc['context_without_ranking', 'buffered_annualized_return_10bps']:.2%} |
| Ranking loss, no context | {indexed.loc['ranking_without_context', 'rank_ic']:.4f} | {indexed.loc['ranking_without_context', 'annualized_return_10bps']:.2%} | {indexed.loc['ranking_without_context', 'sharpe_10bps']:.2f} | {indexed.loc['ranking_without_context', 'buffered_annualized_return_10bps']:.2%} |
| Context + ranking loss | {indexed.loc['context_plus_ranking', 'rank_ic']:.4f} | {indexed.loc['context_plus_ranking', 'annualized_return_10bps']:.2%} | {indexed.loc['context_plus_ranking', 'sharpe_10bps']:.2f} | {indexed.loc['context_plus_ranking', 'buffered_annualized_return_10bps']:.2%} |

The within-date ranking objective raises Rank IC relative to the plain MLP, but this does not become
positive net performance. Adding the current same-date context to that objective lowers Rank IC in
seed 17. Ridge remains ahead of the best neural variant by
{ablation_summary['observations']['ridge_minus_best_neural_net_annualized_return']:.2%} annualized.
The next credible neural step requires true temporal sequences and a fresh holdout, not more tuning
against 2021-2025.
"""
    ablation_report_path = write_text_atomic(
        ablation_report,
        root / "case_studies" / "csi300_alpha" / "market_context_ablation.md",
    )
    print(
        json.dumps(
            {
                "multiseed_csv": str(csv_path),
                "multiseed_json": str(json_path),
                "multiseed_report": str(report_path),
                "ablation_csv": str(ablation_csv),
                "ablation_json": str(ablation_json),
                "ablation_report": str(ablation_report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
