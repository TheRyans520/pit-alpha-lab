from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pitalpha.pipeline import _apply_model_overrides, _apply_model_seed_override, run_experiment


ROOT = Path(__file__).resolve().parents[1]


class PipelineIntegrationTests(unittest.TestCase):
    def test_model_seed_override_is_explicit_and_does_not_change_data_seed(self) -> None:
        from pitalpha.config import load_config

        original = load_config(ROOT / "configs" / "csi300_market_context_seed17.yaml")
        resolved = _apply_model_seed_override(original, 29)
        self.assertEqual(original["experiment"]["seed"], resolved["experiment"]["seed"])
        self.assertEqual(resolved["experiment"]["model_seed_override"], 29)
        self.assertEqual(resolved["model"]["params"]["seed"], 29)
        self.assertTrue(resolved["experiment"]["name"].endswith("-seed29"))
        self.assertTrue(resolved["artifacts"]["root"].endswith("-seed29"))

    def test_market_context_ablation_overrides_are_named_and_recorded(self) -> None:
        from pitalpha.config import load_config

        original = load_config(ROOT / "configs" / "csi300_market_context_seed17.yaml")
        resolved, overrides = _apply_model_overrides(
            original, disable_market_context=True, ranking_weight=0.0
        )
        self.assertFalse(resolved["model"]["params"]["use_market_context"])
        self.assertEqual(resolved["model"]["params"]["ranking_weight"], 0.0)
        self.assertEqual(resolved["experiment"]["ablation_overrides"], overrides)
        self.assertTrue(resolved["experiment"]["name"].endswith("-no-context-no-rank-seed17"))

    def test_synthetic_pipeline_writes_complete_traceable_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(ROOT / "configs" / "demo_synthetic.yaml", output_root=directory)
            run_directory = Path(result["run_directory"])
            manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(manifest["folds"]), 3)
            self.assertTrue((run_directory / "report.md").is_file())
            self.assertTrue((run_directory / "signal_quantiles.csv").is_file())
            self.assertTrue((run_directory / "signal_decay.csv").is_file())
            self.assertTrue((run_directory / "bootstrap_uncertainty.csv").is_file())
            self.assertTrue((run_directory / "cost_break_even.csv").is_file())
            self.assertTrue((run_directory / "data_quality_checks.csv").is_file())
            self.assertEqual(manifest["data_quality"]["status"], "pass")
            predictions = pd.read_parquet(run_directory / "predictions.parquet")
            self.assertEqual(sorted(predictions["fold"].unique()), ["2023", "2024", "2025"])
            daily_returns = pd.read_csv(run_directory / "daily_returns.csv")
            self.assertEqual(float(daily_returns["missing_return_weight"].max()), 0.0)
            self.assertEqual(
                set(daily_returns["strategy"]),
                {"ridge_top_k", "ridge_top_k_buffered", "eligible_equal_weight"},
            )
            holdings = pd.read_parquet(run_directory / "holdings.parquet")
            self.assertEqual(set(holdings["strategy"]), set(daily_returns["strategy"]))
            for artifact in manifest["artifacts"]:
                path = run_directory / artifact["path"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
