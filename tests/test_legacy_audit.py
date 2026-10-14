import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pitalpha.artifacts import sha256_file
from pitalpha.legacy_audit import audit_saved_run


class LegacyAuditTests(unittest.TestCase):
    def fixture(self, root):
        dates = pd.bdate_range("2025-01-06", periods=3)
        pd.DataFrame({
            "datetime": np.repeat(dates, 2), "instrument": ["A", "B"] * 3,
            "score": [2., 1.] * 3, "realized_return_1d": [np.nan, .02, 0., 0., 0., 0.],
        }).to_parquet(root / "predictions.parquet", index=False)
        pd.DataFrame({"datetime": [dates[0]], "instrument": ["B"], "target_weight": [1.]}).to_parquet(root / "holdings.parquet", index=False)
        pd.DataFrame({"datetime": [dates[0]], "strategy": ["ridge_top_k"], "net_return_0bps": [.02]}).to_csv(root / "daily_returns.csv", index=False)
        pd.DataFrame({"strategy": ["ridge_top_k"], "window": ["ALL"], "cost_bps": [0], "days": [1],
                      "annualized_return": [1.02 ** 252 - 1], "sharpe": [np.nan], "maximum_drawdown": [0.]}).to_csv(root / "portfolio_summary.csv", index=False)
        (root / "config.resolved.json").write_text(json.dumps({"data": {"source": "qlib"}, "model": {"name": "ridge"},
                "portfolio": {"rebalance": "first_available_trading_day_each_week", "top_k": 1, "primary_cost_bps": 0}}))
        entries = [{"path": path.name, "sha256": sha256_file(path)} for path in root.iterdir()]
        (root / "run_manifest.json").write_text(json.dumps({"run_id": "test-legacy", "status": "completed", "artifacts": entries}))

    def test_consistent_legacy_returns_do_not_certify_biased_selections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            result = audit_saved_run(root)
            self.assertEqual(result["artifact_hashes_verified"], 5)
            self.assertEqual(result["weekly_selection_differences"][0]["excluded_high_score_names"], ["A"])
            self.assertEqual(result["weekly_selection_differences"][0]["substituted_names"], ["B"])
            self.assertEqual(result["strict_replay"]["status"], "blocked_by_valuation_policy")
            self.assertEqual(result["corrected_market_performance"], "not_established")

    def test_modified_artifact_and_invalid_manifest_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            manifest_path = root / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            original = (root / "daily_returns.csv").read_text()
            (root / "daily_returns.csv").write_text(original + "\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                audit_saved_run(root)
            (root / "daily_returns.csv").write_text(original)
            manifest["artifacts"].insert(0, {"path": "../outside", "sha256": "0" * 64})
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "direct child"):
                audit_saved_run(root)

    def test_refreshed_checksum_does_not_hide_bad_summary_arithmetic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            path = root / "portfolio_summary.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "annualized_return"] = .99
            frame.to_csv(path, index=False)
            manifest = json.loads((root / "run_manifest.json").read_text())
            for entry in manifest["artifacts"]:
                if entry["path"] == path.name:
                    entry["sha256"] = sha256_file(path)
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "summary arithmetic"):
                audit_saved_run(root)
