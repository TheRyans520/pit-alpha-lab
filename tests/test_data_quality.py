from __future__ import annotations

import unittest

import pandas as pd

from pitalpha.data import DataQualityError, audit_panel_quality, require_quality_pass


class DataQualityTests(unittest.TestCase):
    def config(self) -> dict:
        return {
            "data_quality": {
                "minimum_rows": 2,
                "minimum_daily_members": 2,
                "maximum_daily_members": 2,
                "minimum_feature_coverage": 0.5,
                "minimum_label_coverage": 0.5,
                "minimum_realized_return_coverage": 0.5,
                "maximum_absolute_realized_return": 0.5,
                "minimum_ohlc_coverage": 0.5,
            }
        }

    def panel(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                "instrument": ["A", "B"],
                "feature": [1.0, None],
                "label": [0.1, -0.1],
                "realized_return_1d": [0.01, -0.01],
            }
        )

    def test_valid_minimal_panel_passes_with_raw_checks_skipped(self) -> None:
        checks, summary = audit_panel_quality(self.panel(), ["feature"], self.config())
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(checks.loc[checks["check"] == "raw_ohlcv_semantics", "status"].item(), "skipped")
        require_quality_pass(summary)

    def test_duplicate_key_and_extreme_return_fail_gate(self) -> None:
        panel = self.panel()
        panel.loc[1, ["datetime", "instrument"]] = panel.loc[0, ["datetime", "instrument"]]
        panel.loc[1, "realized_return_1d"] = 0.75
        _, summary = audit_panel_quality(panel, ["feature"], self.config())
        self.assertEqual(summary["status"], "fail")
        with self.assertRaises(DataQualityError):
            require_quality_pass(summary)


if __name__ == "__main__":
    unittest.main()
