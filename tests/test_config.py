from __future__ import annotations

import copy
import unittest
from pathlib import Path

from pitalpha.config import ConfigError, config_digest, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_demo_config_is_valid_and_digest_is_stable(self) -> None:
        config = load_config(ROOT / "configs" / "demo_synthetic.yaml")
        self.assertEqual(config["data"]["source"], "synthetic")
        self.assertEqual(config_digest(config), config_digest(copy.deepcopy(config)))
        self.assertEqual(len(config_digest(config)), 64)

    def test_csi300_config_is_valid(self) -> None:
        config = load_config(ROOT / "configs" / "csi300_ridge.yaml")
        self.assertEqual(config["data"]["snapshot_id"], "qlib-cn-2026-09-09-common-2025")

    def test_csi300_lightgbm_config_is_valid(self) -> None:
        config = load_config(ROOT / "configs" / "csi300_lightgbm.yaml")
        self.assertEqual(config["model"]["name"], "lightgbm")

    def test_csi300_mlp_config_is_valid(self) -> None:
        config = load_config(ROOT / "configs" / "csi300_mlp.yaml")
        self.assertEqual(config["model"]["name"], "mlp")

    def test_overlapping_windows_are_rejected(self) -> None:
        config = load_config(ROOT / "configs" / "demo_synthetic.yaml")
        config["splits"]["validation"] = ["2021-12-01", "2022-12-31"]
        with self.assertRaisesRegex(ConfigError, "ordered and disjoint"):
            validate_config(config)

    def test_short_embargo_is_rejected(self) -> None:
        config = load_config(ROOT / "configs" / "demo_synthetic.yaml")
        config["splits"]["embargo_trading_days"] = 4
        with self.assertRaisesRegex(ConfigError, "cover the label horizon"):
            validate_config(config)

    def test_retention_rank_must_cover_top_k(self) -> None:
        config = load_config(ROOT / "configs" / "demo_synthetic.yaml")
        config["portfolio"]["retention_rank"] = config["portfolio"]["top_k"] - 1
        with self.assertRaisesRegex(ConfigError, "at least as large as top_k"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
