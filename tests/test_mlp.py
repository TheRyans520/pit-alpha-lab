from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import pandas as pd


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is not installed")
class MLPTests(unittest.TestCase):
    def test_mlp_is_deterministic_and_ignores_test_labels(self) -> None:
        from pitalpha.models.mlp import fit_predict_mlp

        dates = pd.bdate_range("2024-01-01", periods=40)
        x1 = np.linspace(-1.0, 1.0, len(dates), dtype=np.float32)
        train = pd.DataFrame(
            {
                "datetime": dates,
                "x1": x1,
                "x2": np.sin(x1),
                "label": 0.2 * x1 - 0.1 * np.sin(x1),
            }
        )
        test = pd.DataFrame(
            {
                "datetime": pd.bdate_range("2024-03-01", periods=3),
                "x1": [-0.5, 0.0, 0.5],
                "x2": [-0.4, 0.0, 0.4],
                "label": [1.0, 2.0, 3.0],
            }
        )
        changed = test.copy()
        changed["label"] = [-1e9, 0.0, 1e9]
        params = {
            "hidden_dims": [8],
            "dropout": 0.0,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "batch_size": 16,
            "max_epochs": 3,
            "patience": 2,
            "minimum_delta": 0.0,
            "validation_dates": 5,
            "seed": 17,
            "torch_num_threads": 1,
        }
        first, metadata = fit_predict_mlp(train, test, ["x1", "x2"], params)
        second, _ = fit_predict_mlp(train, changed, ["x1", "x2"], params)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(metadata["device"], "cpu")
        self.assertTrue(metadata["deterministic_algorithms"])
        self.assertGreaterEqual(metadata["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
