from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import pandas as pd


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is not installed")
class MarketContextTests(unittest.TestCase):
    def test_context_is_same_date_only(self) -> None:
        from pitalpha.models.market_context import _build_market_context

        dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]))
        values = np.array([[1.0], [3.0], [10.0], [14.0]], dtype=np.float32)
        changed = values.copy()
        changed[2:] = 1_000.0
        first = _build_market_context(values, dates)
        second = _build_market_context(changed, dates)
        np.testing.assert_array_equal(first.table[0], second.table[0])
        np.testing.assert_allclose(first.table[0], [2.0, 1.0])
        np.testing.assert_array_equal(first.date_codes, [0, 0, 1, 1])

    def test_model_is_deterministic_and_ignores_test_labels(self) -> None:
        from pitalpha.models.market_context import fit_predict_market_context

        train_dates = pd.bdate_range("2024-01-01", periods=32)
        train_rows = []
        for date_index, date in enumerate(train_dates):
            market = np.sin(date_index / 5.0)
            for stock in range(4):
                x1 = (stock - 1.5) / 2.0 + 0.2 * market
                x2 = np.cos(date_index / 7.0 + stock)
                train_rows.append(
                    {"datetime": date, "x1": x1, "x2": x2, "label": 0.15 * x1 - 0.08 * x2}
                )
        train = pd.DataFrame(train_rows)
        test_rows = []
        for date_index, date in enumerate(pd.bdate_range("2024-03-01", periods=3)):
            for stock in range(4):
                test_rows.append(
                    {
                        "datetime": date,
                        "x1": (stock - 1.5) / 2.0,
                        "x2": np.cos(date_index + stock),
                        "label": float(stock),
                    }
                )
        test = pd.DataFrame(test_rows)
        changed = test.copy()
        changed["label"] = np.linspace(-1e9, 1e9, len(changed))
        params = {
            "hidden_dims": [8],
            "context_hidden_dim": 4,
            "use_market_context": True,
            "ranking_weight": 0.1,
            "dropout": 0.0,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "dates_per_batch": 4,
            "prediction_batch_size": 16,
            "max_epochs": 3,
            "patience": 2,
            "minimum_delta": 0.0,
            "validation_dates": 5,
            "seed": 17,
            "torch_num_threads": 1,
        }
        first, metadata = fit_predict_market_context(train, test, ["x1", "x2"], params)
        second, _ = fit_predict_market_context(train, changed, ["x1", "x2"], params)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(metadata["architecture"], "same-date-market-context-gated-mlp")
        self.assertTrue(metadata["use_market_context"])
        self.assertGreater(metadata["parameter_count"], 0)


if __name__ == "__main__":
    unittest.main()
