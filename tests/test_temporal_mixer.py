from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import pandas as pd


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is not installed")
class TemporalMixerTests(unittest.TestCase):
    @staticmethod
    def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
        train_rows: list[dict[str, object]] = []
        for date_index, date in enumerate(pd.bdate_range("2024-01-02", periods=30)):
            for stock in range(3):
                x1 = np.sin(date_index / 4.0) + stock * 0.2
                x2 = np.cos(date_index / 6.0 + stock)
                train_rows.append(
                    {
                        "datetime": date,
                        "instrument": f"S{stock}",
                        "x1": x1,
                        "x2": x2,
                        "label": 0.12 * x1 - 0.07 * x2,
                    }
                )
        test_rows: list[dict[str, object]] = []
        for date_index, date in enumerate(pd.bdate_range("2024-02-13", periods=4)):
            for stock in range(3):
                test_rows.append(
                    {
                        "datetime": date,
                        "instrument": f"S{stock}",
                        "x1": np.sin((date_index + 30) / 4.0) + stock * 0.2,
                        "x2": np.cos((date_index + 30) / 6.0 + stock),
                        "label": float(stock),
                    }
                )
        return pd.DataFrame(train_rows), pd.DataFrame(test_rows)

    @staticmethod
    def _params() -> dict[str, object]:
        return {
            "lookback_sessions": 8,
            "patch_size": 2,
            "hidden_dim": 8,
            "mixer_blocks": 1,
            "token_mlp_dim": 8,
            "channel_mlp_dim": 16,
            "use_last_session_residual": True,
            "dropout": 0.0,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "batch_size": 32,
            "prediction_batch_size": 32,
            "max_epochs": 2,
            "patience": 2,
            "minimum_delta": 0.0,
            "validation_dates": 4,
            "seed": 17,
            "torch_num_threads": 1,
        }

    def test_model_is_deterministic_and_ignores_test_labels(self) -> None:
        from pitalpha.models.temporal_mixer import fit_predict_temporal_mixer

        train, test = self._frames()
        first, metadata = fit_predict_temporal_mixer(
            train, test, ["x1", "x2"], self._params()
        )
        changed = test.copy()
        changed["label"] = np.linspace(-1e9, 1e9, len(changed))
        second, _ = fit_predict_temporal_mixer(
            train, changed, ["x1", "x2"], self._params()
        )

        np.testing.assert_array_equal(first, second)
        self.assertEqual(metadata["architecture"], "causal-temporal-mixer-v0")
        self.assertEqual(metadata["lookback_sessions"], 8)
        self.assertTrue(metadata["use_last_session_residual"])
        self.assertGreater(metadata["parameter_count"], 0)

    def test_future_test_features_cannot_change_first_date_scores(self) -> None:
        from pitalpha.models.temporal_mixer import fit_predict_temporal_mixer

        train, test = self._frames()
        first, _ = fit_predict_temporal_mixer(
            train, test, ["x1", "x2"], self._params()
        )
        changed = test.copy()
        first_date = changed["datetime"].min()
        changed.loc[changed["datetime"] > first_date, ["x1", "x2"]] = 1_000_000.0
        second, _ = fit_predict_temporal_mixer(
            train, changed, ["x1", "x2"], self._params()
        )
        first_date_rows = test["datetime"].eq(first_date).to_numpy()

        np.testing.assert_array_equal(first[first_date_rows], second[first_date_rows])


if __name__ == "__main__":
    unittest.main()
