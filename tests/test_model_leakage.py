from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pitalpha.models import fit_predict_lightgbm, fit_predict_ridge


class ModelLeakageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = ["x1", "x2"]
        self.train = pd.DataFrame(
            {
                "x1": [0.0, 1.0, 2.0, 3.0],
                "x2": [1.0, 1.5, 2.0, 2.5],
                "label": [0.0, 0.1, 0.2, 0.3],
            }
        )

    def test_test_distribution_cannot_refit_preprocessing(self) -> None:
        ordinary = pd.DataFrame({"x1": [1.5], "x2": [1.75], "label": [999.0]})
        extreme = pd.concat(
            [ordinary, pd.DataFrame({"x1": [1e12], "x2": [-1e12], "label": [-999.0]})],
            ignore_index=True,
        )
        first, _ = fit_predict_ridge(self.train, ordinary, self.features, {"alpha": 1.0})
        expanded, _ = fit_predict_ridge(self.train, extreme, self.features, {"alpha": 1.0})
        self.assertAlmostEqual(float(first[0]), float(expanded[0]), places=12)

    def test_test_labels_do_not_affect_predictions(self) -> None:
        test = pd.DataFrame({"x1": [1.5, 2.5], "x2": [1.75, 2.25], "label": [1.0, -1.0]})
        changed = test.copy()
        changed["label"] = [1e9, -1e9]
        first, _ = fit_predict_ridge(self.train, test, self.features, {"alpha": 1.0})
        second, _ = fit_predict_ridge(self.train, changed, self.features, {"alpha": 1.0})
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)

    def test_lightgbm_test_labels_do_not_affect_predictions(self) -> None:
        test = pd.DataFrame({"x1": [1.5, 2.5], "x2": [1.75, 2.25], "label": [1.0, -1.0]})
        changed = test.copy()
        changed["label"] = [1e9, -1e9]
        params = {"n_estimators": 8, "num_leaves": 3, "random_state": 17, "verbosity": -1}
        first, _ = fit_predict_lightgbm(self.train, test, self.features, params)
        second, _ = fit_predict_lightgbm(self.train, changed, self.features, params)
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
