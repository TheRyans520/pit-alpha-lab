from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pitalpha.models import (
    ModelAdapter,
    ModelDescriptor,
    audit_test_label_independence,
    list_model_descriptors,
    resolve_model,
    validate_model_response,
)


class ModelContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame(
            {
                "datetime": pd.date_range("2020-01-01", periods=8),
                "label": np.linspace(-0.04, 0.04, 8),
                "feature": np.linspace(-1.0, 1.0, 8),
            }
        )
        self.test = pd.DataFrame(
            {
                "datetime": pd.date_range("2021-01-01", periods=4),
                "label": [0.02, -0.01, 0.03, -0.02],
                "feature": [-0.5, 0.0, 0.5, 1.0],
            }
        )

    def test_registry_lists_models_without_loading_optional_torch_models(self) -> None:
        descriptors = list_model_descriptors()
        names = {str(row["name"]) for row in descriptors}
        self.assertEqual(
            names,
            {"ridge", "lightgbm", "mlp", "market_context", "temporal_mixer"},
        )

    def test_ridge_satisfies_test_label_independence_contract(self) -> None:
        report = audit_test_label_independence(
            resolve_model("ridge"),
            self.train,
            self.test,
            ["feature"],
            {"alpha": 1.0, "fit_intercept": True},
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.maximum_repeat_difference, 0.0)
        self.assertEqual(report.maximum_label_difference, 0.0)

    def test_contract_detects_a_test_label_leak(self) -> None:
        def leaky(train, test, feature_columns, params):
            del train, feature_columns, params
            return test["label"].to_numpy(dtype=float), {"kind": "deliberately_leaky_test"}

        adapter = ModelAdapter(
            descriptor=ModelDescriptor(
                name="leaky",
                provider="external",
                input_kind="tabular",
                description="Test-only invalid adapter.",
            ),
            fit_predict=leaky,
        )
        report = audit_test_label_independence(
            adapter,
            self.train,
            self.test,
            ["feature"],
            {},
        )
        self.assertFalse(report.passed)
        self.assertGreater(report.maximum_label_difference, 100.0)

    def test_contract_detects_input_mutation(self) -> None:
        def mutating(train, test, feature_columns, params):
            del train, feature_columns, params
            scores = test["feature"].to_numpy(dtype=float, copy=True)
            test.loc[:, "feature"] = 0.0
            return scores, {"kind": "deliberately_mutating_test"}

        adapter = ModelAdapter(
            descriptor=ModelDescriptor(
                name="mutating",
                provider="external",
                input_kind="tabular",
                description="Test-only invalid adapter.",
            ),
            fit_predict=mutating,
        )
        report = audit_test_label_independence(
            adapter,
            self.train,
            self.test,
            ["feature"],
            {},
        )
        self.assertFalse(report.passed)
        self.assertFalse(report.inputs_unchanged)

    def test_invalid_scores_and_metadata_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_model_response(np.array([[1.0]]), {}, expected_rows=1)
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_model_response(np.array([np.nan]), {}, expected_rows=1)
        with self.assertRaisesRegex(ValueError, "JSON-serializable"):
            validate_model_response(np.array([1.0]), {"bad": {1, 2}}, expected_rows=1)

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported model"):
            resolve_model("not-installed")


if __name__ == "__main__":
    unittest.main()
