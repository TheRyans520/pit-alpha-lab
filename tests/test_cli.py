from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from pitalpha.cli import main


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_models_list_command(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "list"])
        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["entry_point_group"], "pitalpha.models")
        self.assertIn("ridge", {row["name"] for row in payload["models"]})

    def test_config_validate_command(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["config", "validate", str(ROOT / "configs" / "demo_synthetic.yaml")])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output.getvalue())["valid"])

    def test_manifest_create_command(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output):
            destination = Path(directory) / "manifest.json"
            code = main(
                [
                    "manifest",
                    "create",
                    str(ROOT / "configs" / "demo_synthetic.yaml"),
                    "--output",
                    str(destination),
                    "--code-revision",
                    "test-revision",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(destination.is_file())
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8"))["code_revision"], "test-revision")

    def test_bad_config_returns_two(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            code = main(["config", "validate", str(ROOT / "does-not-exist.yaml")])
        self.assertEqual(code, 2)
        self.assertIn("does not exist", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
