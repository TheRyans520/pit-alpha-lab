from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pitalpha.artifacts import build_run_manifest, write_json_atomic
from pitalpha.config import config_digest, load_config


ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_manifest_separates_content_and_execution_identity(self) -> None:
        config_path = ROOT / "configs" / "demo_synthetic.yaml"
        config = load_config(config_path)
        created = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
        manifest = build_run_manifest(config, config_path, created_at=created, code_revision="abc123")
        self.assertEqual(manifest["content_id"], config_digest(config))
        self.assertEqual(manifest["created_at_utc"], "2026-09-28T12:00:00Z")
        self.assertIn(config_digest(config)[:12], manifest["run_id"])
        self.assertEqual(manifest["code_revision"], "abc123")
        self.assertEqual(manifest["status"], "planned")

    def test_atomic_json_write_round_trips(self) -> None:
        payload = {"schema_version": 1, "status": "planned"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "manifest.json"
            write_json_atomic(output, payload)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            self.assertFalse(list(output.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
