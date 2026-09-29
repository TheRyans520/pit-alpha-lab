from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pitalpha.data import SnapshotAuditError, load_audited_qlib_panel


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(name: str, checksum: str) -> str:
    value = hashlib.sha256()
    value.update(name.encode("utf-8"))
    value.update(bytes.fromhex(checksum))
    return value.hexdigest()


class AuditedQlibAdapterTests(unittest.TestCase):
    def fixture(self, directory: str) -> tuple[dict, Path, Path]:
        root = Path(directory)
        audit = root / "audits"
        features_dir = audit / "test-tag_features"
        raw_dir = audit / "test-tag_raw"
        manifests = root / "repo" / "manifests"
        features_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)
        manifests.mkdir(parents=True)
        name = "year=2020.parquet"
        keys = {
            "instrument": ["A", "B", "A", "B"],
            "datetime": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]),
        }
        pd.DataFrame({**keys, "F1": [1.0, 2.0, 3.0, 4.0], "F2": [4.0, 3.0, 2.0, 1.0]}).to_parquet(
            features_dir / name, index=False
        )
        pd.DataFrame(
            {
                **keys,
                "open": [10.0, 20.0, 10.1, 19.8],
                "high": [10.2, 20.2, 10.3, 20.0],
                "low": [9.9, 19.8, 10.0, 19.6],
                "close": [10.1, 20.0, 10.2, 19.9],
                "volume": [100.0, 200.0, 110.0, 190.0],
                "factor": [1.0, 1.0, 1.0, 1.0],
                "adjclose": [10.1, 20.0, 10.2, 19.9],
                "vwap": [10.05, 20.0, 10.15, 19.8],
                "label_5d": [-0.2, 0.2, -0.1, 0.1],
                "return_1d": [0.01, -0.01, 0.02, -0.02],
            }
        ).to_parquet(raw_dir / name, index=False)
        feature_sha = sha256(features_dir / name)
        raw_sha = sha256(raw_dir / name)
        metadata = {
            "tag": "test-tag",
            "archive_sha256": "archive",
            "raw_partition_digest": digest(name, raw_sha),
            "features_partition_digest": digest(name, feature_sha),
            "raw_partitions": {name: raw_sha},
            "feature_partitions": {name: feature_sha},
            "feature_names": ["F1", "F2"],
        }
        (audit / "test-tag_extraction_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        manifest = {
            "snapshot_id": "test-snapshot",
            "release_tag": "test-tag",
            "archive_sha256": "archive",
            "raw_partition_digest": metadata["raw_partition_digest"],
            "feature_partition_digest": metadata["features_partition_digest"],
        }
        (manifests / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
        config = {
            "data": {
                "audit_root_env": "TEST_AUDIT_ROOT",
                "release_tag": "test-tag",
                "snapshot_id": "test-snapshot",
                "snapshot_manifest": "manifests/snapshot.json",
                "start": "2020-01-01",
                "end": "2020-12-31",
            },
            "label": {"cross_sectional_winsorization": [0.0, 1.0]},
        }
        return config, root / "repo", audit

    def test_verified_partitions_load_into_canonical_panel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, repository, audit = self.fixture(directory)
            panel, features, provenance = load_audited_qlib_panel(
                config, repository_root=repository, environment={"TEST_AUDIT_ROOT": str(audit)}
            )
            self.assertEqual(features, ("F1", "F2"))
            self.assertEqual(len(panel), 4)
            self.assertIn("forward_return_5d", panel)
            self.assertEqual(provenance["verified_partition_files"], 2)

    def test_modified_partition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, repository, audit = self.fixture(directory)
            path = audit / "test-tag_raw" / "year=2020.parquet"
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(SnapshotAuditError, "checksum mismatch"):
                load_audited_qlib_panel(
                    config, repository_root=repository, environment={"TEST_AUDIT_ROOT": str(audit)}
                )


if __name__ == "__main__":
    unittest.main()
