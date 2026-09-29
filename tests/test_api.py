from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


API_AVAILABLE = importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None


@unittest.skipUnless(API_AVAILABLE, "optional API dependencies are not installed")
class APITests(unittest.TestCase):
    def test_read_only_run_listing_summary_and_equity(self) -> None:
        import httpx

        from pitalpha.api import create_app

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "artifacts" / "demo" / "run-1"
            results = root / "case_studies" / "csi300_alpha" / "results"
            run.mkdir(parents=True)
            results.mkdir(parents=True)
            config = {
                "model": {"name": "ridge"},
                "portfolio": {"primary_cost_bps": 10},
            }
            metrics = {
                "data": {"rows": 2},
                "data_quality": {"status": "pass"},
                "prediction": [{"window": "ALL", "ic": 0.1}],
                "bootstrap_uncertainty": [],
                "signal_decay": [],
                "portfolio": [
                    {"strategy": "ridge_top_k", "window": "ALL", "cost_bps": 10, "sharpe": 1.0}
                ],
                "cost_break_even": [],
            }
            (run / "config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
            (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            pd.DataFrame(
                {
                    "datetime": ["2025-01-02", "2025-01-03"],
                    "strategy": ["ridge_top_k", "ridge_top_k"],
                    "net_return_10bps": [0.01, -0.005],
                    "turnover": [1.0, 0.0],
                }
            ).to_csv(run / "daily_returns.csv", index=False)
            artifacts = []
            for name in ["config.resolved.json", "metrics.json", "daily_returns.csv"]:
                path = run / name
                artifacts.append(
                    {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
                )
            manifest = {
                "status": "completed",
                "run_id": "run-1",
                "created_at_utc": "2025-01-01T00:00:00Z",
                "completed_at_utc": "2025-01-01T00:01:00Z",
                "content_id": "abc",
                "data": {"snapshot_id": "snapshot", "source": "synthetic"},
                "data_quality": {"status": "pass"},
                "artifacts": artifacts,
            }
            (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            pd.DataFrame([{"model": "ridge", "preferred": True}]).to_csv(
                results / "model_comparison_10bps.csv", index=False
            )
            (results / "market_context_multiseed_summary.json").write_text(
                json.dumps({"seed_count": 5, "decision": "reject_for_promotion"}), encoding="utf-8"
            )
            pd.DataFrame([{"model_seed": 17, "annualized_return_10bps": -0.05}]).to_csv(
                results / "market_context_multiseed_10bps.csv", index=False
            )
            (results / "market_context_ablation_summary.json").write_text(
                json.dumps({"decision": "retain_ridge"}), encoding="utf-8"
            )
            pd.DataFrame([{"variant": "ridge", "rank_ic": 0.02}]).to_csv(
                results / "market_context_ablation_10bps.csv", index=False
            )

            async def exercise() -> None:
                transport = httpx.ASGITransport(
                    app=create_app(repo_root=root, artifacts_root=root / "artifacts")
                )
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://testserver"
                ) as client:
                    self.assertEqual((await client.get("/api/health")).status_code, 200)
                    listing = (await client.get("/api/runs")).json()
                    self.assertEqual(listing["count"], 1)
                    models = (await client.get("/api/models")).json()
                    self.assertEqual(models["count"], 5)
                    self.assertFalse(models["external_code_loaded"])
                    self.assertIn("ridge", {row["name"] for row in models["models"]})
                    summary = (await client.get("/api/runs/run-1/summary")).json()
                    self.assertTrue(summary["artifacts_verified"])
                    equity = (
                        await client.get(
                            "/api/runs/run-1/equity",
                            params={"strategy": "ridge_top_k", "cost_bps": 10},
                        )
                    ).json()
                    self.assertAlmostEqual(equity["points"][-1]["wealth"], 1.01 * 0.995)
                    advanced = (await client.get("/api/case-study/market-context")).json()
                    self.assertEqual(advanced["multiseed"]["seed_count"], 5)
                    self.assertEqual(advanced["ablations"][0]["variant"], "ridge")
                    self.assertEqual(
                        (await client.get("/api/runs/..%2Fsecret/summary")).status_code, 404
                    )

            asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
