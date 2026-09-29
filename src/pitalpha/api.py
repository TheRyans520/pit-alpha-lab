"""Read-only FastAPI surface for research artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from pitalpha import __version__
from pitalpha.artifacts import sha256_file
from pitalpha.environment import repository_root
from pitalpha.models import ENTRY_POINT_GROUP, list_model_descriptors


class RunStore:
    """Resolve completed runs by manifest ID without accepting filesystem paths."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @staticmethod
    def _json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"invalid artifact: {path.name}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=500, detail=f"artifact is not a JSON object: {path.name}")
        return payload

    def _index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        if not self.root.is_dir():
            return index
        for manifest_path in self.root.glob("*/*/run_manifest.json"):
            manifest = self._json(manifest_path)
            run_id = manifest.get("run_id")
            if isinstance(run_id, str) and manifest.get("status") == "completed":
                index[run_id] = manifest_path.parent.resolve()
        return index

    def directory(self, run_id: str) -> Path:
        directory = self._index().get(run_id)
        if directory is None:
            raise HTTPException(status_code=404, detail="completed run not found")
        return directory

    def list_runs(self) -> list[dict[str, Any]]:
        rows = []
        for run_id, directory in self._index().items():
            manifest = self._json(directory / "run_manifest.json")
            config = self._json(directory / "config.resolved.json")
            rows.append(
                {
                    "run_id": run_id,
                    "created_at_utc": manifest["created_at_utc"],
                    "completed_at_utc": manifest.get("completed_at_utc"),
                    "snapshot_id": manifest["data"]["snapshot_id"],
                    "data_source": manifest["data"]["source"],
                    "model": config["model"]["name"],
                    "primary_cost_bps": config["portfolio"]["primary_cost_bps"],
                    "quality_status": manifest.get("data_quality", {}).get("status", "unknown"),
                    "content_id": manifest["content_id"],
                }
            )
        return sorted(rows, key=lambda row: (row["created_at_utc"], row["run_id"]), reverse=True)

    def verify(self, directory: Path, manifest: dict[str, Any]) -> bool:
        for artifact in manifest.get("artifacts", []):
            path = directory / str(artifact["path"])
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                return False
        return True


def create_app(
    *,
    repo_root: Path | None = None,
    artifacts_root: Path | None = None,
) -> FastAPI:
    root = (repo_root or repository_root()).resolve()
    store = RunStore((artifacts_root or root / "artifacts").resolve())
    app = FastAPI(
        title="PIT Alpha Lab API",
        version=__version__,
        description="Read-only access to checksum-audited quantitative research runs.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "project": "pit-alpha-lab", "version": __version__, "read_only": True}

    @app.get("/api/runs")
    def runs() -> dict[str, Any]:
        items = store.list_runs()
        return {"runs": items, "count": len(items)}

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        items = list_model_descriptors(include_external_names=False)
        return {
            "models": items,
            "count": len(items),
            "external_entry_point_group": ENTRY_POINT_GROUP,
            "external_code_loaded": False,
        }

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str) -> dict[str, Any]:
        directory = store.directory(run_id)
        manifest = store._json(directory / "run_manifest.json")
        config = store._json(directory / "config.resolved.json")
        metrics = store._json(directory / "metrics.json")
        model_name = str(config["model"]["name"])
        cost = int(config["portfolio"]["primary_cost_bps"])
        portfolio = [
            row
            for row in metrics["portfolio"]
            if row["window"] == "ALL" and int(row["cost_bps"]) == cost
        ]
        prediction = [row for row in metrics["prediction"] if row["window"] == "ALL"]
        return {
            "run_id": run_id,
            "snapshot_id": manifest["data"]["snapshot_id"],
            "model": model_name,
            "primary_cost_bps": cost,
            "artifacts_verified": store.verify(directory, manifest),
            "data": metrics["data"],
            "data_quality": metrics["data_quality"],
            "prediction": prediction[0] if prediction else None,
            "bootstrap_uncertainty": metrics["bootstrap_uncertainty"],
            "signal_decay": metrics["signal_decay"],
            "portfolio": portfolio,
            "cost_break_even": metrics["cost_break_even"],
        }

    @app.get("/api/runs/{run_id}/equity")
    def equity_curve(
        run_id: str,
        strategy: str = Query(..., min_length=1),
        cost_bps: int | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        directory = store.directory(run_id)
        config = store._json(directory / "config.resolved.json")
        selected_cost = int(config["portfolio"]["primary_cost_bps"] if cost_bps is None else cost_bps)
        column = f"net_return_{selected_cost}bps"
        frame = pd.read_csv(directory / "daily_returns.csv", parse_dates=["datetime"])
        if column not in frame:
            raise HTTPException(status_code=400, detail="requested cost is not available")
        selected = frame[frame["strategy"] == strategy].copy()
        if selected.empty:
            raise HTTPException(status_code=404, detail="strategy not found in run")
        selected["wealth"] = (1.0 + selected[column]).cumprod()
        points = [
            {
                "date": timestamp.date().isoformat(),
                "net_return": float(value),
                "wealth": float(wealth),
                "turnover": float(turnover),
            }
            for timestamp, value, wealth, turnover in selected[
                ["datetime", column, "wealth", "turnover"]
            ].itertuples(index=False, name=None)
        ]
        return {"run_id": run_id, "strategy": strategy, "cost_bps": selected_cost, "points": points}

    @app.get("/api/case-study/model-comparison")
    def model_comparison() -> dict[str, Any]:
        path = root / "case_studies" / "csi300_alpha" / "results" / "model_comparison_10bps.csv"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="model comparison is unavailable")
        frame = pd.read_csv(path)
        return {"cost_bps": 10, "models": frame.to_dict(orient="records")}

    @app.get("/api/case-study/market-context")
    def market_context_evidence() -> dict[str, Any]:
        results = root / "case_studies" / "csi300_alpha" / "results"
        paths = {
            "multiseed": results / "market_context_multiseed_summary.json",
            "seeds": results / "market_context_multiseed_10bps.csv",
            "ablation": results / "market_context_ablation_summary.json",
            "ablations": results / "market_context_ablation_10bps.csv",
        }
        if not all(path.is_file() for path in paths.values()):
            raise HTTPException(status_code=404, detail="market-context evidence is unavailable")
        return {
            "multiseed": store._json(paths["multiseed"]),
            "seeds": pd.read_csv(paths["seeds"]).to_dict(orient="records"),
            "ablation": store._json(paths["ablation"]),
            "ablations": pd.read_csv(paths["ablations"]).to_dict(orient="records"),
        }

    return app


app = create_app()
