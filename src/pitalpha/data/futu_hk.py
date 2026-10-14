"""Read-only HK daily-data intake. A downloaded snapshot is not a PIT research dataset."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import socket
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pitalpha.artifacts import sha256_file, write_frame_atomic, write_json_atomic
from pitalpha.artifacts.manifest import source_identity
from pitalpha.environment import repository_root


LIMITATIONS = [
    "Convenience sample selected today; not historical index membership.",
    "No independent exchange calendar, suspension or delisting coverage certification.",
    "Corporate-action history is retrieved now; historical publication times are unknown.",
    "No executable fills, total-return labels or strategy performance are established.",
    "Provider terms govern storage and redistribution; raw data stay local.",
]


def load_request(path: Path) -> dict:
    request = yaml.safe_load(path.read_text(encoding="utf-8"))
    return validate_request(request)


def validate_request(request: dict) -> dict:
    required = {"schema_version", "provider", "symbols", "start", "end", "purpose"}
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError(f"HK request must contain exactly {sorted(required)}")
    if request["schema_version"] != 1 or request["provider"] != "futu_hk":
        raise ValueError("unsupported HK request schema/provider")
    symbols = request["symbols"]
    if (not isinstance(symbols, list) or not 1 <= len(symbols) <= 20
            or any(not isinstance(x, str) or not re.fullmatch(r"HK\.\d{5}", x) for x in symbols)
            or len(set(symbols)) != len(symbols)):
        raise ValueError("pilot requires 1..20 unique HK.00000 symbols")
    for field in ("start", "end"):
        value = request[field]
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("start/end must be quoted ISO dates")
        pd.Timestamp(value)  # Reject invalid calendar dates.
    if request["start"] > request["end"]:
        raise ValueError("start must not follow end")
    if pd.Timestamp(request["end"]).date() >= pd.Timestamp.now(tz="Asia/Hong_Kong").date():
        raise ValueError("end must precede today in Hong Kong; incomplete daily bars are excluded")
    if not isinstance(request["purpose"], str) or not request["purpose"].strip():
        raise ValueError("a nonempty sampling purpose is required")
    return request


def connection_status(port: int = 11111) -> dict:
    if not 1 <= port <= 65535:
        raise ValueError("invalid OpenD port")
    try:
        version = importlib.metadata.version("futu-api")
    except importlib.metadata.PackageNotFoundError:
        version = None
    reachable = False
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            reachable = True
    except OSError:
        pass
    return {"sdk_version": version, "host": "127.0.0.1", "port": port,
            "tcp_reachable": reachable, "login_and_market_permissions": "not_checked",
            "ready_for_connection_attempt": version is not None and reachable}


class FutuQuotes:
    """Only quote endpoints. No trading context, subscriptions or credential handling."""

    def __init__(self, context, sdk):
        self.context, self.sdk = context, sdk
        self.version = importlib.metadata.version("futu-api")
        self.last_request = 0.

    def _call(self, method, **kwargs):
        # Conservative pacing across these endpoints; fail on provider errors, no quota retries.
        time.sleep(max(0., 3.1 - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        result = method(**kwargs)
        if result[0] != self.sdk.RET_OK:
            # Provider error strings may contain account details. Do not persist them.
            raise RuntimeError(f"Futu {method.__name__} failed; inspect local OpenD permissions/logs")
        return result[1:]

    def quota(self):
        used, remaining, details = self._call(self.context.get_history_kl_quota, get_detail=True)[0]
        return int(used), int(remaining), {item["code"] for item in details}

    def bars(self, symbol, request, page_key):
        frame, next_key = self._call(
            self.context.request_history_kline, code=symbol,
            start=request["start"], end=request["end"], ktype=self.sdk.KLType.K_DAY,
            autype=self.sdk.AuType.NONE, fields=[self.sdk.KL_FIELD.ALL],
            max_count=1000, page_req_key=page_key,
        )
        return frame, next_key

    def rehab(self, symbol):
        return self._call(self.context.get_rehab, code=symbol)[0]


@contextmanager
def open_quotes(port: int = 11111):
    status = connection_status(port)
    if not status["ready_for_connection_attempt"]:
        raise RuntimeError("Install requirements-futu.txt, start/login to official OpenD on localhost, then run hk-data doctor")
    import futu
    context = futu.OpenQuoteContext(host="127.0.0.1", port=port)
    try:
        yield FutuQuotes(context, futu)
    finally:
        context.close()


def normalize_bars(frame: pd.DataFrame, request: dict) -> tuple[pd.DataFrame, dict]:
    fields = ["code", "time_key", "open", "high", "low", "close", "volume", "turnover"]
    if frame.empty or not set(fields).issubset(frame.columns):
        raise ValueError("empty bars or missing required OHLCV fields")
    result = frame[fields].copy()
    if result["code"].isna().any() or set(result["code"]) != set(request["symbols"]):
        raise ValueError("returned symbols do not match the request")
    stamps = pd.to_datetime(result["time_key"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    if stamps.isna().any() or not (stamps == stamps.dt.normalize()).all():
        raise ValueError("daily bar timestamps must be nonmissing local midnight dates")
    if not stamps.between(pd.Timestamp(request["start"]), pd.Timestamp(request["end"])).all():
        raise ValueError("provider returned bars outside the requested date range")
    result["date"] = stamps.dt.strftime("%Y-%m-%d")
    if result.duplicated(["code", "date"]).any():
        raise ValueError("duplicate symbol/date, including overlapping pages")
    numeric = fields[2:]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite OHLCV observations")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("nonpositive prices")
    if ((result["high"] < result[["open", "close", "low"]].max(axis=1))
            | (result["low"] > result[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("inconsistent OHLC ordering")
    if (result[["volume", "turnover"]] < 0).any().any():
        raise ValueError("negative volume/turnover")
    result = result.drop(columns="time_key").rename(columns={"code": "instrument"})
    result = result.sort_values(["date", "instrument"]).reset_index(drop=True)
    dates = set(result["date"])
    coverage = []
    for symbol, group in result.groupby("instrument", sort=True):
        coverage.append({"instrument": symbol, "rows": len(group),
                         "first": group["date"].min(), "last": group["date"].max(),
                         "missing_dates_vs_sample_union": sorted(dates - set(group["date"]))})
    return result, {"status": "structural_checks_passed", "rows": len(result), "coverage": coverage,
                    "zero_volume_rows": int((result["volume"] == 0).sum()),
                    "calendar_completeness": "not_established", "research_ready": False,
                    "limitations": LIMITATIONS}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_snapshot(request: dict, client, output_root: Path) -> dict:
    """Persist SDK-decoded responses before checking them; retain failed intake evidence."""
    request = validate_request(request)
    request_bytes = json.dumps(request, sort_keys=True).encode()
    request_id = hashlib.sha256(request_bytes).hexdigest()
    directory = output_root / ("futu-hk-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "kind": "market_data_intake", "status": "collecting",
                "provider": "futu", "sdk_version": client.version, "request": request,
                "request_sha256": request_id, "started_at_utc": _now(),
                "source": source_identity(repository_root()), "adjustment": "NONE",
                "bar_type": "K_DAY", "market_timezone": "Asia/Hong_Kong",
                "historical_availability": "unknown", "research_ready": False,
                "raw_representation": "SDK-decoded tables, not original network packets",
                "limitations": LIMITATIONS, "artifacts": [], "requests": []}
    manifest_path = directory / "snapshot_manifest.json"

    def persist(frame, name):
        path = write_frame_atomic(frame, directory / name)
        manifest["artifacts"].append({"path": name, "sha256": sha256_file(path), "rows": len(frame)})
        write_json_atomic(manifest_path, manifest)

    write_json_atomic(manifest_path, manifest)
    try:
        used, remaining, cached = client.quota()
        needed = len(set(request["symbols"]) - cached)
        manifest["quota"] = {"used": used, "remaining": remaining, "new_symbols_needed": needed}
        if remaining < needed:
            raise ValueError("insufficient historical symbol quota; no bars requested")
        all_frames = []
        for symbol in request["symbols"]:
            key, seen = None, set()
            for page in range(1, 101):
                before = _now()
                frame, next_key = client.bars(symbol, request, key)
                if not isinstance(frame, pd.DataFrame):
                    raise ValueError("invalid bar response type")
                name = f"raw/{symbol}-bars-{page:03d}.parquet"
                manifest["requests"].append({"endpoint": "request_history_kline", "symbol": symbol,
                                             "page": page, "started_at_utc": before,
                                             "received_at_utc": _now(), "response": name})
                persist(frame, name)
                if frame.empty or "code" not in frame or not (frame["code"] == symbol).all():
                    raise ValueError("empty page or wrong symbol returned")
                all_frames.append(frame)
                if next_key is None:
                    break
                if not isinstance(next_key, bytes) or next_key in seen:
                    raise ValueError("invalid or repeated pagination token")
                seen.add(next_key)
                key = next_key
            else:
                raise ValueError("pagination limit exceeded; snapshot incomplete")
            before = _now()
            rehab = client.rehab(symbol)
            if not isinstance(rehab, pd.DataFrame):
                raise ValueError("invalid corporate-action response type")
            name = f"raw/{symbol}-rehab.parquet"
            manifest["requests"].append({"endpoint": "get_rehab", "symbol": symbol,
                                         "started_at_utc": before, "received_at_utc": _now(), "response": name})
            persist(rehab, name)
            if not rehab.empty:
                if "ex_div_date" not in rehab or pd.to_datetime(rehab["ex_div_date"], errors="coerce").isna().any():
                    raise ValueError("invalid corporate-action event dates")
        bars, quality = normalize_bars(pd.concat(all_frames, ignore_index=True), request)
        persist(bars, "bars.parquet")
        quality_path = write_json_atomic(directory / "quality.json", quality)
        manifest["artifacts"].append({"path": "quality.json", "sha256": sha256_file(quality_path)})
        manifest["status"] = "downloaded_unvalidated"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_type"] = type(exc).__name__
        raise
    finally:
        manifest["finished_at_utc"] = _now()
        write_json_atomic(manifest_path, manifest)
    return {"directory": str(directory), "status": manifest["status"], "rows": quality["rows"],
            "research_ready": False, "manifest": str(manifest_path)}


def verify_snapshot(directory: Path) -> dict:
    """Check hashes and rebuild normalized bars/quality from saved raw pages offline."""
    manifest = json.loads((directory / "snapshot_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "downloaded_unvalidated" or manifest.get("kind") != "market_data_intake":
        raise ValueError("snapshot did not complete intake")
    request = validate_request(manifest["request"])
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    if digest != manifest.get("request_sha256") or manifest.get("adjustment") != "NONE":
        raise ValueError("request digest or adjustment contract mismatch")
    paths = set()
    for item in manifest["artifacts"]:
        path = (directory / item["path"]).resolve()
        if not path.is_relative_to(directory.resolve()) or item["path"] in paths:
            raise ValueError("unsafe or duplicate artifact path")
        paths.add(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {item['path']}")
    frames = []
    response_paths = set()
    rehab_symbols = set()
    for response in manifest["requests"]:
        if response["response"] not in paths or response["response"] in response_paths:
            raise ValueError("response is not covered by artifact hashes")
        response_paths.add(response["response"])
        if response["symbol"] not in request["symbols"]:
            raise ValueError("unexpected response symbol")
        if response["endpoint"] == "request_history_kline":
            frame = pd.read_parquet(directory / response["response"])
            if "code" not in frame or not (frame["code"] == response["symbol"]).all():
                raise ValueError("raw response symbol mismatch")
            frames.append(frame)
        elif response["endpoint"] == "get_rehab":
            if response["symbol"] in rehab_symbols:
                raise ValueError("duplicate corporate-action response")
            rehab_symbols.add(response["symbol"])
            rehab = pd.read_parquet(directory / response["response"])
            if not rehab.empty and ("ex_div_date" not in rehab or pd.to_datetime(rehab["ex_div_date"], errors="coerce").isna().any()):
                raise ValueError("invalid corporate-action event dates")
        else:
            raise ValueError("unexpected endpoint")
    if rehab_symbols != set(request["symbols"]):
        raise ValueError("missing corporate-action response")
    if not {"bars.parquet", "quality.json"}.issubset(paths) or not frames:
        raise ValueError("required artifacts missing")
    bars, quality = normalize_bars(pd.concat(frames, ignore_index=True), manifest["request"])
    try:
        pd.testing.assert_frame_equal(bars, pd.read_parquet(directory / "bars.parquet"))
    except AssertionError as exc:
        raise ValueError("normalized bars differ from raw observations") from exc
    if quality != json.loads((directory / "quality.json").read_text(encoding="utf-8")):
        raise ValueError("quality report differs from raw observations")
    return {"status": "integrity_and_structure_passed", "artifacts_verified": len(paths),
            "rows": len(bars), "research_ready": False, "limitations": LIMITATIONS}
