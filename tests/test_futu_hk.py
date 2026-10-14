from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from pitalpha.artifacts import sha256_file
from pitalpha.cli import main
from pitalpha.data.futu_hk import (
    FutuQuotes, collect_snapshot, connection_status, load_request,
    normalize_bars, validate_request, verify_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


def request():
    return {"schema_version": 1, "provider": "futu_hk", "symbols": ["HK.00700"],
            "start": "2024-01-01", "end": "2024-01-05", "purpose": "synthetic contract fixture"}


def frame(day="2024-01-02", symbol="HK.00700"):
    return pd.DataFrame([{"code": symbol, "time_key": day + " 00:00:00",
                          "open": 100., "high": 105., "low": 99., "close": 102.,
                          "volume": 10000, "turnover": 1020000., "pe_ratio": 22.}])


class FakeQuotes:
    version = "offline-fixture-not-market-data"

    def __init__(self):
        self.calls = []

    def quota(self):
        return 0, 100, set()

    def bars(self, symbol, spec, key):
        self.calls.append(key)
        return (frame(), b"page-2") if key is None else (frame("2024-01-03"), None)

    def rehab(self, symbol):
        return pd.DataFrame([{"ex_div_date": "2024-01-03", "per_cash_div": 1.,
                              "forward_adj_factorA": 1., "forward_adj_factorB": -1.}])


class FutuHKTests(unittest.TestCase):
    def test_request_and_offline_plan(self):
        spec = load_request(ROOT / "configs/hk_futu_pilot.yaml")
        self.assertEqual(len(spec["symbols"]), 3)
        with patch("socket.create_connection", side_effect=AssertionError("network")), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["hk-data", "plan", str(ROOT / "configs/hk_futu_pilot.yaml")]), 0)
            self.assertFalse(json.loads(output.getvalue())["network_accessed"])

    def test_invalid_requests_fail_before_any_provider_calls(self):
        for field, value in [("symbols", ["US.AAPL"]), ("symbols", ["HK.00700"] * 2),
                             ("end", "2999-01-01"), ("start", "bad"), ("purpose", ""),
                             ("schema_version", 2), ("end", "2023-01-01")]:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                spec = request()
                spec[field] = value
                client = Mock()
                with self.assertRaises(ValueError):
                    collect_snapshot(spec, client, Path(directory))
                client.quota.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_pagination_raw_retention_and_offline_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeQuotes()
            result = collect_snapshot(request(), client, Path(directory))
            root = Path(result["directory"])
            self.assertEqual(client.calls, [None, b"page-2"])
            self.assertFalse(result["research_ready"])
            self.assertEqual(result["status"], "downloaded_unvalidated")
            self.assertEqual(verify_snapshot(root)["rows"], 2)
            raw = pd.read_parquet(root / "raw/HK.00700-bars-001.parquet")
            clean = pd.read_parquet(root / "bars.parquet")
            self.assertIn("pe_ratio", raw)
            self.assertNotIn("pe_ratio", clean)  # No unverified as-of fundamental features.
            self.assertEqual(clean.iloc[0]["close"], 102.)  # Never silently forward-adjust prices.
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["hk-data", "verify", str(root)]), 0)
                self.assertFalse(json.loads(output.getvalue())["research_ready"])

    def test_quota_failure_has_manifest_and_no_bar_requests(self):
        client = FakeQuotes()
        client.quota = lambda: (100, 0, set())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "quota"):
                collect_snapshot(request(), client, Path(directory))
            self.assertEqual(client.calls, [])
            manifest = json.loads(next(Path(directory).glob("*/snapshot_manifest.json")).read_text())
            self.assertEqual(manifest["status"], "failed")

    def test_cached_symbol_can_be_requested_without_new_quota(self):
        client = FakeQuotes()
        client.quota = lambda: (100, 0, {"HK.00700"})
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(collect_snapshot(request(), client, Path(directory))["rows"], 2)

    def test_provider_failure_retains_raw_but_no_completed_snapshot(self):
        client = FakeQuotes()
        client.rehab = Mock(side_effect=RuntimeError("provider unavailable"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                collect_snapshot(request(), client, Path(directory))
            root = next(Path(directory).iterdir())
            self.assertEqual(len(list((root / "raw").glob("*.parquet"))), 2)
            self.assertFalse((root / "bars.parquet").exists())
            with self.assertRaisesRegex(ValueError, "complete"):
                verify_snapshot(root)

    def test_repeated_page_token_stops(self):
        client = FakeQuotes()
        client.bars = lambda *args: (frame(), b"same")
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "pagination"):
            collect_snapshot(request(), client, Path(directory))

    def test_overlapping_pages_fail_without_deduplication(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_bars(pd.concat([frame(), frame()]), request())

    def test_invalid_market_observations(self):
        for field, value in [("open", -1.), ("close", float("nan")), ("high", 1.),
                             ("low", 200.), ("volume", -1), ("turnover", float("inf")),
                             ("time_key", "2024-01-02 12:00:00"), ("time_key", "2024-02-02 00:00:00"),
                             ("code", "US.AAPL")]:
            with self.subTest(field=field, value=value):
                bad = frame()
                bad.loc[0, field] = value
                with self.assertRaises(ValueError):
                    normalize_bars(bad, request())

    def test_coverage_reports_gaps_without_inventing_suspensions(self):
        spec = request()
        spec["symbols"].append("HK.00005")
        data = pd.concat([frame(), frame("2024-01-03"), frame(symbol="HK.00005")])
        clean, quality = normalize_bars(data, spec)
        self.assertEqual(len(clean), 3)
        self.assertEqual(quality["coverage"][0]["missing_dates_vs_sample_union"], ["2024-01-03"])
        self.assertEqual(quality["calendar_completeness"], "not_established")

    def test_tampered_derived_data_fail_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(collect_snapshot(request(), FakeQuotes(), Path(directory))["directory"])
            path = root / "bars.parquet"
            data = pd.read_parquet(path)
            data.loc[0, "close"] = 101.
            data.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_snapshot(root)
            manifest_path = root / "snapshot_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for item in manifest["artifacts"]:
                if item["path"] == "bars.parquet":
                    item["sha256"] = sha256_file(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "differ"):
                verify_snapshot(root)

    def test_missing_event_file_and_unsafe_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(collect_snapshot(request(), FakeQuotes(), Path(directory))["directory"])
            path = root / "snapshot_manifest.json"
            original = json.loads(path.read_text())
            bad = copy.deepcopy(original)
            bad["requests"] = [x for x in bad["requests"] if x["endpoint"] != "get_rehab"]
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "corporate-action"):
                verify_snapshot(root)
            bad = copy.deepcopy(original)
            bad["artifacts"][0]["path"] = "../outside.parquet"
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                verify_snapshot(root)

    def test_doctor_does_not_claim_login_from_a_socket(self):
        with patch("socket.create_connection") as connect:
            status = connection_status()
            self.assertTrue(status["tcp_reachable"])
            self.assertEqual(status["login_and_market_permissions"], "not_checked")
            connect.assert_called_once_with(("127.0.0.1", 11111), timeout=2)

    def test_sdk_adapter_requests_daily_unadjusted_bars_and_redacts_errors(self):
        sdk = SimpleNamespace(RET_OK=0, KLType=SimpleNamespace(K_DAY="K_DAY"),
                              AuType=SimpleNamespace(NONE="NONE"), KL_FIELD=SimpleNamespace(ALL="ALL"))
        context = Mock()
        context.request_history_kline.return_value = (0, frame(), None)
        with patch("importlib.metadata.version", return_value="fixture"), patch("time.sleep"):
            client = FutuQuotes(context, sdk)
            client.bars("HK.00700", request(), None)
            kwargs = context.request_history_kline.call_args.kwargs
            self.assertEqual(kwargs["autype"], "NONE")
            self.assertEqual(kwargs["ktype"], "K_DAY")
            self.assertEqual(kwargs["max_count"], 1000)
            def denied(**kwargs):
                return -1, "private account information"
            with self.assertRaisesRegex(RuntimeError, "denied failed") as error:
                client._call(denied)
            self.assertNotIn("private", str(error.exception))


if __name__ == "__main__":
    unittest.main()
