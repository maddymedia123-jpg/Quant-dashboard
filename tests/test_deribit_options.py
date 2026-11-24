import asyncio
import json
import pathlib

from core.data import deribit_options
from core.data.deribit_options import compute_max_pain, parse_book_summary, parse_instrument
from core.data.types import OptionsSnapshot

FX = pathlib.Path(__file__).parent / "fixtures"


def test_parse_instrument():
    assert parse_instrument("BTC-25SEP26-105000-C") == ("2026-09-25", 105000.0, "C")
    assert parse_instrument("BTC-9SEP26-80000-P") == ("2026-09-09", 80000.0, "P")


def test_max_pain_simple_book():
    # Calls OI at 100 and 110, puts OI at 90 and 100. Pain is minimised at 100.
    rows = [
        {"strike": 100, "kind": "C", "oi": 10}, {"strike": 110, "kind": "C", "oi": 10},
        {"strike": 90, "kind": "P", "oi": 10}, {"strike": 100, "kind": "P", "oi": 10},
    ]
    assert compute_max_pain(rows) == 100


def test_parse_book_summary_fixture():
    s = parse_book_summary(json.loads((FX / "deribit_book_summary.json").read_text()))
    assert s.available and s.source == "deribit"
    assert 1_000 < s.underlying_price < 1_000_000
    assert s.max_pain is not None and s.max_pain_expiry
    assert 0.1 < s.put_call_oi_ratio < 5
    assert s.iv_atm is not None and 10 < s.iv_atm < 300
    assert s.iv_skew is not None
    assert len(s.expiries) >= 3
    assert all({"expiry", "max_pain", "call_oi", "put_oi"} <= set(e) for e in s.expiries)


def test_fetch_options_uses_okx_after_deribit_failure(monkeypatch):
    class Response:
        def raise_for_status(self):
            raise TimeoutError("Deribit timed out")

    class Client:
        async def get(self, *args, **kwargs):
            return Response()

    async def okx_fallback(_client):
        return OptionsSnapshot(source="okx", underlying_price=100_000, max_pain=100_000,
                               gamma={"available": True, "regime": "positive"})

    monkeypatch.setattr("core.data.okx_options.fetch_okx_options", okx_fallback)
    snapshot = asyncio.run(deribit_options.fetch_options(Client()))

    assert snapshot.available and snapshot.source == "okx"
    assert snapshot.underlying_price == 100_000
    assert snapshot.gamma["available"]
    assert "Deribit failed (TimeoutError: Deribit timed out)" in snapshot.error


def test_fetch_options_uses_okx_after_empty_deribit_chain(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"result": []}

    class Client:
        async def get(self, *args, **kwargs):
            return Response()

    async def okx_fallback(_client):
        return OptionsSnapshot(source="okx", underlying_price=100_000)

    monkeypatch.setattr("core.data.okx_options.fetch_okx_options", okx_fallback)
    snapshot = asyncio.run(deribit_options.fetch_options(Client()))

    assert snapshot.available and snapshot.source == "okx"
    assert "empty options chain" in snapshot.error


def test_fetch_options_preserves_unavailable_values_when_both_sources_fail(monkeypatch):
    class Client:
        async def get(self, *args, **kwargs):
            raise TimeoutError("Deribit timed out")

    async def failed_okx(_client):
        return OptionsSnapshot.unavailable("okx", "connection refused")

    monkeypatch.setattr("core.data.okx_options.fetch_okx_options", failed_okx)
    snapshot = asyncio.run(deribit_options.fetch_options(Client()))

    assert not snapshot.available and snapshot.source == "deribit,okx"
    assert snapshot.max_pain is None and snapshot.underlying_price is None
    assert snapshot.gamma == {"available": False, "regime": "unknown", "note": snapshot.error}
    assert "deribit: TimeoutError: Deribit timed out" in snapshot.error
    assert "okx: connection refused" in snapshot.error
