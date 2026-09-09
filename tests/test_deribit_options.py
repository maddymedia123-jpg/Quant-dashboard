import json
import pathlib

from core.data.deribit_options import compute_max_pain, parse_book_summary, parse_instrument

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
