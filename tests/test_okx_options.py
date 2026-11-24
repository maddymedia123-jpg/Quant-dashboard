import pytest

from core.data.okx_options import parse_okx_options


def test_okx_chain_uses_real_index_oi_and_iv_for_existing_gamma_schema():
    summaries = [
        {"instId": "BTC-USD_UM-261127-95000-P", "markVol": "0.5", "ts": "1795547934332"},
        {"instId": "BTC-USD_UM-261127-105000-C", "markVol": "0.55", "ts": "1795547934332"},
    ]
    open_interest = [
        {"instId": "BTC-USD_UM-261127-95000-P", "oi": "100", "oiCcy": "100"},
        {"instId": "BTC-USD_UM-261127-105000-C", "oi": "200", "oiCcy": "200"},
    ]
    index = {"idxPx": "100000", "ts": "1795547934332"}

    snapshot = parse_okx_options(summaries, open_interest, index)

    assert snapshot.available and snapshot.source == "okx"
    assert snapshot.underlying_price == 100_000
    assert snapshot.iv_atm in (50.0, 55.0)
    assert snapshot.total_put_oi == 100 and snapshot.total_call_oi == 200
    assert snapshot.expiries[0]["expiry"] == "2026-11-27"
    assert snapshot.gamma["available"] and snapshot.gamma["options_used"] == 2
    assert snapshot.gamma_by_desk["weekly"]["available"]


def test_okx_chain_without_real_oi_or_iv_is_rejected():
    summaries = [{"instId": "BTC-USD_UM-261127-95000-P", "markVol": "0.5"}]
    open_interest = [{"instId": "BTC-USD_UM-261127-95000-P", "oi": "0", "oiCcy": "0"}]

    try:
        parse_okx_options(summaries, open_interest, {"idxPx": "100000", "ts": "1795547934332"})
    except ValueError as exc:
        assert "nonzero open interest and mark volatility" in str(exc)
    else:
        raise AssertionError("an options chain without real open interest must not be accepted")


def test_okx_chain_with_invalid_spot_is_rejected_without_inventing_a_price():
    with pytest.raises(ValueError, match="no valid price or timestamp"):
        parse_okx_options([], [], {"idxPx": "nan", "ts": "1795547934332"})
