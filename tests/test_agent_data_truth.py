"""Two things the 20 Oct Director report got wrong because of what it was given, not how it reasoned:

* "neutral funding (0.0%)": every payload number was rounded to four decimal places, and a funding rate
  of 3.5e-05 rounds to 0.0. Small numbers now keep their significant figures.
* "positive spot CVD across all horizons": the field called CVD was an estimate from candle shapes,
  while the real taker flow (taker buy/sell 0.83) said sellers were in control. The field is now the real
  Gate.io perp taker delta, and the estimate is only used - and labelled ESTIMATE - when there is none."""
import os
import tempfile

import pandas as pd
import pytest

os.environ["TI_OFFLINE_FIXTURES"] = "1"
os.environ.setdefault("TI_DATA_DIR", tempfile.mkdtemp())

from core.agents import context  # noqa: E402
from core.config import CATEGORIES  # noqa: E402
from core.data.offline import fixture_market  # noqa: E402
from core.data.types import TakerFlowSnapshot  # noqa: E402
from core.indicators.category import analyze_category  # noqa: E402


@pytest.fixture(scope="module")
def market():
    return fixture_market()


@pytest.fixture(scope="module")
def live(market):
    return analyze_category(CATEGORIES["live"], market.spot.frames, market.futures)


# ---------- funding ----------
def test_small_numbers_keep_their_significant_figures():
    assert context._round(3.5e-05) == 3.5e-05, "a funding rate is not zero"
    assert context._round(-0.000123456) == -0.0001235
    assert context._round(0.8312) == 0.8312 and context._round(83_152.123456) == 83_152.1235
    assert context._round(0.0) == 0.0 and context._round(float("nan")) is None


def test_every_specialist_sees_the_real_funding_rate(market):
    m = market.model_copy(update={"futures": market.futures.model_copy(update={"funding_rate": 3.5e-05,
                                                                              "funding_7d_mean": 4.5e-05})})
    for domain in ("leverage", "sentiment"):
        data, _ = context.domain_payload(domain, m, {})
        assert data["futures"]["funding_rate"] == 3.5e-05, domain
        assert data["futures"]["funding_7d_mean"] == 4.5e-05, domain


# ---------- CVD ----------
def flow(delta_per_bar: float, n: int = 20) -> TakerFlowSnapshot:
    t0 = 1_790_000_000_000
    buy = 100.0
    df = pd.DataFrame({"timestamp": [t0 + i * 3_600_000 for i in range(n)], "buy": [buy] * n,
                       "sell": [buy - delta_per_bar] * n, "delta": [delta_per_bar] * n,
                       "long_liq_usd": [0.0] * n, "short_liq_usd": [0.0] * n})
    return TakerFlowSnapshot(source="gate.io", contract="BTC_USDT", btc_per_contract=0.0001, frames={"1h": df})


def test_cvd_is_the_real_taker_flow_when_the_feed_has_it(market, live):
    m = market.model_copy(update={"taker_flow": flow(-12.5)})
    c = context.cvd_brief(live, m)
    assert c["delta_btc"] == -62.5 and c["taker_buy_btc"] == 500.0 and c["taker_sell_btc"] == 562.5
    assert c["timeframe"] == "1h" and "real taker flow" in c["method"] and "perpetual" in c["method"]
    assert "estimate" not in c


def test_without_taker_data_the_estimate_says_it_is_one(market, live):
    m = market.model_copy(update={"taker_flow": TakerFlowSnapshot.unavailable("gate.io", "down")})
    c = context.cvd_brief(live, m)
    assert "delta_btc" not in c and "estimate" in c and c["method"].startswith("ESTIMATE")
    assert "not real CVD" in c["method"]


def test_the_brief_and_the_flow_specialist_both_get_it(market, live):
    m = market.model_copy(update={"taker_flow": flow(7.0)})
    assert context.category_brief(live, m)["cvd_5bars"]["delta_btc"] == 35.0
    data, _ = context.domain_payload("flow", m, {"live": live})
    assert data["flow_by_category"]["live"]["cvd_5bars"]["delta_btc"] == 35.0
    assert context.common_payload(m, {"live": live})["categories"]["live"]["cvd_5bars"]["delta_btc"] == 35.0


def test_the_candle_estimate_is_not_called_cvd_anywhere():
    from core.agents.prompts import _DOMAINS
    flow_prompt = " ".join(str(v) for v in _DOMAINS["flow"].values())
    assert "spot CVD" not in flow_prompt and "perp flow, not spot" in flow_prompt
    import inspect

    from core.indicators import direction
    assert "CVD rising" not in inspect.getsource(direction) and "candle closes show" in inspect.getsource(direction)
