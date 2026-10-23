"""The promising cell on every venue over Gate's months (11 Jul - 23 Oct 2026), same test as
sine_ny_bigturns.py. Caveat: all venues trade the same bitcoin, so their turns nearly coincide - this checks
that Gate's result is not a quirk of Gate's candles; it is NOT independent evidence that the edge is real."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sine_ny_bigturns import cell  # noqa: E402
from sine_wave_more import DATA  # noqa: E402
from turns import load  # noqa: E402

g = load(DATA / "gate_15m.csv").iloc[:-1].reset_index(drop=True)
lo, hi = int(g.timestamp.iloc[0]), int(g.timestamp.iloc[-1]) + 1
cell(g, lo, hi, "Gate perp")
for v, name in (("binance", "Binance perp"), ("bybit", "Bybit perp"), ("bitget", "Bitget perp"),
                ("coinbase", "Coinbase spot")):
    cell(load(DATA / f"{v}_15m.csv").iloc[:-1].reset_index(drop=True), lo, hi, name)
