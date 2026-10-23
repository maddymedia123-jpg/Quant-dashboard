"""Page Gate.io BTC_USDT perp 1h candles backwards and cache them to research/data/btc_1h.csv.

Throwaway research helper (keyless public REST). Run from the repo root:
    python research/fetch_hourly.py [max_bars]
"""
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.data.gate_candles import parse_candles  # noqa: E402

URL = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
OUT = Path(__file__).parent / "data" / "btc_1h.csv"
H = 3600


def main(max_bars: int = 60_000):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    to = int(time.time()) // H * H
    frames = []
    with httpx.Client(timeout=30, headers={"User-Agent": "trap-intel-research/1.0"}) as c:
        while sum(len(f) for f in frames) < max_bars:
            frm = to - 1999 * H
            r = c.get(URL, params={"contract": "BTC_USDT", "interval": "1h", "from": frm, "to": to})
            if r.status_code != 200:
                print("stop:", r.status_code, r.text[:200])
                break
            rows = r.json()
            if not rows:
                print("stop: empty page at", frm)
                break
            df = parse_candles(rows)
            frames.append(df)
            print(len(df), pd.to_datetime(df.timestamp.min(), unit="ms"), "->",
                  pd.to_datetime(df.timestamp.max(), unit="ms"))
            to = frm - H
            time.sleep(0.3)
    df = pd.concat(frames).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print("saved", len(df), "bars to", OUT)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60_000)
