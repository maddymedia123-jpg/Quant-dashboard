"""Older hourly history for an out-of-sample check: Binance USDT-M BTCUSDT perp 1h, 2020-01 .. 2025-08.

Gate.io only serves the newest 10,000 hourly points ("Candlestick too long ago. Maximum 10000 points
recently are allowed"), so the years before Sep 2025 come from Binance's public monthly dumps
(data.binance.vision). The dashboard does not read Binance (US hosting is refused); this is research only.
Cached to research/data/binance_1h.csv.
"""
import io
import sys
import zipfile
from pathlib import Path

import httpx
import pandas as pd

OUT = Path(__file__).parent / "data" / "binance_1h.csv"
URL = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{y}-{m:02d}.zip"


def main(first=(2020, 1), last=(2025, 8)):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    y, m = first
    with httpx.Client(timeout=60) as c:
        while (y, m) <= last:
            r = c.get(URL.format(y=y, m=m))
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                raw = z.read(z.namelist()[0]).decode()
                rows = [ln.split(",") for ln in raw.splitlines() if ln and ln[0].isdigit()]
                df = pd.DataFrame([[int(r_[0]), float(r_[1]), float(r_[2]), float(r_[3]), float(r_[4]), float(r_[5])]
                                   for r_ in rows], columns=["timestamp", "open", "high", "low", "close", "volume"])
                frames.append(df)
                print(y, m, len(df))
            else:
                print(y, m, "missing", r.status_code)
            m += 1
            if m == 13:
                y, m = y + 1, 1
    df = pd.concat(frames).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print("saved", len(df), "bars to", OUT)


if __name__ == "__main__":
    main()
    sys.exit(0)
