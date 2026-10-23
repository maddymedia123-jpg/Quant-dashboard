"""Binance USDT-M BTCUSDT perp candles at any interval, from the public monthly dumps (research only).

    python research/fetch_binance.py 4h 2020-01 2026-09   -> research/data/binance_4h.csv
"""
import io
import sys
import zipfile
from pathlib import Path

import httpx
import pandas as pd

URL = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/{tf}/BTCUSDT-{tf}-{y}-{m:02d}.zip"


def main(tf: str, first: str, last: str):
    out = Path(__file__).parent / "data" / f"binance_{tf}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    (y, m), end = map(int, first.split("-")), tuple(map(int, last.split("-")))
    frames = []
    with httpx.Client(timeout=60) as c:
        while (y, m) <= end:
            r = c.get(URL.format(tf=tf, y=y, m=m))
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                raw = z.read(z.namelist()[0]).decode()
                rows = [ln.split(",") for ln in raw.splitlines() if ln and ln[0].isdigit()]
                frames.append(pd.DataFrame([[int(x[0]), *map(float, x[1:6])] for x in rows],
                                           columns=["timestamp", "open", "high", "low", "close", "volume"]))
            else:
                print(tf, y, m, "missing", r.status_code)
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    df = pd.concat(frames).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df.to_csv(out, index=False)
    print("saved", len(df), tf, "bars to", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
