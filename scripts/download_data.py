"""Download a little real Binance data into data/ (optional).

The simulator runs on synthetic data; this is for when you want to look at
real prints. It uses the public market-data endpoint, no API key needed.

    python scripts/download_data.py --symbol BTCUSDT --rows 5000

Writes
------
data/<symbol>_aggtrades.csv   Binance aggTrades columns (ms timestamps), which
                              src/mmsim/data.py:load_binance_agg_trades reads
data/<symbol>_book.csv        one depth snapshot: timestamp,side,price,size

Caveat: the REST API gives recent trades and the *current* book, not hours of
historical depth. For a long run with a real book you need a vendor that keeps
L2 history (LOBSTER, Tardis, or your own recording).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://data-api.binance.vision/api/v3"


def fetch(path: str, **params) -> object:
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.load(resp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--rows", type=int, default=5000, help="max trades to fetch (1000 per call)")
    parser.add_argument("--depth", type=int, default=20, help="book levels per side to snapshot")
    args = parser.parse_args()

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    trades: list[dict] = []
    end_time = None
    while len(trades) < args.rows:
        want = min(1000, args.rows - len(trades))
        params = {"symbol": args.symbol, "limit": want}
        if end_time is not None:
            params["endTime"] = end_time
        batch = fetch("aggTrades", **params)
        if not batch:
            break
        trades.extend(batch)
        end_time = batch[0]["T"] - 1  # page backwards in time
    trades.sort(key=lambda t: t["T"])

    trades_path = data_dir / f"{args.symbol}_aggtrades.csv"
    with trades_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
             "transact_time", "is_buyer_maker"]
        )
        for t in trades:
            writer.writerow([t["a"], t["p"], t["q"], t["f"], t["l"], t["T"], t["m"]])

    book = fetch("depth", symbol=args.symbol, limit=args.depth)
    book_path = data_dir / f"{args.symbol}_book.csv"
    # The REST API only returns the *current* book, while the trades span a few
    # minutes before now. We stamp the snapshot at the first trade so the whole
    # window can be replayed with a static book. That is a crutch: for real L2
    # history use a vendor (LOBSTER, Tardis) or record your own.
    ts_ms = int(trades[0]["T"]) if trades else 0
    with book_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "side", "price", "size"])
        for price, size in book["bids"]:
            writer.writerow([ts_ms, "bid", price, size])
        for price, size in book["asks"]:
            writer.writerow([ts_ms, "ask", price, size])

    print(f"wrote {len(trades)} trades -> {trades_path}")
    print(f"wrote a {args.depth}-level snapshot -> {book_path}")
    print("note: the book snapshot is static (stamped at the first trade), because")
    print("      the public API has no historical depth. Fine for a smoke test,")
    print("      not for a real latency/queue study.")


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:  # network blocked, DNS, timeout...
        sys.exit(f"download failed ({exc}). The simulator does not need this file.")
