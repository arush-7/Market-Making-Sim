"""Load real market data into the normalised event stream (plan Phase 1).

Everything downstream only knows about three event types (BookDelta,
BookSnapshot, Trade), so a new data vendor means writing one loader here and
nothing else.

All timestamps are converted to **seconds** (float).
"""

from __future__ import annotations

import csv
from typing import Iterable, Optional, Sequence, Union

from .book import ASK, BID, BookDelta, BookSnapshot, Trade

Event = Union[BookDelta, BookSnapshot, Trade]

_TS_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def _to_seconds(value: float, unit: str) -> float:
    try:
        return float(value) * _TS_SCALE[unit]
    except KeyError:
        raise ValueError(f"unknown timestamp unit {unit!r}; use one of {sorted(_TS_SCALE)}") from None


def _rows(path: str):
    with open(path, newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def _header(path: str) -> set[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {name.strip() for name in next(csv.reader(fh), [])}


def load_depth_csv(path: str, ts_unit: str = "s") -> list[BookDelta]:
    """Incremental price-level updates.

    Columns: ``timestamp,side,price,size`` (size 0 removes the level).
    """
    events: list[BookDelta] = []
    for row in _rows(path):
        side = row["side"].strip().lower()
        if side not in (BID, ASK):
            raise ValueError(f"bad side {row['side']!r} in {path}")
        events.append(
            BookDelta(
                ts=_to_seconds(row["timestamp"], ts_unit),
                side=side,
                price=float(row["price"]),
                size=float(row["size"]),
            )
        )
    return events


def load_snapshot_csv(path: str, ts_unit: str = "s") -> list[BookSnapshot]:
    """Depth snapshots: every row is one level, rows sharing a timestamp form
    one snapshot. Columns: ``timestamp,side,price,size``."""
    grouped: dict[float, dict[str, list[tuple[float, float]]]] = {}
    for row in _rows(path):
        ts = _to_seconds(row["timestamp"], ts_unit)
        side = row["side"].strip().lower()
        if side not in (BID, ASK):
            raise ValueError(f"bad side {row['side']!r} in {path}")
        grouped.setdefault(ts, {BID: [], ASK: []})[side].append(
            (float(row["price"]), float(row["size"]))
        )
    return [
        BookSnapshot(
            ts=ts,
            bids=tuple(sorted(sides[BID], key=lambda x: -x[0])),
            asks=tuple(sorted(sides[ASK], key=lambda x: x[0])),
        )
        for ts, sides in sorted(grouped.items())
    ]


def load_trades_csv(path: str, ts_unit: str = "s") -> list[Trade]:
    """Tape prints. Columns: ``timestamp,price,size,aggressor``."""
    events: list[Trade] = []
    for row in _rows(path):
        aggressor = row["aggressor"].strip().lower()
        if aggressor not in ("buy", "sell"):
            raise ValueError(f"bad aggressor {row['aggressor']!r} in {path}")
        events.append(
            Trade(
                ts=_to_seconds(row["timestamp"], ts_unit),
                price=float(row["price"]),
                size=float(row["size"]),
                aggressor=aggressor,
            )
        )
    return events


def load_binance_agg_trades(path: str) -> list[Trade]:
    """Binance ``aggTrades`` CSV: ts are milliseconds, ``is_buyer_maker`` tells
    you the taker side (buyer was the maker => the taker sold)."""
    events: list[Trade] = []
    for row in _rows(path):
        is_buyer_maker = str(row["is_buyer_maker"]).strip().lower() in ("true", "1")
        events.append(
            Trade(
                ts=float(row["transact_time"]) * 1e-3,
                price=float(row["price"]),
                size=float(row["quantity"]),
                aggressor="sell" if is_buyer_maker else "buy",
            )
        )
    return events


def merge_events(*streams: Iterable[Event]) -> list[Event]:
    """Merge event streams in timestamp order (stable on ties)."""
    merged: list[Event] = []
    for stream in streams:
        merged.extend(stream)
    merged.sort(key=lambda e: e.ts)
    return merged


def load_events(
    depth_csv: Optional[str] = None,
    trades_csv: Optional[str] = None,
    ts_unit: str = "s",
    snapshots: bool = False,
    trades_format: str = "auto",
) -> list[Event]:
    """Convenience wrapper: give it files, get one ordered event stream.

    ``trades_format`` is ``"plain"``, ``"binance"`` or ``"auto"`` (detects the
    Binance aggTrades header), so a downloaded Binance file just works.
    """
    streams: list[Sequence[Event]] = []
    if depth_csv:
        streams.append(
            load_snapshot_csv(depth_csv, ts_unit) if snapshots else load_depth_csv(depth_csv, ts_unit)
        )
    if trades_csv:
        fmt = trades_format
        if fmt == "auto":
            fmt = "binance" if "is_buyer_maker" in _header(trades_csv) else "plain"
        if fmt == "binance":
            streams.append(load_binance_agg_trades(trades_csv))
        elif fmt == "plain":
            streams.append(load_trades_csv(trades_csv, ts_unit))
        else:
            raise ValueError(f"unknown trades_format {trades_format!r}")
    if not streams:
        raise ValueError("supply at least one of depth_csv / trades_csv")
    return merge_events(*streams)
