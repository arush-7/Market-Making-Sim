"""Order book reconstruction (plan Phase 1).

The book is two dictionaries mapping ``price -> resting size``. Everything
else (best bid/ask, mid, depth, imbalance) is derived from them. Both
snapshots and incremental price-level updates are supported, so the same
class replays a synthetic market and a real exchange depth feed.

Price levels are keyed by float price that has already been rounded to the
tick (see :func:`round_to_tick`). Two prices that round to the same tick are
the same level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

BID = "bid"
ASK = "ask"

BUY = "buy"  # taker bought -> hit the ask side
SELL = "sell"  # taker sold -> hit the bid side


@dataclass(frozen=True)
class BookDelta:
    """One price level changed. ``size <= 0`` removes the level."""

    ts: float
    side: str
    price: float
    size: float


@dataclass(frozen=True)
class BookSnapshot:
    """A full top-of-book state. Replaces the whole book."""

    ts: float
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Trade:
    """A print on the tape.

    ``aggressor`` is the taker side: ``"sell"`` means a seller hit the bid and
    the print happened at or below the previous best bid.
    """

    ts: float
    price: float
    size: float
    aggressor: str


def round_to_tick(price: float, tick: float, mode: str = "nearest") -> float:
    """Snap ``price`` to the tick grid.

    mode: "nearest" | "down" | "up". The trailing ``round(..., 10)`` stops
    floats like 100.30000000000001 from leaking into the book keys.
    """
    steps = price / tick
    if mode == "down":
        n = math.floor(steps + 1e-9)
    elif mode == "up":
        n = math.ceil(steps - 1e-9)
    else:
        n = math.floor(steps + 0.5)
    return round(n * tick, 10)


class OrderBook:
    """L2 book: price -> size for each side, plus cached sorted prices."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ts: float = 0.0
        self.update_count: int = 0
        self._bid_prices: Optional[list[float]] = None
        self._ask_prices: Optional[list[float]] = None

    # ------------------------------------------------------------------
    # updates
    # ------------------------------------------------------------------
    def apply(self, event: object) -> None:
        """Apply a BookDelta or BookSnapshot."""
        if isinstance(event, BookDelta):
            self.ts = event.ts
            if event.side == BID:
                if event.size <= 0:
                    self.bids.pop(event.price, None)
                else:
                    self.bids[event.price] = event.size
                self._bid_prices = None
            elif event.side == ASK:
                if event.size <= 0:
                    self.asks.pop(event.price, None)
                else:
                    self.asks[event.price] = event.size
                self._ask_prices = None
            else:
                raise ValueError(f"unknown side {event.side!r}")
        elif isinstance(event, BookSnapshot):
            self.ts = event.ts
            self.bids = {float(p): float(s) for p, s in event.bids if s > 0}
            self.asks = {float(p): float(s) for p, s in event.asks if s > 0}
            self._bid_prices = None
            self._ask_prices = None
        else:
            raise TypeError(f"cannot apply {type(event).__name__} to the book")
        self.update_count += 1

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self._bid_prices = None
        self._ask_prices = None

    def _sorted_bids(self) -> list[float]:
        if self._bid_prices is None:
            self._bid_prices = sorted(self.bids, reverse=True)
        return self._bid_prices

    def _sorted_asks(self) -> list[float]:
        if self._ask_prices is None:
            self._ask_prices = sorted(self.asks)
        return self._ask_prices

    # ------------------------------------------------------------------
    # top of book
    # ------------------------------------------------------------------
    @property
    def best_bid(self) -> Optional[float]:
        prices = self._sorted_bids()
        return prices[0] if prices else None

    @property
    def best_ask(self) -> Optional[float]:
        prices = self._sorted_asks()
        return prices[0] if prices else None

    @property
    def mid(self) -> Optional[float]:
        """Mid price, or None if either side is empty."""
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return 0.5 * (bb + ba)

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    def is_crossed(self) -> bool:
        """Sanity check used to validate book reconstruction."""
        bb, ba = self.best_bid, self.best_ask
        return bb is not None and ba is not None and bb >= ba

    # ------------------------------------------------------------------
    # depth
    # ------------------------------------------------------------------
    def size_at(self, side: str, price: float) -> float:
        """Resting size at one price level (0 if the level does not exist)."""
        book = self.bids if side == BID else self.asks
        return book.get(price, 0.0)

    def depth(self, side: str, levels: int = 5) -> list[tuple[float, float]]:
        """Top ``levels`` of one side as ``[(price, size), ...]``, best first."""
        prices = self._sorted_bids() if side == BID else self._sorted_asks()
        book = self.bids if side == BID else self.asks
        return [(p, book[p]) for p in prices[:levels]]

    def imbalance(self, levels: int = 5) -> float:
        """Signed volume imbalance in [-1, 1]; positive means bid-heavy."""
        bid_size = sum(s for _, s in self.depth(BID, levels))
        ask_size = sum(s for _, s in self.depth(ASK, levels))
        total = bid_size + ask_size
        if total <= 0:
            return 0.0
        return (bid_size - ask_size) / total

    def volume_within(self, side: str, price: float) -> float:
        """Total resting size at ``price`` or better (used for queue maths)."""
        if side == BID:
            return sum(s for p, s in self.bids.items() if p >= price)
        return sum(s for p, s in self.asks.items() if p <= price)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"OrderBook(ts={self.ts:.3f}, best_bid={self.best_bid}, "
            f"best_ask={self.best_ask}, levels={len(self.bids)}/{len(self.asks)})"
        )
