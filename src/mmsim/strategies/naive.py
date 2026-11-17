"""The naive baseline (plan Phase 3).

Quote a fixed distance either side of the mid, always, whatever the volatility
or the inventory. This is the benchmark every smarter strategy has to beat:
if Avellaneda-Stoikov cannot beat "just quote four cents wide", it is not
adding value.

Deliberately dumb, except for hard inventory limits (which a real desk would
also enforce).
"""

from __future__ import annotations

from typing import Optional

from ..book import BID, OrderBook, round_to_tick
from .base import Quotes, Strategy


class FixedSpreadStrategy(Strategy):
    name = "naive_fixed_spread"

    def __init__(
        self,
        half_spread: float = 0.04,
        size: float = 1.0,
        tick: float = 0.01,
        max_inventory: float = 10.0,
    ) -> None:
        self.half_spread = float(half_spread)
        self.size = float(size)
        self.tick = float(tick)
        self.max_inventory = float(max_inventory)

    def quotes(self, ts: float, book: OrderBook, inventory: float) -> Quotes:
        mid = book.mid
        if mid is None:
            return Quotes(None, None, self.size)

        bid: Optional[float] = round_to_tick(mid - self.half_spread, self.tick)
        ask: Optional[float] = round_to_tick(mid + self.half_spread, self.tick)
        if bid is not None and ask - bid < self.tick:
            ask = bid + self.tick
        if bid is not None and bid <= 0:
            bid = None

        # hard limits: stop adding to a position that is already maxed out
        if inventory >= self.max_inventory:
            bid = None
        if inventory <= -self.max_inventory:
            ask = None
        return Quotes(bid, ask, self.size)

    def __repr__(self) -> str:  # pragma: no cover
        return f"FixedSpreadStrategy(half_spread={self.half_spread}, size={self.size})"


__all__ = ["FixedSpreadStrategy"]
