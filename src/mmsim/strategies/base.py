"""Strategy interface and shared helpers.

A strategy is asked one question: *given the book and my inventory, what two
prices do I want to show?* Returning ``None`` for one side means "pull that
quote". Everything about who fills and when lives in the exchange.
"""

from __future__ import annotations

import bisect
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..book import OrderBook, Trade


@dataclass(frozen=True)
class Quotes:
    """The two prices (and size) the strategy wants to show."""

    bid: Optional[float]
    ask: Optional[float]
    size: float


class Strategy(ABC):
    name = "base"

    @abstractmethod
    def quotes(self, ts: float, book: OrderBook, inventory: float) -> Quotes:
        """Target quotes at time ``ts``. Must not use any future information."""

    def on_trade(self, trade: Trade) -> None:
        """Called for every tape print (see order-flow imbalance)."""

    def on_fill(self, fill) -> None:
        """Called for every fill of your own orders."""

    def reset(self) -> None:
        """Clear any internal state between runs."""


class RollingVolatility:
    """Realised volatility from the mid, using only data up to ``end_ts``.

    sigma = sqrt( sum(dmid^2) / span )  -> price units per sqrt(second)

    This is the irregular-interval version of realised variance: every mid
    change contributes squared move per unit time, so the units come out right
    no matter how often the book updates.

    Use ``vol_lag`` to pretend the estimate is stale (a real system recomputes
    sigma every few seconds, so during a volatility spike it is always behind).
    """

    def __init__(self, window_seconds: float = 120.0, sample_interval: float = 0.5) -> None:
        self.window = float(window_seconds)
        self.sample_interval = float(sample_interval)
        self._ts: deque[float] = deque()
        self._mid: deque[float] = deque()
        self._last_sample_ts: Optional[float] = None

    def update(self, ts: float, mid: Optional[float]) -> None:
        if mid is None:
            return
        if self._last_sample_ts is not None and ts - self._last_sample_ts < self.sample_interval:
            return
        self._last_sample_ts = ts
        self._ts.append(ts)
        self._mid.append(mid)
        cutoff = ts - self.window
        while len(self._ts) > 2 and self._ts[0] < cutoff:
            self._ts.popleft()
            self._mid.popleft()

    def estimate(self, end_ts: Optional[float] = None) -> Optional[float]:
        ts = list(self._ts)
        mid = list(self._mid)
        if end_ts is not None:
            idx = bisect.bisect_right(ts, end_ts)
            ts, mid = ts[:idx], mid[:idx]
        if len(ts) < 4:
            return None
        span = ts[-1] - ts[0]
        if span <= 0:
            return None
        squared = 0.0
        for a, b in zip(mid, mid[1:]):
            d = b - a
            squared += d * d
        return math.sqrt(squared / span)


class OrderFlowImbalance:
    """Signed taker-volume imbalance over the last ``window`` seconds, in [-1, 1].

    ``> 0`` means buyers have been the aggressive side recently. When informed
    flow is present the mid keeps moving that way for a while, so a strategy
    can lean its quotes in that direction instead of being run over.
    """

    def __init__(self, window: float = 10.0) -> None:
        self.window = float(window)
        self._events: deque[tuple[float, float]] = deque()

    def update(self, trade: Trade) -> None:
        signed = trade.size if trade.aggressor == "buy" else -trade.size
        self._events.append((trade.ts, signed))

    def value(self, ts: float) -> float:
        cutoff = ts - self.window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        total = 0.0
        signed = 0.0
        for _, s in self._events:
            total += abs(s)
            signed += s
        if total <= 0:
            return 0.0
        return signed / total
