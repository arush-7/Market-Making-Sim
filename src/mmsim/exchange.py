"""The simulated venue (plan Phase 2 and Phase 5).

Responsibilities:

* hold the maker's resting orders,
* apply one-way latency to submissions and cancels,
* decide fills via a swappable :mod:`mmsim.fills` model,
* keep the cash / inventory / fee books.

Latency is modelled as a single fixed delay in seconds (e.g. 0.02 = 20 ms).
An order submitted at ``t`` cannot trade before ``t + latency``; a cancel sent
at ``t`` keeps the order alive until ``t + cancel_latency``. Real latency is
variable and asymmetric; that is a stated limitation.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

from .book import ASK, BID, OrderBook, Trade
from .fills import FillModel

SIGN = {BID: 1.0, ASK: -1.0}  # signing a fill: buy adds inventory, sell removes


@dataclass
class Order:
    """One resting maker order."""

    oid: int
    side: str
    price: float
    size: float
    ts_submitted: float
    ts_live: float
    filled: float = 0.0
    queue_ahead: float = 0.0
    cancel_at: Optional[float] = None

    @property
    def remaining(self) -> float:
        return self.size - self.filled


@dataclass(frozen=True)
class Fill:
    """A completed (possibly partial) execution of one order."""

    ts: float
    oid: int
    side: str
    price: float
    size: float
    mid: float
    fee: float

    @property
    def sign(self) -> float:
        return SIGN[self.side]

    @property
    def notional(self) -> float:
        return self.price * self.size


class SimulatedExchange:
    """Owns your orders, your cash and your inventory."""

    def __init__(
        self,
        book: OrderBook,
        fill_model: FillModel,
        latency: float = 0.02,
        cancel_latency: Optional[float] = None,
        maker_fee_bps: float = -0.5,
    ) -> None:
        self.book = book
        self.fill_model = fill_model
        self.latency = float(latency)
        self.cancel_latency = float(latency if cancel_latency is None else cancel_latency)
        # maker_fee_bps < 0 is a rebate you receive, > 0 is a fee you pay
        self.maker_fee_bps = float(maker_fee_bps)

        self.live: dict[int, Order] = {}
        self.fills: list[Fill] = []
        self.cash = 0.0
        self.inventory = 0.0
        self.fees = 0.0
        self.orders_submitted = 0

        self._pending: list[tuple[float, int, Order]] = []
        self._next_oid = 1

    # ------------------------------------------------------------------
    # order entry
    # ------------------------------------------------------------------
    def submit(self, side: str, price: float, size: float, ts: float) -> Order:
        if size <= 0:
            raise ValueError("order size must be positive")
        order = Order(
            oid=self._next_oid,
            side=side,
            price=float(price),
            size=float(size),
            ts_submitted=ts,
            ts_live=ts + self.latency,
        )
        self._next_oid += 1
        self.orders_submitted += 1
        heapq.heappush(self._pending, (order.ts_live, order.oid, order))
        return order

    def cancel(self, order: Order, ts: float) -> None:
        """Cancel request: the order keeps trading until it takes effect."""
        if order.cancel_at is None:
            order.cancel_at = ts + self.cancel_latency

    def cancel_all(self, ts: float) -> None:
        for order in list(self.live.values()):
            self.cancel(order, ts)
        # kill orders that have not arrived yet
        for _, _, order in self._pending:
            if order.cancel_at is None:
                order.cancel_at = ts

    def pending_orders(self) -> list[Order]:
        """Submitted but still in flight (cancelled ones excluded)."""
        return [order for _, _, order in self._pending if order.cancel_at is None]

    # ------------------------------------------------------------------
    # clock
    # ------------------------------------------------------------------
    def advance(self, ts: float) -> None:
        """Activate orders whose latency has elapsed, then drop cancelled ones."""
        while self._pending and self._pending[0][0] <= ts:
            _, _, order = heapq.heappop(self._pending)
            if order.cancel_at is not None and order.cancel_at <= ts:
                continue  # cancelled before it ever reached the venue
            self.live[order.oid] = order
            self.fill_model.on_order_live(order, self.book)

        if self.live:
            dead = [
                oid
                for oid, order in self.live.items()
                if order.cancel_at is not None and order.cancel_at <= ts
            ]
            for oid in dead:
                self.live.pop(oid, None)

    # ------------------------------------------------------------------
    # fills
    # ------------------------------------------------------------------
    def process_trade(self, trade: Trade) -> list[Fill]:
        """Let a tape print fill your resting orders."""
        if not self.live:
            return []
        mid = self.book.mid
        new_fills: list[Fill] = []
        for order in list(self.live.values()):
            if order.cancel_at is not None:
                continue  # being pulled; conservative
            qty = self.fill_model.on_trade(order, trade, self.book)
            if qty <= 0:
                continue
            qty = min(qty, order.remaining)
            order.filled += qty
            self.inventory += SIGN[order.side] * qty
            self.cash -= SIGN[order.side] * order.price * qty
            fee = order.price * qty * self.maker_fee_bps / 10_000.0
            # cash stays pure trade cash; fees are netted at valuation time
            self.fees += fee
            fill_mid = mid if mid is not None else order.price
            new_fills.append(
                Fill(trade.ts, order.oid, order.side, order.price, qty, fill_mid, fee)
            )
            if order.remaining <= 1e-12:
                self.live.pop(order.oid, None)
        self.fills.extend(new_fills)
        return new_fills

    # ------------------------------------------------------------------
    # valuation
    # ------------------------------------------------------------------
    def equity(self, mid: Optional[float]) -> float:
        """Cash + inventory marked to mid, minus fees paid (plus rebates earned).

        ``cash`` excludes fees on purpose, so the PnL decomposition in
        metrics.py (spread + inventory drift - fees) adds up exactly.
        """
        mark = mid if mid is not None else 0.0
        return self.cash + self.inventory * mark - self.fees

    def has_quotes(self) -> tuple[bool, bool]:
        """(has live bid, has live ask)."""
        has_bid = any(o.side == BID for o in self.live.values())
        has_ask = any(o.side == ASK for o in self.live.values())
        return has_bid, has_ask
