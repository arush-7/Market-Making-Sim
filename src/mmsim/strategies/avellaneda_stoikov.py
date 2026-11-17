"""Avellaneda-Stoikov market maker (plan Phase 4).

Two formulas do all the work. With inventory ``q``, volatility ``sigma`` (per
sqrt(second)), risk aversion ``gamma``, fill-intensity decay ``k`` and time
left in the horizon ``tau = T - t``:

    reservation price   r = mid - q * gamma * sigma^2 * tau
    optimal spread      delta = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/k)

and the quotes are ``r - delta/2`` (bid) and ``r + delta/2`` (ask).

Reading it:

* ``q * gamma * sigma^2 * tau`` is the *skew*. Long inventory -> your fair
  value is below the mid, so you quote down: sell more eagerly, buy less.
* ``gamma * sigma^2 * tau`` widens the spread with volatility and time left.
  More uncertainty -> demand more compensation per trade.
* ``(2/gamma) * ln(1 + gamma/k)`` is the "pure market making" spread: it
  depends on how fast fill probability decays with distance (``k``). Larger
  ``k`` (fills die off quickly) -> tighter spread is affordable.

Everything here is causal: ``sigma`` comes from :class:`RollingVolatility`,
which only sees the past (optionally with a deliberate lag).
"""

from __future__ import annotations

import math
from typing import Optional

from ..book import BID, OrderBook, round_to_tick
from .base import OrderFlowImbalance, Quotes, RollingVolatility, Strategy


class AvellanedaStoikov(Strategy):
    name = "avellaneda_stoikov"

    def __init__(
        self,
        gamma: float = 0.25,
        # k is the fitted fill-intensity decay of the default synthetic market
        # (experiments/run_experiments.py recalibrates it from the data)
        k: float = 26.5,
        horizon: float = 120.0,
        size: float = 1.0,
        tick: float = 0.01,
        max_inventory: float = 10.0,
        vol_window: float = 120.0,
        vol_lag: float = 0.0,
        fallback_sigma: float = 0.015,  # used until enough mid history exists
        ofi_weight: float = 0.0,
        ofi_window: float = 10.0,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        self.gamma = float(gamma)
        self.k = float(k)
        self.horizon = float(horizon)
        self.size = float(size)
        self.tick = float(tick)
        self.max_inventory = float(max_inventory)
        self.vol_lag = float(vol_lag)
        self.fallback_sigma = float(fallback_sigma)
        self.ofi_weight = float(ofi_weight)  # in bps of mid, per unit imbalance

        self.vol = RollingVolatility(window_seconds=vol_window)
        self.ofi = OrderFlowImbalance(window=ofi_window) if ofi_weight else None

        # diagnostics, useful for plots and debugging
        self.last_sigma: Optional[float] = None
        self.last_reservation: Optional[float] = None
        self.last_half_spread: Optional[float] = None

    # ------------------------------------------------------------------
    def on_trade(self, trade) -> None:
        if self.ofi is not None:
            self.ofi.update(trade)

    def reset(self) -> None:
        self.vol = RollingVolatility(window_seconds=self.vol.window)
        if self.ofi is not None:
            self.ofi = OrderFlowImbalance(window=self.ofi.window)
        self.last_sigma = None
        self.last_reservation = None
        self.last_half_spread = None

    # ------------------------------------------------------------------
    def reservation_price(self, mid: float, inventory: float, sigma: float, tau: float) -> float:
        return mid - inventory * self.gamma * sigma * sigma * tau

    def optimal_half_spread(self, sigma: float, tau: float) -> float:
        return 0.5 * (
            self.gamma * sigma * sigma * tau + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        )

    # ------------------------------------------------------------------
    def quotes(self, ts: float, book: OrderBook, inventory: float) -> Quotes:
        mid = book.mid
        if mid is None:
            return Quotes(None, None, self.size)

        self.vol.update(ts, mid)
        sigma = self.vol.estimate(ts - self.vol_lag)
        if sigma is None or sigma <= 0:
            sigma = self.fallback_sigma
        self.last_sigma = sigma

        tau = self.horizon  # rolling horizon: this is a perpetual maker, not one session
        reservation = self.reservation_price(mid, inventory, sigma, tau)

        if self.ofi is not None:
            # lean quotes in the direction of recent aggressive flow
            reservation += self.ofi_weight * 1e-4 * mid * self.ofi.value(ts)

        half_spread = self.optimal_half_spread(sigma, tau)
        self.last_reservation = reservation
        self.last_half_spread = half_spread

        bid: Optional[float] = round_to_tick(reservation - half_spread, self.tick)
        ask: Optional[float] = round_to_tick(reservation + half_spread, self.tick)
        if ask - bid < self.tick:
            ask = bid + self.tick
        if bid <= 0:
            bid = self.tick

        if inventory >= self.max_inventory:
            bid = None
        if inventory <= -self.max_inventory:
            ask = None
        return Quotes(bid, ask, self.size)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AvellanedaStoikov(gamma={self.gamma}, k={self.k}, "
            f"horizon={self.horizon}, vol_lag={self.vol_lag}, ofi_weight={self.ofi_weight})"
        )


__all__ = ["AvellanedaStoikov"]
