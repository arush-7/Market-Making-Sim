"""Inventory limits and a kill switch (plan Phase 5).

The point of risk limits is not to make money; it is to make sure a bad
simulation run cannot produce a fantasy PnL by accumulating an absurd
position. In a real desk these limits are hard, enforced pre-trade.

``max_inventory`` is in units of the asset (not notional). A quote that would
push ``|inventory|`` past the limit is not sent.
"""

from __future__ import annotations

from typing import Optional

from .book import BID


class RiskManager:
    def __init__(
        self,
        max_inventory: float = 10.0,
        max_drawdown: Optional[float] = None,
        capital: float = 100_000.0,
    ) -> None:
        self.max_inventory = float(max_inventory)
        self.max_drawdown = max_drawdown
        self.capital = capital
        self.peak_equity = 0.0
        self.killed = False

    def allows(self, side: str, qty: float, inventory: float) -> bool:
        """Would a new order of ``qty`` on ``side`` breach the position limit?"""
        signed = qty if side == BID else -qty
        return abs(inventory + signed) <= self.max_inventory + 1e-9

    def update(self, equity: float) -> bool:
        """Feed the equity curve in. Returns True once the kill switch fired."""
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.max_drawdown is not None:
            if self.peak_equity - equity >= self.max_drawdown:
                self.killed = True
        return self.killed

    def drawdown(self, equity: float) -> float:
        return self.peak_equity - equity
