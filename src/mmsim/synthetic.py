"""A synthetic limit order book with *known* parameters (plan Phase 7).

Why bother, when you could download BTCUSDT?

* it runs instantly and deterministically (seeded),
* every parameter is known, so a wrong A-S implementation shows up as
  "behaves nothing like the theory predicts",
* you can create a calm period and a volatility spike on demand, which is
  impossible to schedule on real data.

The model, one step of ``dt`` seconds:

1. A snapshot of the book is published around the latent mid:
   ``depth_levels`` price levels per side, each with a random resting size.
2. A Poisson number of aggressive orders arrive. Each one picks a depth
   ``d ~ Geometric(p)`` and prints at that level, so most trades hit the top
   of book and few reach deep. That is what makes fill intensity decay roughly
   exponentially in quote distance - the ``lambda(delta) = A * exp(-k*delta)``
   assumption behind Avellaneda-Stoikov.
3. A fraction of those trades are *informed*: they know an information shock
   and trade in its direction. Part of the shock hits the mid immediately,
   the rest decays in over ``info_decay`` seconds. This is the source of
   adverse selection: your fill is followed by the mid moving away from you.
   It is also why order-flow imbalance predicts the next few seconds.
4. Finally the uninformed mid diffuses by ``sigma * sqrt(dt)``.

Known simplifications (state these in an interview): trades do not consume the
book, the book is rebuilt from scratch every step, and participants do not
react to your quotes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

from .book import ASK, BID, BookSnapshot, Trade


@dataclass
class SyntheticConfig:
    """Knobs of the fake market. Defaults: 1 hour, calm then volatile."""

    duration: float = 3600.0
    dt: float = 0.25  # book update interval (seconds)
    mid_start: float = 100.0
    tick: float = 0.01

    # uninformed volatility, in price units per sqrt(second)
    sigma: float = 0.012
    sigma_volatile: float = 0.040
    regime_change_ts: float = 1800.0  # set >= duration for a single calm regime

    trade_rate: float = 1.5  # aggressive orders per second
    informed_fraction: float = 0.40
    info_bps: float = 1.5  # size of one information shock, bps of mid
    immediate_fraction: float = 0.35  # how much of the shock hits at once
    info_decay: float = 4.0  # seconds for the rest of the shock to arrive

    depth_levels: int = 8
    level_size: float = 8.0  # resting size per level (scaled per level/step)
    depth_p: float = 0.30  # Geometric(p) depth of an aggressive order
    trade_size_frac: tuple[float, float] = (0.15, 0.70)

    seed: int = 7


@dataclass
class SyntheticMarket:
    events: list[object]
    trades: list[Trade]
    config: SyntheticConfig

    @property
    def duration(self) -> float:
        return self.config.duration

    def sigma_at(self, ts: float) -> float:
        """The uninformed volatility the generator used at ``ts``."""
        return _sigma_at(self.config, ts)


def _sigma_at(cfg: SyntheticConfig, ts: float) -> float:
    return cfg.sigma_volatile if ts >= cfg.regime_change_ts else cfg.sigma


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's method. Fine for the small rates used here."""
    if lam <= 0:
        return 0
    limit = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _geometric(rng: random.Random, p: float) -> int:
    """Number of trials until the first success, >= 1."""
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    r = rng.random()
    return max(1, int(math.ceil(math.log(1.0 - r) / math.log(1.0 - p))))


def build_synthetic_market(cfg: SyntheticConfig | None = None) -> SyntheticMarket:
    cfg = cfg or SyntheticConfig()
    rng = random.Random(cfg.seed)
    tick = cfg.tick
    dt = cfg.dt
    decay = math.exp(-dt / cfg.info_decay) if cfg.info_decay > 0 else 0.0
    steps = int(cfg.duration / dt)

    events: list[object] = []
    trades: list[Trade] = []
    mid = cfg.mid_start
    transient = 0.0  # information pressure that has not hit the mid yet

    for step in range(steps):
        ts = step * dt
        sigma = _sigma_at(cfg, ts)

        # 1. publish a book around the current mid
        best_bid = math.floor(mid / tick + 1e-9) * tick - tick
        best_ask = math.ceil(mid / tick - 1e-9) * tick + tick
        bids: list[tuple[float, float]] = []
        asks: list[tuple[float, float]] = []
        for i in range(cfg.depth_levels):
            bid_size = cfg.level_size * rng.uniform(0.4, 1.6) * (1.0 + 0.10 * i)
            ask_size = cfg.level_size * rng.uniform(0.4, 1.6) * (1.0 + 0.10 * i)
            bids.append((round(best_bid - i * tick, 10), round(bid_size, 4)))
            asks.append((round(best_ask + i * tick, 10), round(ask_size, 4)))
        events.append(BookSnapshot(ts=ts, bids=tuple(bids), asks=tuple(asks)))

        # 2. & 3. aggressive orders, some of them informed
        for _ in range(_poisson(rng, cfg.trade_rate * dt)):
            t_trade = ts + rng.random() * dt
            if rng.random() < cfg.informed_fraction:
                shock = rng.gauss(0.0, cfg.info_bps * 1e-4 * mid)
                aggressor = "buy" if shock > 0 else "sell"
                mid += cfg.immediate_fraction * shock
                transient += (1.0 - cfg.immediate_fraction) * shock
            else:
                aggressor = "buy" if rng.random() < 0.5 else "sell"

            depth = min(_geometric(rng, cfg.depth_p), cfg.depth_levels)
            if aggressor == "sell":
                price = bids[depth - 1][0]
                level_size = bids[depth - 1][1]
            else:
                price = asks[depth - 1][0]
                level_size = asks[depth - 1][1]
            size = level_size * rng.uniform(*cfg.trade_size_frac)

            trade = Trade(ts=t_trade, price=price, size=round(size, 6), aggressor=aggressor)
            trades.append(trade)
            events.append(trade)

        # 4. uninformed diffusion + the decaying information pressure
        mid += rng.gauss(0.0, sigma * math.sqrt(dt))
        mid += transient * (1.0 - decay)
        transient *= decay
        if mid < 10 * tick:
            mid = 10 * tick

    events.sort(key=lambda e: e.ts)  # trades inside a step have ts > the snapshot's
    return SyntheticMarket(events=events, trades=trades, config=cfg)


def trade_distances(trades: Iterable[Trade], mid_by_time) -> list[float]:
    """|trade price - mid| for each trade, using ``mid_by_time(ts)``.

    Used to calibrate ``k``: the fill-intensity decay of the A-S model.
    """
    return [abs(t.price - mid_by_time(t.ts)) for t in trades]


__all__ = ["SyntheticConfig", "SyntheticMarket", "build_synthetic_market", "trade_distances"]
