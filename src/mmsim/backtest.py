"""The event loop (plan Phase 6).

One pass over the event stream. For every event:

1. the exchange activates any order whose latency has elapsed,
2. the event is applied (book update, or a tape print that may fill you),
3. the mid is recorded and inventory PnL is accumulated,
4. every ``quote_interval`` the strategy is asked for new quotes and the
   difference between "wanted" and "live" is cancelled / submitted.

The strategy only ever sees the book as of the current event, so there is no
look-ahead. Latency is applied by the exchange, not by the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .book import ASK, BID, OrderBook, Trade
from .exchange import Fill, SimulatedExchange
from .fills import make_fill_model
from .metrics import Metrics, compute_metrics
from .risk import RiskManager
from .strategies.base import Quotes, Strategy


@dataclass
class BacktestConfig:
    """Everything about the venue that is not the strategy."""

    latency: float = 0.02  # one-way, seconds
    cancel_latency: Optional[float] = None
    maker_fee_bps: float = -0.5  # negative = rebate you receive
    queue_model: bool = True  # True: queue fills, False: optimistic fills
    max_inventory: float = 10.0
    quote_interval: float = 0.25  # how often the strategy is polled
    sample_interval: float = 1.0  # equity curve resolution
    sharpe_period: float = 60.0  # Sharpe sampling, must be >> tick
    markout_horizons: tuple[float, ...] = (1.0, 10.0, 60.0)
    kill_drawdown: Optional[float] = None  # in dollars
    initial_capital: float = 100_000.0


@dataclass
class BacktestResult:
    strategy_name: str
    fill_model: str
    config: BacktestConfig
    fills: list[Fill]
    mid_ts: list[float]
    mid: list[float]
    eq_ts: list[float]
    equity: list[float]
    inventory_ts: list[float]
    inventory: list[float]
    spread_pnl: float
    inventory_pnl: float
    fee_pnl: float
    total_pnl: float
    orders_submitted: int
    quote_uptime: float
    duration: float
    avg_quoted_spread_bps: float
    crossed_books: int
    missing_mid: int
    stopped_early: bool = False

    def mid_at(self, ts: float) -> Optional[float]:
        import bisect

        i = bisect.bisect_right(self.mid_ts, ts) - 1
        return self.mid[i] if i >= 0 else None

    def report(self) -> Metrics:
        return compute_metrics(self)


class Backtester:
    def __init__(self, strategy: Strategy, config: Optional[BacktestConfig] = None, tick: float = 0.01):
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.tick = float(tick)

    # ------------------------------------------------------------------
    def run(self, events: Iterable[object]) -> BacktestResult:
        cfg = self.config
        events = list(events)
        if not events:
            raise ValueError("no events to simulate")

        self.strategy.reset()
        book = OrderBook()
        fill_model = make_fill_model("queue" if cfg.queue_model else "simple")
        exchange = SimulatedExchange(
            book,
            fill_model,
            latency=cfg.latency,
            cancel_latency=cfg.cancel_latency,
            maker_fee_bps=cfg.maker_fee_bps,
        )
        risk = RiskManager(cfg.max_inventory, cfg.kill_drawdown, cfg.initial_capital)

        t_start, t_end = events[0].ts, events[-1].ts
        mid_ts: list[float] = []
        mid_vals: list[float] = []
        eq_ts: list[float] = [t_start]
        eq_vals: list[float] = [0.0]
        inv_ts: list[float] = [t_start]
        inv_vals: list[float] = [0.0]

        spread_pnl = 0.0
        inventory_pnl = 0.0
        uptime = 0.0
        quoted_spreads: list[float] = []
        crossed_books = 0
        missing_mid = 0
        stopped = False

        last_mid: Optional[float] = None
        prev_ts = t_start
        next_quote = t_start
        next_sample = t_start + cfg.sample_interval

        for ev in events:
            ts = ev.ts
            exchange.advance(ts)

            if isinstance(ev, Trade):
                for flush in exchange.process_trade(ev):
                    spread_pnl += flush.sign * (flush.mid - flush.price) * flush.size
                    self.strategy.on_fill(flush)
                self.strategy.on_trade(ev)
            else:
                book.apply(ev)
                if book.is_crossed():
                    crossed_books += 1

            mid = book.mid
            if mid is None:
                missing_mid += 1
            elif mid != last_mid:
                if last_mid is not None:
                    inventory_pnl += exchange.inventory * (mid - last_mid)
                last_mid = mid
                mid_ts.append(ts)
                mid_vals.append(mid)

            # poll the strategy and reconcile the book
            if not stopped and ts >= next_quote:
                next_quote = ts + cfg.quote_interval
                target = self.strategy.quotes(ts, book, exchange.inventory)
                if target.bid is not None and target.ask is not None and mid:
                    quoted_spreads.append((target.ask - target.bid) / mid * 1e4)
                self._reconcile(exchange, target, risk, ts)

            # quote uptime over the elapsed slice
            has_bid, has_ask = exchange.has_quotes()
            if has_bid and has_ask:
                uptime += ts - prev_ts
            prev_ts = ts

            # mark-to-market sample on a fixed grid
            if ts >= next_sample:
                while next_sample <= ts:
                    next_sample += cfg.sample_interval
                equity = exchange.equity(mid)
                eq_ts.append(ts)
                eq_vals.append(equity)
                inv_ts.append(ts)
                inv_vals.append(exchange.inventory)
                if risk.update(equity) and not stopped:
                    stopped = True
                    exchange.cancel_all(ts)

        final_mid = last_mid
        final_equity = exchange.equity(final_mid)
        eq_ts.append(t_end)
        eq_vals.append(final_equity)
        inv_ts.append(t_end)
        inv_vals.append(exchange.inventory)

        fees = exchange.fees
        return BacktestResult(
            strategy_name=getattr(self.strategy, "name", type(self.strategy).__name__),
            fill_model=exchange.fill_model.name,
            config=cfg,
            fills=exchange.fills,
            mid_ts=mid_ts,
            mid=mid_vals,
            eq_ts=eq_ts,
            equity=eq_vals,
            inventory_ts=inv_ts,
            inventory=inv_vals,
            spread_pnl=spread_pnl,
            inventory_pnl=inventory_pnl,
            fee_pnl=-fees,
            total_pnl=final_equity - 0.0,
            orders_submitted=exchange.orders_submitted,
            quote_uptime=uptime,
            duration=t_end - t_start,
            avg_quoted_spread_bps=(
                sum(quoted_spreads) / len(quoted_spreads) if quoted_spreads else float("nan")
            ),
            crossed_books=crossed_books,
            missing_mid=missing_mid,
            stopped_early=stopped,
        )

    # ------------------------------------------------------------------
    def _reconcile(
        self,
        exchange: SimulatedExchange,
        target: Quotes,
        risk: RiskManager,
        ts: float,
    ) -> None:
        """Make the venue's live orders match ``target``.

        Orders already at the right price are left alone (that preserves queue
        position). A price change is a cancel/replace, and both legs take
        latency, which is exactly when you get picked off.
        """
        live: dict[str, list] = {BID: [], ASK: []}
        for order in exchange.live.values():
            live[order.side].append(order)
        for order in exchange.pending_orders():
            live[order.side].append(order)

        for side, price in ((BID, target.bid), (ASK, target.ask)):
            orders = live[side]
            if price is None or not risk.allows(side, target.size, exchange.inventory):
                for order in orders:
                    exchange.cancel(order, ts)
                continue

            keep = None
            for order in orders:
                if keep is None and abs(order.price - price) < self.tick * 0.5:
                    keep = order
                else:
                    exchange.cancel(order, ts)
            if keep is None:
                exchange.submit(side, price, target.size, ts)


def run_backtest(
    events: Iterable[object],
    strategy: Strategy,
    config: Optional[BacktestConfig] = None,
    tick: float = 0.01,
) -> BacktestResult:
    return Backtester(strategy, config, tick).run(events)


__all__ = ["BacktestConfig", "BacktestResult", "Backtester", "run_backtest"]
