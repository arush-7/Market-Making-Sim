"""Performance measurement (plan Phase 6).

Every number a resume bullet or an interview answer might touch:

* total PnL and its decomposition: spread capture / inventory / fees
* Sharpe from equity sampled on a fixed grid (never tick by tick)
* markouts at 1s / 10s / 60s -> adverse selection
* inventory distribution and max drawdown
* fill rate and quote uptime

The decomposition is exact, not a heuristic. Writing ``m_f`` for the mid at
the moment of a fill and ``p_f`` for your fill price:

    spread capture = sum over fills of sign * (m_f - p_f) * q_f
    inventory PnL  = sum over time of inventory * (mid_t - mid_{t-1})
    fees           = -sum of fees paid (rebates count positive)

and ``total = spread + inventory + fees``. The identity holds because marking
to mid is the same thing as crediting the spread at the fill and the drift
afterwards. ``Metrics.decomposition_error`` checks it on every run.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev
from typing import Optional, Sequence

SECONDS_PER_YEAR = 365.0 * 24 * 3600


# ----------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------
def resample(
    ts: Sequence[float], values: Sequence[float], period: float = 1.0
) -> tuple[list[float], list[float]]:
    """Forward-fill ``values`` onto a regular grid of ``period`` seconds."""
    if len(ts) < 2 or period <= 0:
        return [], []
    t0, t1 = ts[0], ts[-1]
    n = int((t1 - t0) / period)
    out_ts: list[float] = []
    out_v: list[float] = []
    j = 0
    for k in range(n + 1):
        target = t0 + k * period
        while j + 1 < len(ts) and ts[j + 1] <= target:
            j += 1
        out_ts.append(target)
        out_v.append(values[j])
    return out_ts, out_v


def sharpe_ratio(
    ts: Sequence[float],
    equity: Sequence[float],
    period_seconds: float = 60.0,
    annualization: float = SECONDS_PER_YEAR,
) -> float:
    """Annualised Sharpe of PnL increments sampled every ``period_seconds``.

    PnL increments are in dollars; dividing by a constant capital would not
    change the ratio, so it is omitted. Compute this on 1-minute (or coarser)
    returns. Tick-by-tick Sharpes are meaningless noise and are the classic
    way to make a backtest look far better than it is.
    """
    _, eq = resample(ts, equity, period_seconds)
    if len(eq) < 4:
        return float("nan")
    rets = [b - a for a, b in zip(eq, eq[1:])]
    sd = pstdev(rets)
    if sd <= 1e-12:
        return float("nan")
    return mean(rets) / sd * math.sqrt(annualization / period_seconds)


def max_drawdown(equity: Sequence[float], capital: float) -> tuple[float, float]:
    """Peak-to-trough drawdown in dollars and in bps of capital."""
    peak = -math.inf
    worst = 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak - v > worst:
            worst = peak - v
    return worst, (worst / capital * 1e4 if capital else float("nan"))


def markouts(
    fills: Sequence,
    mid_ts: Sequence[float],
    mid: Sequence[float],
    horizons: Sequence[float] = (1.0, 10.0, 60.0),
) -> dict[str, float]:
    """Average mid move after each fill, in bps, signed so positive is good.

    For a buy, a positive markout means the mid rose after you bought.
    Consistently negative markouts mean you are being adversely selected:
    your fills are followed by the price moving against you.

    Fills too close to the end of the data are skipped (no future mid yet).
    """
    out: dict[str, float] = {}
    if not fills or not mid_ts:
        return {f"{h:g}s": float("nan") for h in horizons}
    last_ts = mid_ts[-1]
    for h in horizons:
        vals: list[float] = []
        for f in fills:
            target = f.ts + h
            if target > last_ts:
                continue
            i = bisect.bisect_right(mid_ts, target) - 1
            if i < 0 or f.mid <= 0:
                continue
            vals.append(f.sign * (mid[i] - f.mid) / f.mid * 1e4)
        out[f"{h:g}s"] = mean(vals) if vals else float("nan")
    return out


def realized_volatility(
    mid_ts: Sequence[float], mid: Sequence[float], sample_seconds: float = 1.0
) -> float:
    """Std of mid changes per ``sample_seconds``, in price units per sqrt(sec)."""
    _, vals = resample(mid_ts, mid, sample_seconds)
    if len(vals) < 3:
        return float("nan")
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    return pstdev(diffs) / math.sqrt(sample_seconds)


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
@dataclass
class Metrics:
    strategy: str = ""
    fill_model: str = ""
    duration_hours: float = float("nan")

    total_pnl: float = 0.0
    spread_pnl: float = 0.0
    inventory_pnl: float = 0.0
    fee_pnl: float = 0.0
    decomposition_error: float = 0.0

    sharpe: float = float("nan")
    max_drawdown: float = 0.0
    max_drawdown_bps: float = 0.0

    fills: int = 0
    fill_rate: float = float("nan")
    fills_per_hour: float = float("nan")
    quote_uptime: float = float("nan")

    avg_inventory: float = 0.0
    inventory_std: float = 0.0
    max_abs_inventory: float = 0.0
    final_inventory: float = 0.0

    avg_quoted_spread_bps: float = float("nan")
    pnl_per_fill: float = float("nan")
    markouts: dict[str, float] = field(default_factory=dict)

    def row(self) -> dict:
        """Flat dict for CSVs / tables (markouts get their own columns)."""
        out = {k: v for k, v in asdict(self).items() if k != "markouts"}
        out.update({f"markout_{k}": v for k, v in self.markouts.items()})
        return out


def compute_metrics(result) -> Metrics:
    """Turn a :class:`~mmsim.backtest.BacktestResult` into a report."""
    m = Metrics()
    m.strategy = result.strategy_name
    m.fill_model = result.fill_model
    m.duration_hours = result.duration / 3600.0

    m.total_pnl = result.total_pnl
    m.spread_pnl = result.spread_pnl
    m.inventory_pnl = result.inventory_pnl
    m.fee_pnl = result.fee_pnl
    m.decomposition_error = result.total_pnl - (
        result.spread_pnl + result.inventory_pnl + result.fee_pnl
    )

    m.sharpe = sharpe_ratio(result.eq_ts, result.equity, result.config.sharpe_period)
    m.max_drawdown, m.max_drawdown_bps = max_drawdown(result.equity, result.config.initial_capital)

    m.fills = len(result.fills)
    m.fill_rate = m.fills / result.orders_submitted if result.orders_submitted else float("nan")
    m.fills_per_hour = m.fills / m.duration_hours if m.duration_hours > 0 else float("nan")
    m.quote_uptime = (
        result.quote_uptime / result.duration if result.duration > 0 else float("nan")
    )

    if result.inventory:
        m.avg_inventory = mean(result.inventory)
        m.inventory_std = pstdev(result.inventory) if len(result.inventory) > 1 else 0.0
        m.max_abs_inventory = max(abs(v) for v in result.inventory)
        m.final_inventory = result.inventory[-1]

    m.avg_quoted_spread_bps = result.avg_quoted_spread_bps
    m.pnl_per_fill = m.total_pnl / m.fills if m.fills else float("nan")
    m.markouts = markouts(
        result.fills, result.mid_ts, result.mid, result.config.markout_horizons
    )
    return m


__all__ = [
    "Metrics",
    "compute_metrics",
    "markouts",
    "max_drawdown",
    "realized_volatility",
    "resample",
    "sharpe_ratio",
]
