"""Phase 6 checks: the PnL identity, markout signs, Sharpe, drawdown."""

import random

import pytest

from mmsim.backtest import BacktestConfig, Backtester
from mmsim.book import BID
from mmsim.exchange import Fill
from mmsim.metrics import max_drawdown, markouts, sharpe_ratio
from mmsim.strategies import AvellanedaStoikov, FixedSpreadStrategy
from mmsim.synthetic import SyntheticConfig, build_synthetic_market


@pytest.fixture(scope="module")
def small_market():
    return build_synthetic_market(SyntheticConfig(duration=600.0, dt=0.5, seed=11))


@pytest.mark.parametrize(
    "strategy",
    [
        FixedSpreadStrategy(half_spread=0.03, size=1.0),
        AvellanedaStoikov(gamma=0.5, k=20.0, horizon=300.0, size=1.0),
    ],
)
def test_pnl_decomposition_is_exact(small_market, strategy):
    cfg = BacktestConfig(queue_model=True, quote_interval=0.5)
    result = Backtester(strategy, cfg).run(small_market.events)
    report = result.report()

    assert result.fills, "expected some fills in the test market"
    assert abs(report.decomposition_error) < 1e-6
    # and the total is what cash + mark-to-mid says it is
    assert report.total_pnl == pytest.approx(
        report.spread_pnl + report.inventory_pnl + report.fee_pnl, abs=1e-6
    )


def test_markout_sign_convention():
    fill = Fill(ts=10.0, oid=1, side=BID, price=99.99, size=1.0, mid=100.0, fee=0.0)
    mid_ts = [0.0, 10.0, 60.0]
    mid = [100.0, 100.0, 100.10]  # mid rises after the buy -> good for us
    result = markouts([fill], mid_ts, mid, horizons=(50.0,))
    assert result["50s"] == pytest.approx(10.0, abs=1e-9)  # +10 bps


def test_sharpe_is_nan_for_a_flat_curve():
    ts = [float(i) for i in range(0, 601)]
    equity = [0.0] * len(ts)
    assert sharpe_ratio(ts, equity, 60.0) != sharpe_ratio(ts, equity, 60.0)  # NaN


def test_sharpe_is_finite_for_a_noisy_curve():
    rng = random.Random(0)
    ts = [float(i) for i in range(0, 3601)]
    equity = []
    value = 0.0
    for _ in ts:
        value += 0.002 + rng.gauss(0.0, 0.01)
        equity.append(value)
    result = sharpe_ratio(ts, equity, 60.0)
    assert result == result  # not NaN
    assert result > 0


def test_max_drawdown():
    equity = [0.0, 5.0, 2.0, 4.0, -1.0, 0.0]
    dd, dd_bps = max_drawdown(equity, capital=100.0)
    assert dd == pytest.approx(6.0)  # peak 5 -> trough -1
    assert dd_bps == pytest.approx(600.0)
