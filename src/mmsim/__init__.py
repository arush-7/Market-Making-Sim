"""mmsim - a small market-making simulator.

Layout follows the project plan:

    book.py       L2 book reconstruction from updates
    data.py       CSV feeds -> normalised event stream
    synthetic.py  a market with known parameters (no downloads needed)
    exchange.py   your orders, latency, fills, fees
    fills.py      swappable fill models (simple / queue)
    strategies/   naive fixed-spread baseline + Avellaneda-Stoikov
    risk.py       inventory limits, kill switch
    backtest.py   the event loop
    metrics.py    PnL, markouts, inventory stats, Sharpe
    calibration.py fitting k from observed trade distances
"""

from .book import ASK, BID, BookDelta, BookSnapshot, OrderBook, Trade, round_to_tick
from .backtest import BacktestConfig, BacktestResult, Backtester, run_backtest
from .calibration import fit_k
from .exchange import Fill, Order, SimulatedExchange
from .fills import FillModel, QueueFillModel, SimpleFillModel, make_fill_model
from .metrics import Metrics, compute_metrics
from .risk import RiskManager
from .synthetic import SyntheticConfig, SyntheticMarket, build_synthetic_market

__all__ = [
    "ASK",
    "BID",
    "BookDelta",
    "BookSnapshot",
    "Trade",
    "OrderBook",
    "round_to_tick",
    "Order",
    "Fill",
    "SimulatedExchange",
    "FillModel",
    "SimpleFillModel",
    "QueueFillModel",
    "make_fill_model",
    "RiskManager",
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "Metrics",
    "compute_metrics",
    "fit_k",
    "SyntheticConfig",
    "SyntheticMarket",
    "build_synthetic_market",
]

__version__ = "0.1.0"
