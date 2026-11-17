# Market Making Simulator

A market-making backtest engine in Python: rebuild an L2 book, quote both sides,
get filled in a queue, and explain where the PnL actually came from.

Built from `plan.md`. It runs out of the box on a **synthetic market** (no
downloads, deterministic, known parameters) and can also replay real Binance
trades with a static book.

```
naive fixed-spread           PnL  +6.62   (sd across seeds 20.48)   max DD 17.91
Avellaneda-Stoikov           PnL  +6.66   (sd across seeds  0.83)   max DD  0.75
```

Same average PnL. One of them is a directional bet that happens to break even.

---

## Quickstart

```bash
python -m pytest -q                        # 32 tests, <1s
python experiments/run_experiments.py      # all experiments, ~40s
```

Everything is standard library except `pytest` (and only for the tests).

```python
import sys; sys.path.insert(0, "src")
from mmsim import Backtester, BacktestConfig, SyntheticConfig, build_synthetic_market
from mmsim.strategies import AvellanedaStoikov, FixedSpreadStrategy

market = build_synthetic_market(SyntheticConfig(seed=7))     # 1 hour, calm then volatile
result = Backtester(AvellanedaStoikov(), BacktestConfig()).run(market.events)

print(result.report().row())
```

`WALKTHROUGH.md` explains every concept, formula and file in this repo. Read it
before an interview.

---

## What it does, phase by phase

| Phase | What | Where |
|---|---|---|
| 1 | Rebuild the book from snapshots and price-level deltas | `src/mmsim/book.py`, `data.py` |
| 2 | Simulated venue: your orders, latency, fills, fees | `exchange.py`, `fills.py` |
| 3 | Naive fixed-spread baseline | `strategies/naive.py` |
| 4 | Avellaneda-Stoikov, with `sigma` from past data and `k` fitted from trades | `strategies/avellaneda_stoikov.py`, `calibration.py` |
| 5 | Latency, fees/rebates, inventory limits, queue-position fills | `exchange.py`, `risk.py` |
| 6 | PnL decomposition, markouts, inventory stats, parameter sweeps | `metrics.py`, `experiments/` |
| 7 | Synthetic market with known parameters | `synthetic.py` |

## Results

5 synthetic markets (different seeds), 1 hour each, 1 unit size, 20 ms latency,
0.5 bps maker rebate, queue fill model. Numbers are means across seeds.

| | naive fixed-spread | Avellaneda-Stoikov |
|---|---|---|
| Total PnL | +6.62 | +6.66 |
| PnL sd across seeds | **20.48** | **0.83** |
| Spread capture | +7.86 | +7.49 |
| Inventory PnL | −2.18 | −1.82 |
| Fills | 222 | 231 |
| Fill rate (fills / orders) | 1.5% | 1.2% |
| Quote uptime | 0.70 | 0.99 |
| Inventory sd | 2.75 | 0.99 |
| Max drawdown | 17.91 | 0.75 |
| Avg quoted spread | 8.4 bps | 10.2 bps |
| Markout 1s / 10s / 60s | −0.18 / −0.63 / −1.22 bps | −0.24 / −0.61 / −0.02 bps |
| Sharpe (1-min, annualised) | 41 | 464 |

Read that as: the naive MM spends 30% of its time not quoting (it is pinned at
its inventory limit), its PnL is a directional position, and it loses twice as
much at 60 s markouts. A-S holds inventory near zero, quotes 99% of the time,
and its PnL is spread capture rather than luck.

**Sharpe warning.** 464 is not a real-world number. It is 60 observations of a
toy PnL path; treat it as a way to rank strategies, never as a claim. Real
market making Sharpe ratios live in the single digits, and real PnL has costs
this model does not have (adverse selection from real flow, fees tiers,
liquidation slippage).

### The experiments (each one is a talking point)

**1. Risk aversion `gamma`** — total PnL barely changes (+6.7 → +4.3), but
inventory sd falls 2.31 → 0.48 and quoted spread widens 8.1 → 18.2 bps.
`gamma` buys stability, not returns.

| gamma | PnL | PnL sd | fills | inv sd | max DD | spread |
|---|---|---|---|---|---|---|
| 0.05 | +5.62 | 3.74 | 287 | 2.31 | 3.98 | 8.1 |
| 0.25 | +6.66 | 0.83 | 231 | 0.99 | 0.75 | 10.2 |
| 1.00 | +4.25 | 0.59 | 144 | 0.48 | 0.51 | 18.2 |

**2. Latency** — 0 → 100 ms costs 33% of PnL (+7.42 → +4.97) and 26% of fills,
with no change in quoted spread: you are simply filled worse. This is why
backtests that ignore latency flatter themselves.

**3. Simple vs queue fills** — the same A-S strategy makes **+66.06** with the
optimistic fill model and **+6.66** with queue position. About 90% of that PnL
was fake. Fill counts: 1685 vs 231.

**4. Calm vs volatile** — A-S widens from 8.1 to 12.5 bps, fills drop 317 → 149,
PnL 9.45 → 4.40, drawdown stays ~0.7. The naive MM's drawdown jumps 8.33 →
21.70 because it keeps quoting the same 8 bps into a market that moves 5x more.

**5. Sigma estimate lag** — on a market where volatility jumps 7x after 30
minutes, a stale estimator (600 s window, 30 s lag) keeps quotes too tight
through the transition: 60 s markouts −0.69 bps vs +0.20 bps with a 120 s
window, drawdown 1.90 vs 0.77. The cost is real but modest, because the
inventory skew already provides some protection.

**6. Order-flow imbalance** — a small weight helps the markouts (−0.20 → −0.04
bps at 10 s) but does not move total PnL (+6.68 → +6.76, inside noise). A large
weight (10 bps) overtrades: 642 fills, spread capture collapses to +0.92.
Honest conclusion: the signal is real in this market but too weak to pay for
itself at a 1-unit size.

Full tables: `results/summary.md`, raw numbers: `results/*.csv`.

## Assumptions and limitations

These are the ones an interviewer will probe. They are deliberate.

| Area | Choice here | Limitation |
|---|---|---|
| Fill model (simple) | Fill when a trade prints at/through your price | Optimistic: ignores queue position, inflates PnL ~10x |
| Fill model (queue) | Join back of queue at your price; fill after the volume ahead trades | Cannot see hidden orders or cancels ahead of you |
| Market impact | None (1 unit of a 8-unit book level) | Breaks down immediately if you scale size |
| Latency | Fixed one-way delay, default 20 ms, swept 0–100 ms | Real latency is variable and asymmetric |
| Cancels | Take effect after the latency delay; the order keeps trading meanwhile | Real cancels can fail or race with fills |
| Fees | Fixed 0.5 bps maker rebate | Real tiers depend on volume and venue |
| Valuation | Mark inventory to mid | Hides the cost of liquidating at the bid/ask |
| Other participants | Flow is exogenous; it does not react to your quotes | Unrealistic at size |
| Data | Synthetic GBM + informed flow, or Binance trades with a *static* book | No real L2 history, no real queue dynamics |
| Sharpe | 1-minute PnL increments, annualised | 1 hour of toy data: noisy and optimistic |

Not modelled at all: funding/borrow, exchange outages, self-trade prevention,
tick-size regimes, order-priority by size, multiple assets, hedging.

## Common mistakes (from `plan.md`) and where they are handled

| Mistake | Handling |
|---|---|
| Reporting the optimistic fill model | Both models ship; experiment 3 quantifies the difference (10x) |
| Look-ahead in volatility | `RollingVolatility.estimate(end_ts)` only sees past samples; `tests/test_strategy.py` proves it |
| Ignoring latency | Latency is in the exchange, not the strategy; swept 0–100 ms |
| Tick-level Sharpe | Sharpe is computed from 1-minute samples (`BacktestConfig.sharpe_period`) |
| Trusting the book blindly | `tests/test_book.py` rebuilds a book from deltas and diffs it against snapshots; the backtest counts crossed books |
| PnL that does not add up | `decomposition_error` is ~1e-13 on every run (exact identity, see walkthrough) |

## Repo map

```
src/mmsim/
  book.py           OrderBook, BookDelta, BookSnapshot, Trade, tick maths
  data.py           CSV loaders (plain + Binance aggTrades) -> event stream
  synthetic.py      market generator with known parameters
  exchange.py       orders, latency, queues, cash/inventory/fees
  fills.py          SimpleFillModel, QueueFillModel
  strategies/       base.py (interface, vol, OFI), naive.py, avellaneda_stoikov.py
  risk.py           inventory limits, drawdown kill switch
  backtest.py       event loop -> BacktestResult -> Metrics
  metrics.py        PnL decomposition, markouts, Sharpe, drawdown
  calibration.py    fit k from trade distances
tests/              32 tests: book, fills, strategy maths, PnL identity, data
experiments/        run_experiments.py -> results/*.csv + summary.md
scripts/            download_data.py (optional real Binance sample)
data/               gitignored; see data/README.md for formats
```

## What I would do next

1. **Real L2 replay** — record your own book or buy LOBSTER history; the queue
   model only becomes interesting with real depth.
2. **Order-flow imbalance as a separate alpha** rather than a skew term, fitted
   on data instead of swept by hand.
3. **Multi-asset / hedging** — the inventory problem is much more tractable when
   you can hedge the exposure elsewhere.
4. **Execution-cost model** — liquidation at the book instead of at the mid, so
   the PnL stops flattering itself.
