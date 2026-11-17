"""Every experiment from the project plan, in one command.

    python experiments/run_experiments.py             # 5 seeds per config
    python experiments/run_experiments.py --seeds 3    # faster

Each configuration is run on several synthetic markets (different seeds) and
the reported numbers are the mean across seeds, with the spread across seeds
next to it. One seed is one story; five seeds is evidence.

Outputs
-------
results/calibration.txt   the fitted k
results/*.csv             one row per configuration (means and std devs)
results/summary.md        markdown tables, the ones quoted in README.md
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import statistics as stats
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from mmsim import (  # noqa: E402
    BacktestConfig,
    Backtester,
    Metrics,
    OrderBook,
    SyntheticConfig,
    Trade,
    build_synthetic_market,
    fit_k,
)
from mmsim.synthetic import trade_distances  # noqa: E402
from mmsim.strategies import AvellanedaStoikov, FixedSpreadStrategy  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS: tuple[int, ...] = (7, 11, 23, 42, 99)
NAIVE_HALF_SPREAD = 0.04
AS_DEFAULTS = dict(gamma=0.25, horizon=120.0)


# ----------------------------------------------------------------------
# markets
# ----------------------------------------------------------------------
MARKET_BUILDERS: dict[str, Callable[[int], SyntheticConfig]] = {
    # the default: one calm hour, then a volatility spike
    "default": lambda seed: SyntheticConfig(seed=seed),
    # calm all the way through (sigma_volatile never kicks in)
    "calm": lambda seed: SyntheticConfig(seed=seed, sigma_volatile=0.012, regime_change_ts=1e12),
    # volatile from the start
    "volatile": lambda seed: SyntheticConfig(seed=seed, sigma=0.040, sigma_volatile=0.040),
    # a violent spike after 30 minutes, used for the sigma-lag experiment
    "spike": lambda seed: SyntheticConfig(seed=seed, sigma=0.008, sigma_volatile=0.055),
}

_market_cache: dict[tuple[str, int], object] = {}


def market(name: str, seed: int):
    key = (name, seed)
    if key not in _market_cache:
        _market_cache[key] = build_synthetic_market(MARKET_BUILDERS[name](seed))
    return _market_cache[key]


def mid_series(mkt) -> tuple[list[float], list[float]]:
    """Rebuild the mid path once per market (used for k calibration)."""
    book = OrderBook()
    ts: list[float] = []
    mids: list[float] = []
    for ev in mkt.events:
        if isinstance(ev, Trade):
            continue
        book.apply(ev)
        m = book.mid
        if m is not None and (not mids or mids[-1] != m):
            ts.append(ev.ts)
            mids.append(m)
    return ts, mids


def calibrate_k(mkt) -> float | None:
    ts, mids = mid_series(mkt)

    def mid_at(t: float) -> float:
        i = bisect.bisect_right(ts, t) - 1
        return mids[max(i, 0)]

    distances = trade_distances(mkt.trades, mid_at)
    return fit_k(distances, bucket_width=0.01, max_distance=0.25, min_count=20)


# ----------------------------------------------------------------------
# running / aggregating
# ----------------------------------------------------------------------
@dataclass
class ConfigResult:
    params: dict[str, object]
    metrics: list[Metrics] = field(default_factory=list)


def evaluate(
    params: dict,
    strategy_factory: Callable[[], object],
    market_name: str,
    seeds: Sequence[int],
    config: BacktestConfig | None = None,
) -> ConfigResult:
    res = ConfigResult(params=dict(params))
    for seed in seeds:
        mkt = market(market_name, seed)
        strategy = strategy_factory()
        res.metrics.append(Backtester(strategy, config or BacktestConfig()).run(mkt.events).report())
    return res


SCALARS = [
    "total_pnl",
    "spread_pnl",
    "inventory_pnl",
    "fee_pnl",
    "sharpe",
    "max_drawdown",
    "fills",
    "fill_rate",
    "quote_uptime",
    "inventory_std",
    "max_abs_inventory",
    "avg_quoted_spread_bps",
    "pnl_per_fill",
]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return stats.fmean(values) if values else float("nan")


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return stats.pstdev(values) if len(values) > 1 else 0.0


def summarize(result: ConfigResult) -> dict:
    row: dict[str, object] = dict(result.params)
    row["n_seeds"] = len(result.metrics)
    for name in SCALARS:
        vals = [getattr(m, name) for m in result.metrics]
        row[f"{name}_mean"] = _mean(vals)
        row[f"{name}_std"] = _std(vals)
    horizons = sorted({h for m in result.metrics for h in m.markouts})
    for h in horizons:
        vals = [m.markouts.get(h, float("nan")) for m in result.metrics]
        row[f"markout_{h}_mean"] = _mean(vals)
        row[f"markout_{h}_std"] = _std(vals)
    return row


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def md_table(rows: Sequence[dict], columns: Sequence[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(title for title, _, _ in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, rule]
    for row in rows:
        cells = []
        for _, key, fmt in columns:
            value = row.get(key, "")
            if isinstance(value, (int, float)):
                if isinstance(value, float) and math.isnan(value):
                    cells.append("n/a")
                else:
                    cells.append(fmt.format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


PNL_COLS = [
    ("config", "label", "{}"),
    ("PnL $", "total_pnl_mean", "{:+.2f}"),
    ("PnL sd", "total_pnl_std", "{:.2f}"),
    ("spread $", "spread_pnl_mean", "{:+.2f}"),
    ("inv $", "inventory_pnl_mean", "{:+.2f}"),
    ("fills", "fills_mean", "{:.0f}"),
    ("uptime", "quote_uptime_mean", "{:.2f}"),
    ("inv sd", "inventory_std_mean", "{:.2f}"),
    ("maxDD $", "max_drawdown_mean", "{:.2f}"),
    ("spr bps", "avg_quoted_spread_bps_mean", "{:.1f}"),
]


# ----------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------
def exp_baseline(seeds: Sequence[int], k: float | None) -> list[dict]:
    rows = [
        summarize(
            evaluate(
                {"label": "naive fixed 8bps", "strategy": "naive"},
                lambda: FixedSpreadStrategy(half_spread=NAIVE_HALF_SPREAD),
                "default",
                seeds,
            )
        ),
        summarize(
            evaluate(
                {"label": "Avellaneda-Stoikov", "strategy": "avellaneda_stoikov"},
                lambda: AvellanedaStoikov(k=k, **AS_DEFAULTS),
                "default",
                seeds,
            )
        ),
    ]
    return rows


def exp_gamma(seeds, k) -> list[dict]:
    rows = []
    for gamma in (0.05, 0.1, 0.25, 0.5, 1.0):
        rows.append(
            summarize(
                evaluate(
                    {"label": f"gamma={gamma:g}", "gamma": gamma},
                    lambda g=gamma: AvellanedaStoikov(gamma=g, k=k, horizon=AS_DEFAULTS["horizon"]),
                    "default",
                    seeds,
                )
            )
        )
    return rows


def exp_latency(seeds, k) -> list[dict]:
    rows = []
    for latency in (0.0, 0.01, 0.02, 0.05, 0.10):
        rows.append(
            summarize(
                evaluate(
                    {"label": f"{latency * 1000:.0f} ms", "latency_ms": latency * 1000},
                    lambda: AvellanedaStoikov(k=k, **AS_DEFAULTS),
                    "default",
                    seeds,
                    BacktestConfig(latency=latency),
                )
            )
        )
    return rows


def exp_fill_model(seeds, k) -> list[dict]:
    rows = []
    for name, factory in (
        ("naive", lambda: FixedSpreadStrategy(half_spread=NAIVE_HALF_SPREAD)),
        ("AS", lambda: AvellanedaStoikov(k=k, **AS_DEFAULTS)),
    ):
        for model, queue in (("simple", False), ("queue", True)):
            rows.append(
                summarize(
                    evaluate(
                        {"label": f"{name} / {model}", "strategy": name, "fill_model": model},
                        factory,
                        "default",
                        seeds,
                        BacktestConfig(queue_model=queue),
                    )
                )
            )
    return rows


def exp_regimes(seeds, k) -> list[dict]:
    rows = []
    for regime in ("calm", "default", "volatile"):
        for name, factory in (
            ("naive", lambda: FixedSpreadStrategy(half_spread=NAIVE_HALF_SPREAD)),
            ("AS", lambda: AvellanedaStoikov(k=k, **AS_DEFAULTS)),
        ):
            rows.append(
                summarize(
                    evaluate(
                        {"label": f"{name} / {regime}", "strategy": name, "regime": regime},
                        factory,
                        regime,
                        seeds,
                    )
                )
            )
    return rows


def exp_sigma_lag(seeds, k) -> list[dict]:
    rows = []
    for window, lag in ((30, 0), (120, 0), (120, 30), (600, 0), (600, 30)):
        rows.append(
            summarize(
                evaluate(
                    {
                        "label": f"window={window}s lag={lag}s",
                        "vol_window_s": window,
                        "vol_lag_s": lag,
                    },
                    lambda w=window, l=lag: AvellanedaStoikov(
                        k=k, **AS_DEFAULTS, vol_window=w, vol_lag=l
                    ),
                    "spike",
                    seeds,
                )
            )
        )
    return rows


def exp_ofi(seeds, k) -> list[dict]:
    rows = []
    for weight in (0, 1, 3, 10):
        rows.append(
            summarize(
                evaluate(
                    {"label": f"ofi={weight:g} bps", "ofi_weight": weight},
                    lambda w=weight: AvellanedaStoikov(k=k, **AS_DEFAULTS, ofi_weight=w),
                    "spike",
                    seeds,
                )
            )
        )
    return rows


# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="how many synthetic markets per config")
    args = parser.parse_args()
    seeds = SEEDS[: max(1, args.seeds)]

    t0 = time.time()
    mkt = market("default", seeds[0])
    k = calibrate_k(mkt)
    k_for_strategy = k if k else 26.5
    print(f"calibrated k = {k} (using {k_for_strategy})")

    sections: list[tuple[str, list[dict]]] = [
        ("1. Baseline vs naive", exp_baseline(seeds, k_for_strategy)),
        ("2. Risk aversion gamma", exp_gamma(seeds, k_for_strategy)),
        ("3. Latency", exp_latency(seeds, k_for_strategy)),
        ("4. Simple vs queue fills", exp_fill_model(seeds, k_for_strategy)),
        ("5. Calm vs volatile", exp_regimes(seeds, k_for_strategy)),
        ("6. Volatility spike and a lagging sigma", exp_sigma_lag(seeds, k_for_strategy)),
        ("7. Order-flow imbalance signal", exp_ofi(seeds, k_for_strategy)),
    ]

    names = [
        "baseline",
        "gamma",
        "latency",
        "fill_model",
        "regimes",
        "sigma_lag",
        "ofi",
    ]

    md: list[str] = [
        "# Experiment results",
        "",
        f"Synthetic market, seeds {list(seeds)}. Values are means across seeds;",
        "`PnL sd` is the spread of total PnL across seeds (model risk, not sampling error).",
        "",
        f"Fitted fill-intensity decay: **k = {k:.2f}** per price unit.",
        "",
    ]

    for (title, rows), name in zip(sections, names):
        write_csv(RESULTS / f"{name}.csv", rows)
        md.append(f"## {title}")
        md.append("")
        md.append(md_table(rows, PNL_COLS))
        md.append("")

        print(f"\n{title}")
        print(md_table(rows, PNL_COLS))

    with (RESULTS / "summary.md").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with (RESULTS / "calibration.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"fit_k from trade distances on the default market: {k:.4f}\n")

    print(f"\nwrote {RESULTS}/summary.md ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
