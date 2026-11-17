"""Phase 3-4 checks: strategy maths, no look-ahead, k calibration."""

import math
import random

import pytest

from mmsim.book import BID, BookSnapshot, OrderBook
from mmsim.calibration import fit_k
from mmsim.strategies import AvellanedaStoikov, FixedSpreadStrategy, RollingVolatility


def make_book(mid=100.0, half_spread=0.01):
    book = OrderBook()
    book.apply(
        BookSnapshot(
            ts=0.0,
            bids=((mid - half_spread, 5.0), (mid - 2 * half_spread, 5.0)),
            asks=((mid + half_spread, 5.0), (mid + 2 * half_spread, 5.0)),
        )
    )
    return book


def test_reservation_price_skews_against_inventory():
    s = AvellanedaStoikov(gamma=0.5, k=20.0, horizon=300.0)
    flat = s.reservation_price(100.0, 0.0, 0.01, 300.0)
    long_pos = s.reservation_price(100.0, 5.0, 0.01, 300.0)
    short_pos = s.reservation_price(100.0, -5.0, 0.01, 300.0)

    assert flat == pytest.approx(100.0)
    assert long_pos < flat < short_pos
    # the skew is exactly q * gamma * sigma^2 * tau
    assert flat - long_pos == pytest.approx(5.0 * 0.5 * 0.01**2 * 300.0)


def test_optimal_spread_matches_the_formula():
    s = AvellanedaStoikov(gamma=0.5, k=20.0)
    sigma, tau = 0.012, 300.0
    expected = 0.5 * (0.5 * sigma**2 * tau + (2 / 0.5) * math.log(1 + 0.5 / 20.0))
    assert s.optimal_half_spread(sigma, tau) == pytest.approx(expected)


def test_spread_widens_with_volatility():
    s = AvellanedaStoikov(gamma=0.5, k=20.0)
    calm = s.optimal_half_spread(0.01, 300.0)
    wild = s.optimal_half_spread(0.05, 300.0)
    assert wild > calm


def test_quotes_are_skewed_when_long_and_respect_inventory_limit():
    s = AvellanedaStoikov(gamma=0.5, k=20.0, max_inventory=10.0)
    book = make_book()
    flat = s.quotes(1.0, book, inventory=0.0)
    long_pos = s.quotes(1.0, book, inventory=6.0)

    assert long_pos.bid < flat.bid and long_pos.ask < flat.ask
    assert long_pos.bid < long_pos.ask

    maxed = s.quotes(1.0, book, inventory=10.0)
    assert maxed.bid is None  # cannot add to a maxed-out long
    assert maxed.ask is not None


def test_naive_baseline_quotes_symmetric_around_mid():
    s = FixedSpreadStrategy(half_spread=0.04)
    q = s.quotes(0.0, make_book(mid=100.0), inventory=0.0)
    assert q.bid == pytest.approx(99.96)
    assert q.ask == pytest.approx(100.04)
    assert q.ask - q.bid == pytest.approx(0.08)


def test_volatility_estimate_has_no_lookahead():
    vol = RollingVolatility(window_seconds=60.0, sample_interval=0.5)
    mid = 100.0
    for i in range(40):
        mid += 0.01 if i % 2 == 0 else -0.01
        vol.update(i * 0.5, mid)

    past_estimate = vol.estimate(end_ts=10.0)
    for i in range(40, 80):  # a later volatility explosion
        mid += 1.0 if i % 2 == 0 else -1.0
        vol.update(i * 0.5, mid)

    assert vol.estimate(end_ts=10.0) == pytest.approx(past_estimate)


def test_volatility_recovers_known_sigma():
    rng = random.Random(0)
    sigma = 0.02  # price per sqrt(second)
    vol = RollingVolatility(window_seconds=600.0, sample_interval=1.0)
    mid = 100.0
    for i in range(601):
        mid += rng.gauss(0.0, sigma)
        vol.update(i * 1.0, mid)
    est = vol.estimate()
    assert 0.7 * sigma < est < 1.3 * sigma


def test_fit_k_recovers_the_generating_decay():
    rng = random.Random(1)
    true_k = 25.0
    distances = [rng.expovariate(true_k) for _ in range(200_000)]
    fitted = fit_k(distances, bucket_width=0.01, max_distance=0.2, min_count=50)
    assert fitted is not None
    assert abs(fitted - true_k) / true_k < 0.15
