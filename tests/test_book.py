"""Phase 1 checks: does the book reconstruct correctly?"""

import pytest

from mmsim.book import ASK, BID, BookDelta, BookSnapshot, OrderBook, round_to_tick
from mmsim.synthetic import SyntheticConfig, build_synthetic_market


def test_snapshot_sets_top_of_book():
    book = OrderBook()
    book.apply(
        BookSnapshot(
            ts=1.0,
            bids=((99.5, 3.0), (99.0, 1.0)),
            asks=((100.5, 2.0), (101.0, 4.0)),
        )
    )
    assert book.best_bid == 99.5
    assert book.best_ask == 100.5
    assert book.mid == 100.0
    assert book.spread == 1.0
    assert not book.is_crossed()


def test_deltas_add_change_and_remove_levels():
    book = OrderBook()
    book.apply(BookDelta(ts=1.0, side=BID, price=99.0, size=2.0))
    book.apply(BookDelta(ts=1.1, side=ASK, price=100.0, size=2.0))
    assert (book.best_bid, book.best_ask) == (99.0, 100.0)

    # better bid arrives
    book.apply(BookDelta(ts=1.2, side=BID, price=99.5, size=1.0))
    assert book.best_bid == 99.5

    # size change on an existing level
    book.apply(BookDelta(ts=1.3, side=BID, price=99.5, size=4.0))
    assert book.size_at(BID, 99.5) == 4.0

    # size 0 removes the level, best bid falls back
    book.apply(BookDelta(ts=1.4, side=BID, price=99.5, size=0.0))
    assert book.best_bid == 99.0
    assert book.size_at(BID, 99.5) == 0.0


def test_large_synthetic_feed_never_crosses():
    market = build_synthetic_market(SyntheticConfig(duration=120.0, dt=0.5, seed=3))
    book = OrderBook()
    mid_before = None
    for ev in market.events:
        if isinstance(ev, BookSnapshot):
            book.apply(ev)
            assert not book.is_crossed()
            assert book.mid is not None
            if mid_before is not None:
                # the mid does not teleport: at most a few ticks per step
                assert abs(book.mid - mid_before) < 0.5
            mid_before = book.mid


def test_delta_stream_reconstructs_the_snapshot():
    """Simulate an exchange sending only changes; the rebuilt book must match."""
    snapshots = [
        BookSnapshot(0.0, ((99.0, 1.0), (98.0, 2.0)), ((100.0, 1.5), (101.0, 2.0))),
        BookSnapshot(1.0, ((99.5, 1.0), (99.0, 0.5)), ((100.0, 2.0),)),
    ]
    reference = OrderBook()
    reference.apply(snapshots[-1])

    rebuilt = OrderBook()
    rebuilt.apply(snapshots[0])
    # diff the two snapshots into deltas, the way a real feed would
    deltas = []
    for side, old, new in (
        (BID, snapshots[0].bids, snapshots[1].bids),
        (ASK, snapshots[0].asks, snapshots[1].asks),
    ):
        old_map, new_map = dict(old), dict(new)
        for price in set(old_map) | set(new_map):
            if old_map.get(price) != new_map.get(price):
                deltas.append(BookDelta(1.0, side, price, new_map.get(price, 0.0)))
    for d in deltas:
        rebuilt.apply(d)

    assert rebuilt.best_bid == reference.best_bid
    assert rebuilt.best_ask == reference.best_ask
    assert rebuilt.depth(BID) == reference.depth(BID)
    assert rebuilt.depth(ASK) == reference.depth(ASK)


def test_round_to_tick():
    assert round_to_tick(100.004, 0.01) == pytest.approx(100.0)
    assert round_to_tick(100.006, 0.01) == pytest.approx(100.01)
    assert round_to_tick(100.019, 0.01, "down") == pytest.approx(100.01)
    assert round_to_tick(100.011, 0.01, "up") == pytest.approx(100.02)


def test_imbalance_extremes():
    book = OrderBook()
    book.apply(BookSnapshot(0, ((1.0, 10.0),), ((2.0, 0.0),)))
    assert book.imbalance() == pytest.approx(1.0)
