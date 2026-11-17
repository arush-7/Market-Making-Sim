"""Phase 2 checks: order lifecycle, latency and both fill models."""

import pytest

from mmsim.book import ASK, BID, BookSnapshot, OrderBook, Trade
from mmsim.exchange import SimulatedExchange
from mmsim.fills import QueueFillModel, SimpleFillModel


def make_book(bid_size=5.0):
    book = OrderBook()
    book.apply(
        BookSnapshot(
            ts=0.0,
            bids=((99.99, bid_size), (99.98, 5.0)),
            asks=((100.01, 5.0), (100.02, 5.0)),
        )
    )
    return book


def test_order_is_not_live_before_latency():
    book = make_book()
    ex = SimulatedExchange(book, SimpleFillModel(), latency=0.05)
    ex.submit(BID, 99.99, 1.0, ts=0.0)
    ex.advance(0.04)
    assert not ex.live  # still in flight
    ex.advance(0.05)
    assert len(ex.live) == 1


def test_simple_fill_needs_the_trade_to_reach_your_price():
    book = make_book()
    ex = SimulatedExchange(book, SimpleFillModel(), latency=0.0)
    ex.submit(BID, 99.99, 1.0, ts=0.0)
    ex.advance(0.0)

    fills = ex.process_trade(Trade(0.1, 100.00, 2.0, "sell"))  # above your bid
    assert fills == []

    fills = ex.process_trade(Trade(0.2, 99.99, 2.0, "sell"))  # at your price
    assert len(fills) == 1
    assert fills[0].size == 1.0  # capped by your order, not the trade


def test_partial_fills_accumulate_and_complete():
    book = make_book()
    ex = SimulatedExchange(book, SimpleFillModel(), latency=0.0)
    ex.submit(ASK, 100.01, 2.0, ts=0.0)
    ex.advance(0.0)

    f1 = ex.process_trade(Trade(0.1, 100.02, 0.5, "buy"))
    assert f1[0].size == 0.5
    f2 = ex.process_trade(Trade(0.2, 100.01, 3.0, "buy"))
    assert f2[0].size == pytest.approx(1.5)  # remaining of the 2.0 order
    assert not ex.live  # fully filled -> gone
    assert ex.inventory == pytest.approx(-2.0)


def test_queue_model_waits_for_the_volume_ahead():
    book = make_book(bid_size=5.0)
    ex = SimulatedExchange(book, QueueFillModel(), latency=0.0)
    ex.submit(BID, 99.99, 1.0, ts=0.0)
    ex.advance(0.0)
    assert ex.live[1].queue_ahead == 5.0

    # trades eat the queue first, you get nothing yet
    assert ex.process_trade(Trade(0.1, 99.99, 3.0, "sell")) == []
    assert ex.live[1].queue_ahead == pytest.approx(2.0)

    # this one clears the rest of the queue and fills your order
    fills = ex.process_trade(Trade(0.2, 99.99, 5.0, "sell"))
    assert len(fills) == 1
    assert fills[0].size == pytest.approx(1.0)


def test_queue_model_does_not_fill_on_a_trade_that_never_reaches_you():
    book = make_book(bid_size=0.0)
    ex = SimulatedExchange(book, QueueFillModel(), latency=0.0)
    ex.submit(BID, 99.99, 1.0, ts=0.0)
    ex.advance(0.0)
    assert ex.process_trade(Trade(0.1, 100.00, 100.0, "buy")) == []


def test_cancel_takes_effect_after_latency():
    book = make_book()
    ex = SimulatedExchange(book, SimpleFillModel(), latency=0.0, cancel_latency=0.10)
    order = ex.submit(BID, 99.99, 1.0, ts=0.0)
    ex.advance(0.0)

    ex.cancel(order, ts=0.0)
    ex.advance(0.05)
    assert order.oid in ex.live  # still exposed

    ex.advance(0.10)
    assert order.oid not in ex.live


def test_cash_inventory_and_fees():
    book = make_book()
    # 1 bp maker rebate: we are *paid* to provide liquidity
    ex = SimulatedExchange(book, SimpleFillModel(), latency=0.0, maker_fee_bps=-1.0)
    ex.submit(BID, 100.00, 10.0, ts=0.0)
    ex.advance(0.0)
    ex.process_trade(Trade(0.1, 100.00, 10.0, "sell"))

    assert ex.inventory == pytest.approx(10.0)
    assert ex.cash == pytest.approx(-1000.0)  # trade cash only, no fees
    assert ex.fees == pytest.approx(-0.1)  # negative fee = rebate received
    # marking at the mid (100.0) the position is worth 1000, so we are up the rebate
    assert ex.equity(100.0) == pytest.approx(0.1)
