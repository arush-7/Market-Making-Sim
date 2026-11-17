"""Phase 1 checks for the CSV loaders."""

import pytest

from mmsim.data import (
    load_binance_agg_trades,
    load_depth_csv,
    load_events,
    load_trades_csv,
    merge_events,
)


def test_depth_csv(tmp_path):
    path = tmp_path / "depth.csv"
    path.write_text("timestamp,side,price,size\n1.5,bid,99.0,2\n1.5,ask,101.0,3\n2.0,bid,99.0,0\n")
    deltas = load_depth_csv(str(path))
    assert len(deltas) == 3
    assert deltas[0].ts == pytest.approx(1.5)
    assert deltas[0].side == "bid"
    assert deltas[2].size == 0.0


def test_trades_csv_and_ms_conversion(tmp_path):
    path = tmp_path / "trades.csv"
    path.write_text("timestamp,price,size,aggressor\n1500,100.0,0.5,sell\n", encoding="utf-8")
    trades = load_trades_csv(str(path), ts_unit="ms")
    assert trades[0].ts == pytest.approx(1.5)
    assert trades[0].aggressor == "sell"


def test_load_events_auto_detects_binance_agg_trades(tmp_path):
    path = tmp_path / "agg.csv"
    path.write_text(
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "1,100.0,0.5,1,1,1700000000123,true\n"
    )
    events = load_events(trades_csv=str(path))
    assert len(events) == 1
    assert events[0].aggressor == "sell"


def test_binance_agg_trades_maps_taker_side(tmp_path):
    path = tmp_path / "agg.csv"
    path.write_text(
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "1,100.0,0.5,1,1,1700000000123,true\n"
        "2,100.1,0.2,2,2,1700000001123,false\n"
    )
    trades = load_binance_agg_trades(str(path))
    assert trades[0].ts == pytest.approx(1700000000.123)
    assert trades[0].aggressor == "sell"  # buyer was the maker -> taker sold
    assert trades[1].aggressor == "buy"


def test_merge_events_orders_by_time(tmp_path):
    depth = tmp_path / "d.csv"
    depth.write_text("timestamp,side,price,size\n2,bid,99,1\n")
    trades = tmp_path / "t.csv"
    trades.write_text("timestamp,price,size,aggressor\n1,99,1,sell\n")
    events = load_events(str(depth), str(trades))
    assert [e.ts for e in events] == [1.0, 2.0]

    other = load_trades_csv(str(trades))
    assert merge_events(other, events)[0].ts == 1.0
