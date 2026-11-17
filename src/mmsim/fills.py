"""Fill models (plan Phase 2 and Phase 5).

A fill model answers one question: *a trade just printed, how much of this
resting order of mine did it fill?*  Models are swappable so the same backtest
can be run with an optimistic and a conservative assumption.

    simple : fill whenever a trade prints at or through your price.
             Ignores that other orders are ahead of you -> optimistic.
    queue  : you join the back of the queue at your price level; fills only
             happen after the volume resting ahead of you has traded.
             -> pessimistic, and closer to how a real passive order behaves.

Neither model claims to know about hidden orders, cancels ahead of you, or
market impact. Those are stated limitations in the README.
"""

from __future__ import annotations

from .book import ASK, BID


class FillModel:
    """Interface: decide how much one trade fills one order."""

    name = "base"

    def on_trade(self, order, trade, book) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def on_order_live(self, order, book) -> None:
        """Called when an order becomes live; lets a model set queue position."""


class SimpleFillModel(FillModel):
    """Fill when a trade prints at or through your price.

    Optimistic: it assumes you are first in line at your price level.
    """

    name = "simple"

    def on_trade(self, order, trade, book) -> float:
        if order.remaining <= 0:
            return 0.0
        if order.side == BID:
            if trade.price > order.price:
                return 0.0
        else:
            if trade.price < order.price:
                return 0.0
        return min(order.remaining, trade.size)


class QueueFillModel(FillModel):
    """Join the back of the queue; fill once the volume ahead is consumed.

    ``order.queue_ahead`` is set from the book when the order becomes live.
    Each trade at (or through) that price consumes queue ahead of you first.
    """

    name = "queue"

    def on_order_live(self, order, book) -> None:
        order.queue_ahead = book.size_at(order.side, order.price)

    def on_trade(self, order, trade, book) -> float:
        if order.remaining <= 0:
            return 0.0
        # A trade must print at your price, or through it (sweeping past you).
        if order.side == BID:
            if trade.price > order.price:
                return 0.0
        else:
            if trade.price < order.price:
                return 0.0

        volume = trade.size
        if order.queue_ahead > 0:
            consumed = min(order.queue_ahead, volume)
            order.queue_ahead -= consumed
            volume -= consumed
        if volume <= 1e-12:
            return 0.0
        return min(order.remaining, volume)


FILL_MODELS = {
    "simple": SimpleFillModel,
    "queue": QueueFillModel,
}


def make_fill_model(name: str) -> FillModel:
    try:
        return FILL_MODELS[name]()
    except KeyError:
        raise ValueError(f"unknown fill model {name!r}; choose from {sorted(FILL_MODELS)}") from None
