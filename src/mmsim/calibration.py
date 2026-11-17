"""Fit ``k`` from observed trade distances (plan Phase 4).

Avellaneda-Stoikov assumes your fill intensity decays exponentially with the
distance ``delta`` between your quote and the mid:

    lambda(delta) = A * exp(-k * delta)

so ``log(lambda)`` is linear in ``delta``. We approximate ``lambda`` by
counting how many trades printed at each distance bucket (trades per unit
time, per bucket, cancels out in the slope), then least-squares fit a line.
``k = -slope``.

This is a crude estimator on purpose: one bucket width, no time weighting. Say
so if you are asked. It is still far better than guessing ``k``.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional


def fit_k(
    distances: Iterable[float],
    bucket_width: float = 0.01,
    max_distance: float = 0.25,
    min_count: int = 20,
) -> Optional[float]:
    """Least-squares ``k`` from a list of |trade price - mid| observations.

    Returns ``None`` if there is not enough data to fit.
    """
    buckets: dict[int, int] = {}
    for d in distances:
        if d < 0 or d > max_distance:
            continue
        idx = int(d / bucket_width)
        buckets[idx] = buckets.get(idx, 0) + 1

    points = [
        ((idx + 0.5) * bucket_width, math.log(count))
        for idx, count in sorted(buckets.items())
        if count >= min_count
    ]
    if len(points) < 3:
        return None

    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None

    slope = (n * sxy - sx * sy) / denom
    if slope >= 0:
        return None  # intensity is not decaying: model assumption does not fit
    return -slope


__all__ = ["fit_k"]
