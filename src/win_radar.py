"""
WIN Radar - mathematical layer on top of orderflow.py.

This module does NOT modify the original orderflow engine.

It measures:
    - aggressive buying/selling efficiency
    - price response to aggressive flow
    - absorption / loss of efficiency
    - movement state
    - radar state

The calculations are intentionally transparent.
No trading order or automatic buy/sell recommendation is generated.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_array(values, dtype=float):
    """Convert a bar field to a numpy array safely."""
    return np.asarray(values, dtype=dtype)


def _rolling_median(values, window=20):
    """
    Rolling median including the current bar.

    A small and transparent implementation is used so the module
    does not require pandas.
    """
    values = _safe_array(values)
    out = np.full(values.size, np.nan, dtype=float)

    if values.size == 0:
        return out

    window = max(int(window), 1)

    for i in range(values.size):
        start = max(0, i - window + 1)
        sample = values[start:i + 1]
        finite = sample[np.isfinite(sample)]

        if finite.size:
            out[i] = np.median(finite)

    return out


def _safe_ratio(numerator, denominator, floor=1e-9):
    """Safe division preserving the sign of the numerator."""
    denominator = np.maximum(np.abs(denominator), floor)
    return numerator / denominator


# ---------------------------------------------------------------------------
# 1. Efficiency
# ---------------------------------------------------------------------------

def efficiency(b, window=20):
    """
    Measure price displacement produced by aggressive volume.

    Buy efficiency:
        upward price movement / aggressive buying volume.

    Sell efficiency:
        downward price movement / aggressive selling volume.

    The raw values are also normalized against their recent median,
    producing an easier-to-read efficiency score.

    Interpretation:

        > 1.0
            stronger response than the recent reference.

        around 1.0
            normal response.

        < 1.0
            weaker response.

        near 0
            aggressive volume produced little directional movement.
    """

    close = _safe_array(b["close"])
    ask = _safe_array(b["ask_volume"])
    bid = _safe_array(b["bid_volume"])

    n = close.size

    if n == 0:
        empty = np.array([], dtype=float)
        return {
            "buy_efficiency": empty,
            "sell_efficiency": empty,
            "buy_raw_efficiency": empty,
            "sell_raw_efficiency": empty,
            "net_efficiency": empty,
            "price_move": empty,
        }

    price_move = np.zeros(n, dtype=float)

    if n > 1:
        price_move[1:] = close[1:] - close[:-1]

    # Only movement in the corresponding direction counts.
    buy_move = np.maximum(price_move, 0.0)
    sell_move = np.maximum(-price_move, 0.0)

    # Price displacement per aggressive contract.
    buy_raw = _safe_ratio(buy_move, ask)
    sell_raw = _safe_ratio(sell_move, bid)

    # Recent typical response.
    buy_ref = _rolling_median(buy_raw, window)
    sell_ref = _rolling_median(sell_raw, window)

    # Ignore zero references when normalising.
    buy_ref = np.maximum(buy_ref, 1e-9)
    sell_ref = np.maximum(sell_ref, 1e-9)

    buy_eff = buy_raw / buy_ref
    sell_eff = sell_raw / sell_ref

    # When there is no directional movement, efficiency is zero.
    buy_eff = np.where(buy_move > 0, buy_eff, 0.0)
    sell_eff = np.where(sell_move > 0, sell_eff, 0.0)

    net = buy_eff - sell_eff

    return {
        "buy_efficiency": buy_eff,
        "sell_efficiency": sell_eff,
        "buy_raw_efficiency": buy_raw,
        "sell_raw_efficiency": sell_raw,
        "net_efficiency": net,
        "price_move": price_move,
    }


# ---------------------------------------------------------------------------
# 2. Absorption / loss of efficiency
# ---------------------------------------------------------------------------

def absorption_signal(
    b,
    window=20,
    aggression_threshold=1.5,
    displacement_threshold=0.35,
):
    """
    Detect high aggression with weak price response.

    Buy absorption:
        strong aggressive buying
        +
        weak upward price response.

    Sell absorption:
        strong aggressive selling
        +
        weak downward price response.

    This is intentionally different from orderflow.py's existing
    absorption calculation.
    """

    close = _safe_array(b["close"])
    ask = _safe_array(b["ask_volume"])
    bid = _safe_array(b["bid_volume"])

    n = close.size

    if n == 0:
        empty_bool = np.array([], dtype=bool)
        empty_float = np.array([], dtype=float)

        return {
            "buy_absorption": empty_bool,
            "sell_absorption": empty_bool,
            "buy_pressure": empty_float,
            "sell_pressure": empty_float,
            "response_ratio": empty_float,
            "buy_loss": empty_float,
            "sell_loss": empty_float,
            "score": empty_float,
        }

    price_move = np.zeros(n, dtype=float)

    if n > 1:
        price_move[1:] = close[1:] - close[:-1]

    buy_move = np.maximum(price_move, 0.0)
    sell_move = np.maximum(-price_move, 0.0)
    move_abs = np.abs(price_move)

    ask_ref = _rolling_median(ask, window)
    bid_ref = _rolling_median(bid, window)
    move_ref = _rolling_median(move_abs, window)

    ask_ref = np.maximum(ask_ref, 1e-9)
    bid_ref = np.maximum(bid_ref, 1e-9)
    move_ref = np.maximum(move_ref, 1e-9)

    # How unusual
