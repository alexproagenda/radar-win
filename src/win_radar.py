"""
WIN Radar - mathematical layer on top of orderflow.py.

This module does NOT modify the original orderflow engine.

It measures:
    - aggressive buying/selling efficiency
    - displacement efficiency
    - absorption / loss of efficiency
    - previous movement state
    - radar state

The first version is intentionally transparent.
No trading order or buy/sell recommendation is generated here.
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
    """Simple rolling median without external dependencies."""
    values = _safe_array(values)
    out = np.full(values.size, np.nan, dtype=float)

    if values.size == 0:
        return out

    window = max(int(window), 1)

    for i in range(values.size):
        start = max(0, i - window + 1)
        out[i] = np.median(values[start:i + 1])

    return out


def _normalise(value, reference, floor=1e-9):
    """Return value/reference while protecting against zero."""
    return value / max(abs(reference), floor)


# ---------------------------------------------------------------------------
# 1. Efficiency
# ---------------------------------------------------------------------------

def efficiency(b, window=20):
    """
    Measure how efficiently aggressive flow moves price.

    Positive result:
        aggressive buying is producing upward displacement.

    Negative result:
        aggressive selling is producing downward displacement.

    Near zero:
        aggression exists but price is not responding efficiently.

    Returns arrays so every bar can be inspected.
    """

    close = _safe_array(b["close"])
    delta = _safe_array(b["delta"])

    n = close.size

    if n == 0:
        return {
            "buy_efficiency": np.array([]),
            "sell_efficiency": np.array([]),
            "net_efficiency": np.array([]),
            "price_move": np.array([]),
        }

    # Bar-to-bar price displacement.
    price_move = np.zeros(n, dtype=float)
    if n > 1:
        price_move[1:] = close[1:] - close[:-1]

    # Aggressive volume by side.
    ask = _safe_array(b["ask_volume"])
    bid = _safe_array(b["bid_volume"])

    # Rolling references make the measure adapt to different activity levels.
    ask_ref = _rolling_median(ask, window)
    bid_ref = _rolling_median(bid, window)

    move_abs = np.abs(price_move)

    # A positive price response to aggressive buying.
    buy_response = np.maximum(price_move, 0.0)
    sell_response = np.maximum(-price_move, 0.0)

    buy_eff = np.zeros(n, dtype=float)
    sell_eff = np.zeros(n, dtype=float)

    for i in range(n):
        buy_eff[i] = _normalise(
            buy_response[i],
            ask_ref[i],
        ) * _normalise(
            ask[i],
            ask_ref[i],
        )

        sell_eff[i] = _normalise(
            sell_response[i],
            bid_ref[i],
        ) * _normalise(
            bid[i],
            bid_ref[i],
        )

    # Directional net efficiency.
    net = buy_eff - sell_eff

    return {
        "buy_efficiency": buy_eff,
        "sell_efficiency": sell_eff,
        "net_efficiency": net,
        "price_move": price_move,
        "delta": delta,
    }


# ---------------------------------------------------------------------------
# 2. Absorption / loss of efficiency
# ---------------------------------------------------------------------------

def absorption_signal(b, window=20, aggression_threshold=1.5,
                      displacement_threshold=0.35):
    """
    Detect situations where aggression is large but price displacement
    is unusually small.

    This is deliberately different from orderflow.py's existing absorption
    flag. Here we specifically compare aggression against price response.

    Returns:
        buy_absorption
        sell_absorption
        buy_loss
        sell_loss
        score
    """

    close = _safe_array(b["close"])
    ask = _safe_array(b["ask_volume"])
    bid = _safe_array(b["bid_volume"])

    n = close.size

    if n == 0:
        return {
            "buy_absorption": np.array([], dtype=bool),
            "sell_absorption": np.array([], dtype=bool),
            "buy_loss": np.array([], dtype=float),
            "sell_loss": np.array([], dtype=float),
            "score": np.array([], dtype=float),
        }

    price_move = np.zeros(n, dtype=float)
    if n > 1:
        price_move[1:] = close[1:] - close[:-1]

    move_abs = np.abs(price_move)

    ask_ref = _rolling_median(ask, window)
    bid_ref = _rolling_median(bid, window)
    move_ref = _rolling_median(move_abs, window)

    buy_pressure = np.zeros(n, dtype=float)
    sell_pressure = np.zeros(n, dtype=float)

    buy_absorption = np.zeros(n, dtype=bool)
    sell_absorption = np.zeros(n, dtype=bool)

    buy_loss = np.zeros(n, dtype=float)
    sell_loss = np.zeros(n, dtype=float)

    for i in range(n):
        buy_pressure[i] = _normalise(ask[i], ask_ref[i])
        sell_pressure[i] = _normalise(bid[i], bid_ref[i])

        displacement = _normalise(
            move_abs[i],
            move_ref[i],
        )

        # High aggression + weak price response.
        buy_loss[i] = buy_pressure[i] / max(displacement, 0.10)
        sell_loss[i] = sell_pressure[i] / max(displacement, 0.10)

        buy_absorption[i] = (
            buy_pressure[i] >= aggression_threshold
            and displacement <= displacement_threshold
            and ask[i] > bid[i]
        )

        sell_absorption[i] = (
            sell_pressure[i] >= aggression_threshold
            and displacement <= displacement_threshold
            and bid[i] > ask[i]
        )

    score = buy_loss - sell_loss

    return {
        "buy_absorption": buy_absorption,
        "sell_absorption": sell_absorption,
        "buy_loss": buy_loss,
        "sell_loss": sell_loss,
        "score": score,
    }


# ---------------------------------------------------------------------------
# 3. Movement state
# ---------------------------------------------------------------------------

def movement_state(b, lookback=3):
    """
    Classify the recent price movement.

    Returns:
        +1 = UP
         0 = NEUTRAL
        -1 = DOWN
    """

    close = _safe_array(b["close"])
    n = close.size

    state = np.zeros(n, dtype=np.int8)

    if n <= lookback:
        return state

    for i in range(lookback, n):
        move = close[i] - close[i - lookback]

        if move > 0:
            state[i] = 1
        elif move < 0:
            state[i] = -1

    return state


def movement_labels(state):
    """Convert numeric movement state to readable labels."""
    state = np.asarray(state)

    return np.where(
        state > 0,
        "UP",
        np.where(state < 0, "DOWN", "NEUTRAL"),
    )


# ---------------------------------------------------------------------------
# 4. Radar signal
# ---------------------------------------------------------------------------

def radar_signal(b, window=20, lookback=3):
    """
    Combine efficiency, absorption and movement.

    Possible states:

        CONTINUATION
        ATTENTION
        LOSS_OF_EFFICIENCY
        CONFIRMED_MOVE
        NEUTRAL

    Important:
        These are analytical states, not trading orders.
    """

    eff = efficiency(b, window=window)
    absorb = absorption_signal(b, window=window)
    movement = movement_state(b, lookback=lookback)

    net = eff["net_efficiency"]

    n = len(net)

    labels = np.full(n, "NEUTRAL", dtype=object)

    for i in range(n):
        state = movement[i]

        buy_eff = eff["buy_efficiency"][i]
        sell_eff = eff["sell_efficiency"][i]

        buy_abs = absorb["buy_absorption"][i]
        sell_abs = absorb["sell_absorption"][i]

        # ---------------------------------------------------------------
        # Confirmed directional movement
        # ---------------------------------------------------------------

        if state > 0 and buy_eff > sell_eff and buy_eff > 1.0:
            labels[i] = "CONFIRMED_MOVE"

        elif state < 0 and sell_eff > buy_eff and sell_eff > 1.0:
            labels[i] = "CONFIRMED_MOVE"

        # ---------------------------------------------------------------
        # Loss of efficiency / absorption
        # ---------------------------------------------------------------

        if state > 0 and (buy_abs or buy_eff < sell_eff):
            labels[i] = "LOSS_OF_EFFICIENCY"

        elif state < 0 and (sell_abs or sell_eff < buy_eff):
            labels[i] = "LOSS_OF_EFFICIENCY"

        # ---------------------------------------------------------------
        # Continuation
        # ---------------------------------------------------------------

        if state > 0 and buy_eff > sell_eff and not buy_abs:
            labels[i] = "CONTINUATION"

        elif state < 0 and sell_eff > buy_eff and not sell_abs:
            labels[i] = "CONTINUATION"

        # ---------------------------------------------------------------
        # Attention
        # ---------------------------------------------------------------

        if buy_abs or sell_abs:
            labels[i] = "ATTENTION"

    return {
        "movement": movement,
        "movement_label": movement_labels(movement),

        "buy_efficiency": eff["buy_efficiency"],
        "sell_efficiency": eff["sell_efficiency"],
        "net_efficiency": eff["net_efficiency"],

        "buy_absorption": absorb["buy_absorption"],
        "sell_absorption": absorb["sell_absorption"],

        "buy_loss": absorb["buy_loss"],
        "sell_loss": absorb["sell_loss"],

        "signal": labels,
    }


# ---------------------------------------------------------------------------
# 5. Latest radar snapshot
# ---------------------------------------------------------------------------

def latest(b, window=20, lookback=3):
    """
    Return only the latest Radar state.

    Useful for API/UI integration.
    """

    radar = radar_signal(
        b,
        window=window,
        lookback=lookback,
    )

    if len(radar["signal"]) == 0:
        return {
            "signal": "NEUTRAL",
            "movement": "NEUTRAL",
            "buy_efficiency": 0.0,
            "sell_efficiency": 0.0,
            "net_efficiency": 0.0,
            "buy_absorption": False,
            "sell_absorption": False,
        }

    i = -1

    return {
        "signal": str(radar["signal"][i]),
        "movement": str(radar["movement_label"][i]),
        "buy_efficiency": round(float(radar["buy_efficiency"][i]), 4),
        "sell_efficiency": round(float(radar["sell_efficiency"][i]), 4),
        "net_efficiency": round(float(radar["net_efficiency"][i]), 4),
        "buy_absorption": bool(radar["buy_absorption"][i]),
        "sell_absorption": bool(radar["sell_absorption"][i]),
        "buy_loss": round(float(radar["buy_loss"][i]), 4),
        "sell_loss": round(float(radar["sell_loss"][i]), 4),
    }
