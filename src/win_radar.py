"""
WIN Radar - camada matemática sobre o orderflow.py.

Este módulo NÃO modifica o motor original.

Ele mede:
- eficiência de compra e venda
- resposta do preço ao fluxo agressivo
- absorção / perda de eficiência
- estado do movimento
- sinal do Radar

As análises são matemáticas e não geram ordens automáticas.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_array(values, dtype=float):
    """Converte um campo das barras para numpy com segurança."""
    return np.asarray(values, dtype=dtype)


def _rolling_median(values, window=20):
    """
    Calcula a mediana móvel incluindo a barra atual.
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
    """Divisão segura."""
    denominator = np.maximum(np.abs(denominator), floor)

    return numerator / denominator


# ---------------------------------------------------------------------------
# 1. Efficiency
# ---------------------------------------------------------------------------

def efficiency(b, window=20):
    """
    Mede quanto o preço se deslocou em relação ao volume agressivo.

    Compra:
        movimento para cima / volume agressor comprador.

    Venda:
        movimento para baixo / volume agressor vendedor.
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

    buy_move = np.maximum(price_move, 0.0)
    sell_move = np.maximum(-price_move, 0.0)

    buy_raw = _safe_ratio(buy_move, ask)
    sell_raw = _safe_ratio(sell_move, bid)

    buy_ref = _rolling_median(buy_raw, window)
    sell_ref = _rolling_median(sell_raw, window)

    buy_ref = np.maximum(buy_ref, 1e-9)
    sell_ref = np.maximum(sell_ref, 1e-9)

    buy_eff = buy_raw / buy_ref
    sell_eff = sell_raw / sell_ref

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
    Detecta forte agressão com pouca resposta do preço.

    Compra:
        muita agressão compradora
        +
        pouca alta do preço.

    Venda:
        muita agressão vendedora
        +
        pouca queda do preço.
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

    move_abs = np.abs(price_move)

    ask_ref = _rolling_median(ask, window)
    bid_ref = _rolling_median(bid, window)
    move_ref = _rolling_median(move_abs, window)

    ask_ref = np.maximum(ask_ref, 1e-9)
    bid_ref = np.maximum(bid_ref, 1e-9)
    move_ref = np.maximum(move_ref, 1e-9)

    # Quanto a agressão está acima do comportamento recente?
    buy_pressure = ask / ask_ref
    sell_pressure = bid / bid_ref

    # Quanto o preço se movimentou em relação ao normal recente?
    response_ratio = move_abs / move_ref

    # Perda de eficiência:
    # muita agressão + pouca resposta = perda de eficiência.
    buy_loss = buy_pressure / np.maximum(response_ratio, 0.25)
    sell_loss = sell_pressure / np.maximum(response_ratio, 0.25)

    # Possível absorção compradora.
    buy_absorption = (
        (buy_pressure >= aggression_threshold)
        & (response_ratio <= displacement_threshold)
        & (ask > bid)
    )

    # Possível absorção vendedora.
    sell_absorption = (
        (sell_pressure >= aggression_threshold)
        & (response_ratio <= displacement_threshold)
        & (bid > ask)
    )

    score = buy_loss - sell_loss

    return {
        "buy_absorption": buy_absorption,
        "sell_absorption": sell_absorption,
        "buy_pressure": buy_pressure,
        "sell_pressure": sell_pressure,
        "response_ratio": response_ratio,
        "buy_loss": buy_loss,
        "sell_loss": sell_loss,
        "score": score,
    }
    # ---------------------------------------------------------------------------
# 3. Movement state
# ---------------------------------------------------------------------------

def movement_state(b, lookback=3):
    """
    Classifica o movimento recente do preço.

        +1 = alta
         0 = neutro
        -1 = queda
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
    """Converte o estado numérico em texto."""

    state = np.asarray(state)

    return np.where(
        state > 0,
        "UP",
        np.where(
            state < 0,
            "DOWN",
            "NEUTRAL",
        ),
    )


# ---------------------------------------------------------------------------
# 4. Radar signal
# ---------------------------------------------------------------------------

def radar_signal(b, window=20, lookback=3):
    """
    Combina movimento, eficiência e absorção.
    """

    eff = efficiency(b, window=window)

    absorb = absorption_signal(
        b,
        window=window,
    )

    movement = movement_state(
        b,
        lookback=lookback,
    )

    n = len(movement)

    labels = np.full(
        n,
        "NEUTRAL",
        dtype=object,
    )
    for i in range(n):

        state = movement[i]

        buy_eff = eff["buy_efficiency"][i]
        sell_eff = eff["sell_efficiency"][i]

        buy_abs = absorb["buy_absorption"][i]
        sell_abs = absorb["sell_absorption"][i]

        if buy_abs or sell_abs:
            labels[i] = "ATTENTION"
            continue
        if state > 0 and buy_eff < 1.0:
            labels[i] = "LOSS_OF_EFFICIENCY"
            continue

        if state < 0 and sell_eff < 1.0:
            labels[i] = "LOSS_OF_EFFICIENCY"
            continue
        if state > 0 and buy_eff >
        sell_eff and buy_eff >= 1.5:
            labels[i] = "CONFIRMED_MOVE"
            continue

        if state < 0 and sell_eff >
        buy_eff and sell_eff >= 1.5:
            labels[i] ="CONFIRMED_MOVE"
            continue
        if state > 0 and buy_eff > 
        sell_eff:
            labels[i] = "CONTINUATION"
            continue

        if state < 0 and sell_eff >
        buy_eff:
            labels[i] = "CONTINUATION"
            continue
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
    Retorna somente o estado mais recente do Radar.

    Útil para futura integração com API ou interface.
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
            "buy_loss": 0.0,
            "sell_loss": 0.0,
        }

    i = -1

    return {
        "signal": str(radar["signal"][i]),
        "movement": str(radar["movement_label"][i]),

        "buy_efficiency": round(
            float(radar["buy_efficiency"][i]),
            4,
        ),

        "sell_efficiency": round(
            float(radar["sell_efficiency"][i]),
            4,
        ),

        "net_efficiency": round(
            float(radar["net_efficiency"][i]),
            4,
        ),

        "buy_absorption": bool(
            radar["buy_absorption"][i]
        ),

        "sell_absorption": bool(
            radar["sell_absorption"][i]
        ),

        "buy_loss": round(
            float(radar["buy_loss"][i]),
            4,
        ),
            

        "sell_loss": round(
            float(radar["sell_loss"]
[i]),
            4,
        ),
    }
      
        
