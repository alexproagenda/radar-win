import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from win_radar import efficiency, absorption_signal


def make_bars(close, ask, bid):
    close = np.array(close, dtype=float)
    ask = np.array(ask, dtype=float)
    bid = np.array(bid, dtype=float)

    return {
        "close": close,
        "ask_volume": ask,
        "bid_volume": bid,
        "delta": ask - bid,
    }


# ---------------------------------------------------------
# TESTE 1 — muita compra + preço sobe
# Esperamos boa eficiência compradora
# ---------------------------------------------------------

bars_buy = make_bars(
    close=[100, 101, 102, 104, 106, 108],
    ask=[100, 120, 150, 180, 200, 220],
    bid=[50, 50, 50, 50, 50, 50],
)

result_buy = efficiency(bars_buy)

print("\nTESTE 1 — COMPRA EFICIENTE")
print("Buy efficiency:", result_buy["buy_efficiency"])
print("Sell efficiency:", result_buy["sell_efficiency"])


# ---------------------------------------------------------
# TESTE 2 — muita compra + preço quase não anda
# Esperamos perda de eficiência / absorção
# ---------------------------------------------------------

bars_absorption = make_bars(
    close=[100, 100, 100, 100.1, 100, 100],
    ask=[100, 120, 150, 200, 250, 300],
    bid=[50, 50, 50, 50, 50, 50],
)

result_absorption = absorption_signal(bars_absorption)

print("\nTESTE 2 — POSSÍVEL ABSORÇÃO DE COMPRA")
print("Buy absorption:", result_absorption["buy_absorption"])
print("Buy loss:", result_absorption["buy_loss"])


# ---------------------------------------------------------
# TESTE 3 — muita venda + preço cai
# Esperamos boa eficiência vendedora
# ---------------------------------------------------------

bars_sell = make_bars(
    close=[100, 99, 98, 96, 94, 92],
    ask=[50, 50, 50, 50, 50, 50],
    bid=[100, 120, 150, 180, 200, 220],
)

result_sell = efficiency(bars_sell)

print("\nTESTE 3 — VENDA EFICIENTE")
print("Buy efficiency:", result_sell["buy_efficiency"])
print("Sell efficiency:", result_sell["sell_efficiency"])


print("\nTESTES CRIADOS COM SUCESSO.")
