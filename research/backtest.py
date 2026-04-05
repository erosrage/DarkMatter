import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from strategy import get_grid


def backtest(prices, sweep=0.2, steps=8, order_size=20, reset_interval=None,
             ma_period=20, atr_period=14, pyramid=True,
             circuit_breaker=0.20, atr_multiplier=8.0,
             min_sweep=0.05, max_sweep=0.50):
    """
    Backtest the grid strategy on a price series.

    Optimizations:
      1. Trend filter     — buys only placed when price > MA (ma_period).
      2. ATR-based sweep  — grid width scales with volatility (atr_period, atr_multiplier).
      3. Pyramid sizing   — deeper buy levels use proportionally larger order sizes.
      4. Stale orders     — handled implicitly: grid resets cancel unfilled levels.
      5. Circuit breaker  — halts all trading if portfolio drops > circuit_breaker below start.

    Set any period to None/0 to disable that feature.
    reset_interval: candles between grid resets (None = never reset after first set).
    """
    INITIAL_CASH = 100_000
    cash = INITIAL_CASH
    holdings = 0.0
    profit = 0.0
    halted = False

    executed_buys = []
    executed_sells = []

    buys, sells = get_grid(prices[0], sweep, steps)

    for i, price in enumerate(prices):

        # ── 5. Circuit breaker ────────────────────────────────────────────────
        if circuit_breaker:
            portfolio_value = cash + holdings * price
            if portfolio_value < INITIAL_CASH * (1 - circuit_breaker):
                halted = True
                break

        # ── 1. Trend filter ───────────────────────────────────────────────────
        if ma_period and i >= ma_period:
            ma = sum(prices[i - ma_period:i]) / ma_period
            trend_up = price > ma
        else:
            trend_up = True  # not enough history yet — allow trades

        # ── 2. ATR-based sweep ────────────────────────────────────────────────
        if atr_period and i >= atr_period:
            ranges = [abs(prices[j] - prices[j - 1]) for j in range(i - atr_period + 1, i + 1)]
            atr = sum(ranges) / atr_period
            dynamic_sweep = max(min_sweep, min(max_sweep, (atr / price) * atr_multiplier)) if price > 0 else sweep
        else:
            dynamic_sweep = sweep

        # ── 4. Stale order cancellation / grid reset ──────────────────────────
        if i == 0:
            buys, sells = get_grid(price, dynamic_sweep, steps)
        elif reset_interval and i % reset_interval == 0:
            buys, sells = get_grid(price, dynamic_sweep, steps)

        # ── Buy levels ────────────────────────────────────────────────────────
        if trend_up:
            for idx, b in enumerate(buys[:]):
                # ── 3. Pyramid sizing: deeper levels get larger orders ─────
                size = order_size * (idx + 1) if pyramid else order_size
                if price <= b and cash >= b * size:
                    cash -= b * size
                    holdings += size
                    executed_buys.append((i, price))
                    buys.remove(b)

        # ── Sell levels ───────────────────────────────────────────────────────
        for s in sells[:]:
            if price >= s and holdings >= order_size:
                cash += s * order_size
                holdings -= order_size
                profit += s * order_size
                executed_sells.append((i, price))
                sells.remove(s)

    final_value = cash + holdings * prices[-1]

    return {
        "final_value": final_value,
        "profit": profit,
        "cash": cash,
        "holdings": holdings,
        "buy_points": executed_buys,
        "sell_points": executed_sells,
        "halted": halted,
    }


def plot_trades(prices, buy_points, sell_points):
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 6))
    plt.plot(prices, label="Price", color="blue")

    if buy_points:
        x_buys, y_buys = zip(*buy_points)
        plt.scatter(x_buys, y_buys, color="green", marker="^", s=100, label="Buys")
    if sell_points:
        x_sells, y_sells = zip(*sell_points)
        plt.scatter(x_sells, y_sells, color="red", marker="v", s=100, label="Sells")

    plt.title("Backtest Trades on Price Chart")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "eth_usd_2y.csv")
    df = pd.read_csv(filename)

    if "close" not in df.columns:
        raise ValueError("CSV must contain a 'close' column")

    prices = df["close"].tolist()
    result = backtest(prices)

    print("\n=== Backtest Results ===")
    print(f"Final Value: {result['final_value']:.2f}")
    print(f"Profit:      {result['profit']:.2f}")
    print(f"Cash:        {result['cash']:.2f}")
    print(f"Holdings:    {result['holdings']:.4f}")
    print(f"Halted:      {result['halted']}")

    plot_trades(prices, result["buy_points"], result["sell_points"])
