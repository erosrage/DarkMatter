import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from strategy import get_sd_levels


def backtest(prices, steps=3, std_multiplier=1.0, window=168, order_size=20,
             reset_interval=None, ma_period=168, atr_period=48, pyramid=True,
             circuit_breaker=0.20):
    """
    Backtest the standard-deviation grid strategy on a price series.

    Levels are placed at mean ± N*σ of a rolling `window`-candle lookback.

    steps          : number of buy/sell levels on each side
    std_multiplier : spacing between levels in σ units
    window         : rolling lookback for mean/σ (in candles)
    order_size     : base order size (units)
    reset_interval : candles between level resets (None = reset every candle)
    ma_period      : trend filter — buys only when price > MA (None = off)
    atr_period     : unused directly (levels are SD-based) but kept for API parity
    pyramid        : deeper buy levels use larger order sizes
    circuit_breaker: halt if portfolio drops this fraction below starting value
    """
    INITIAL_CASH = 100_000
    cash         = INITIAL_CASH
    holdings     = 0.0
    profit       = 0.0
    halted       = False

    executed_buys  = []
    executed_sells = []

    buys, sells = [], []

    for i, price in enumerate(prices):

        # ── Circuit breaker ───────────────────────────────────────────────────
        if circuit_breaker:
            if cash + holdings * price < INITIAL_CASH * (1 - circuit_breaker):
                halted = True
                break

        # ── Trend filter ──────────────────────────────────────────────────────
        if ma_period and i >= ma_period:
            ma       = sum(prices[i - ma_period:i]) / ma_period
            trend_up = price > ma
        else:
            trend_up = True

        # ── Level reset ───────────────────────────────────────────────────────
        history = prices[max(0, i - window):i + 1]
        if i == 0 or (reset_interval and i % reset_interval == 0):
            buys, sells = get_sd_levels(history, steps, std_multiplier, window)

        # ── Buy levels ────────────────────────────────────────────────────────
        if trend_up:
            for idx, b in enumerate(buys[:]):
                size = order_size * (idx + 1) if pyramid else order_size
                if price <= b and cash >= b * size:
                    cash     -= b * size
                    holdings += size
                    executed_buys.append((i, price))
                    buys.remove(b)

        # ── Sell levels ───────────────────────────────────────────────────────
        for s in sells[:]:
            if price >= s and holdings >= order_size:
                cash     += s * order_size
                holdings -= order_size
                profit   += s * order_size
                executed_sells.append((i, price))
                sells.remove(s)

    final_value = cash + holdings * prices[-1]

    return {
        "final_value":  final_value,
        "profit":       profit,
        "cash":         cash,
        "holdings":     holdings,
        "buy_points":   executed_buys,
        "sell_points":  executed_sells,
        "halted":       halted,
    }


def plot_trades(prices, buy_points, sell_points, dates=None,
                initial_cash=100_000, final_value=None, profit=None):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))

    x = dates if dates is not None else list(range(len(prices)))
    ax.plot(x, prices, label="Price", color="blue")

    if buy_points:
        x_buys = [dates[i] if dates is not None else i for i, _ in buy_points]
        y_buys = [p for _, p in buy_points]
        ax.scatter(x_buys, y_buys, color="green", marker="^", s=100, label=f"Buys ({len(buy_points)})")

    if sell_points:
        x_sells = [dates[i] if dates is not None else i for i, _ in sell_points]
        y_sells = [p for _, p in sell_points]
        ax.scatter(x_sells, y_sells, color="red", marker="v", s=100, label=f"Sells ({len(sell_points)})")

    # ── P&L annotation box ────────────────────────────────────────────────────
    if final_value is not None:
        pnl        = final_value - initial_cash
        return_pct = pnl / initial_cash * 100
        pnl_color  = "#1a7a1a" if pnl >= 0 else "#cc0000"
        sign       = "+" if pnl >= 0 else ""

        info = (
            f"Starting Balance:  ${initial_cash:>12,.2f}\n"
            f"Final Value:       ${final_value:>12,.2f}\n"
            f"Profit / Loss:     ${pnl:>+12,.2f}  ({sign}{return_pct:.2f}%)"
        )
        if profit is not None:
            info += f"\nRealised Profit:   ${profit:>12,.2f}"

        ax.annotate(
            info,
            xy=(0.01, 0.97),
            xycoords="axes fraction",
            va="top", ha="left",
            fontsize=9,
            fontfamily="monospace",
            color=pnl_color,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor=pnl_color),
        )

    if dates is not None:
        fig.autofmt_xdate()

    ax.set_title("SD Strategy — Backtest Trades")
    ax.set_xlabel("Date" if dates is not None else "Candle")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "research", "data", "2y", "hourly"
    )
    # filename = os.path.join(data_dir, "eth_usd_1h.csv")
    filename = "C:\\Users\\Michael\\OneDrive\\Desktop\\Projects\\TradingBot\\trading_bot\\research\\data\\2y\\hourly\\eth_usd_1h.csv"

    df = pd.read_csv(filename, parse_dates=["date"])

    if "close" not in df.columns:
        raise ValueError("CSV must contain a 'close' column")

    prices = df["close"].tolist()
    dates  = df["date"].tolist()
    result = backtest(prices)

    num_buys  = len(result["buy_points"])
    num_sells = len(result["sell_points"])
    return_pct = (result["final_value"] - 100_000) / 100_000 * 100

    print("\n=== SD Strategy Backtest Results ===")
    print(f"Starting Capital: $100,000.00")
    print(f"Final Value:      ${result['final_value']:,.2f}  ({return_pct:+.2f}%)")
    print(f"Realised Profit:  ${result['profit']:,.2f}")
    print(f"Cash Remaining:   ${result['cash']:,.2f}")
    print(f"Holdings:         {result['holdings']:.4f} units")
    print(f"Buys Executed:    {num_buys}")
    print(f"Sells Executed:   {num_sells}")
    print(f"Total Trades:     {num_buys + num_sells}")
    print(f"Halted:           {result['halted']}")

    plot_trades(
        prices, result["buy_points"], result["sell_points"],
        dates        = dates,
        initial_cash = 100_000,
        final_value  = result["final_value"],
        profit       = result["profit"],
    )
