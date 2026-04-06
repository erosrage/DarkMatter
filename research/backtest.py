import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from strategy import get_grid

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "2y", "hourly")

MARKET_CONFIGS = {
    "ETH": {"sweep": 0.20, "steps": 8,  "ma_period": 336, "reset_interval": None, "counter_mult": 3.0, "order_size_pct": 0.15, "file": "eth_usd_1h.csv"},
    "BNB": {"sweep": 0.08, "steps": 12, "ma_period": None, "reset_interval": None, "counter_mult": 3.0, "order_size_pct": 0.15, "file": "bnb_usd_1h.csv"},
    "SOL": {"sweep": 0.20, "steps": 4,  "ma_period": None, "reset_interval": None, "counter_mult": 3.0, "order_size_pct": 0.15, "file": "sol_usd_1h.csv"},
}


def backtest(prices, sweep=0.08, steps=8, order_size_pct=0.05, reset_interval=6,
             ma_period=168, atr_period=14,
             circuit_breaker=0.20, atr_multiplier=2.0,
             min_sweep=0.02, max_sweep=0.15, reactive=True,
             counter_multiplier=1.0):
    """
    Backtest the grid strategy on a price series.

    Order sizing is a percentage of current cash balance, so profits compound
    and losses self-limit as the portfolio grows or shrinks.

      order_size_pct: fraction of current cash to deploy per grid level (e.g. 0.05 = 5%)
    """
    INITIAL_CASH = 100_000
    cash = INITIAL_CASH
    holdings = 0.0
    profit = 0.0
    halted = False

    executed_buys = []
    executed_sells = []

    def compute_sweep(i, price):
        if atr_period and i >= atr_period:
            ranges = [abs(prices[j] - prices[j - 1]) for j in range(i - atr_period + 1, i + 1)]
            atr = sum(ranges) / atr_period
            return max(min_sweep, min(max_sweep, (atr / price) * atr_multiplier)) if price > 0 else sweep
        return sweep

    def make_grid(i, price, current_cash):
        sw = compute_sweep(i, price)
        step_pct = sw / steps
        order_usd = current_cash * order_size_pct
        buys, sells = get_grid(price, sw, steps)
        orders = []
        for b in buys:
            orders.append({"side": "buy",  "price": b, "qty": order_usd / b, "step_pct": step_pct})
        for s in sells:
            orders.append({"side": "sell", "price": s, "qty": order_usd / s, "step_pct": step_pct})
        return orders

    active_orders = make_grid(0, prices[0], cash)

    for i, price in enumerate(prices):

        # ── Circuit breaker ────────────────────────────────────────────────────
        if circuit_breaker:
            portfolio_value = cash + holdings * price
            if portfolio_value < INITIAL_CASH * (1 - circuit_breaker):
                halted = True
                break

        # ── Trend filter ───────────────────────────────────────────────────────
        if ma_period and i >= ma_period:
            ma = sum(prices[i - ma_period:i]) / ma_period
            trend_up = price > ma
        else:
            trend_up = True

        # ── Grid reset ──────────────────────────────────────────────────────────
        if i > 0 and reset_interval and i % reset_interval == 0:
            active_orders = make_grid(i, price, cash)

        # ── Check fills ────────────────────────────────────────────────────────
        filled = []
        remaining = []
        for order in active_orders:
            if order["side"] == "buy" and price <= order["price"]:
                cost = order["price"] * order["qty"]
                if cash >= cost:
                    cash -= cost
                    holdings += order["qty"]
                    executed_buys.append((i, price))
                    filled.append(order)
                else:
                    remaining.append(order)
            elif order["side"] == "sell" and price >= order["price"]:
                if holdings >= order["qty"]:
                    cash += order["price"] * order["qty"]
                    holdings -= order["qty"]
                    profit += order["price"] * order["qty"]
                    executed_sells.append((i, price))
                    filled.append(order)
                else:
                    remaining.append(order)
            else:
                remaining.append(order)

        active_orders = remaining

        # ── Reactive re-entry ──────────────────────────────────────────────────
        if reactive:
            # Build set of existing (side, rounded_price) to avoid stacking duplicates
            existing = {(o["side"], round(o["price"], 4)) for o in active_orders}
            for order in filled:
                sp = order["step_pct"]
                if order["side"] == "buy":
                    counter_price = order["price"] * (1 + sp * counter_multiplier)
                    key = ("sell", round(counter_price, 4))
                    if key not in existing:
                        # sell same qty bought — profit = qty × (sell_price - buy_price)
                        active_orders.append({"side": "sell", "price": counter_price,
                                              "qty": order["qty"], "step_pct": sp})
                        existing.add(key)
                elif trend_up:  # sell filled → re-buy only in uptrend
                    counter_price = order["price"] * (1 - sp * counter_multiplier)
                    key = ("buy", round(counter_price, 4))
                    if key not in existing:
                        order_usd = cash * order_size_pct
                        active_orders.append({"side": "buy",  "price": counter_price,
                                              "qty": order_usd / counter_price, "step_pct": sp})
                        existing.add(key)

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


def plot_trades(prices, buy_points, sell_points, dates=None, title="Backtest Trades", metrics=None):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))

    x = dates if dates is not None else list(range(len(prices)))
    ax.plot(x, prices, label="Price", color="blue")

    if buy_points:
        x_buys = [dates[i] if dates is not None else i for i, _ in buy_points]
        y_buys = [p for _, p in buy_points]
        ax.scatter(x_buys, y_buys, color="green", marker="^", s=60, label=f"Buys ({len(buy_points)})")

    if sell_points:
        x_sells = [dates[i] if dates is not None else i for i, _ in sell_points]
        y_sells = [p for _, p in sell_points]
        ax.scatter(x_sells, y_sells, color="red", marker="v", s=60, label=f"Sells ({len(sell_points)})")

    if metrics:
        sign = "+" if metrics["return_pct"] >= 0 else ""
        text = (
            f"Final Value:  ${metrics['final_value']:>12,.2f}\n"
            f"Return:       {sign}{metrics['return_pct']:.2f}%\n"
            f"Cash:         ${metrics['cash']:>12,.2f}\n"
            f"Holdings val: ${metrics['holdings_val']:>12,.2f}\n"
            f"Gross Rev:    ${metrics['gross_revenue']:>12,.2f}\n"
            f"Trades:       {metrics['buys']} buys / {metrics['sells']} sells\n"
            f"Halted:       {'Yes' if metrics['halted'] else 'No'}"
        )
        ax.text(
            0.01, 0.97, text,
            transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    if dates is not None:
        fig.autofmt_xdate()

    ax.set_title(title)
    ax.set_xlabel("Date" if dates is not None else "Candle")
    ax.set_ylabel("Price")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    INITIAL_CASH = 100_000
    results = {}

    print("\n=== Backtest Results (2-Year Hourly) ===\n")
    print(f"{'Symbol':<8} {'Final Value':>14} {'Return':>9} {'Buys':>7} {'Sells':>7} {'Halted':>8}")
    print("-" * 58)

    for symbol, cfg in MARKET_CONFIGS.items():
        filepath = os.path.join(DATA_DIR, cfg["file"])
        df = pd.read_csv(filepath, parse_dates=["date"])
        if "close" not in df.columns:
            print(f"{symbol}: missing 'close' column, skipping.")
            continue

        prices = df["close"].tolist()
        dates  = df["date"].tolist()

        result = backtest(
            prices,
            sweep=cfg["sweep"],
            steps=cfg["steps"],
            ma_period=cfg["ma_period"],
            reset_interval=cfg["reset_interval"],
            counter_multiplier=cfg["counter_mult"],
            order_size_pct=cfg["order_size_pct"],
        )

        return_pct = (result["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
        results[symbol] = result

        print(f"{symbol:<8} ${result['final_value']:>13,.2f} {return_pct:>+8.2f}% "
              f"{len(result['buy_points']):>7} {len(result['sell_points']):>7} "
              f"{'YES' if result['halted'] else 'no':>8}")

    print()

    # Detailed breakdown per symbol
    for symbol, cfg in MARKET_CONFIGS.items():
        if symbol not in results:
            continue
        result = results[symbol]
        filepath = os.path.join(DATA_DIR, cfg["file"])
        df = pd.read_csv(filepath, parse_dates=["date"])
        last_price = df["close"].iloc[-1]
        holdings_val = result["holdings"] * last_price

        return_pct = (result["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
        print(f"--- {symbol} ---")
        print(f"  Final Value:   ${result['final_value']:>12,.2f}  ({return_pct:+.2f}%)")
        print(f"  Cash:          ${result['cash']:>12,.2f}")
        print(f"  Holdings val:  ${holdings_val:>12,.2f}  ({result['holdings']:.6f} units @ ${last_price:,.2f})")
        print(f"  Gross Revenue: ${result['profit']:>12,.2f}")
        print(f"  Trades:        {len(result['buy_points'])} buys / {len(result['sell_points'])} sells")
        print(f"  Halted:        {result['halted']}")
        print()

    # Plot each symbol
    for symbol, cfg in MARKET_CONFIGS.items():
        if symbol not in results:
            continue
        filepath = os.path.join(DATA_DIR, cfg["file"])
        df = pd.read_csv(filepath, parse_dates=["date"])
        prices = df["close"].tolist()
        dates  = df["date"].tolist()
        result = results[symbol]
        cfg2 = MARKET_CONFIGS[symbol]
        last_price = df["close"].iloc[-1]
        holdings_val = result["holdings"] * last_price
        return_pct = (result["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
        metrics = {
            "final_value":   result["final_value"],
            "return_pct":    return_pct,
            "cash":          result["cash"],
            "holdings_val":  holdings_val,
            "gross_revenue": result["profit"],
            "buys":          len(result["buy_points"]),
            "sells":         len(result["sell_points"]),
            "halted":        result["halted"],
        }
        plot_trades(prices, result["buy_points"], result["sell_points"],
                    dates=dates,
                    title=f"{symbol} — sweep={cfg2['sweep']} steps={cfg2['steps']} ma={cfg2['ma_period']} cm={cfg2['counter_mult']}",
                    metrics=metrics)
