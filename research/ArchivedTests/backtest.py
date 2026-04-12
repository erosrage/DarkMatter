import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from strategy import get_grid

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "2y", "hourly")

MARKET_CONFIGS = {
    "ETH": {"sweep": 0.10, "steps": 4,  "ma_period": 336, "reset_interval": 24, "counter_mult": 3.0, "order_size_pct": 0.15, "file": "eth_usd_1h.csv"},
    "BNB": {"sweep": 0.06, "steps": 12, "ma_period": None, "reset_interval": 24, "counter_mult": 3.0, "order_size_pct": 0.15, "file": "bnb_usd_1h.csv"},
    "SOL": {"sweep": 0.20, "steps": 4,  "ma_period": None, "reset_interval": 24, "counter_mult": 3.0, "order_size_pct": 0.15, "file": "sol_usd_1h.csv"},
}


def backtest(prices, sweep=0.08, steps=8, order_size_pct=0.05, reset_interval=6,
             ma_period=168, atr_period=14,
             circuit_breaker=0.20, atr_multiplier=2.0,
             min_sweep=0.02, max_sweep=0.15, reactive=True,
             counter_multiplier=1.0, initial_cash=100_000,
             buy_ma_period=None, drift_reset_pct=None,
             skip_buys_in_downtrend=False,
             price_decimals=None, qty_decimals=None):
    """
    Backtest the grid strategy on a price series.

    Order sizing is a percentage of current cash balance, so profits compound
    and losses self-limit as the portfolio grows or shrinks.

      order_size_pct:          fraction of current cash to deploy per grid level (e.g. 0.05 = 5%)
      initial_cash:            starting capital (default 100,000)
      buy_ma_period:           short MA that gates initial BUY order placement on grid resets
      drift_reset_pct:         reset grid when price drifts >this fraction from grid centre
      skip_buys_in_downtrend:  if True, buy orders will not execute when price < buy_ma_period MA
                               (orders remain queued but are skipped each candle during downtrend)
    """
    INITIAL_CASH = initial_cash
    cash = INITIAL_CASH
    holdings = 0.0
    profit = 0.0
    halted = False

    executed_buys = []
    executed_sells = []
    grid_centre = prices[0]   # tracks the price around which current grid is centred

    def compute_sweep(i, price):
        if atr_period and i >= atr_period:
            ranges = [abs(prices[j] - prices[j - 1]) for j in range(i - atr_period + 1, i + 1)]
            atr = sum(ranges) / atr_period
            return max(min_sweep, min(max_sweep, (atr / price) * atr_multiplier)) if price > 0 else sweep
        return sweep

    def rp(p):
        return round(p, price_decimals) if price_decimals is not None else p

    def rq(q):
        return round(q, qty_decimals) if qty_decimals is not None else q

    def make_grid(i, price, current_cash, buys_allowed=True):
        nonlocal grid_centre
        sw = compute_sweep(i, price)
        step_pct = sw / steps
        order_usd = current_cash * order_size_pct
        buys, sells = get_grid(price, sw, steps)
        orders = []
        if buys_allowed:
            for b in buys:
                b = rp(b)
                orders.append({"side": "buy",  "price": b, "qty": rq(order_usd / b), "step_pct": step_pct})
        for s in sells:
            s = rp(s)
            orders.append({"side": "sell", "price": s, "qty": rq(order_usd / s), "step_pct": step_pct})
        grid_centre = price
        return orders

    active_orders = make_grid(0, prices[0], cash)

    for i, price in enumerate(prices):

        # ── Circuit breaker ────────────────────────────────────────────────────
        if circuit_breaker:
            portfolio_value = cash + holdings * price
            if portfolio_value < INITIAL_CASH * (1 - circuit_breaker):
                halted = True
                break

        # ── Long-term trend filter (for reactive counter-buys) ─────────────────
        if ma_period and i >= ma_period:
            ma = sum(prices[i - ma_period:i]) / ma_period
            trend_up = price > ma
        else:
            trend_up = True

        # ── Short-term buy gate (for initial grid BUY placement) ───────────────
        if buy_ma_period and i >= buy_ma_period:
            short_ma = sum(prices[i - buy_ma_period:i]) / buy_ma_period
            buys_allowed = price > short_ma
        else:
            buys_allowed = True

        # ── Price-drift reset ──────────────────────────────────────────────────
        drift_triggered = (
            drift_reset_pct and grid_centre > 0 and
            abs(price - grid_centre) / grid_centre > drift_reset_pct
        )

        # ── Time-based grid reset ──────────────────────────────────────────────
        if i > 0 and (drift_triggered or (reset_interval and i % reset_interval == 0)):
            active_orders = make_grid(i, price, cash, buys_allowed=buys_allowed)

        # ── Check fills ────────────────────────────────────────────────────────
        filled = []
        remaining = []
        for order in active_orders:
            if order["side"] == "buy" and price <= order["price"]:
                # skip execution if downtrend gating is active
                if skip_buys_in_downtrend and not buys_allowed:
                    remaining.append(order)
                    continue
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
                    counter_price = rp(order["price"] * (1 + sp * counter_multiplier))
                    key = ("sell", round(counter_price, 4))
                    if key not in existing:
                        active_orders.append({"side": "sell", "price": counter_price,
                                              "qty": order["qty"], "step_pct": sp})
                        existing.add(key)
                elif trend_up:  # sell filled → re-buy only in uptrend
                    counter_price = rp(order["price"] * (1 - sp * counter_multiplier))
                    key = ("buy", round(counter_price, 4))
                    if key not in existing:
                        order_usd = cash * order_size_pct
                        active_orders.append({"side": "buy",  "price": counter_price,
                                              "qty": rq(order_usd / counter_price), "step_pct": sp})
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


def backtest_multi(market_configs, data_dir, initial_cash=30_000,
                   circuit_breaker=0.20, reactive=True):
    """
    Backtest all markets simultaneously against a single shared cash pool,
    mirroring production where all markets draw from the same account balance.

    Order sizing uses portfolio_val * order_size_pct at each timestep, so the
    effective per-order USD shrinks as cash is consumed across markets.
    """
    # ── Load and align price series on common timestamps ──────────────────────
    frames = {}
    for symbol, cfg in market_configs.items():
        filepath = os.path.join(data_dir, cfg["file"])
        df = pd.read_csv(filepath, parse_dates=["date"])
        if "close" not in df.columns:
            print(f"{symbol}: missing 'close' column, skipping.")
            continue
        frames[symbol] = df.set_index("date")["close"]

    combined = pd.DataFrame(frames).dropna()
    dates = combined.index.tolist()
    symbols = list(frames.keys())

    # ── Shared state ──────────────────────────────────────────────────────────
    cash = float(initial_cash)
    holdings      = {m: 0.0 for m in symbols}
    active_orders = {m: [] for m in symbols}
    executed_buys  = {m: [] for m in symbols}
    executed_sells = {m: [] for m in symbols}
    gross_revenue  = {m: 0.0 for m in symbols}
    price_history  = {m: [] for m in symbols}
    halted = False

    def portfolio_val(prices):
        return cash + sum(holdings[m] * prices[m] for m in symbols)

    def make_orders(m, price, pval):
        cfg = market_configs[m]
        sweep = cfg["sweep"]
        steps = cfg["steps"]
        step_pct = sweep / steps
        order_usd = pval * cfg["order_size_pct"]
        buys, sells = get_grid(price, sweep, steps)
        orders = []
        for b in buys:
            orders.append({"side": "buy",  "price": b, "qty": order_usd / b, "step_pct": step_pct})
        for s in sells:
            orders.append({"side": "sell", "price": s, "qty": order_usd / s, "step_pct": step_pct})
        return orders

    # Initialise grids at t=0
    row0 = combined.iloc[0]
    pval0 = initial_cash  # no holdings yet
    for m in symbols:
        active_orders[m] = make_orders(m, row0[m], pval0)

    # ── Main simulation loop ───────────────────────────────────────────────────
    for i in range(len(dates)):
        row = combined.iloc[i]
        prices = {m: float(row[m]) for m in symbols}

        for m in symbols:
            price_history[m].append(prices[m])

        pval = portfolio_val(prices)

        # Circuit breaker — halt everything
        if circuit_breaker and pval < initial_cash * (1 - circuit_breaker):
            halted = True
            break

        # Trend filter per market
        trend_up = {}
        for m in symbols:
            ma_period = market_configs[m].get("ma_period")
            if ma_period and i >= ma_period:
                hist = price_history[m]
                trend_up[m] = prices[m] > sum(hist[-ma_period:]) / ma_period
            else:
                trend_up[m] = True

        # ── Fill orders and reactive re-entry ─────────────────────────────────
        # Process markets in order; shared cash is deducted immediately so later
        # markets see the reduced balance — same effect as concurrent Robinhood orders
        # competing for buying power.
        for m in symbols:
            price = prices[m]
            cfg   = market_configs[m]
            filled    = []
            remaining = []

            for order in active_orders[m]:
                if order["side"] == "buy" and price <= order["price"]:
                    cost = order["price"] * order["qty"]
                    if cash >= cost:
                        cash -= cost
                        holdings[m] += order["qty"]
                        executed_buys[m].append((i, price))
                        filled.append(order)
                    else:
                        remaining.append(order)  # insufficient shared cash
                elif order["side"] == "sell" and price >= order["price"]:
                    if holdings[m] >= order["qty"]:
                        proceeds = order["price"] * order["qty"]
                        cash += proceeds
                        holdings[m] -= order["qty"]
                        gross_revenue[m] += proceeds
                        executed_sells[m].append((i, price))
                        filled.append(order)
                    else:
                        remaining.append(order)
                else:
                    remaining.append(order)

            active_orders[m] = remaining

            if reactive:
                existing = {(o["side"], round(o["price"], 4)) for o in active_orders[m]}
                cm = cfg.get("counter_mult", 1.0)
                for order in filled:
                    sp = order["step_pct"]
                    if order["side"] == "buy":
                        cp = order["price"] * (1 + sp * cm)
                        key = ("sell", round(cp, 4))
                        if key not in existing:
                            active_orders[m].append({"side": "sell", "price": cp,
                                                      "qty": order["qty"], "step_pct": sp})
                            existing.add(key)
                    elif trend_up[m]:
                        cp = order["price"] * (1 - sp * cm)
                        key = ("buy", round(cp, 4))
                        if key not in existing:
                            order_usd = pval * cfg["order_size_pct"]
                            active_orders[m].append({"side": "buy", "price": cp,
                                                      "qty": order_usd / cp, "step_pct": sp})
                            existing.add(key)

    last_prices = {m: float(combined.iloc[-1][m]) for m in symbols}
    final_value = cash + sum(holdings[m] * last_prices[m] for m in symbols)

    return {
        "final_value":    final_value,
        "cash":           cash,
        "holdings":       holdings,
        "last_prices":    last_prices,
        "gross_revenue":  gross_revenue,
        "executed_buys":  executed_buys,
        "executed_sells": executed_sells,
        "halted":         halted,
        "dates":          dates,
        "combined":       combined,
    }


if __name__ == "__main__":
    INITIAL_CASH = 30_000

    result = backtest_multi(MARKET_CONFIGS, DATA_DIR, initial_cash=INITIAL_CASH)

    final_value  = result["final_value"]
    return_pct   = (final_value - INITIAL_CASH) / INITIAL_CASH * 100
    last_prices  = result["last_prices"]
    symbols      = list(result["holdings"].keys())

    print("\n=== Backtest Results — Shared Capital (2-Year Hourly) ===")
    print(f"  Starting Capital: ${INITIAL_CASH:>10,.2f}")
    print(f"  Final Value:      ${final_value:>10,.2f}  ({return_pct:+.2f}%)")
    print(f"  Cash remaining:   ${result['cash']:>10,.2f}")
    print(f"  Halted:           {'YES' if result['halted'] else 'No'}")
    print()

    print(f"{'Symbol':<8} {'Holdings Val':>14} {'Units':>14} {'Last Price':>12} {'Buys':>7} {'Sells':>7} {'Gross Rev':>14}")
    print("-" * 82)
    for m in symbols:
        hval = result["holdings"][m] * last_prices[m]
        buys  = len(result["executed_buys"][m])
        sells = len(result["executed_sells"][m])
        print(f"{m:<8} ${hval:>13,.2f} {result['holdings'][m]:>14.6f} ${last_prices[m]:>11,.2f} "
              f"{buys:>7} {sells:>7} ${result['gross_revenue'][m]:>13,.2f}")

    # Plot each symbol
    for m in symbols:
        combined = result["combined"]
        prices = combined[m].tolist()
        dates  = result["dates"]
        hval   = result["holdings"][m] * last_prices[m]
        cfg2   = MARKET_CONFIGS[m]
        metrics = {
            "final_value":   final_value,
            "return_pct":    return_pct,
            "cash":          result["cash"],
            "holdings_val":  hval,
            "gross_revenue": result["gross_revenue"][m],
            "buys":          len(result["executed_buys"][m]),
            "sells":         len(result["executed_sells"][m]),
            "halted":        result["halted"],
        }
        plot_trades(prices, result["executed_buys"][m], result["executed_sells"][m],
                    dates=dates,
                    title=f"{m} — shared capital  sweep={cfg2['sweep']} steps={cfg2['steps']} cm={cfg2['counter_mult']}",
                    metrics=metrics)
