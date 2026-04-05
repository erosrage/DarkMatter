import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from strategy import get_grid

def backtest(prices, sweep=0.2, steps=8, order_size=20):
    cash = 100000
    holdings = 0
    profit = 0

    base_price = prices[0]
    buys, sells = get_grid(base_price, sweep, steps)

    # Track executed trades for plotting
    executed_buys = []
    executed_sells = []

    for i, price in enumerate(prices):
        # Buy levels
        for b in buys[:]:
            if price <= b and cash >= b * order_size:
                cash -= b * order_size
                holdings += order_size
                executed_buys.append((i, price))  # record index and price
                buys.remove(b)

        # Sell levels
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
        "sell_points": executed_sells
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
    filename = "C:\\Users\\Michael\\OneDrive\\Desktop\\Projects\\TradingBot\\trading_bot\\eth_usd_2y.csv"
    df = pd.read_csv(filename)

    if "close" not in df.columns:
        raise ValueError("CSV must contain a 'close' column")

    prices = df["close"].tolist()
    result = backtest(prices)

    print("\n=== Backtest Results ===")
    print(f"Final Value: {result['final_value']}")
    print(f"Profit:      {result['profit']}")
    print(f"Cash:        {result['cash']}")
    print(f"Holdings:    {result['holdings']}")

    # Plot
    plot_trades(prices, result["buy_points"], result["sell_points"])