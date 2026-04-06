"""
Parameter search for ETH, BNB, SOL grid bot.
Sweeps the most impactful variables and reports what actually produces profit.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from itertools import product
from backtest import backtest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "2y", "hourly")

TARGETS = {
    "ETH": {"file": "eth_usd_1h.csv", "ma_default": 168},
    "BNB": {"file": "bnb_usd_1h.csv", "ma_default": None},
    "SOL": {"file": "sol_usd_1h.csv", "ma_default": 168},
}

# Search space — keep it tractable
SWEEP_VALUES          = [0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
STEPS_VALUES          = [4, 6, 8, 12]
MA_VALUES             = [24, 48, 72, 168, 336, None]
RESET_VALUES          = [6, 12, 24, 48, None]
COUNTER_MULT_VALUES   = [1.0, 1.5, 2.0, 3.0]
ORDER_SIZE_PCT_VALUES = [0.01, 0.02, 0.05, 0.10, 0.15]

INITIAL_CASH = 100_000

def run_search(symbol, prices):
    best_results = []

    combos = list(product(SWEEP_VALUES, STEPS_VALUES, MA_VALUES, RESET_VALUES, COUNTER_MULT_VALUES, ORDER_SIZE_PCT_VALUES))
    total = len(combos)
    print(f"  {symbol}: testing {total} combinations...", flush=True)

    for sweep, steps, ma_period, reset_interval, cm, order_size_pct in combos:
        r = backtest(
            prices,
            sweep=sweep,
            steps=steps,
            ma_period=ma_period,
            reset_interval=reset_interval,
            counter_multiplier=cm,
            order_size_pct=order_size_pct,
            atr_multiplier=2.0,
            circuit_breaker=0.20,
            reactive=True,
        )
        return_pct = (r["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
        trades = len(r["buy_points"]) + len(r["sell_points"])
        best_results.append({
            "symbol": symbol,
            "sweep": sweep,
            "steps": steps,
            "ma_period": ma_period,
            "reset_interval": reset_interval,
            "counter_mult": cm,
            "order_size_pct": order_size_pct,
            "return_pct": return_pct,
            "final_value": r["final_value"],
            "trades": trades,
            "halted": r["halted"],
        })

    df = pd.DataFrame(best_results).sort_values("return_pct", ascending=False)
    return df


all_results = []

for symbol, cfg in TARGETS.items():
    path = os.path.join(DATA_DIR, cfg["file"])
    df_prices = pd.read_csv(path)
    prices = df_prices["close"].dropna().tolist()

    df_results = run_search(symbol, prices)
    all_results.append(df_results)

    top = df_results.head(5)
    print(f"\n  Top 5 for {symbol}:")
    print(f"  {'Return':>8}  {'Sweep':>6}  {'Steps':>5}  {'MA':>6}  {'Reset':>6}  {'CntMult':>7}  {'OrdPct':>7}  {'Trades':>7}  Halted")
    for _, row in top.iterrows():
        ma_str    = str(int(row["ma_period"])) if pd.notna(row["ma_period"]) else "None"
        reset_str = str(int(row["reset_interval"])) if pd.notna(row["reset_interval"]) else "None"
        print(f"  {row['return_pct']:>+7.2f}%  {row['sweep']:>6.2f}  {int(row['steps']):>5}  "
              f"{ma_str:>6}  {reset_str:>6}  {row['counter_mult']:>7.1f}  {row['order_size_pct']:>7.2f}  "
              f"{int(row['trades']):>7}  {'YES' if row['halted'] else 'no'}")
    print()

# Combined: best overall
combined = pd.concat(all_results)
profitable = combined[combined["return_pct"] > 0]
print(f"\n{'='*60}")
print(f"Profitable combos: {len(profitable)} / {len(combined)} ({100*len(profitable)/len(combined):.1f}%)")
print(f"{'='*60}\n")

# Best per symbol summary
print("OPTIMAL CONFIGS:")
for symbol in TARGETS:
    sym_df = combined[combined["symbol"] == symbol].sort_values("return_pct", ascending=False)
    best = sym_df.iloc[0]
    ma_str    = str(int(best["ma_period"])) if pd.notna(best["ma_period"]) else "None"
    reset_str = str(int(best["reset_interval"])) if pd.notna(best["reset_interval"]) else "None"
    print(f'  "{symbol}": {{"sweep": {best["sweep"]}, "steps": {int(best["steps"])}, '
          f'"ma_period": {ma_str}, "reset_interval": {reset_str}, '
          f'"counter_mult": {best["counter_mult"]}, "order_size_pct": {best["order_size_pct"]}}}  # {best["return_pct"]:+.2f}%')
