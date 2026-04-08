"""
Shared-capital parameter search for ETH, BNB, SOL grid bot.

Search strategy:
  - Shared params:     reset_interval, order_size_pct
  - Per-market params: sweep, steps  (ma_period and counter_mult kept at
                       known-good defaults to keep the space tractable)

Each combination is evaluated via backtest_multi() so all three markets
compete for the same $30k pool — matching production behaviour.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from itertools import product
from backtest import backtest_multi

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data", "2y", "hourly")
INITIAL_CASH = 30_000

# ── Fixed per-market params (not swept) ───────────────────────────────────────
MA_PERIOD    = {"ETH": 336, "BNB": None, "SOL": None}
COUNTER_MULT = {"ETH": 3.0, "BNB": 3.0,  "SOL": 3.0}

# ── Search space ──────────────────────────────────────────────────────────────
SWEEP_VALUES        = [0.06, 0.10, 0.15, 0.20]
STEPS_VALUES        = [4, 8, 12]
RESET_VALUES        = [24, 48, 168, None]          # hours; None = never reset
ORDER_SIZE_VALUES   = [0.04, 0.06, 0.08, 0.10, 0.15]

SYMBOLS = ["ETH", "BNB", "SOL"]
FILES   = {"ETH": "eth_usd_1h.csv", "BNB": "bnb_usd_1h.csv", "SOL": "sol_usd_1h.csv"}

# Per-market sweep × steps combos
per_market_combos = list(product(SWEEP_VALUES, STEPS_VALUES))  # (sweep, steps)
# All ways to assign a per-market combo to each of the 3 symbols
market_assignments = list(product(per_market_combos, repeat=len(SYMBOLS)))
# Shared param combos
shared_combos = list(product(RESET_VALUES, ORDER_SIZE_VALUES))

total = len(market_assignments) * len(shared_combos)
print(f"\nSearching {len(market_assignments)} market-param combos × "
      f"{len(shared_combos)} shared-param combos = {total:,} total runs")
print(f"Starting capital: ${INITIAL_CASH:,}\n")

results = []
done = 0

for shared in shared_combos:
    reset_interval, order_size_pct = shared

    for assignment in market_assignments:
        cfg = {}
        for sym, (sweep, steps) in zip(SYMBOLS, assignment):
            cfg[sym] = {
                "sweep":          sweep,
                "steps":          steps,
                "ma_period":      MA_PERIOD[sym],
                "reset_interval": reset_interval,
                "counter_mult":   COUNTER_MULT[sym],
                "order_size_pct": order_size_pct,
                "file":           FILES[sym],
            }

        r = backtest_multi(cfg, DATA_DIR, initial_cash=INITIAL_CASH)
        return_pct = (r["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
        total_trades = sum(
            len(r["executed_buys"][s]) + len(r["executed_sells"][s]) for s in SYMBOLS
        )

        results.append({
            "return_pct":    return_pct,
            "final_value":   r["final_value"],
            "cash":          r["cash"],
            "halted":        r["halted"],
            "trades":        total_trades,
            "reset":         reset_interval,
            "order_pct":     order_size_pct,
            "ETH_sweep":     assignment[0][0], "ETH_steps": assignment[0][1],
            "BNB_sweep":     assignment[1][0], "BNB_steps": assignment[1][1],
            "SOL_sweep":     assignment[2][0], "SOL_steps": assignment[2][1],
        })

        done += 1
        if done % 500 == 0:
            best_so_far = max(results, key=lambda x: x["return_pct"])
            print(f"  [{done:>6}/{total}]  best so far: {best_so_far['return_pct']:+.2f}%", flush=True)

df = pd.DataFrame(results).sort_values("return_pct", ascending=False)

# ── Summary ───────────────────────────────────────────────────────────────────
profitable = df[df["return_pct"] > 0]
print(f"\n{'='*90}")
print(f"Profitable combos: {len(profitable):,} / {len(df):,}  ({100*len(profitable)/len(df):.1f}%)")
print(f"{'='*90}\n")

print("TOP 15 CONFIGURATIONS:")
print(f"{'Rank':>4}  {'Return':>8}  {'FinalVal':>10}  {'Reset':>6}  {'OrdPct':>7}  "
      f"{'ETH sw/st':>10}  {'BNB sw/st':>10}  {'SOL sw/st':>10}  {'Trades':>7}  Halt")
print("-" * 100)

for rank, (_, row) in enumerate(df.head(15).iterrows(), 1):
    reset_str = str(int(row["reset"])) if pd.notna(row["reset"]) else "None"
    print(f"{rank:>4}  {row['return_pct']:>+7.2f}%  ${row['final_value']:>9,.0f}  "
          f"{reset_str:>6}  {row['order_pct']:>7.2f}  "
          f"{row['ETH_sweep']:.2f}/{int(row['ETH_steps']):>2}  "
          f"{row['BNB_sweep']:.2f}/{int(row['BNB_steps']):>2}  "
          f"{row['SOL_sweep']:.2f}/{int(row['SOL_steps']):>2}  "
          f"{int(row['trades']):>7}  {'YES' if row['halted'] else 'no'}")

print()

# Best config as ready-to-paste MARKET_CONFIGS
best = df.iloc[0]
reset_val = str(int(best["reset"])) if pd.notna(best["reset"]) else "None"
print("BEST CONFIG (paste into backtest.py / main.py):")
print(f"""
MARKET_CONFIGS = {{
    "ETH": {{"sweep": {best['ETH_sweep']}, "steps": {int(best['ETH_steps'])},  "ma_period": {MA_PERIOD['ETH']}, "reset_interval": {reset_val}, "counter_mult": {COUNTER_MULT['ETH']}, "order_size_pct": {best['order_pct']}, "file": "eth_usd_1h.csv"}},
    "BNB": {{"sweep": {best['BNB_sweep']}, "steps": {int(best['BNB_steps'])}, "ma_period": {MA_PERIOD['BNB']}, "reset_interval": {reset_val}, "counter_mult": {COUNTER_MULT['BNB']}, "order_size_pct": {best['order_pct']}, "file": "bnb_usd_1h.csv"}},
    "SOL": {{"sweep": {best['SOL_sweep']}, "steps": {int(best['SOL_steps'])},  "ma_period": {MA_PERIOD['SOL']}, "reset_interval": {reset_val}, "counter_mult": {COUNTER_MULT['SOL']}, "order_size_pct": {best['order_pct']}, "file": "sol_usd_1h.csv"}},
}}  # {best['return_pct']:+.2f}% return
""")

# Save full results
out_path = os.path.join(os.path.dirname(__file__), "param_search_results.csv")
df.to_csv(out_path, index=False)
print(f"Full results saved to {out_path}")
