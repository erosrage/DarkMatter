"""
Parameter search on 15-minute data for ETH, BNB, SOL.

Tunable levers (and their effect):
  sweep          — grid width each side; wider = bigger profit per fill but fewer fills
  steps          — number of grid levels; more = finer grid, more fills, more capital tied up
  reset_interval — candles between full grid resets; shorter = grid stays near price; longer = waits
  order_size_pct — fraction of portfolio per order level; higher = more exposure
  counter_mult   — how far counter-order is placed after a fill (× step_pct); higher = bigger profit target

Search is multi-market (shared $30k pool) to mirror production.
~576 runs, completes in under a minute.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from itertools import product
from research.ArchivedTests.backtest import backtest_multi

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data", "15m")
INITIAL_CASH = 30_000

MA_PERIOD = {"ETH": 1344, "BNB": None, "SOL": None}
FILES = {
    "ETH": "eth_usd_15m.csv",
    "BNB": "bnb_usd_15m.csv",
    "SOL": "sol_usd_15m.csv",
}
SYMBOLS = ["ETH", "BNB", "SOL"]

# ── Steps fixed at production-proven values ───────────────────────────────────
FIXED_STEPS = {"ETH": 4, "BNB": 12, "SOL": 4}

# ── Search space (~243 runs, ~5 minutes) ──────────────────────────────────────
# Sweep: the single biggest lever — wider = larger profit per round trip, fewer fills
ETH_SWEEP_VALS = [0.06, 0.10, 0.15]
BNB_SWEEP_VALS = [0.04, 0.06, 0.08]   # BNB trades tight ranges
SOL_SWEEP_VALS = [0.10, 0.20, 0.30]   # SOL is the most volatile

# Shared params
RESET_VALS      = [48, 96, None]    # 15m candles: 12h, 24h, never
ORDER_SIZE_VALS = [0.05, 0.10, 0.15]
COUNTER_VALS    = [2.0, 3.0, 4.0]

sweep_combos  = list(product(ETH_SWEEP_VALS, BNB_SWEEP_VALS, SOL_SWEEP_VALS))  # 27
shared_combos = list(product(RESET_VALS, ORDER_SIZE_VALS, COUNTER_VALS))         # 27

total = len(sweep_combos) * len(shared_combos)
print(f"\nSearch space: {len(sweep_combos)} sweep combos x {len(shared_combos)} shared combos = {total:,} runs")
print(f"Steps fixed: ETH={FIXED_STEPS['ETH']}, BNB={FIXED_STEPS['BNB']}, SOL={FIXED_STEPS['SOL']}")
print(f"Starting capital: ${INITIAL_CASH:,}  |  Data: 15m  |  Window: ~60 days\n")

results = []
done = 0

for eth_sw, bnb_sw, sol_sw in sweep_combos:
    sweeps = {"ETH": eth_sw, "BNB": bnb_sw, "SOL": sol_sw}

    for reset, order_pct, counter_m in shared_combos:
        cfg = {}
        for sym in SYMBOLS:
            cfg[sym] = {
                "sweep":          sweeps[sym],
                "steps":          FIXED_STEPS[sym],
                "ma_period":      MA_PERIOD[sym],
                "reset_interval": reset,
                "counter_mult":   counter_m,
                "order_size_pct": order_pct,
                "file":           FILES[sym],
            }

        r = backtest_multi(cfg, DATA_DIR, initial_cash=INITIAL_CASH)
        ret_pct = (r["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
        trades  = sum(
            len(r["executed_buys"][s]) + len(r["executed_sells"][s]) for s in SYMBOLS
        )

        results.append({
            "return_pct":  ret_pct,
            "final_value": r["final_value"],
            "cash":        r["cash"],
            "halted":      r["halted"],
            "trades":      trades,
            "reset":       reset,
            "order_pct":   order_pct,
            "counter_m":   counter_m,
            "ETH_sweep": eth_sw,
            "BNB_sweep": bnb_sw,
            "SOL_sweep": sol_sw,
        })

        done += 1
        if done % 50 == 0:
            best_so_far = max(results, key=lambda x: x["return_pct"])
            print(f"  [{done:>4}/{total}]  best so far: {best_so_far['return_pct']:+.2f}%", flush=True)

df = pd.DataFrame(results).sort_values("return_pct", ascending=False)

# ── Results ───────────────────────────────────────────────────────────────────
profitable = df[df["return_pct"] > 0]
print(f"\n{'='*110}")
print(f"Total runs: {len(df):,}  |  Profitable: {len(profitable):,} ({100*len(profitable)/len(df):.1f}%)")
print(f"{'='*110}\n")

print("TOP 20 CONFIGURATIONS:")
hdr = (f"{'#':>3}  {'Return':>8}  {'FinalVal':>10}  {'Reset':>6}  "
       f"{'OrdPct':>7}  {'CM':>5}  "
       f"{'ETH_sw':>7}  {'BNB_sw':>7}  {'SOL_sw':>7}  {'Trades':>7}  Halt")
print(hdr)
print("-" * len(hdr))

for rank, (_, row) in enumerate(df.head(20).iterrows(), 1):
    rst = str(int(row["reset"])) if pd.notna(row["reset"]) else "None"
    print(f"{rank:>3}  {row['return_pct']:>+7.2f}%  ${row['final_value']:>9,.0f}  "
          f"{rst:>6}  {row['order_pct']:>7.3f}  {row['counter_m']:>5.1f}  "
          f"{row['ETH_sweep']:.2f}     {row['BNB_sweep']:.2f}     {row['SOL_sweep']:.2f}  "
          f"{int(row['trades']):>7}  {'YES' if row['halted'] else 'no'}")

# ── Parameter sensitivity ──────────────────────────────────────────────────────
print("\n\n--- Parameter Sensitivity (mean return by value) ---\n")

for param, col in [("Reset interval", "reset"), ("Order size %", "order_pct"), ("Counter mult", "counter_m"),
                   ("ETH sweep", "ETH_sweep"), ("BNB sweep", "BNB_sweep"), ("SOL sweep", "SOL_sweep")]:
    grp = df.groupby(col)["return_pct"].mean().sort_index()
    vals = "  ".join(f"{k}={v:+.2f}%" for k, v in grp.items())
    print(f"  {param:<16}: {vals}")

# ── Best config ────────────────────────────────────────────────────────────────
best = df.iloc[0]
rst_val = str(int(best["reset"])) if pd.notna(best["reset"]) else "None"
print(f"\n\nBEST CONFIG  ({best['return_pct']:+.2f}% return, {int(best['trades'])} trades, halted={'Yes' if best['halted'] else 'No'}):")
print(f"""
MARKET_CONFIGS_15M = {{
    "ETH": {{"sweep": {best['ETH_sweep']}, "steps": {FIXED_STEPS['ETH']},  "ma_period": 1344, "reset_interval": {rst_val}, "counter_mult": {best['counter_m']}, "order_size_pct": {best['order_pct']}, "file": "eth_usd_15m.csv"}},
    "BNB": {{"sweep": {best['BNB_sweep']}, "steps": {FIXED_STEPS['BNB']}, "ma_period": None,  "reset_interval": {rst_val}, "counter_mult": {best['counter_m']}, "order_size_pct": {best['order_pct']}, "file": "bnb_usd_15m.csv"}},
    "SOL": {{"sweep": {best['SOL_sweep']}, "steps": {FIXED_STEPS['SOL']},  "ma_period": None,  "reset_interval": {rst_val}, "counter_mult": {best['counter_m']}, "order_size_pct": {best['order_pct']}, "file": "sol_usd_15m.csv"}},
}}
""")

# Save results
out = os.path.join(os.path.dirname(__file__), "param_search_15m_results.csv")
df.to_csv(out, index=False)
print(f"Full results saved to {out}")
