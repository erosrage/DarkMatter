"""
Grid search over sweep/steps/reset_interval across all crypto CSVs.
Runs two passes — baseline (no enhancements) vs enhanced (all optimizations) —
then reports the delta so you can see exactly what the improvements bought.

Prefers 5y data over 2y when both are available for a given symbol.

Score = return_pct / num_trades  (maximise return, minimise churn)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import pandas as pd
from research.ArchivedTests.backtest import backtest

INITIAL_CASH = 100_000
ORDER_SIZE   = 20

SWEEP_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
STEPS_VALUES = [2, 4, 6, 8, 10, 12, 16]
# Reset intervals in hours: 1 day, 3 days, 1 week, 2 weeks, 1 month
RESET_VALUES = [None, 24, 72, 168, 336, 720]

# MA/ATR periods in hours: MA=168h (1 week), ATR=48h (2 days)
ENHANCED_KWARGS = dict(
    ma_period       = 168,
    atr_period      = 48,
    pyramid         = True,
    circuit_breaker = 0.20,
    atr_multiplier  = 8.0,
)

BASELINE_KWARGS = dict(
    ma_period       = None,
    atr_period      = None,
    pyramid         = False,
    circuit_breaker = None,
)

base_dir    = os.path.dirname(os.path.abspath(__file__))
data_dir    = os.path.join(base_dir, "data", "2y", "hourly")
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

symbol_files = {}
for path in sorted(glob.glob(os.path.join(data_dir, "*_1h.csv"))):
    sym = os.path.basename(path).replace("_usd_1h.csv", "").upper()
    symbol_files[sym] = ("1h", path)

print(f"Found {len(symbol_files)} symbols with hourly data\n")


def best_params(prices, extra_kwargs):
    best = None
    for sweep in SWEEP_VALUES:
        for steps in STEPS_VALUES:
            for reset in RESET_VALUES:
                r = backtest(
                    prices, sweep=sweep, steps=steps,
                    order_size=ORDER_SIZE, reset_interval=reset,
                    **extra_kwargs
                )
                num_trades = len(r["buy_points"]) + len(r["sell_points"])
                return_pct = (r["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
                score = return_pct / num_trades if num_trades > 0 else 0
                if best is None or score > best["score"]:
                    best = dict(sweep=sweep, steps=steps, reset_interval=reset,
                                return_pct=return_pct, num_trades=num_trades,
                                final_value=r["final_value"], score=score,
                                halted=r.get("halted", False))
    return best


baseline_rows = []
enhanced_rows = []

for symbol, (interval, csv_path) in sorted(symbol_files.items()):
    df = pd.read_csv(csv_path)
    if "close" not in df.columns or len(df) < 30:
        print(f"[SKIP] {symbol}")
        continue

    prices = df["close"].dropna().tolist()

    b = best_params(prices, BASELINE_KWARGS)
    e = best_params(prices, ENHANCED_KWARGS)

    b["symbol"] = symbol
    b["interval"] = interval
    e["symbol"] = symbol
    e["interval"] = interval
    baseline_rows.append(b)
    enhanced_rows.append(e)

    delta = e["return_pct"] - b["return_pct"]
    rst_e = f"{e['reset_interval']:4d}h" if e["reset_interval"] else " None"
    print(
        f"{symbol:12s}  "
        f"base={b['return_pct']:+8.2f}%  "
        f"enhanced={e['return_pct']:+8.2f}%  "
        f"delta={delta:+7.2f}%  "
        f"[sweep={e['sweep']:.2f} steps={e['steps']:2d} reset={rst_e}]"
        + (" HALTED" if e["halted"] else "")
    )


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

base_df = pd.DataFrame(baseline_rows).sort_values("return_pct", ascending=False)
enh_df  = pd.DataFrame(enhanced_rows).sort_values("return_pct", ascending=False)

avg_base        = base_df["return_pct"].mean()
avg_enh         = enh_df["return_pct"].mean()
avg_base_trades = base_df["num_trades"].mean()
avg_enh_trades  = enh_df["num_trades"].mean()

print(f"\nAverage return  — baseline: {avg_base:+.2f}%   enhanced: {avg_enh:+.2f}%   delta: {avg_enh - avg_base:+.2f}%")
print(f"Average trades  — baseline: {avg_base_trades:.1f}      enhanced: {avg_enh_trades:.1f}")

print("\nTop 15 coins (enhanced):")
print(
    enh_df[["symbol", "interval", "sweep", "steps", "reset_interval", "return_pct", "num_trades", "score"]]
    .head(15)
    .to_string(index=False)
)

print("\nBottom 10 coins (enhanced):")
print(
    enh_df[["symbol", "interval", "sweep", "steps", "reset_interval", "return_pct", "num_trades", "score"]]
    .tail(10)
    .to_string(index=False)
)

print("\nBiggest improvements (enhanced vs baseline):")
compare = enh_df[["symbol", "return_pct"]].copy().rename(columns={"return_pct": "enhanced_return"})
compare["baseline_return"] = base_df.set_index("symbol").loc[compare["symbol"].values, "return_pct"].values
compare["delta"] = compare["enhanced_return"] - compare["baseline_return"]
compare = compare.sort_values("delta", ascending=False)
print(compare.head(15).to_string(index=False))

print("\nOptimal parameters for top performers (enhanced):")
top = enh_df.head(15)[["symbol", "interval", "sweep", "steps", "reset_interval", "return_pct"]]
print(top.to_string(index=False))

# Save both result sets
base_df.to_csv(os.path.join(results_dir, "grid_search_baseline.csv"), index=False)
enh_df.to_csv(os.path.join(results_dir, "grid_search_enhanced.csv"), index=False)
print(f"\nResults saved to research/results/")
