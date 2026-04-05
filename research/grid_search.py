"""
Grid search over sweep/steps/reset_interval across all crypto CSVs.
Runs two passes — baseline (no enhancements) vs enhanced (all 5 optimizations) —
then reports the delta so you can see exactly what the improvements bought.

Score = return_pct / num_trades  (maximise return, minimise churn)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import pandas as pd
from backtest import backtest

INITIAL_CASH = 100_000
ORDER_SIZE   = 20

SWEEP_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
STEPS_VALUES = [2, 4, 6, 8, 10, 12, 16]
RESET_VALUES = [None, 7, 14, 30, 60, 90]

# Fixed defaults for the 5 new features (applied in enhanced pass only)
ENHANCED_KWARGS = dict(
    ma_period       = 20,
    atr_period      = 14,
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

base_dir     = os.path.dirname(os.path.abspath(__file__))
data_dir     = os.path.join(base_dir, "data")
results_dir  = os.path.join(base_dir, "results")
csv_files    = sorted(glob.glob(os.path.join(data_dir, "*_usd_2y.csv")))
print(f"Found {len(csv_files)} CSVs\n")


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


baseline_rows  = []
enhanced_rows  = []

for csv_path in csv_files:
    symbol = os.path.basename(csv_path).replace("_usd_2y.csv", "").upper()
    df = pd.read_csv(csv_path)
    if "close" not in df.columns or len(df) < 30:
        print(f"[SKIP] {symbol}")
        continue

    prices = df["close"].dropna().tolist()

    b = best_params(prices, BASELINE_KWARGS)
    e = best_params(prices, ENHANCED_KWARGS)

    b["symbol"] = symbol
    e["symbol"] = symbol
    baseline_rows.append(b)
    enhanced_rows.append(e)

    delta = e["return_pct"] - b["return_pct"]
    rst_b = f"{b['reset_interval']:3d}d" if b["reset_interval"] else "None"
    rst_e = f"{e['reset_interval']:3d}d" if e["reset_interval"] else "None"
    print(
        f"{symbol:12s}  "
        f"base={b['return_pct']:+7.2f}% ({b['num_trades']:3d}t)  "
        f"enhanced={e['return_pct']:+7.2f}% ({e['num_trades']:3d}t)  "
        f"delta={delta:+7.2f}%  "
        f"[sweep={e['sweep']:.2f} steps={e['steps']} reset={rst_e}]"
    )


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)

base_df = pd.DataFrame(baseline_rows).sort_values("return_pct", ascending=False)
enh_df  = pd.DataFrame(enhanced_rows).sort_values("return_pct", ascending=False)

avg_base = base_df["return_pct"].mean()
avg_enh  = enh_df["return_pct"].mean()
avg_base_trades = base_df["num_trades"].mean()
avg_enh_trades  = enh_df["num_trades"].mean()

print(f"\nAverage return  — baseline: {avg_base:+.2f}%   enhanced: {avg_enh:+.2f}%   delta: {avg_enh - avg_base:+.2f}%")
print(f"Average trades  — baseline: {avg_base_trades:.1f}      enhanced: {avg_enh_trades:.1f}")

print("\nTop 10 coins (enhanced):")
print(
    enh_df[["symbol", "sweep", "steps", "reset_interval", "return_pct", "num_trades", "score"]]
    .head(10)
    .to_string(index=False)
)

print("\nBottom 10 coins (enhanced):")
print(
    enh_df[["symbol", "sweep", "steps", "reset_interval", "return_pct", "num_trades", "score"]]
    .tail(10)
    .to_string(index=False)
)

print("\nBiggest improvements (enhanced vs baseline):")
compare = enh_df[["symbol", "return_pct"]].copy()
compare = compare.rename(columns={"return_pct": "enhanced_return"})
compare["baseline_return"] = base_df.set_index("symbol").loc[compare["symbol"].values, "return_pct"].values
compare["delta"] = compare["enhanced_return"] - compare["baseline_return"]
compare = compare.sort_values("delta", ascending=False)
print(compare.head(10).to_string(index=False))

# Save both result sets
base_df.to_csv(os.path.join(results_dir, "grid_search_baseline.csv"), index=False)
enh_df.to_csv(os.path.join(results_dir, "grid_search_enhanced.csv"), index=False)
print(f"\nResults saved to research/results/")
