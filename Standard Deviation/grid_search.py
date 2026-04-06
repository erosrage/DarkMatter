"""
Grid search for the Standard Deviation strategy across all hourly CSVs.

Search space:
  window        — rolling lookback for mean/σ (hours)
  steps         — levels on each side
  std_multiplier— σ spacing between levels
  reset_interval— hours between level resets (None = reset every candle)

Two passes per symbol:
  baseline — no enhancements (no trend filter, no pyramid, no circuit breaker)
  enhanced — MA trend filter + pyramid sizing + circuit breaker

Score = return_pct (raw return, no trade-count penalty)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import pandas as pd
from backtest import backtest

INITIAL_CASH = 100_000
ORDER_SIZE   = 20

WINDOW_VALUES  = [48, 168, 336, 720]          # 2d, 1wk, 2wk, 1mo
STEPS_VALUES   = [2, 3, 4, 6]
STD_VALUES     = [0.5, 1.0, 1.5, 2.0]
RESET_VALUES   = [None, 24, 168, 720]         # None, 1d, 1wk, 1mo

ENHANCED_KWARGS = dict(
    ma_period       = 168,
    atr_period      = 48,
    pyramid         = True,
    circuit_breaker = 0.20,
)

BASELINE_KWARGS = dict(
    ma_period       = None,
    atr_period      = None,
    pyramid         = False,
    circuit_breaker = None,
)

data_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "research", "data", "2y", "hourly")
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)

symbol_files = {}
for path in sorted(glob.glob(os.path.join(data_dir, "*_usd_1h.csv"))):
    sym = os.path.basename(path).replace("_usd_1h.csv", "").upper()
    symbol_files[sym] = path

total_combos = len(WINDOW_VALUES) * len(STEPS_VALUES) * len(STD_VALUES) * len(RESET_VALUES)
print(f"Found {len(symbol_files)} symbols  |  {total_combos} combinations per pass\n")


def best_params(prices, extra_kwargs):
    best = None
    for window in WINDOW_VALUES:
        for steps in STEPS_VALUES:
            for std_mult in STD_VALUES:
                for reset in RESET_VALUES:
                    r = backtest(
                        prices,
                        steps          = steps,
                        std_multiplier = std_mult,
                        window         = window,
                        order_size     = ORDER_SIZE,
                        reset_interval = reset,
                        **extra_kwargs,
                    )
                    num_trades = len(r["buy_points"]) + len(r["sell_points"])
                    return_pct = (r["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
                    # Score = raw return (no trade-count penalty)
                    score = return_pct
                    if best is None or score > best["score"]:
                        best = dict(
                            window         = window,
                            steps          = steps,
                            std_multiplier = std_mult,
                            reset_interval = reset,
                            return_pct     = return_pct,
                            num_trades     = num_trades,
                            final_value    = r["final_value"],
                            score          = score,
                            halted         = r.get("halted", False),
                        )
    return best


baseline_rows = []
enhanced_rows = []

for symbol, csv_path in sorted(symbol_files.items()):
    df = pd.read_csv(csv_path)
    if "close" not in df.columns or len(df) < 200:
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
    rst_e = f"{e['reset_interval']:4d}h" if e["reset_interval"] else " None"
    print(
        f"{symbol:10s}  "
        f"base={b['return_pct']:+8.2f}%  "
        f"enhanced={e['return_pct']:+8.2f}%  "
        f"delta={delta:+7.2f}%  "
        f"[win={e['window']:3d}h  steps={e['steps']}  std={e['std_multiplier']:.1f}  reset={rst_e}]"
        + (" HALTED" if e["halted"] else "")
    )


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

base_df = pd.DataFrame(baseline_rows).sort_values("return_pct", ascending=False)
enh_df  = pd.DataFrame(enhanced_rows).sort_values("return_pct", ascending=False)

print(f"\nAverage return  — baseline: {base_df['return_pct'].mean():+.2f}%   "
      f"enhanced: {enh_df['return_pct'].mean():+.2f}%   "
      f"delta: {enh_df['return_pct'].mean() - base_df['return_pct'].mean():+.2f}%")
print(f"Average trades  — baseline: {base_df['num_trades'].mean():.1f}   "
      f"enhanced: {enh_df['num_trades'].mean():.1f}")

print("\nTop 15 coins (enhanced):")
print(
    enh_df[["symbol", "window", "steps", "std_multiplier", "reset_interval",
            "return_pct", "num_trades"]]
    .head(15)
    .to_string(index=False)
)

print("\nBottom 10 coins (enhanced):")
print(
    enh_df[["symbol", "window", "steps", "std_multiplier", "reset_interval",
            "return_pct", "num_trades"]]
    .tail(10)
    .to_string(index=False)
)

print("\nBiggest improvements (enhanced vs baseline):")
compare = enh_df[["symbol", "return_pct"]].copy().rename(columns={"return_pct": "enhanced"})
compare["baseline"] = base_df.set_index("symbol").loc[compare["symbol"].values, "return_pct"].values
compare["delta"]    = compare["enhanced"] - compare["baseline"]
print(compare.sort_values("delta", ascending=False).head(15).to_string(index=False))

print("\nOptimal parameters — top 15 (enhanced):")
print(
    enh_df[["symbol", "window", "steps", "std_multiplier", "reset_interval", "return_pct"]]
    .head(15)
    .to_string(index=False)
)

base_df.to_csv(os.path.join(results_dir, "sd_baseline.csv"),  index=False)
enh_df.to_csv(os.path.join(results_dir, "sd_enhanced.csv"),   index=False)
print(f"\nResults saved to Standard Deviation/results/")
