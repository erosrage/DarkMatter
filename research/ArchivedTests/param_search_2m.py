"""
Comprehensive parameter search on 2-minute data — INDEPENDENT markets, 100k each.

Each symbol is backtested independently with its own $100,000 starting capital.
No shared pool — maximises return extraction per symbol.

Search covers:
  sweep          — grid width (key profit lever)
  steps          — grid levels (fill frequency vs capital efficiency)
  reset_interval — candles between resets (12h=360, 24h=720, 6h=180, never)
  order_size_pct — fraction of cash per grid level
  counter_mult   — profit target multiplier on reactive counter-orders
  atr_multiplier — dynamic sweep sensitivity (0=disabled, else scales sweep by ATR)

~2,916 runs per symbol (targeted: ETH, BNB, SOL).
Full symbol sweep uses best per-symbol config from search.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from itertools import product
from research.ArchivedTests.backtest import backtest

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data", "2m")
INITIAL_CASH = 100_000
EXCLUDE      = {"usdc", "arb", "eigen", "ena"}   # empty/stablecoin data

# ── Scaled to 2m resolution (1h = 30 candles) ─────────────────────────────────
# reset: 6h=180, 12h=360, 24h=720, never=None
# ma:    ETH 336h = 10080 candles (too large for 34d data → skip or use None)
# atr:   14h = 420 candles

# ── Search space ──────────────────────────────────────────────────────────────
SWEEP_VALS      = [0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
STEPS_VALS      = [4, 8, 12]
RESET_VALS      = [180, 360, 720, None]   # 6h, 12h, 24h, never
ORDER_SIZE_VALS = [0.03, 0.05, 0.08, 0.10, 0.15]
COUNTER_VALS    = [2.0, 3.0, 4.0, 5.0]

# ATR dynamic sweep: use atr_period=420 (14h) or disable (atr_period=None)
ATR_MULT_VALS   = [0.0, 2.0, 3.0]   # 0.0 = disabled (use static sweep)

# Fixed params
CIRCUIT_BREAKER = 0.20
MIN_SWEEP       = 0.02
MAX_SWEEP       = 0.30
ATR_PERIOD      = 420   # 14h at 2m resolution (only used when atr_mult > 0)

# Primary symbols to run full search on
PRIMARY_SYMBOLS = ["ETH", "BNB", "SOL"]


def load_prices(symbol):
    fname = f"{symbol.lower()}_usd_2m.csv"
    path  = os.path.join(DATA_DIR, fname)
    df    = pd.read_csv(path, parse_dates=["date"])
    df    = df.dropna(subset=["close"])
    prices = df["close"].tolist()
    dates  = df["date"].tolist()
    return prices, dates


def run_search(symbol, prices):
    """Grid search over all parameter combinations for a single symbol."""
    results = []
    total = len(SWEEP_VALS) * len(STEPS_VALS) * len(RESET_VALS) * len(ORDER_SIZE_VALS) * len(COUNTER_VALS) * len(ATR_MULT_VALS)
    done  = 0

    for sweep, steps, reset, order_pct, cm, atr_mult in product(
            SWEEP_VALS, STEPS_VALS, RESET_VALS, ORDER_SIZE_VALS, COUNTER_VALS, ATR_MULT_VALS):

        use_atr    = atr_mult > 0
        atr_period = ATR_PERIOD if use_atr else None

        res = backtest(
            prices,
            sweep=sweep,
            steps=steps,
            order_size_pct=order_pct,
            reset_interval=reset,
            ma_period=None,           # MA skipped — 336h > 34-day window
            atr_period=atr_period,
            atr_multiplier=atr_mult if use_atr else 2.0,
            min_sweep=MIN_SWEEP,
            max_sweep=MAX_SWEEP,
            circuit_breaker=CIRCUIT_BREAKER,
            reactive=True,
            counter_multiplier=cm,
            initial_cash=INITIAL_CASH,
        )

        ret_pct  = (res["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
        trades   = len(res["buy_points"]) + len(res["sell_points"])

        results.append({
            "return_pct":  ret_pct,
            "final_value": res["final_value"],
            "cash":        res["cash"],
            "profit":      res["profit"],
            "halted":      res["halted"],
            "trades":      trades,
            "sweep":       sweep,
            "steps":       steps,
            "reset":       reset,
            "order_pct":   order_pct,
            "counter_m":   cm,
            "atr_mult":    atr_mult,
        })

        done += 1
        if done % 500 == 0:
            best = max(results, key=lambda x: x["return_pct"])
            print(f"  [{symbol}] [{done:>5}/{total}]  best so far: {best['return_pct']:+.2f}%", flush=True)

    return pd.DataFrame(results).sort_values("return_pct", ascending=False)


def print_results(symbol, df):
    profitable = df[df["return_pct"] > 0]
    halted_pct = 100 * df["halted"].sum() / len(df)
    print(f"\n{'='*115}")
    print(f"  {symbol}  |  Runs: {len(df):,}  |  Profitable: {len(profitable):,} ({100*len(profitable)/len(df):.1f}%)  |  Halted: {halted_pct:.1f}%")
    print(f"{'='*115}\n")

    print(f"  TOP 15 CONFIGURATIONS - {symbol}")
    hdr = (f"  {'#':>3}  {'Return':>9}  {'FinalVal':>12}  {'Sweep':>6}  {'Steps':>5}  "
           f"{'Reset':>6}  {'OrdPct':>7}  {'CM':>5}  {'ATR':>5}  {'Trades':>7}  Halt")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for rank, (_, row) in enumerate(df.head(15).iterrows(), 1):
        rst = str(int(row["reset"])) if pd.notna(row["reset"]) else "None"
        print(f"  {rank:>3}  {row['return_pct']:>+8.2f}%  ${row['final_value']:>11,.0f}  "
              f"{row['sweep']:.2f}    {int(row['steps']):>5}  "
              f"{rst:>6}  {row['order_pct']:>7.3f}  {row['counter_m']:>5.1f}  "
              f"{row['atr_mult']:>5.1f}  {int(row['trades']):>7}  {'YES' if row['halted'] else 'no'}")

    # Sensitivity
    print(f"\n  --- Parameter Sensitivity - {symbol} (mean return) ---")
    for label, col in [("Sweep", "sweep"), ("Steps", "steps"), ("Reset", "reset"),
                        ("Order%", "order_pct"), ("CounterM", "counter_m"), ("ATR mult", "atr_mult")]:
        grp = df.groupby(col)["return_pct"].mean().sort_index()
        vals = "  ".join(f"{k}={v:+.2f}%" for k, v in grp.items())
        print(f"    {label:<10}: {vals}")


def best_config_snippet(symbol, row):
    rst = str(int(row["reset"])) if pd.notna(row["reset"]) else "None"
    atr_p = ATR_PERIOD if row["atr_mult"] > 0 else "None"
    return (f'  "{symbol}": {{'
            f'"sweep": {row["sweep"]}, "steps": {int(row["steps"])}, '
            f'"reset_interval": {rst}, "order_size_pct": {row["order_pct"]}, '
            f'"counter_mult": {row["counter_m"]}, "atr_mult": {row["atr_mult"]}, '
            f'"atr_period": {atr_p}, '
            f'"file": "{symbol.lower()}_usd_2m.csv"}}')


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMARY SEARCH — ETH, BNB, SOL  (full grid)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*115}")
print(f"  2m PARAMETER SEARCH — Independent markets, ${INITIAL_CASH:,} each")
total_runs = (len(SWEEP_VALS) * len(STEPS_VALS) * len(RESET_VALS) *
              len(ORDER_SIZE_VALS) * len(COUNTER_VALS) * len(ATR_MULT_VALS))
print(f"  Search space: {total_runs:,} runs per symbol  |  Data: ~34 days @ 2m")
print(f"{'='*115}\n")

best_per_sym = {}
all_dfs = {}

for sym in PRIMARY_SYMBOLS:
    print(f"\n  Loading {sym}...")
    prices, dates = load_prices(sym)
    print(f"  {sym}: {len(prices):,} candles  ({str(dates[0])[:10]} to {str(dates[-1])[:10]})")
    print(f"  Running {total_runs:,} parameter combinations...")

    df = run_search(sym, prices)
    all_dfs[sym] = df
    best_per_sym[sym] = df.iloc[0]
    print_results(sym, df)

    out_path = os.path.join(os.path.dirname(__file__), f"param_search_2m_{sym.lower()}_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# BEST CONFIGS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*115}")
print("  BEST CONFIGS PER SYMBOL")
print(f"{'='*115}\n")

for sym in PRIMARY_SYMBOLS:
    row = best_per_sym[sym]
    rst = str(int(row["reset"])) if pd.notna(row["reset"]) else "None"
    print(f"  {sym}: {row['return_pct']:+.2f}%  sweep={row['sweep']}  steps={int(row['steps'])}  "
          f"reset={rst}  order_pct={row['order_pct']}  cm={row['counter_m']}  atr={row['atr_mult']}")

print("\n\nCopy-paste config:")
print("OPTIMISED_CFG_2M = {")
for sym in PRIMARY_SYMBOLS:
    print(best_config_snippet(sym, best_per_sym[sym]))
print("}")


# ═══════════════════════════════════════════════════════════════════════════════
# FULL SYMBOL SWEEP — use best per-symbol params as template, adapt per symbol
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*115}")
print("  FULL SYMBOL SWEEP — All available 2m symbols, $100k each")
print(f"{'='*115}\n")

# Use median best config from primary symbols as default
import statistics
best_sweep   = statistics.median(best_per_sym[s]["sweep"]     for s in PRIMARY_SYMBOLS)
best_steps   = int(statistics.median(best_per_sym[s]["steps"] for s in PRIMARY_SYMBOLS))
best_reset   = best_per_sym["ETH"]["reset"]   # use ETH's best reset as default
best_pct     = statistics.median(best_per_sym[s]["order_pct"]  for s in PRIMARY_SYMBOLS)
best_cm      = statistics.median(best_per_sym[s]["counter_m"]  for s in PRIMARY_SYMBOLS)
best_atr     = statistics.median(best_per_sym[s]["atr_mult"]   for s in PRIMARY_SYMBOLS)
best_atr_p   = ATR_PERIOD if best_atr > 0 else None

sweep_cfg = dict(
    sweep=best_sweep, steps=best_steps, order_size_pct=best_pct,
    reset_interval=best_reset,
    ma_period=None,
    atr_period=best_atr_p,
    atr_multiplier=best_atr if best_atr > 0 else 2.0,
    min_sweep=MIN_SWEEP, max_sweep=MAX_SWEEP,
    circuit_breaker=CIRCUIT_BREAKER,
    reactive=True, counter_multiplier=best_cm,
    initial_cash=INITIAL_CASH,
)

print(f"  Using median best config: sweep={best_sweep} steps={best_steps} "
      f"reset={best_reset} order_pct={best_pct} cm={best_cm} atr={best_atr}\n")

sweep_rows = []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith("_usd_2m.csv"):
        continue
    symbol = fname.replace("_usd_2m.csv", "").upper()
    if symbol.lower() in EXCLUDE:
        continue

    try:
        prices, dates = load_prices(symbol)
        if len(prices) < 500:
            continue

        res      = backtest(prices, **sweep_cfg)
        ret_pct  = (res["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
        trades   = len(res["buy_points"]) + len(res["sell_points"])
        hold_val = res["holdings"] * prices[-1]

        sweep_rows.append({
            "symbol":      symbol,
            "return_pct":  ret_pct,
            "final_value": res["final_value"],
            "cash":        res["cash"],
            "hold_val":    hold_val,
            "profit":      res["profit"],
            "buys":        len(res["buy_points"]),
            "sells":       len(res["sell_points"]),
            "halted":      res["halted"],
            "candles":     len(prices),
        })
    except Exception as e:
        print(f"  ERR {symbol}: {e}")

sweep_rows.sort(key=lambda r: r["return_pct"], reverse=True)

print(f"  {'Symbol':<10} {'Return':>9} {'Final Val':>13} {'Cash':>13} {'Hold Val':>12} "
      f"{'GrossRev':>13} {'Buys':>6} {'Sells':>6} {'Halt':>5}")
print("  " + "-" * 100)
for r in sweep_rows:
    sign = "+" if r["return_pct"] >= 0 else ""
    print(f"  {r['symbol']:<10} {sign}{r['return_pct']:>8.2f}%  ${r['final_value']:>12,.0f}  "
          f"${r['cash']:>12,.0f}  ${r['hold_val']:>11,.0f}  "
          f"${r['profit']:>12,.0f}  {r['buys']:>6}  {r['sells']:>6}  "
          f"{'YES' if r['halted'] else 'No':>5}")

winners = [r for r in sweep_rows if r["return_pct"] > 0 and not r["halted"]]
losers  = [r for r in sweep_rows if r["return_pct"] <= 0 or r["halted"]]
halted  = [r for r in sweep_rows if r["halted"]]
if sweep_rows:
    med = sorted(r["return_pct"] for r in sweep_rows)[len(sweep_rows) // 2]
    print(f"\n  Total: {len(sweep_rows)}  |  Winners: {len(winners)}  |  Losers: {len(losers)}  |  Halted: {len(halted)}")
    print(f"  Median return: {med:+.2f}%")
    print(f"  Best:   {sweep_rows[0]['symbol']} {sweep_rows[0]['return_pct']:+.2f}%")
    print(f"  Worst:  {sweep_rows[-1]['symbol']} {sweep_rows[-1]['return_pct']:+.2f}%")
    avg = sum(r["return_pct"] for r in winners) / len(winners) if winners else 0
    print(f"  Avg winner return: {avg:+.2f}%")
