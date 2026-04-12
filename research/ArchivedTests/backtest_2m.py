"""
Backtest the grid strategy on 2-minute data (Mar 6 - Apr 9, 2026 — ~34 days).

Time-based parameter scaling (1h = 30 x 2m candles):
  reset_interval: 24h -> 720 candles (production), 12h -> 360 candles (optimised)
  ma_period:      ETH 336h -> 10,080 candles
  atr_period:     14h -> 420 candles

Runs:
  1. Multi-market shared-capital — production config scaled to 2m
  2. Multi-market shared-capital — optimised config (from 15m param search, scaled to 2m)
  3. Individual single-market sweep across all 39 available symbols
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from backtest import backtest, backtest_multi

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data", "2m")
INITIAL_CASH = 100_000
EXCLUDE      = {"usdc"}   # stablecoin — no grid profit possible

# ── Production config scaled to 2m ────────────────────────────────────────────
PROD_CFG = {
    "ETH": {"sweep": 0.10, "steps": 4,  "ma_period": 10080, "reset_interval": 720,
            "counter_mult": 3.0, "order_size_pct": 0.15, "file": "eth_usd_2m.csv"},
    "BNB": {"sweep": 0.06, "steps": 12, "ma_period": None,  "reset_interval": 720,
            "counter_mult": 3.0, "order_size_pct": 0.15, "file": "bnb_usd_2m.csv"},
    "SOL": {"sweep": 0.20, "steps": 4,  "ma_period": None,  "reset_interval": 720,
            "counter_mult": 3.0, "order_size_pct": 0.15, "file": "sol_usd_2m.csv"},
}

# ── Optimised config (15m search best result, scaled to 2m) ───────────────────
OPT_CFG = {
    "ETH": {"sweep": 0.06, "steps": 4,  "ma_period": 10080, "reset_interval": 360,
            "counter_mult": 4.0, "order_size_pct": 0.05, "file": "eth_usd_2m.csv"},
    "BNB": {"sweep": 0.08, "steps": 12, "ma_period": None,  "reset_interval": 360,
            "counter_mult": 4.0, "order_size_pct": 0.05, "file": "bnb_usd_2m.csv"},
    "SOL": {"sweep": 0.10, "steps": 4,  "ma_period": None,  "reset_interval": 360,
            "counter_mult": 4.0, "order_size_pct": 0.05, "file": "sol_usd_2m.csv"},
}

# ── Default single-market config for the full-symbol sweep ────────────────────
DEFAULT_CFG = dict(
    sweep=0.10, steps=8, order_size_pct=0.10,
    reset_interval=360,          # 12h at 2m resolution
    ma_period=None,
    atr_period=420,              # 14h at 2m resolution
    atr_multiplier=2.0,
    min_sweep=0.02, max_sweep=0.20,
    circuit_breaker=0.20,
    reactive=True, counter_multiplier=3.0,
)


def load_prices(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["close"])
    return df["close"].tolist(), df["date"].tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Multi-market — production config (scaled to 2m)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== [1] Multi-Market Shared-Capital — PRODUCTION Config (2m, ~34 days) ===")
print(f"  Capital: ${INITIAL_CASH:,}   Markets: ETH, BNB, SOL\n")

r_prod = backtest_multi(PROD_CFG, DATA_DIR, initial_cash=INITIAL_CASH)

fv_prod   = r_prod["final_value"]
ret_prod  = (fv_prod - INITIAL_CASH) / INITIAL_CASH * 100
lp_prod   = r_prod["last_prices"]
syms_prod = list(r_prod["holdings"].keys())

print(f"  Starting Capital: ${INITIAL_CASH:>10,.2f}")
print(f"  Final Value:      ${fv_prod:>10,.2f}  ({ret_prod:+.2f}%)")
print(f"  Cash remaining:   ${r_prod['cash']:>10,.2f}")
print(f"  Halted:           {'YES' if r_prod['halted'] else 'No'}")
print()
print(f"  {'Symbol':<8} {'Holdings Val':>14} {'Units':>14} {'Last Price':>12} {'Buys':>7} {'Sells':>7} {'Gross Rev':>14}")
print("  " + "-" * 83)
for m in syms_prod:
    hval  = r_prod["holdings"][m] * lp_prod[m]
    buys  = len(r_prod["executed_buys"][m])
    sells = len(r_prod["executed_sells"][m])
    print(f"  {m:<8} ${hval:>13,.2f} {r_prod['holdings'][m]:>14.6f} ${lp_prod[m]:>11,.2f} "
          f"{buys:>7} {sells:>7} ${r_prod['gross_revenue'][m]:>13,.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Multi-market — optimised config
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n=== [2] Multi-Market Shared-Capital — OPTIMISED Config (2m, ~34 days) ===")
print(f"  Capital: ${INITIAL_CASH:,}   Markets: ETH, BNB, SOL\n")

r_opt = backtest_multi(OPT_CFG, DATA_DIR, initial_cash=INITIAL_CASH)

fv_opt   = r_opt["final_value"]
ret_opt  = (fv_opt - INITIAL_CASH) / INITIAL_CASH * 100
lp_opt   = r_opt["last_prices"]
syms_opt = list(r_opt["holdings"].keys())

print(f"  Starting Capital: ${INITIAL_CASH:>10,.2f}")
print(f"  Final Value:      ${fv_opt:>10,.2f}  ({ret_opt:+.2f}%)")
print(f"  Cash remaining:   ${r_opt['cash']:>10,.2f}")
print(f"  Halted:           {'YES' if r_opt['halted'] else 'No'}")
print()
print(f"  {'Symbol':<8} {'Holdings Val':>14} {'Units':>14} {'Last Price':>12} {'Buys':>7} {'Sells':>7} {'Gross Rev':>14}")
print("  " + "-" * 83)
for m in syms_opt:
    hval  = r_opt["holdings"][m] * lp_opt[m]
    buys  = len(r_opt["executed_buys"][m])
    sells = len(r_opt["executed_sells"][m])
    print(f"  {m:<8} ${hval:>13,.2f} {r_opt['holdings'][m]:>14.6f} ${lp_opt[m]:>11,.2f} "
          f"{buys:>7} {sells:>7} ${r_opt['gross_revenue'][m]:>13,.2f}")

print(f"\n  Optimised vs Production delta: {ret_opt - ret_prod:+.2f} percentage points")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Individual symbol sweep
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n=== [3] Individual Symbol Backtest Sweep (2m, ~34 days) ===")
print(f"  Capital per symbol: ${INITIAL_CASH:,}  (independent)")
print(f"  Config: sweep={DEFAULT_CFG['sweep']} steps={DEFAULT_CFG['steps']} "
      f"order_size_pct={DEFAULT_CFG['order_size_pct']} cm={DEFAULT_CFG['counter_multiplier']} "
      f"reset={DEFAULT_CFG['reset_interval']} candles\n")

rows = []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith("_usd_2m.csv"):
        continue
    symbol = fname.replace("_usd_2m.csv", "").upper()
    if symbol.lower() in EXCLUDE:
        continue

    try:
        prices, dates = load_prices(fname)
        if len(prices) < 500:
            continue

        res = backtest(prices, **DEFAULT_CFG, initial_cash=INITIAL_CASH)
        final_val    = res["final_value"]
        ret_pct      = (final_val - INITIAL_CASH) / INITIAL_CASH * 100
        buys         = len(res["buy_points"])
        sells        = len(res["sell_points"])
        holdings_val = res["holdings"] * prices[-1]

        rows.append({
            "symbol":       symbol,
            "final_value":  final_val,
            "return_pct":   ret_pct,
            "cash":         res["cash"],
            "holdings_val": holdings_val,
            "gross_rev":    res["profit"],
            "buys":         buys,
            "sells":        sells,
            "halted":       res["halted"],
            "candles":      len(prices),
            "start":        str(dates[0])[:10],
            "end":          str(dates[-1])[:10],
        })
    except Exception as e:
        print(f"  ERR {symbol}: {e}")

rows.sort(key=lambda r: r["return_pct"], reverse=True)

print(f"  {'Symbol':<10} {'Return':>9} {'Final Val':>12} {'Cash':>12} {'Hold Val':>11} "
      f"{'GrossRev':>12} {'Buys':>6} {'Sells':>6} {'Halt':>5}")
print("  " + "-" * 97)
for r in rows:
    halt_str = "YES" if r["halted"] else "No"
    sign     = "+" if r["return_pct"] >= 0 else ""
    print(f"  {r['symbol']:<10} {sign}{r['return_pct']:>8.2f}% ${r['final_value']:>11,.2f} "
          f"${r['cash']:>11,.2f} ${r['holdings_val']:>10,.2f} "
          f"${r['gross_rev']:>11,.2f} {r['buys']:>6} {r['sells']:>6} {halt_str:>5}")

winners = [r for r in rows if r["return_pct"] > 0 and not r["halted"]]
losers  = [r for r in rows if r["return_pct"] <= 0 or r["halted"]]
halted  = [r for r in rows if r["halted"]]
med_ret = sorted(r["return_pct"] for r in rows)[len(rows) // 2]

print(f"\n  Total: {len(rows)}  |  Winners: {len(winners)}  |  Losers: {len(losers)}  |  Halted: {len(halted)}")
print(f"  Median return: {med_ret:+.2f}%")
if rows:
    print(f"  Best:  {rows[0]['symbol']} {rows[0]['return_pct']:+.2f}%")
    print(f"  Worst: {rows[-1]['symbol']} {rows[-1]['return_pct']:+.2f}%")

# ── Cross-interval comparison ──────────────────────────────────────────────────
print("\n\n=== Interval Comparison Summary ===")
print(f"  {'Interval':<10} {'Window':>10} {'Candles':>9} {'Multi-Mkt (prod)':>18} {'Multi-Mkt (opt)':>16}")
print(f"  {'--------':<10} {'------':>10} {'-------':>9} {'----------------':>18} {'---------------':>16}")
print(f"  {'15m':<10} {'60 days':>10} {'~5,672':>9} {'   +7.07%':>18} {'  +18.15%':>16}")
print(f"  {'2m':<10} {'~34 days':>10} {'~24,400':>9} {ret_prod:>+17.2f}% {ret_opt:>+15.2f}%")
