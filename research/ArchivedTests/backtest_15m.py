"""
Backtest the grid strategy on 15-minute data.

Scales all time-based parameters by 4 (since 1h = 4 x 15m):
  reset_interval: 24h -> 96 candles
  ma_period:      ETH 336h -> 1344 candles
  atr_period:     14 bars (hourly) -> 56 bars (15m)

Runs:
  1. Multi-market shared-capital backtest on production markets (ETH, BNB, SOL)
  2. Individual single-market sweep across all 40 available symbols
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from research.ArchivedTests.backtest import backtest, backtest_multi

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "15m")
INITIAL_CASH = 30_000

# Production market configs scaled for 15m (x4 all period parameters)
MARKET_CONFIGS_15M = {
    "ETH":  {"sweep": 0.10, "steps": 4,  "ma_period": 1344, "reset_interval": 96, "counter_mult": 3.0, "order_size_pct": 0.15, "file": "eth_usd_15m.csv"},
    "BNB":  {"sweep": 0.06, "steps": 12, "ma_period": None,  "reset_interval": 96, "counter_mult": 3.0, "order_size_pct": 0.15, "file": "bnb_usd_15m.csv"},
    "SOL":  {"sweep": 0.20, "steps": 4,  "ma_period": None,  "reset_interval": 96, "counter_mult": 3.0, "order_size_pct": 0.15, "file": "sol_usd_15m.csv"},
}

# Default single-market config (applied to all symbols in the sweep)
DEFAULT_CFG = dict(
    sweep=0.10, steps=8, order_size_pct=0.10,
    reset_interval=96, ma_period=None,
    atr_period=56, atr_multiplier=2.0,
    min_sweep=0.02, max_sweep=0.20,
    circuit_breaker=0.20,
    reactive=True, counter_multiplier=3.0,
)


def load_prices(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["close"])
    return df["close"].tolist(), df["date"].tolist()


# ── 1. Multi-market shared-capital backtest ───────────────────────────────────
print("\n=== Multi-Market Shared-Capital Backtest (15m, 60 days) ===")
print(f"  Capital: ${INITIAL_CASH:,}   Markets: ETH, BNB, SOL\n")

result = backtest_multi(MARKET_CONFIGS_15M, DATA_DIR, initial_cash=INITIAL_CASH)

final_value = result["final_value"]
return_pct  = (final_value - INITIAL_CASH) / INITIAL_CASH * 100
last_prices = result["last_prices"]
symbols     = list(result["holdings"].keys())

print(f"  Starting Capital: ${INITIAL_CASH:>10,.2f}")
print(f"  Final Value:      ${final_value:>10,.2f}  ({return_pct:+.2f}%)")
print(f"  Cash remaining:   ${result['cash']:>10,.2f}")
print(f"  Halted:           {'YES' if result['halted'] else 'No'}")
print()

print(f"{'Symbol':<8} {'Holdings Val':>14} {'Units':>14} {'Last Price':>12} {'Buys':>7} {'Sells':>7} {'Gross Rev':>14}")
print("-" * 85)
for m in symbols:
    hval  = result["holdings"][m] * last_prices[m]
    buys  = len(result["executed_buys"][m])
    sells = len(result["executed_sells"][m])
    print(f"{m:<8} ${hval:>13,.2f} {result['holdings'][m]:>14.6f} ${last_prices[m]:>11,.2f} "
          f"{buys:>7} {sells:>7} ${result['gross_revenue'][m]:>13,.2f}")


# ── 2. Individual symbol sweep across all 40 symbols ─────────────────────────
print("\n\n=== Individual Symbol Backtest Sweep (15m, 60 days) ===")
print(f"  Capital per symbol: ${INITIAL_CASH:,}  (independent, not shared)")
print(f"  Config: sweep={DEFAULT_CFG['sweep']} steps={DEFAULT_CFG['steps']} "
      f"order_size_pct={DEFAULT_CFG['order_size_pct']} cm={DEFAULT_CFG['counter_multiplier']}\n")

# Exclude stablecoins — no grid profit possible
EXCLUDE = {"usdc", "trump"}  # trump has sparse data

rows = []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith("_usd_15m.csv"):
        continue
    symbol = fname.replace("_usd_15m.csv", "").upper()
    if symbol.lower() in EXCLUDE:
        continue

    try:
        prices, dates = load_prices(fname)
        if len(prices) < 200:
            continue

        res = backtest(prices, **DEFAULT_CFG)
        final_val   = res["final_value"]
        ret_pct     = (final_val - INITIAL_CASH) / INITIAL_CASH * 100
        buys        = len(res["buy_points"])
        sells       = len(res["sell_points"])
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
        })
    except Exception as e:
        print(f"  ERR {symbol}: {e}")

# Sort by return descending
rows.sort(key=lambda r: r["return_pct"], reverse=True)

print(f"{'Symbol':<10} {'Return':>9} {'Final Val':>12} {'Cash':>12} {'Hold Val':>11} "
      f"{'GrossRev':>12} {'Buys':>6} {'Sells':>6} {'Halt':>5}")
print("-" * 95)
for r in rows:
    halt_str = "YES" if r["halted"] else "No"
    sign = "+" if r["return_pct"] >= 0 else ""
    print(f"{r['symbol']:<10} {sign}{r['return_pct']:>8.2f}% ${r['final_value']:>11,.2f} "
          f"${r['cash']:>11,.2f} ${r['holdings_val']:>10,.2f} "
          f"${r['gross_rev']:>11,.2f} {r['buys']:>6} {r['sells']:>6} {halt_str:>5}")

winners = [r for r in rows if r["return_pct"] > 0 and not r["halted"]]
losers  = [r for r in rows if r["return_pct"] <= 0 or r["halted"]]
halted  = [r for r in rows if r["halted"]]

print(f"\n  Total symbols: {len(rows)}  |  Winners: {len(winners)}  |  "
      f"Losers: {len(losers)}  |  Halted: {len(halted)}")
print(f"  Median return: {sorted([r['return_pct'] for r in rows])[len(rows)//2]:+.2f}%")
if rows:
    best  = rows[0]
    worst = rows[-1]
    print(f"  Best:  {best['symbol']} {best['return_pct']:+.2f}%")
    print(f"  Worst: {worst['symbol']} {worst['return_pct']:+.2f}%")
