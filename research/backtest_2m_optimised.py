"""
Optimised 2m backtest — per-symbol independent $100k, best configs from 3-round search.

Search process:
  Round 1: 4,320 combos per symbol (ETH/BNB/SOL) — identified key param directions
  Round 2: Extended range on ETH + AVAX — found sweep/steps breakpoints
  Round 3: 800 combos, 13 symbols — final per-symbol global optimum

All markets run independently with their own $100,000 starting capital.
Robinhood has no $100k limit per trade restriction, so this mirrors real operation.

Optimised vs baseline improvements:
  ONDO:  +14.97% -> +30.91%  (+15.94pp)
  AVAX:  +18.98% -> +25.71%  (+6.73pp)
  ETC:   +16.11% -> +23.65%  (+7.54pp)
  LINK:  +15.20% -> +18.60%  (+3.40pp)
  DOGE:  +8.18%  -> +17.40%  (+9.22pp)
  LDO:   +12.51% -> +17.19%  (+4.68pp)
  ETH:   +13.61% -> +16.65%  (+3.04pp)
  BTC:   +8.25%  -> +14.85%  (+6.60pp)
  BNB:   +3.87%  -> +5.44%   (+1.57pp)
  SOL:   +8.67%  -> +8.79%   (+0.12pp)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from research.ArchivedTests.backtest import backtest

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data", "2m")
INITIAL_CASH = 100_000

# ── Per-symbol optimised configs (global best across all 3 search rounds) ──────
# Key insights from 13,000+ backtests:
#   counter_mult=8-12: biggest single lever — counter orders capture much larger moves
#   order_pct=0.25-0.50: with many steps, excess levels aren't placed; concentrates
#                         capital at the closest (most likely to fill) grid levels
#   steps=20-48: more levels gives finer grid; high order_pct auto-selects closest ones
#   reset=None: avoids disrupting profitable open reactive orders
#   sweep=0.10-0.15 for most assets (except ETH which benefits from tight 0.06)
OPTIMISED_CFG = {
    # Global best (across round 1: 4,320 combos, round 2: extended range, round 3: 800 combos, round 4: 400 ultra-fine)
    "ONDO":    dict(sweep=0.15, steps=32, order_size_pct=0.50, counter_multiplier=10.0),  # +35.56%
    "AVAX":    dict(sweep=0.12, steps=20, order_size_pct=0.20, counter_multiplier=12.0),  # +26.34%
    "ETC":     dict(sweep=0.15, steps=16, order_size_pct=0.50, counter_multiplier=8.0),   # +25.00%
    "LINK":    dict(sweep=0.15, steps=12, order_size_pct=0.25, counter_multiplier=5.0),   # +18.60%
    "DOGE":    dict(sweep=0.12, steps=32, order_size_pct=0.50, counter_multiplier=6.0),   # +18.47%
    "LDO":     dict(sweep=0.15, steps=12, order_size_pct=0.20, counter_multiplier=8.0),   # +17.19%
    "ETH":     dict(sweep=0.06, steps=20, order_size_pct=0.25, counter_multiplier=5.0),   # +16.65% (round 3 beats ultra-fine)
    "VIRTUAL": dict(sweep=0.15, steps=12, order_size_pct=0.20, counter_multiplier=8.0),   # +15.90%
    "BTC":     dict(sweep=0.10, steps=48, order_size_pct=0.50, counter_multiplier=6.0),   # +14.99%
    "TON":     dict(sweep=0.15, steps=12, order_size_pct=0.25, counter_multiplier=8.0),   # +14.21%
    "ADA":     dict(sweep=0.15, steps=12, order_size_pct=0.20, counter_multiplier=4.0),   # +12.15%
    "SOL":     dict(sweep=0.15, steps=20, order_size_pct=0.10, counter_multiplier=8.0),   # +8.79%
    "BNB":     dict(sweep=0.15, steps=12, order_size_pct=0.25, counter_multiplier=8.0),   # +5.44%
}

# Original baseline (default config from round-1 sweep, same params all symbols)
BASELINE_RETURN = {
    "ONDO": 14.97, "AVAX": 18.98, "ETC": 16.11, "LINK": 15.20, "DOGE": 8.18,
    "LDO": 12.51, "ETH": 13.61, "VIRTUAL": 11.41, "BTC": 8.25, "TON": 9.26,
    "ADA": 9.28, "SOL": 8.67, "BNB": 3.87,
}

SHARED_PARAMS = dict(
    reset_interval=None,
    ma_period=None,
    atr_period=None,
    circuit_breaker=0.20,
    reactive=True,
    min_sweep=0.005,
    max_sweep=0.50,
    initial_cash=INITIAL_CASH,
)


def load_prices(symbol):
    fname = f"{symbol.lower()}_usd_2m.csv"
    path  = os.path.join(DATA_DIR, fname)
    df    = pd.read_csv(path, parse_dates=["date"])
    df    = df.dropna(subset=["close"])
    return df["close"].tolist(), df["date"].tolist()


print(f"\n{'='*115}")
print(f"  OPTIMISED 2m BACKTEST -- Independent $100,000 per symbol  |  ~34 days")
print(f"  Config: per-symbol tuned params  |  reset=None  |  reactive=True")
print(f"{'='*115}\n")

print(f"  {'Symbol':<10} {'Baseline':>10} {'Optimised':>11} {'Delta':>8} "
      f"{'FinalVal':>13} {'Buys':>6} {'Sells':>6} {'GrossRev':>13} {'Halt':>5}")
print(f"  {'-'*100}")

results = []
for sym, cfg in OPTIMISED_CFG.items():
    try:
        prices, dates = load_prices(sym)
    except Exception as e:
        print(f"  {sym:<10} ERR: {e}")
        continue

    r       = backtest(prices, **cfg, **SHARED_PARAMS)
    ret     = (r["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
    hold_v  = r["holdings"] * prices[-1]
    buys    = len(r["buy_points"])
    sells   = len(r["sell_points"])
    base    = BASELINE_RETURN.get(sym)
    delta   = ret - base if base is not None else 0.0
    delta_s = f"{delta:>+.2f}pp" if base is not None else "  N/A"
    base_s  = f"{base:>+.2f}%" if base is not None else "  N/A"

    results.append({
        "symbol":      sym,
        "return_pct":  ret,
        "final_value": r["final_value"],
        "cash":        r["cash"],
        "hold_val":    hold_v,
        "gross_rev":   r["profit"],
        "buys":        buys,
        "sells":       sells,
        "halted":      r["halted"],
        "delta":       delta,
        "baseline":    base,
    })

    print(f"  {sym:<10} {base_s:>10} {ret:>+10.2f}% {delta_s:>8} "
          f"${r['final_value']:>12,.0f} {buys:>6} {sells:>6} "
          f"${r['profit']:>12,.0f} {'YES' if r['halted'] else 'No':>5}")

print(f"\n{'='*115}")

total_initial = INITIAL_CASH * len(results)
total_final   = sum(r["final_value"] for r in results)
total_profit  = total_final - total_initial
total_ret     = total_profit / total_initial * 100
avg_ret       = sum(r["return_pct"] for r in results) / len(results)
avg_delta     = sum(r["delta"] for r in results) / len(results)
winners       = [r for r in results if r["return_pct"] > 0 and not r["halted"]]
halted_n      = sum(1 for r in results if r["halted"])

print(f"\n  AGGREGATE SUMMARY ({len(results)} markets, ${total_initial:,.0f} total capital)")
print(f"  Total initial:  ${total_initial:>14,.0f}")
print(f"  Total final:    ${total_final:>14,.0f}")
print(f"  Total profit:   ${total_profit:>14,.0f}  ({total_ret:>+.2f}%)")
print(f"  Avg per market: {avg_ret:>+14.2f}%")
print(f"  Avg improvement vs baseline: {avg_delta:>+.2f}pp")
print(f"  Winners: {len(winners)}/{len(results)}  |  Halted: {halted_n}")

# Ranked by return
results.sort(key=lambda r: r["return_pct"], reverse=True)
print(f"\n  Ranked by return:")
for r in results:
    sign = "+" if r["return_pct"] >= 0 else ""
    print(f"    {r['symbol']:<10} {sign}{r['return_pct']:>7.2f}%  delta={r['delta']:>+.2f}pp")

print(f"\n{'='*115}")
print("  CONFIG REFERENCE")
print(f"{'='*115}")
for sym, cfg in OPTIMISED_CFG.items():
    if sym not in {r["symbol"] for r in results}:
        continue
    print(f'  "{sym}": sweep={cfg["sweep"]}, steps={cfg["steps"]}, '
          f'order_pct={cfg["order_size_pct"]}, counter_mult={cfg["counter_multiplier"]}')
