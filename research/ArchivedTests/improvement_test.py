"""
Iteration 2: Test targeted improvements against baseline.

Improvements tested:
  1. buy_ma_period  — short MA gates initial BUY order placement (avoids downtrends)
  2. drift_reset_pct — price-drift triggered grid reset (keeps grid near current price)
  3. Top market selection — AVAX, ETC, LINK, ONDO, LDO instead of ETH/BNB/SOL

Each test runs independently at $100,000 per symbol.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from research.ArchivedTests.backtest import backtest

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data", "2m")
INITIAL_CASH = 100_000

# Best params from search (round 1)
BEST_PARAMS = {
    "ETH": dict(sweep=0.04, steps=12, reset_interval=None, order_size_pct=0.15, counter_multiplier=4.0, ma_period=None, atr_period=None),
    "BNB": dict(sweep=0.20, steps=12, reset_interval=None, order_size_pct=0.15, counter_multiplier=5.0, ma_period=None, atr_period=None),
    "SOL": dict(sweep=0.15, steps=12, reset_interval=None, order_size_pct=0.15, counter_multiplier=5.0, ma_period=None, atr_period=None),
    # Top performers from full sweep
    "AVAX": dict(sweep=0.15, steps=12, reset_interval=None, order_size_pct=0.15, counter_multiplier=5.0, ma_period=None, atr_period=None),
    "ETC":  dict(sweep=0.15, steps=12, reset_interval=None, order_size_pct=0.15, counter_multiplier=5.0, ma_period=None, atr_period=None),
    "LINK": dict(sweep=0.15, steps=12, reset_interval=None, order_size_pct=0.15, counter_multiplier=5.0, ma_period=None, atr_period=None),
    "ONDO": dict(sweep=0.15, steps=12, reset_interval=None, order_size_pct=0.15, counter_multiplier=5.0, ma_period=None, atr_period=None),
    "LDO":  dict(sweep=0.15, steps=12, reset_interval=None, order_size_pct=0.15, counter_multiplier=5.0, ma_period=None, atr_period=None),
}

# Short MA periods to test (2m candles)
# 30m=15, 1h=30, 2h=60, 4h=120, 8h=240, 12h=360
BUY_MA_OPTIONS = [None, 15, 30, 60, 120, 240, 360]

# Drift reset percentages to test
DRIFT_OPTIONS  = [None, 0.03, 0.05, 0.08, 0.10, 0.15]


def load_prices(symbol):
    fname = f"{symbol.lower()}_usd_2m.csv"
    path  = os.path.join(DATA_DIR, fname)
    df    = pd.read_csv(path, parse_dates=["date"])
    df    = df.dropna(subset=["close"])
    return df["close"].tolist()


def run(prices, base_params, **overrides):
    p = {**base_params, **overrides}
    r = backtest(prices, initial_cash=INITIAL_CASH, circuit_breaker=0.20,
                 reactive=True, min_sweep=0.02, max_sweep=0.30, **p)
    return (r["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100, r["halted"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASELINE vs BUY MA FILTER — test on BNB (downtrend) and ETH (uptrend)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*100)
print("  TEST 1: Short-term Buy MA Filter Effect")
print("  Objective: Gate initial BUY placement when price < short MA (avoid buying downtrends)")
print("="*100)

for sym in ["BNB", "SOL", "ETH"]:
    prices = load_prices(sym)
    base   = BEST_PARAMS[sym]
    print(f"\n  {sym}:")
    print(f"    {'buy_ma':>10}  {'Return':>10}  {'Halt':>5}")
    print(f"    {'-'*30}")
    for bma in BUY_MA_OPTIONS:
        ret, halt = run(prices, base, buy_ma_period=bma)
        label = str(bma) if bma else "None (baseline)"
        print(f"    {label:>10}  {ret:>+9.2f}%  {'YES' if halt else 'No':>5}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BASELINE vs DRIFT RESET — test on all primary symbols
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*100)
print("  TEST 2: Price-Drift Grid Reset Effect")
print("  Objective: Reset grid immediately when price drifts >X% from grid centre")
print("="*100)

for sym in ["ETH", "BNB", "SOL"]:
    prices = load_prices(sym)
    base   = BEST_PARAMS[sym]
    print(f"\n  {sym}:")
    print(f"    {'drift_pct':>10}  {'Return':>10}  {'Halt':>5}")
    print(f"    {'-'*30}")
    for drift in DRIFT_OPTIONS:
        ret, halt = run(prices, base, drift_reset_pct=drift)
        label = f"{drift*100:.0f}%" if drift else "None (baseline)"
        print(f"    {label:>10}  {ret:>+9.2f}%  {'YES' if halt else 'No':>5}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1b. DYNAMIC BUY GATING — skip_buys_in_downtrend (blocks fill execution, not placement)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*100)
print("  TEST 1b: Dynamic Buy Execution Gate (skip_buys_in_downtrend=True)")
print("  Objective: Block buy ORDER FILLS (not placement) when price < short MA")
print("="*100)

for sym in ["BNB", "SOL", "ETH"]:
    prices = load_prices(sym)
    base   = BEST_PARAMS[sym]
    print(f"\n  {sym}:")
    print(f"    {'buy_ma':>10}  {'skip_buys':>10}  {'Return':>10}  {'Halt':>5}")
    print(f"    {'-'*45}")
    for bma in [None, 15, 30, 60, 120, 240, 360]:
        for skip in [False, True]:
            ret, halt = run(prices, base, buy_ma_period=bma, skip_buys_in_downtrend=skip)
            if bma is None and skip:
                continue  # skip_buys with no MA = no effect
            label = str(bma) if bma else "None (base)"
            print(f"    {label:>10}  {str(skip):>10}  {ret:>+9.2f}%  {'YES' if halt else 'No':>5}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. COMBINED: Best buy_ma + drift_reset on all symbols
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*100)
print("  TEST 3: Combined buy_ma + drift_reset search on BNB and SOL")
print("="*100)

for sym in ["BNB", "SOL"]:
    prices = load_prices(sym)
    base   = BEST_PARAMS[sym]
    print(f"\n  {sym}  (finding best buy_ma + drift combo):")
    print(f"    {'buy_ma':>8}  {'drift%':>8}  {'Return':>10}  {'Halt':>5}")
    print(f"    {'-'*38}")
    rows = []
    for bma in BUY_MA_OPTIONS:
        for drift in DRIFT_OPTIONS:
            ret, halt = run(prices, base, buy_ma_period=bma, drift_reset_pct=drift)
            rows.append((ret, bma if bma else 0, drift if drift else 0.0, halt))
    rows.sort(reverse=True)
    for ret, bma, drift, halt in rows[:10]:
        bma_s   = str(int(bma)) if bma else "None"
        drift_s = f"{drift*100:.0f}%" if drift else "None"
        print(f"    {bma_s:>8}  {drift_s:>8}  {ret:>+9.2f}%  {'YES' if halt else 'No':>5}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TOP PERFORMERS — AVAX, ETC, LINK, ONDO, LDO with per-symbol param search
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*100)
print("  TEST 4: Per-symbol param refinement for top performers (AVAX, ETC, LINK, ONDO, LDO)")
print("="*100)

TOP_SYMBOLS = ["AVAX", "ETC", "LINK", "ONDO", "LDO", "VIRTUAL", "SYRUP", "ADA", "TON", "BTC", "DOGE"]
SWEEP_FINE  = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
STEPS_FINE  = [8, 12, 16]
CM_FINE     = [4.0, 5.0, 6.0]
BMA_FINE    = [None, 30, 60, 120]

results_top = []
for sym in TOP_SYMBOLS:
    try:
        prices = load_prices(sym)
    except Exception:
        continue
    if len(prices) < 500:
        continue

    best_ret   = -9999
    best_cfg   = {}
    for sw in SWEEP_FINE:
        for st in STEPS_FINE:
            for cm in CM_FINE:
                for bma in BMA_FINE:
                    ret, halt = run(prices,
                                    dict(sweep=sw, steps=st, reset_interval=None,
                                         order_size_pct=0.15, counter_multiplier=cm,
                                         ma_period=None, atr_period=None),
                                    buy_ma_period=bma)
                    if not halt and ret > best_ret:
                        best_ret = ret
                        best_cfg = dict(sweep=sw, steps=st, cm=cm, bma=bma)

    results_top.append((sym, best_ret, best_cfg))

results_top.sort(key=lambda x: x[1], reverse=True)

print(f"\n  {'Symbol':<10}  {'BestReturn':>11}  {'Config'}")
print(f"  {'-'*80}")
for sym, ret, cfg in results_top:
    print(f"  {sym:<10}  {ret:>+10.2f}%  sweep={cfg.get('sweep')} steps={cfg.get('steps')} "
          f"cm={cfg.get('cm')} buy_ma={cfg.get('bma')}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FINAL: Run optimised configs on all top symbols and compare to round-1 baseline
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*100)
print("  FINAL: Optimised vs Round-1 baseline comparison")
print("  Top symbols with individually tuned params")
print("="*100)

baseline_by_sym = {
    "ETH": 13.61, "BNB": 3.87, "SOL": 8.67,
    "AVAX": 18.98, "ETC": 16.11, "LINK": 15.20, "ONDO": 14.97, "LDO": 12.51,
}

print(f"\n  {'Symbol':<10}  {'Baseline':>10}  {'Optimised':>11}  {'Delta':>8}")
print(f"  {'-'*50}")
for sym, ret, cfg in results_top:
    base = baseline_by_sym.get(sym, None)
    base_s = f"{base:+.2f}%" if base is not None else "    N/A"
    delta  = ret - base if base is not None else 0
    delta_s = f"{delta:+.2f}%" if base is not None else "    N/A"
    print(f"  {sym:<10}  {base_s:>10}  {ret:>+10.2f}%  {delta_s:>8}")
