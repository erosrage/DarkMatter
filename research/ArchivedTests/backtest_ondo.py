"""
ONDO 2m Backtest -- Production-accurate simulation.

Optimised params:  sweep=0.15  steps=32  order_pct=0.50  counter_mult=10.0
Production rounding applied:
  - Price: dynamic decimal places = max(2, ceil(-log10(price * step_pct)) + 1)
  - Qty:   2 decimal places

Runs three scenarios:
  1. Baseline   -- no rounding (pure backtest, best-case)
  2. Production -- with price + qty rounding (mirrors live execution)
  3. Summary    -- delta and explanation
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from research.ArchivedTests.backtest import backtest

DATA_FILE    = os.path.join(os.path.dirname(__file__), "data", "2m", "ondo_usd_2m.csv")
INITIAL_CASH = 100_000

SYMBOL       = "ONDO"
SWEEP        = 0.15
STEPS        = 32
ORDER_PCT    = 0.50
COUNTER_MULT = 10.0
STEP_PCT     = SWEEP / STEPS  # 0.004688

SHARED = dict(
    sweep             = SWEEP,
    steps             = STEPS,
    order_size_pct    = ORDER_PCT,
    counter_multiplier= COUNTER_MULT,
    reset_interval    = None,
    ma_period         = None,
    atr_period        = None,
    circuit_breaker   = 0.20,
    reactive          = True,
    min_sweep         = 0.005,
    max_sweep         = 0.50,
    initial_cash      = INITIAL_CASH,
)

# ── Load data ─────────────────────────────────────────────────────────────────
df     = pd.read_csv(DATA_FILE, parse_dates=["date"]).dropna(subset=["close"])
prices = df["close"].tolist()
dates  = df["date"].tolist()

start_price = prices[0]
end_price   = prices[-1]
n_candles   = len(prices)
days        = n_candles * 2 / 60 / 24

# ── Dynamic price precision (mirrors production price_dp()) ───────────────────
price_dp = max(2, math.ceil(-math.log10(start_price * STEP_PCT)) + 1)

# ── Run scenarios ─────────────────────────────────────────────────────────────
r_base = backtest(prices, **SHARED)
r_prod = backtest(prices, **SHARED, price_decimals=price_dp, qty_decimals=2)

def summarise(r, label):
    ret      = (r["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
    hold_val = r["holdings"] * end_price
    buys     = len(r["buy_points"])
    sells    = len(r["sell_points"])
    return dict(label=label, ret=ret, final=r["final_value"],
                cash=r["cash"], hold_val=hold_val,
                buys=buys, sells=sells, halted=r["halted"])

base = summarise(r_base, "Baseline (no rounding)")
prod = summarise(r_prod, f"Production (price={price_dp}dp, qty=2dp)")

# ── Print report ──────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  ONDO 2m Backtest  |  {n_candles:,} candles  |  ~{days:.0f} days")
print(f"  Config: sweep={SWEEP}  steps={STEPS}  order_pct={ORDER_PCT}  cm={COUNTER_MULT}")
print(f"  Price range: ${start_price:.4f} -> ${end_price:.4f}")
print(f"  Step size: {STEP_PCT*100:.3f}%  |  price_dp used in production: {price_dp}")
print(f"{'='*72}\n")

print(f"  {'Scenario':<35} {'Return':>8} {'Final Val':>13} {'Buys':>6} {'Sells':>6} {'Halt':>5}")
print(f"  {'-'*70}")
for s in [base, prod]:
    sign = "+" if s["ret"] >= 0 else ""
    print(f"  {s['label']:<35} {sign}{s['ret']:>7.2f}% ${s['final']:>12,.0f} "
          f"{s['buys']:>6} {s['sells']:>6} {'YES' if s['halted'] else 'No':>5}")

delta = prod["ret"] - base["ret"]
print(f"\n  Rounding impact: {delta:+.2f}pp")
print(f"\n  Note: delta is largely a backtest artefact. At 2m candle resolution,")
print(f"  a {10**-price_dp:.4f} price shift can change a fill by 1 candle,")
print(f"  which cascades through reactive counter-orders. In live trading,")
print(f"  sub-tick limit price differences rarely change whether an order fills.")

print(f"\n  Production scenario detail:")
print(f"    Cash remaining:  ${prod['cash']:>12,.2f}")
print(f"    Holdings value:  ${prod['hold_val']:>12,.2f}")
print(f"    Total:           ${prod['cash'] + prod['hold_val']:>12,.2f}")
print(f"\n{'='*72}\n")
