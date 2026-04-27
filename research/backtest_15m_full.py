"""
15m Full Universe Backtest — All Available Crypto Data

Runs the reactive grid strategy across every 15m CSV found in data/15m/.
Each symbol runs independently with $100,000 starting capital.

Optimised params (from research/param_search_full.py — 80/20 temporal split):
  SOL:   sweep=0.15  steps=20  order_pct=0.50  cm=12
  ONDO:  sweep=0.05  steps=8   order_pct=0.50  cm=10
  ADA:   sweep=0.25  steps=48  order_pct=0.50  cm=12
  ETC:   sweep=0.15  steps=20  order_pct=0.50  cm=12

All other symbols use a balanced default:
  sweep=0.10  steps=12  order_pct=0.50  cm=10

Shared params: reset=None  MA=None  ATR=None  circuit_breaker=20%  reactive=True

Output: sorted results table + summary bar chart (results/backtest_15m_universe.png)
"""

import sys
import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Import canonical backtest ──────────────────────────────────────────────────
_BT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ArchivedTests")
if _BT_DIR not in sys.path:
    sys.path.insert(0, _BT_DIR)
from backtest import backtest  # noqa: E402

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "15m")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

INITIAL_CASH  = 100_000
SKIP_SYMBOLS  = {"USDC"}        # stablecoins — skip
TRAIN_RATIO   = 0.80

# ── Per-symbol optimised configs (from param_search_full.py) ──────────────────
OPTIMISED = {
    "SOL":  dict(sweep=0.15, steps=20, order_size_pct=0.50, counter_multiplier=12.0),
    "ONDO": dict(sweep=0.05, steps=8,  order_size_pct=0.50, counter_multiplier=10.0),
    "ADA":  dict(sweep=0.25, steps=48, order_size_pct=0.50, counter_multiplier=12.0),
    "ETC":  dict(sweep=0.15, steps=20, order_size_pct=0.50, counter_multiplier=12.0),
}

# Default for all other symbols
DEFAULT_CFG = dict(sweep=0.10, steps=12, order_size_pct=0.50, counter_multiplier=10.0)

SHARED = dict(
    reset_interval=None, ma_period=None, atr_period=None,
    circuit_breaker=0.20, reactive=True,
    min_sweep=0.005, max_sweep=0.50,
    initial_cash=INITIAL_CASH,
)

# ── Visual theme ───────────────────────────────────────────────────────────────
BG_DARK  = "#0f1117"
BG_PANEL = "#1a1d2e"
BUY_CLR  = "#22c55e"
SELL_CLR = "#ef4444"
OPT_CLR  = "#f59e0b"


# ── Portfolio-history simulation ───────────────────────────────────────────────

def run_sim(prices, sweep, steps, order_size_pct, counter_multiplier,
            initial_cash=100_000, circuit_breaker=0.20):
    """
    Reactive grid simulation returning per-candle portfolio value history.
    Mirrors the logic in backtest_15m.py run_with_history() exactly.
    """
    step_pct  = sweep / steps
    cash      = float(initial_cash)
    holdings  = 0.0
    port_hist = []

    order_usd = cash * order_size_pct
    active = []
    for i in range(1, steps + 1):
        pct = sweep * i / steps
        bp  = prices[0] * (1 - pct)
        sp  = prices[0] * (1 + pct)
        active.append({"side": "buy",  "price": bp, "qty": order_usd / bp,  "step_pct": step_pct})
        active.append({"side": "sell", "price": sp, "qty": order_usd / sp, "step_pct": step_pct})

    for i, price in enumerate(prices):
        port_val = cash + holdings * price
        if circuit_breaker and port_val < initial_cash * (1 - circuit_breaker):
            port_hist.extend([port_val] * (len(prices) - i))
            break

        filled    = []
        remaining = []
        for order in active:
            if order["side"] == "buy" and price <= order["price"]:
                cost = order["price"] * order["qty"]
                if cash >= cost:
                    cash     -= cost
                    holdings += order["qty"]
                    filled.append(order)
                else:
                    remaining.append(order)
            elif order["side"] == "sell" and price >= order["price"]:
                if holdings >= order["qty"]:
                    cash     += order["price"] * order["qty"]
                    holdings -= order["qty"]
                    filled.append(order)
                else:
                    remaining.append(order)
            else:
                remaining.append(order)

        active   = remaining
        existing = {(o["side"], round(o["price"], 8)) for o in active}
        for order in filled:
            sp2 = order["step_pct"]
            if order["side"] == "buy":
                cp  = order["price"] * (1 + sp2 * counter_multiplier)
                key = ("sell", round(cp, 8))
                if key not in existing:
                    active.append({"side": "sell", "price": cp,
                                   "qty": order["qty"], "step_pct": sp2})
                    existing.add(key)
            else:
                cp  = order["price"] * (1 - sp2 * counter_multiplier)
                key = ("buy", round(cp, 8))
                if key not in existing:
                    now = cash * order_size_pct
                    if now > 0 and cash >= now:
                        active.append({"side": "buy", "price": cp,
                                       "qty": now / cp, "step_pct": sp2})
                        existing.add(key)

        port_hist.append(cash + holdings * price)

    return port_hist


def drawdown(port_hist):
    arr  = np.array(port_hist, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / peak * 100
    return float(dd.min())


# ── Load helpers ───────────────────────────────────────────────────────────────

def load_prices(symbol):
    fname = f"{symbol.lower()}_usd_15m.csv"
    path  = os.path.join(DATA_DIR, fname)
    df    = pd.read_csv(path, parse_dates=["date"])
    df    = df.dropna(subset=["close"])
    return df["close"].tolist(), df["date"].tolist()


# ── Discover all available 15m symbols ────────────────────────────────────────
available = sorted(os.listdir(DATA_DIR))
ALL_SYMBOLS = [
    f.replace("_usd_15m.csv", "").upper()
    for f in available
    if f.endswith("_usd_15m.csv")
]

print(f"\n{'='*80}")
print(f"  15m FULL UNIVERSE BACKTEST  —  {len(ALL_SYMBOLS)} symbols found in data/15m/")
print(f"  Initial capital: ${INITIAL_CASH:,} per symbol  |  reactive=True  |  no MA/ATR")
print(f"  * = optimised config  (all others: sweep=10% steps=12 order=50% cm=10)")
print(f"{'='*80}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL BACKTESTS
# ═══════════════════════════════════════════════════════════════════════════════

results = []

for sym in ALL_SYMBOLS:
    if sym in SKIP_SYMBOLS:
        print(f"  {sym:<7}: skipped (stablecoin)")
        continue

    cfg = OPTIMISED.get(sym, DEFAULT_CFG)
    is_opt = sym in OPTIMISED

    try:
        prices, dates = load_prices(sym)
    except Exception as e:
        print(f"  {sym:<7}: ERR loading — {e}")
        continue

    if len(prices) < 20:
        print(f"  {sym:<7}: too few candles ({len(prices)}), skipping")
        continue

    # Full period
    try:
        r = backtest(prices, **cfg, **SHARED)
    except Exception as e:
        print(f"  {sym:<7}: ERR backtest — {e}")
        continue

    ret      = (r["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
    hold_val = r["holdings"] * prices[-1]

    # Portfolio history for drawdown
    try:
        ph   = run_sim(prices, **cfg,
                       initial_cash=INITIAL_CASH, circuit_breaker=0.20)
        max_dd = drawdown(ph)
    except Exception:
        max_dd = float("nan")

    # 80/20 train/val split
    n_train = int(len(prices) * TRAIN_RATIO)
    try:
        rt = backtest(prices[:n_train], **cfg, **SHARED)
        rv = backtest(prices[n_train:], **cfg, **SHARED)
        train_roi  = (rt["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
        val_roi    = (rv["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
        val_halted = rv["halted"]
    except Exception:
        train_roi = val_roi = 0.0
        val_halted = False

    n_candles   = len(prices)
    date_start  = str(dates[0])[:10] if dates else "?"
    date_end    = str(dates[-1])[:10] if dates else "?"

    results.append({
        "symbol":      sym,
        "is_opt":      is_opt,
        "cfg":         cfg,
        "return_pct":  ret,
        "train_roi":   train_roi,
        "val_roi":     val_roi,
        "final_value": r["final_value"],
        "cash":        r["cash"],
        "hold_val":    hold_val,
        "buys":        len(r["buy_points"]),
        "sells":       len(r["sell_points"]),
        "halted":      r["halted"],
        "val_halted":  val_halted,
        "n_candles":   n_candles,
        "date_start":  date_start,
        "date_end":    date_end,
        "max_dd":      max_dd,
    })

    opt_tag  = "*" if is_opt else " "
    dd_s     = f"{max_dd:.1f}%" if not math.isnan(max_dd) else "  N/A"
    n_buys   = len(r["buy_points"])
    n_sells  = len(r["sell_points"])
    print(f"  {opt_tag}{sym:<6}: full={ret:>+7.2f}%  val={val_roi:>+7.2f}%"
          f"  buys={n_buys:>5}  sells={n_sells:>5}"
          f"  maxDD={dd_s:>7}  halt={'Y' if r['halted'] else 'N'}"
          f"  n={n_candles:>6,}")


# ═══════════════════════════════════════════════════════════════════════════════
# RANKED RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

results.sort(key=lambda x: x["return_pct"], reverse=True)
n   = len(results)
tot = INITIAL_CASH * n

print(f"\n{'='*145}")
print(f"  RANKED RESULTS (highest full-period return -> lowest)")
print(f"{'='*145}\n")
print(f"  {'#':>3}  {'Sym':<6} {'Opt':>4}  {'FullROI':>9} {'TrainROI':>9} {'ValROI':>9}"
      f"  {'FinalVal':>13}  {'Cash':>12}  {'HoldVal':>12}"
      f"  {'Buys':>5} {'Sells':>6}  {'MaxDD':>7}  {'Halt':>5}  {'n':>6}  Period")
print(f"  {'-'*139}")

for rank, r in enumerate(results, 1):
    opt_s = "YES" if r["is_opt"] else "—"
    dd_s  = f"{r['max_dd']:>6.1f}%" if not math.isnan(r["max_dd"]) else "   N/A"
    print(
        f"  {rank:>3}. {r['symbol']:<6} {opt_s:>4}  "
        f"{r['return_pct']:>+8.2f}%  {r['train_roi']:>+8.2f}%  {r['val_roi']:>+8.2f}%  "
        f"${r['final_value']:>12,.2f}  ${r['cash']:>11,.2f}  ${r['hold_val']:>11,.2f}  "
        f"{r['buys']:>5} {r['sells']:>6}  {dd_s}  "
        f"{'YES' if r['halted'] else 'No':>5}  {r['n_candles']:>6,}  "
        f"{r['date_start']}->{r['date_end']}"
    )


# ── Aggregate summary ─────────────────────────────────────────────────────────
print(f"\n{'='*145}")

total_final      = sum(r["final_value"] for r in results)
avg_ret          = sum(r["return_pct"]  for r in results) / n
avg_val          = sum(r["val_roi"]     for r in results) / n
profitable_full  = sum(1 for r in results if r["return_pct"] > 0)
profitable_val   = sum(1 for r in results if r["val_roi"]    > 0)
halted_count     = sum(1 for r in results if r["halted"])

print(f"\n  AGGREGATE  ({n} symbols  |  ${tot:,.0f} total capital)")
print(f"  Total initial:     ${tot:>15,.0f}")
print(f"  Total final:       ${total_final:>15,.2f}   ({(total_final - tot) / tot * 100:>+.2f}%)")
print(f"  Avg return (full): {avg_ret:>+15.2f}%")
print(f"  Avg return (val):  {avg_val:>+15.2f}%")
print(f"  Profitable (full): {profitable_full:>3} / {n}")
print(f"  Profitable (val):  {profitable_val:>3} / {n}")
print(f"  Halted:            {halted_count:>3} / {n}")
print(f"  Best  (full): {results[0]['symbol']}  {results[0]['return_pct']:>+.2f}%")
print(f"  Worst (full): {results[-1]['symbol']}  {results[-1]['return_pct']:>+.2f}%")
print(f"  Best  (val):  {max(results, key=lambda x: x['val_roi'])['symbol']}"
      f"  {max(r['val_roi'] for r in results):>+.2f}%")
print(f"  Worst (val):  {min(results, key=lambda x: x['val_roi'])['symbol']}"
      f"  {min(r['val_roi'] for r in results):>+.2f}%")

print(f"\n  CONFIG REFERENCE")
print(f"  {'Sym':<6} {'Sweep':>7} {'Steps':>6} {'StepPct':>8} {'OrderPct':>9} {'CM':>5} {'CounterGap':>11}")
print(f"  {'-'*58}")
for r in results:
    cfg = r["cfg"]
    sp  = cfg["sweep"] / cfg["steps"]
    opt_tag = "*" if r["is_opt"] else " "
    print(f"  {opt_tag}{r['symbol']:<5} {cfg['sweep']*100:>6.1f}%  {cfg['steps']:>6}"
          f"  {sp*100:>7.3f}%  {cfg['order_size_pct']*100:>8.0f}%  {cfg['counter_multiplier']:>5.0f}"
          f"  {sp*cfg['counter_multiplier']*100:>9.2f}%")
print(f"  Shared: reset=None  MA=None  ATR=None  circuit_breaker=20%  reactive=True\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating summary chart...")

sns.set_theme(style="darkgrid", palette="deep")

syms       = [r["symbol"]     for r in results]
full_rets  = [r["return_pct"] for r in results]
val_rets   = [r["val_roi"]    for r in results]
max_dds    = [r["max_dd"] if not math.isnan(r["max_dd"]) else 0.0 for r in results]
is_opt     = [r["is_opt"]     for r in results]

full_colors = [OPT_CLR if opt else (BUY_CLR if v >= 0 else SELL_CLR)
               for opt, v in zip(is_opt, full_rets)]
val_colors  = [OPT_CLR if opt else (BUY_CLR if v >= 0 else SELL_CLR)
               for opt, v in zip(is_opt, val_rets)]

fig = plt.figure(figsize=(22, 18), facecolor=BG_DARK)
fig.suptitle(
    f"15m Reactive Grid Backtest — Full Universe ({n} symbols, $100k each)\n"
    f"Default: sweep=10% steps=12 order=50% cm=10  |  gold = optimised params",
    color="white", fontsize=14, fontweight="bold", y=0.99,
)

# Layout: 3 rows
gs = plt.GridSpec(3, 1, figure=fig, hspace=0.55, top=0.94, bottom=0.05,
                  left=0.06, right=0.97)

# ── Row 0: Full-period return ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(BG_PANEL)
x_pos = range(len(syms))
bars  = ax1.bar(x_pos, full_rets, color=full_colors, edgecolor="#222", linewidth=0.5, zorder=3)
for bar, val, sym in zip(bars, full_rets, syms):
    ypos = bar.get_height() + (0.5 if val >= 0 else -0.5)
    va   = "bottom" if val >= 0 else "top"
    ax1.text(bar.get_x() + bar.get_width() / 2, ypos,
             f"{val:+.1f}%", ha="center", va=va, fontsize=6.5, color="white")
ax1.axhline(0, color="white", lw=0.8, alpha=0.5)
ax1.set_xticks(list(x_pos))
ax1.set_xticklabels(syms, rotation=45, ha="right", fontsize=8, color="white")
ax1.tick_params(axis="y", colors="white", labelsize=8)
ax1.set_title("Full-Period Return per Symbol  (sorted highest -> lowest)",
              color="white", fontsize=11, pad=6)
ax1.set_ylabel("Return %", color="white", fontsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
for sp in ax1.spines.values():
    sp.set_edgecolor("#333")

# ── Row 1: Validation return (last 20%) ───────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(BG_PANEL)
bars2 = ax2.bar(x_pos, val_rets, color=val_colors, edgecolor="#222", linewidth=0.5, zorder=3)
for bar, val in zip(bars2, val_rets):
    ypos = bar.get_height() + (0.3 if val >= 0 else -0.3)
    va   = "bottom" if val >= 0 else "top"
    ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
             f"{val:+.1f}%", ha="center", va=va, fontsize=6.5, color="white")
ax2.axhline(0, color="white", lw=0.8, alpha=0.5)
ax2.set_xticks(list(x_pos))
ax2.set_xticklabels(syms, rotation=45, ha="right", fontsize=8, color="white")
ax2.tick_params(axis="y", colors="white", labelsize=8)
ax2.set_title("Validation Return — last 20% of data  (out-of-sample)",
              color="white", fontsize=11, pad=6)
ax2.set_ylabel("Val Return %", color="white", fontsize=9)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
for sp in ax2.spines.values():
    sp.set_edgecolor("#333")

# ── Row 2: Max drawdown ────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor(BG_PANEL)
dd_colors = [SELL_CLR] * len(max_dds)
bars3 = ax3.bar(x_pos, max_dds, color=dd_colors, edgecolor="#222", linewidth=0.5, zorder=3, alpha=0.85)
for bar, val in zip(bars3, max_dds):
    ypos = bar.get_height() - 0.5
    ax3.text(bar.get_x() + bar.get_width() / 2, ypos,
             f"{val:.1f}%", ha="center", va="top", fontsize=6.5, color="white")
ax3.axhline(0, color="white", lw=0.8, alpha=0.5)
ax3.set_xticks(list(x_pos))
ax3.set_xticklabels(syms, rotation=45, ha="right", fontsize=8, color="white")
ax3.tick_params(axis="y", colors="white", labelsize=8)
ax3.set_title("Maximum Drawdown from Rolling Peak  (lower = safer)",
              color="white", fontsize=11, pad=6)
ax3.set_ylabel("Max Drawdown %", color="white", fontsize=9)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
for sp in ax3.spines.values():
    sp.set_edgecolor("#333")

out_path = os.path.join(RESULTS_DIR, "backtest_15m_universe.png")
plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG_DARK)
plt.close()
print(f"Chart saved -> {out_path}\n")
