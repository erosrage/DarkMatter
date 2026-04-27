"""
Optimised 15m backtest -- per-symbol independent $100k, best configs from full brute-force search.

Search: research/param_search_full.py  (1,050 combos/pair, 80/20 temporal split)
MA and ATR universally excluded (confirmed unhelpful across all symbols).

Optimised params (val ROI from 80/20 split on ~60 days of 15m data):
  SOL:   sweep=0.15  steps=20  order_pct=0.50  cm=12   val=+12.79%  full=+34.00%
  ONDO:  sweep=0.05  steps=8   order_pct=0.50  cm=10   val=+11.97%  full=+35.68%
  ADA:   sweep=0.25  steps=48  order_pct=0.50  cm=12   val=+11.00%  full=+32.34%
  ETC:   sweep=0.15  steps=20  order_pct=0.50  cm=12   val=+7.72%   full=+53.88%

All markets run independently with their own $100,000 starting capital.
"""

import sys
import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns

# ── Backtest import ─────────────────────────────────────────────────────────────
_BT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ArchivedTests")
if _BT_DIR not in sys.path:
    sys.path.insert(0, _BT_DIR)
from backtest import backtest  # noqa: E402

DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "15m")
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

INITIAL_CASH = 100_000

# ── Seaborn theme ───────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="deep")
PALETTE  = {"SOL": "#9945FF", "ONDO": "#2563EB", "ADA": "#0D9488", "ETC": "#EA580C"}
BUY_CLR  = "#22c55e"
SELL_CLR = "#ef4444"
PORT_CLR = "#f59e0b"
BG_DARK  = "#0f1117"
BG_PANEL = "#1a1d2e"

# ── Optimised configs ────────────────────────────────────────────────────────────
OPTIMISED_CFG = {
    "SOL":  dict(sweep=0.15, steps=20, order_size_pct=0.50, counter_multiplier=12.0),
    "ONDO": dict(sweep=0.05, steps=8,  order_size_pct=0.50, counter_multiplier=10.0),
    "ADA":  dict(sweep=0.25, steps=48, order_size_pct=0.50, counter_multiplier=12.0),
    "ETC":  dict(sweep=0.15, steps=20, order_size_pct=0.50, counter_multiplier=12.0),
}
VAL_ROI = {"SOL": 12.79, "ONDO": 11.97, "ADA": 11.00, "ETC": 7.72}

SHARED_PARAMS = dict(
    reset_interval=None, ma_period=None, atr_period=None,
    circuit_breaker=0.20, reactive=True,
    min_sweep=0.005, max_sweep=0.50, initial_cash=INITIAL_CASH,
)
TRAIN_RATIO = 0.80


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_prices(symbol):
    fname = f"{symbol.lower()}_usd_15m.csv"
    path  = os.path.join(DATA_DIR, fname)
    df    = pd.read_csv(path, parse_dates=["date"])
    df    = df.dropna(subset=["close"])
    return df["close"].tolist(), df["date"].tolist()


def run_with_history(prices, sweep, steps, order_size_pct, counter_multiplier,
                     initial_cash=100_000, circuit_breaker=0.20):
    """
    Reactive grid backtest that returns portfolio-value history and detailed
    trade events (candle_idx, fill_price, qty).  Mirrors the archived backtest
    logic exactly so results match param_search_full.py.
    """
    step_pct  = sweep / steps
    cash      = float(initial_cash)
    holdings  = 0.0
    port_hist = []
    buy_ev    = []   # (candle_idx, fill_price, qty)
    sell_ev   = []

    order_usd = cash * order_size_pct
    active    = []
    for i in range(1, steps + 1):
        pct        = sweep * i / steps
        buy_price  = prices[0] * (1 - pct)
        sell_price = prices[0] * (1 + pct)
        active.append({"side": "buy",  "price": buy_price,
                        "qty": order_usd / buy_price,  "step_pct": step_pct})
        active.append({"side": "sell", "price": sell_price,
                        "qty": order_usd / sell_price, "step_pct": step_pct})

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
                    buy_ev.append((i, order["price"], order["qty"]))
                    filled.append(order)
                else:
                    remaining.append(order)
            elif order["side"] == "sell" and price >= order["price"]:
                if holdings >= order["qty"]:
                    cash     += order["price"] * order["qty"]
                    holdings -= order["qty"]
                    sell_ev.append((i, order["price"], order["qty"]))
                    filled.append(order)
                else:
                    remaining.append(order)
            else:
                remaining.append(order)

        active   = remaining
        existing = {(o["side"], round(o["price"], 8)) for o in active}

        for order in filled:
            sp = order["step_pct"]
            if order["side"] == "buy":
                cp  = order["price"] * (1 + sp * counter_multiplier)
                key = ("sell", round(cp, 8))
                if key not in existing:
                    active.append({"side": "sell", "price": cp,
                                   "qty": order["qty"], "step_pct": sp})
                    existing.add(key)
            else:
                cp  = order["price"] * (1 - sp * counter_multiplier)
                key = ("buy", round(cp, 8))
                if key not in existing:
                    now = cash * order_size_pct
                    if now > 0 and cash >= now:
                        active.append({"side": "buy", "price": cp,
                                       "qty": now / cp, "step_pct": sp})
                        existing.add(key)

        port_hist.append(cash + holdings * price)

    return buy_ev, sell_ev, port_hist


def match_roundtrips(buy_events, sell_events):
    """Pair each sell to the most recent preceding unmatched buy."""
    trips     = []
    used_buys = set()
    for si, sp, sq in sell_events:
        best = None
        for j, (bi, bp, bq) in enumerate(buy_events):
            if j in used_buys:
                continue
            if bi < si and (best is None or bi > buy_events[best][0]):
                best = j
        if best is not None:
            bi, bp, bq = buy_events[best]
            used_buys.add(best)
            trips.append((bi, bp, si, sp, (sp - bp) / bp * 100))
    return trips


def drawdown_series(port_hist):
    arr  = np.array(port_hist, dtype=float)
    peak = np.maximum.accumulate(arr)
    return (arr - peak) / peak * 100


def fmt_k(v, _):
    return f"${v/1e3:.0f}k"


# ═══════════════════════════════════════════════════════════════════════════════
# RUN BACKTESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nRunning backtests...")

full_results  = []
split_results = []

for sym, cfg in OPTIMISED_CFG.items():
    try:
        prices, dates = load_prices(sym)
    except Exception as e:
        print(f"  {sym}: ERR loading data -- {e}")
        continue

    buy_ev, sell_ev, port_hist = run_with_history(prices, **cfg, initial_cash=INITIAL_CASH)

    # Use the canonical backtest for exact cash/holdings/profit figures
    r_exact = backtest(prices, **cfg, **SHARED_PARAMS)
    fv      = r_exact["final_value"]
    ret     = (fv - INITIAL_CASH) / INITIAL_CASH * 100
    hold_v  = r_exact["holdings"] * prices[-1]

    dd    = drawdown_series(port_hist)
    trips = match_roundtrips(buy_ev, sell_ev)

    full_results.append({
        "symbol":     sym,
        "return_pct": ret,
        "final_value": fv,
        "cash":        r_exact["cash"],
        "hold_val":    hold_v,
        "buys":        len(buy_ev),
        "sells":       len(sell_ev),
        "halted":      r_exact["halted"],
        "val_roi":     VAL_ROI.get(sym),
        "n_candles":   len(prices),
        "cfg":         cfg,
        "prices":      prices,
        "dates":       dates,
        "port_hist":   port_hist,
        "buy_events":  buy_ev,
        "sell_events": sell_ev,
        "drawdown":    dd,
        "trips":       trips,
        "max_dd":      float(dd.min()),
    })

    n_train = int(len(prices) * TRAIN_RATIO)
    rt = backtest(prices[:n_train], **cfg, **SHARED_PARAMS)
    rv = backtest(prices[n_train:], **cfg, **SHARED_PARAMS)
    split_results.append({
        "symbol":     sym,
        "train_roi":  (rt["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100,
        "val_roi":    (rv["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100,
        "n_train":    n_train,
        "n_val":      len(prices) - n_train,
        "val_halted": rv["halted"],
    })

    print(f"  {sym}: full={ret:>+.2f}%  buys={len(buy_ev)}  sells={len(sell_ev)}"
          f"  max_dd={dd.min():.1f}%  round-trips={len(trips)}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  OPTIMISED 15m BACKTEST  --  Independent $100,000 per symbol  |  ~59 days")
print(f"  reset=None  |  MA/ATR=None  |  reactive=True  |  circuit_breaker=20%")
print(f"{'='*110}\n")
print(f"  {'Symbol':<8} {'Val ROI':>9} {'Full ROI':>10} {'FinalVal':>14} {'Cash':>14} "
      f"{'HoldVal':>12} {'Buys':>6} {'Sells':>6} {'MaxDD':>8} {'Halt':>5}")
print(f"  {'-'*104}")
for r in full_results:
    val_s = f"{r['val_roi']:>+.2f}%" if r["val_roi"] is not None else "  N/A"
    print(f"  {r['symbol']:<8} {val_s:>9} {r['return_pct']:>+9.2f}%  ${r['final_value']:>12,.2f}"
          f"  ${r['cash']:>12,.2f}  ${r['hold_val']:>10,.2f}  {r['buys']:>6}  {r['sells']:>6}"
          f"  {r['max_dd']:>7.1f}%  {'YES' if r['halted'] else 'No':>5}")

print(f"\n{'='*110}")
total_initial = INITIAL_CASH * len(full_results)
total_final   = sum(r["final_value"] for r in full_results)
avg_ret       = sum(r["return_pct"] for r in full_results) / len(full_results)
print(f"\n  AGGREGATE  ({len(full_results)} markets, ${total_initial:,.0f} total capital)")
print(f"  Total initial:  ${total_initial:>14,.0f}")
print(f"  Total final:    ${total_final:>14,.0f}  ({(total_final-total_initial)/total_initial*100:>+.2f}%)")
print(f"  Avg per market: {avg_ret:>+14.2f}%\n")

print(f"  80/20 TRAIN/VAL SPLIT")
print(f"  {'Symbol':<8} {'TrainROI':>10} {'TrainCandles':>14} {'ValROI':>10} {'ValCandles':>12} {'ValHalt':>8}")
print(f"  {'-'*68}")
for r in split_results:
    print(f"  {r['symbol']:<8} {r['train_roi']:>+9.2f}%  {r['n_train']:>14,}  {r['val_roi']:>+9.2f}%"
          f"  {r['n_val']:>12,}  {'YES' if r['val_halted'] else 'No':>8}")

print(f"\n  CONFIG REFERENCE")
print(f"  {'Symbol':<8} {'Sweep':>7} {'Steps':>7} {'StepPct':>9} {'OrderPct':>10} {'CM':>5} {'CounterGap':>12}")
print(f"  {'-'*64}")
for r in full_results:
    cfg = r["cfg"]
    sp  = cfg["sweep"] / cfg["steps"]
    print(f"  {r['symbol']:<8} {cfg['sweep']*100:>6.1f}%  {cfg['steps']:>7}  {sp*100:>8.3f}%"
          f"  {cfg['order_size_pct']*100:>9.0f}%  {cfg['counter_multiplier']:>5.0f}  "
          f"{sp*cfg['counter_multiplier']*100:>10.2f}%")
print(f"  Shared: reset=None  MA=None  ATR=None  circuit_breaker=20%  reactive=True\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating charts...")


# ── Figures 1–4: Per-symbol detail ────────────────────────────────────────────
for r in full_results:
    sym      = r["symbol"]
    prices   = r["prices"]
    dates    = r["dates"]
    port_h   = r["port_hist"]
    buy_ev   = r["buy_events"]
    sell_ev  = r["sell_events"]
    dd       = r["drawdown"]
    trips    = r["trips"]
    cfg      = r["cfg"]
    clr      = PALETTE[sym]
    n        = len(prices)
    xs       = np.arange(n)

    date_labels = [str(d)[:10] for d in dates]
    tick_step   = max(1, n // 8)
    tick_pos    = list(range(0, n, tick_step))
    tick_lbl    = [date_labels[i] for i in tick_pos]

    step_pct    = cfg["sweep"] / cfg["steps"]
    counter_gap = step_pct * cfg["counter_multiplier"]
    win_trips   = [t for t in trips if t[4] > 0]
    avg_gain    = float(np.mean([t[4] for t in trips])) if trips else 0.0
    best_trip   = max(trips, key=lambda t: t[4]) if trips else None

    # Pre-compute safe strings for stats panel
    win_rate_s  = f"{len(win_trips)/len(trips)*100:.0f}%" if trips else "N/A"
    avg_gain_s  = f"{avg_gain:>+.2f}%"
    best_trip_s = f"{best_trip[4]:>+.2f}%" if best_trip else "N/A"

    fig = plt.figure(figsize=(20, 18), facecolor=BG_DARK)
    fig.suptitle(
        f"{sym}  |  15m Grid Trader  |  "
        f"sweep={cfg['sweep']*100:.0f}%  steps={cfg['steps']}  "
        f"CM={cfg['counter_multiplier']:.0f}  order={cfg['order_size_pct']*100:.0f}%  |  "
        f"Full ROI: {r['return_pct']:>+.2f}%  |  Val ROI: {r['val_roi']:>+.2f}%",
        color="white", fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.36,
                           top=0.94, bottom=0.06, left=0.07, right=0.97)

    # ─ [0, :] Price chart + trade markers (full width) ──────────────────────
    ax_price = fig.add_subplot(gs[0, :])
    ax_price.set_facecolor(BG_PANEL)
    ax_price.plot(xs, prices, color=clr, linewidth=0.9, alpha=0.9, label="Price", zorder=2)

    # Shaded region between first buy and first sell (first round-trip)
    if trips:
        bi0, bp0, si0, sp0, _ = trips[0]
        ax_price.axvspan(bi0, si0, alpha=0.08, color=BUY_CLR, zorder=1)

    # Buy markers
    if buy_ev:
        bx = [e[0] for e in buy_ev]
        by = [e[1] for e in buy_ev]
        ax_price.scatter(bx, by, marker="^", color=BUY_CLR, s=130, zorder=5,
                         label=f"Buy ({len(buy_ev)})", edgecolors="white", linewidths=0.4)
        for xi, yi in zip(bx, by):
            ax_price.annotate(f"${yi:,.2f}", (xi, yi), textcoords="offset points",
                              xytext=(4, -16), fontsize=7.5, color=BUY_CLR, alpha=0.9)

    # Sell markers
    if sell_ev:
        sx = [e[0] for e in sell_ev]
        sy = [e[1] for e in sell_ev]
        ax_price.scatter(sx, sy, marker="v", color=SELL_CLR, s=130, zorder=5,
                         label=f"Sell ({len(sell_ev)})", edgecolors="white", linewidths=0.4)
        for xi, yi in zip(sx, sy):
            ax_price.annotate(f"${yi:,.2f}", (xi, yi), textcoords="offset points",
                              xytext=(4, 8), fontsize=7.5, color=SELL_CLR, alpha=0.9)

    # Counter-gap reference lines from last buy
    if buy_ev:
        ref = buy_ev[-1][1]
        ax_price.axhline(ref * (1 + counter_gap),      color=SELL_CLR, lw=0.7, ls="--", alpha=0.5,
                          label=f"+{counter_gap*100:.1f}% counter-sell")
        ax_price.axhline(ref * (1 + counter_gap * 2),  color=SELL_CLR, lw=0.5, ls=":",  alpha=0.3)
        ax_price.axhline(ref * (1 - counter_gap),      color=BUY_CLR,  lw=0.7, ls="--", alpha=0.5,
                          label=f"-{counter_gap*100:.1f}% counter-buy")

    ax_price.set_xticks(tick_pos)
    ax_price.set_xticklabels(tick_lbl, rotation=22, ha="right", fontsize=8, color="white")
    ax_price.tick_params(axis="y", colors="white", labelsize=8)
    ax_price.set_title("Price Series with Trade Execution & Counter-Order Levels",
                        color="white", fontsize=11, pad=6)
    ax_price.set_ylabel("Price (USD)", color="white", fontsize=9)
    ax_price.legend(loc="upper right", fontsize=7.5, framealpha=0.35)
    for sp_ in ax_price.spines.values():
        sp_.set_edgecolor("#333")

    # ─ [1, 0] Portfolio value over time ──────────────────────────────────────
    ax_port = fig.add_subplot(gs[1, 0])
    ax_port.set_facecolor(BG_PANEL)
    ax_port.plot(xs, port_h, color=PORT_CLR, linewidth=1.2, zorder=3)
    ax_port.fill_between(xs, INITIAL_CASH, port_h,
                         where=np.array(port_h) >= INITIAL_CASH,
                         alpha=0.28, color=BUY_CLR, zorder=2)
    ax_port.fill_between(xs, INITIAL_CASH, port_h,
                         where=np.array(port_h) < INITIAL_CASH,
                         alpha=0.28, color=SELL_CLR, zorder=2)
    ax_port.axhline(INITIAL_CASH, color="white", lw=0.8, ls="--", alpha=0.45, label="$100k start")
    ax_port.axhline(r["final_value"], color=PORT_CLR, lw=0.6, ls=":", alpha=0.5,
                    label=f"Final ${r['final_value']/1e3:.1f}k")
    ax_port.set_title("Portfolio Value Over Time", color="white", fontsize=10, pad=5)
    ax_port.set_ylabel("USD", color="white", fontsize=8)
    ax_port.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax_port.tick_params(colors="white", labelsize=7)
    ax_port.set_xticks(tick_pos[::2])
    ax_port.set_xticklabels(tick_lbl[::2], rotation=22, ha="right", fontsize=7, color="white")
    ax_port.legend(fontsize=7, framealpha=0.35)
    for sp_ in ax_port.spines.values():
        sp_.set_edgecolor("#333")

    # ─ [1, 1] Drawdown from peak ──────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[1, 1])
    ax_dd.set_facecolor(BG_PANEL)
    ax_dd.fill_between(xs, 0, dd, color=SELL_CLR, alpha=0.55, zorder=2)
    ax_dd.plot(xs, dd, color=SELL_CLR, linewidth=0.9, zorder=3)
    ax_dd.axhline(0, color="white", lw=0.5, alpha=0.35)
    ax_dd.axhline(r["max_dd"], color=SELL_CLR, lw=0.6, ls="--", alpha=0.5,
                  label=f"Max {r['max_dd']:.1f}%")
    ax_dd.set_title(f"Drawdown from Rolling Peak  (worst: {r['max_dd']:.1f}%)",
                     color="white", fontsize=10, pad=5)
    ax_dd.set_ylabel("Drawdown %", color="white", fontsize=8)
    ax_dd.tick_params(colors="white", labelsize=7)
    ax_dd.set_xticks(tick_pos[::2])
    ax_dd.set_xticklabels(tick_lbl[::2], rotation=22, ha="right", fontsize=7, color="white")
    ax_dd.legend(fontsize=7, framealpha=0.35)
    for sp_ in ax_dd.spines.values():
        sp_.set_edgecolor("#333")

    # ─ [1, 2] Cumulative realized P&L ────────────────────────────────────────
    ax_pnl = fig.add_subplot(gs[1, 2])
    ax_pnl.set_facecolor(BG_PANEL)
    if trips:
        notional = INITIAL_CASH * cfg["order_size_pct"]
        cum      = 0.0
        cx, cy   = [], []
        for bi, bp, si, sp2, pct in trips:
            cum += notional * (pct / 100)
            cx.append(si); cy.append(cum)
        ax_pnl.step(cx, cy, color=PORT_CLR, linewidth=1.4, where="post")
        ax_pnl.fill_between(cx, 0, cy, step="post", alpha=0.25, color=PORT_CLR)
        ax_pnl.scatter(cx, cy, color=PORT_CLR, s=45, zorder=4)
        ax_pnl.axhline(0, color="white", lw=0.5, ls="--", alpha=0.35)
    ax_pnl.set_title("Cumulative Realized P&L (matched round-trips)",
                      color="white", fontsize=10, pad=5)
    ax_pnl.set_ylabel("USD Profit", color="white", fontsize=8)
    ax_pnl.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_pnl.tick_params(colors="white", labelsize=7)
    for sp_ in ax_pnl.spines.values():
        sp_.set_edgecolor("#333")

    # ─ [2, 0] Round-trip gain per trade ──────────────────────────────────────
    ax_bars = fig.add_subplot(gs[2, 0])
    ax_bars.set_facecolor(BG_PANEL)
    if trips:
        pcts   = [t[4] for t in trips]
        labels = [f"T{i+1}" for i in range(len(trips))]
        colors = [BUY_CLR if p >= 0 else SELL_CLR for p in pcts]
        bars   = ax_bars.bar(labels, pcts, color=colors, edgecolor="#222", linewidth=0.6)
        for bar, pct in zip(bars, pcts):
            ypos = bar.get_height() + (0.1 if pct >= 0 else -0.3)
            va   = "bottom" if pct >= 0 else "top"
            ax_bars.text(bar.get_x() + bar.get_width() / 2, ypos,
                         f"{pct:+.1f}%", ha="center", va=va, fontsize=8, color="white")
        ax_bars.axhline(0, color="white", lw=0.6, alpha=0.4)
    ax_bars.set_title(f"Round-Trip P&L per Trade  (avg {avg_gain_s})",
                       color="white", fontsize=10, pad=5)
    ax_bars.set_ylabel("Gain %", color="white", fontsize=8)
    ax_bars.tick_params(colors="white", labelsize=8)
    for sp_ in ax_bars.spines.values():
        sp_.set_edgecolor("#333")

    # ─ [2, 1] Buy vs sell price distribution ─────────────────────────────────
    ax_dist = fig.add_subplot(gs[2, 1])
    ax_dist.set_facecolor(BG_PANEL)
    if buy_ev:
        buy_prices  = [e[1] for e in buy_ev]
        sell_prices = [e[1] for e in sell_ev]
        prange      = max(buy_prices + sell_prices) - min(buy_prices + sell_prices)
        bins        = max(6, int(prange / (prange / max(8, len(buy_ev + sell_ev)))))
        ax_dist.hist(buy_prices,  bins=bins, color=BUY_CLR,  alpha=0.65,
                     label=f"Buys  ({len(buy_ev)})",  edgecolor="#111")
        ax_dist.hist(sell_prices, bins=bins, color=SELL_CLR, alpha=0.65,
                     label=f"Sells ({len(sell_ev)})", edgecolor="#111")
        ax_dist.axvline(prices[0],  color="white",  lw=0.8, ls="--", alpha=0.5, label=f"Open ${prices[0]:,.2f}")
        ax_dist.axvline(prices[-1], color=clr,       lw=0.8, ls="--", alpha=0.7, label=f"Last ${prices[-1]:,.2f}")
    ax_dist.set_title("Fill Price Distribution (Buys vs Sells)",
                       color="white", fontsize=10, pad=5)
    ax_dist.set_xlabel("Fill Price (USD)", color="white", fontsize=8)
    ax_dist.set_ylabel("Count", color="white", fontsize=8)
    ax_dist.tick_params(colors="white", labelsize=7)
    ax_dist.legend(fontsize=7, framealpha=0.35)
    for sp_ in ax_dist.spines.values():
        sp_.set_edgecolor("#333")

    # ─ [2, 2] Stats text panel ────────────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.set_facecolor(BG_PANEL)
    ax_stats.axis("off")

    stats_txt = (
        f"  GRID CONFIG\n"
        f"  Sweep:        {cfg['sweep']*100:.1f}%\n"
        f"  Steps:        {cfg['steps']}\n"
        f"  Step size:    {step_pct*100:.3f}%\n"
        f"  Counter gap:  {counter_gap*100:.2f}%  (CM={cfg['counter_multiplier']:.0f})\n"
        f"  Order size:   {cfg['order_size_pct']*100:.0f}%  (2 levels placed)\n"
        f"\n"
        f"  PERFORMANCE\n"
        f"  Full ROI:     {r['return_pct']:>+.2f}%\n"
        f"  Val ROI:      {r['val_roi']:>+.2f}%\n"
        f"  Final value:  ${r['final_value']:>10,.2f}\n"
        f"  Cash:         ${r['cash']:>10,.2f}\n"
        f"  Holdings val: ${r['hold_val']:>10,.2f}\n"
        f"  Max drawdown: {r['max_dd']:>+.2f}%\n"
        f"\n"
        f"  TRADES\n"
        f"  Buys:         {len(buy_ev)}\n"
        f"  Sells:        {len(sell_ev)}\n"
        f"  Round-trips:  {len(trips)}\n"
        f"  Win rate:     {win_rate_s}\n"
        f"  Avg gain:     {avg_gain_s} per trip\n"
        f"  Best trip:    {best_trip_s}\n"
        f"  Candles:      {n:,}  (~{n*15/60/24:.0f} days)\n"
    )
    ax_stats.text(0.04, 0.97, stats_txt, transform=ax_stats.transAxes,
                  fontsize=8.8, verticalalignment="top", color="white",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#252840", alpha=0.7, edgecolor="#444"))
    for sp_ in ax_stats.spines.values():
        sp_.set_edgecolor("#333")

    out_path = os.path.join(RESULTS_DIR, f"backtest_15m_{sym.lower()}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: research/results/backtest_15m_{sym.lower()}.png")
    plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Summary comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("  Generating summary figure...")

syms     = [r["symbol"]      for r in full_results]
colors   = [PALETTE[s]       for s in syms]
full_roi = [r["return_pct"]  for r in full_results]
v_roi    = [r["val_roi"]     for r in full_results]
max_dds  = [abs(r["max_dd"]) for r in full_results]
n_buys   = [r["buys"]        for r in full_results]
n_sells  = [r["sells"]       for r in full_results]
cash_v   = [r["cash"]        for r in full_results]
hold_v   = [r["hold_val"]    for r in full_results]
x        = np.arange(len(syms))
w        = 0.38

fig2 = plt.figure(figsize=(20, 13), facecolor=BG_DARK)
fig2.suptitle("15m Grid Trader  --  Strategy Summary  ($100k per symbol, ~59 days)",
              color="white", fontsize=16, fontweight="bold", y=0.98)
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.50, wspace=0.38,
                         top=0.92, bottom=0.09, left=0.07, right=0.97)


def style_ax(ax, title):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color="white", fontsize=11, pad=6)
    ax.tick_params(colors="white", labelsize=8)
    for sp_ in ax.spines.values():
        sp_.set_edgecolor("#333")


# ─ [0,0] ROI grouped bar (full vs val) ───────────────────────────────────────
ax1 = fig2.add_subplot(gs2[0, 0])
style_ax(ax1, "ROI: Full Dataset vs Validation (held-out 20%)")
ax1.bar(x - w/2, full_roi, w, color=colors, alpha=0.92, edgecolor="#111", label="Full (~59d)")
ax1.bar(x + w/2, v_roi,    w, color=colors, alpha=0.40, edgecolor="#111", label="Val (held-out 20%)")
for i, (f, v) in enumerate(zip(full_roi, v_roi)):
    ax1.text(i - w/2, f + 0.4, f"{f:>+.1f}%", ha="center", va="bottom", fontsize=8.5,
             color="white", fontweight="bold")
    ax1.text(i + w/2, v + 0.4, f"{v:>+.1f}%", ha="center", va="bottom", fontsize=8, color="#ccc")
ax1.set_xticks(x)
ax1.set_xticklabels(syms, color="white", fontsize=11)
ax1.set_ylabel("Return %", color="white", fontsize=9)
ax1.legend(fontsize=8, framealpha=0.35)
ax1.axhline(0, color="white", lw=0.4, alpha=0.3)

# ─ [0,1] Trade counts grouped bar ────────────────────────────────────────────
ax2 = fig2.add_subplot(gs2[0, 1])
style_ax(ax2, "Trade Count (Buys vs Sells)")
ax2.bar(x - w/2, n_buys,  w, color=BUY_CLR,  alpha=0.88, edgecolor="#111", label="Buys")
ax2.bar(x + w/2, n_sells, w, color=SELL_CLR, alpha=0.88, edgecolor="#111", label="Sells")
for i, (b, s) in enumerate(zip(n_buys, n_sells)):
    ax2.text(i - w/2, b + 0.1, str(b), ha="center", va="bottom", fontsize=10,
             color=BUY_CLR, fontweight="bold")
    ax2.text(i + w/2, s + 0.1, str(s), ha="center", va="bottom", fontsize=10,
             color=SELL_CLR, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(syms, color="white", fontsize=11)
ax2.set_ylabel("# Trades", color="white", fontsize=9)
ax2.legend(fontsize=8, framealpha=0.35)

# ─ [0,2] Stacked final portfolio breakdown ────────────────────────────────────
ax3 = fig2.add_subplot(gs2[0, 2])
style_ax(ax3, "Final Portfolio: Cash vs Holdings")
ax3.bar(x, cash_v, color="#60a5fa", alpha=0.9, edgecolor="#111", label="Cash")
ax3.bar(x, hold_v, bottom=cash_v, color=PORT_CLR, alpha=0.9, edgecolor="#111", label="Holdings val")
ax3.axhline(INITIAL_CASH, color="white", lw=0.9, ls="--", alpha=0.5, label="$100k start")
for i, r in enumerate(full_results):
    ax3.text(i, r["final_value"] + 600,
             f"${r['final_value']/1e3:.1f}k\n{r['return_pct']:>+.1f}%",
             ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(syms, color="white", fontsize=11)
ax3.set_ylabel("USD", color="white", fontsize=9)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax3.legend(fontsize=8, framealpha=0.35)

# ─ [1,0] Max drawdown bars ────────────────────────────────────────────────────
ax4 = fig2.add_subplot(gs2[1, 0])
style_ax(ax4, "Max Drawdown from Peak")
neg_dds = [-abs(r["max_dd"]) for r in full_results]
ax4.bar(x, neg_dds, color=colors, alpha=0.85, edgecolor="#111")
for i, v in enumerate(neg_dds):
    ax4.text(i, v - 0.1, f"{v:.1f}%", ha="center", va="top", fontsize=9,
             color="white", fontweight="bold")
ax4.axhline(0, color="white", lw=0.4, alpha=0.3)
ax4.set_xticks(x)
ax4.set_xticklabels(syms, color="white", fontsize=11)
ax4.set_ylabel("Drawdown %", color="white", fontsize=9)

# ─ [1,1] Config parameter heatmap ─────────────────────────────────────────────
ax5 = fig2.add_subplot(gs2[1, 1])
ax5.set_facecolor(BG_PANEL)
ax5.set_title("Config Parameter Heatmap (colour = normalised value)",
              color="white", fontsize=11, pad=6)
param_labels = ["Sweep %", "Steps", "Step %", "Counter\nGap %", "Order %", "CM"]
param_data   = []
for r in full_results:
    cfg = r["cfg"]
    sp  = cfg["sweep"] / cfg["steps"]
    cg  = sp * cfg["counter_multiplier"]
    param_data.append([
        cfg["sweep"] * 100,
        cfg["steps"],
        sp * 100,
        cg * 100,
        cfg["order_size_pct"] * 100,
        cfg["counter_multiplier"],
    ])
df_params = pd.DataFrame(param_data, index=syms, columns=param_labels)
df_norm   = (df_params - df_params.min()) / (df_params.max() - df_params.min() + 1e-9)
im = ax5.imshow(df_norm.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax5.set_xticks(range(len(param_labels)))
ax5.set_xticklabels(param_labels, color="white", fontsize=8.5)
ax5.set_yticks(range(len(syms)))
ax5.set_yticklabels(syms, color="white", fontsize=11)
for i in range(len(syms)):
    for j in range(len(param_labels)):
        raw  = df_params.values[i, j]
        norm = df_norm.values[i, j]
        fmt  = f"{raw:.2f}" if raw < 10 else f"{raw:.0f}"
        ax5.text(j, i, fmt, ha="center", va="center", fontsize=9,
                 color="black" if norm > 0.55 else "white", fontweight="bold")
cb = plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
for sp_ in ax5.spines.values():
    sp_.set_edgecolor("#333")

# ─ [1,2] Aggregate text summary ───────────────────────────────────────────────
ax6 = fig2.add_subplot(gs2[1, 2])
ax6.set_facecolor(BG_PANEL)
ax6.axis("off")
total_i  = INITIAL_CASH * len(full_results)
total_f  = sum(r["final_value"] for r in full_results)
avg_full = sum(r["return_pct"] for r in full_results) / len(full_results)
avg_val  = sum(r["val_roi"]    for r in full_results) / len(full_results)
avg_dd   = sum(abs(r["max_dd"]) for r in full_results) / len(full_results)
best_sym = max(full_results, key=lambda r: r["return_pct"])
safe_sym = min(full_results, key=lambda r: abs(r["max_dd"]))

summary = (
    f"  AGGREGATE SUMMARY\n"
    f"  Symbols:        {len(full_results)}\n"
    f"  Capital each:   $100,000\n"
    f"  Total capital:  ${total_i:>12,.0f}\n"
    f"  Total final:    ${total_f:>12,.0f}\n"
    f"  Total profit:   ${total_f-total_i:>12,.0f}\n"
    f"  Combined ROI:   {(total_f-total_i)/total_i*100:>+.2f}%\n"
    f"\n"
    f"  Avg full ROI:   {avg_full:>+.2f}%\n"
    f"  Avg val ROI:    {avg_val:>+.2f}%\n"
    f"  Avg max DD:     {avg_dd:.1f}%\n"
    f"\n"
    f"  Period:         ~59 days (15m candles)\n"
    f"  Candles/symbol: ~5,672\n"
    f"  Val window:     ~1,135 candles (20%)\n"
    f"\n"
    f"  BEST ROI\n"
    f"  {best_sym['symbol']}:  {best_sym['return_pct']:>+.2f}%\n"
    f"\n"
    f"  LOWEST DRAWDOWN\n"
    f"  {safe_sym['symbol']}:  {safe_sym['max_dd']:.1f}% max DD\n"
    f"\n"
    f"  Shared config:\n"
    f"  reset=None  MA=None\n"
    f"  ATR=None  cb=20%\n"
    f"  reactive=True\n"
)
ax6.text(0.04, 0.97, summary, transform=ax6.transAxes,
         fontsize=9, verticalalignment="top", color="white",
         fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#252840",
                   alpha=0.75, edgecolor="#444"))
for sp_ in ax6.spines.values():
    sp_.set_edgecolor("#333")

out_path = os.path.join(RESULTS_DIR, "backtest_15m_summary.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"  Saved: research/results/backtest_15m_summary.png")
plt.show()
plt.close(fig2)

print("\nDone. Charts saved to research/results/")
