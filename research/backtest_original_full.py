"""
Full universe backtest of Original trader_robinhood.py Strategy — all 40 15m datasets.

Each symbol is run independently with $10,000 starting capital.
Strategy parameters are exact copies from trader_robinhood.py:
  - 20% sweep, $10/order chunk, hourly cycle (4x15m), 0.26% fee both sides
  - Weighted-average buy price (avgBuy), Market/Recovery sell modes
  - Dynamic minProf 0.7-5.0%

Output: ranked text report + summary charts
"""

import os
import bisect
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────
SWEEP        = 20.0
ORDER_SIZE   = 10.0
RESERVE      = 0.0
INTERVAL_C   = 4            # candles per reset (4 x 15m = 1h)
FEE_BUY      = 1.0026
FEE_SELL     = 0.9974
INITIAL_CASH = 10_000       # per symbol, independent
CIRCUIT_BRK  = 0.30

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data", "15m")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helpers (exact copies from trader_robinhood.py) ───────────────────
def _interval(balance):
    chunks = balance / ORDER_SIZE
    return SWEEP / chunks if chunks >= 1 else SWEEP


def _stable(n, spread_pct):
    return [i * spread_pct / 100.0 for i in range(1, int(n) + 1)]


def _buy_rates(base, table):
    return [(1.0 - e) * base for e in table]


def _sell_rates(base, table):
    rates = [base]
    for e in range(len(table) - 1):
        rates.append((1.0 + table[e]) * base)
    return rates


def make_buy_orders(usd_balance, ask):
    """Returns (prices_asc, qtys_asc) as lists sorted ascending by price."""
    n = int(usd_balance / ORDER_SIZE)
    if n < 1:
        return [], []
    table  = _stable(n, _interval(usd_balance))[:n]
    prices = _buy_rates(ask, table)
    prices = [p for p in prices if p > 0]
    qtys   = [ORDER_SIZE / p for p in prices]
    pairs  = sorted(zip(prices, qtys))
    if not pairs:
        return [], []
    p_out, q_out = zip(*pairs)
    return list(p_out), list(q_out)


def make_sell_orders(coin_bal, sell_base):
    """Returns (prices_asc, qtys_asc) as lists sorted ascending by price."""
    if coin_bal <= 0 or sell_base <= 0:
        return [], []
    notional = coin_bal * sell_base
    n = int(notional / ORDER_SIZE)
    if n < 1:
        return [], []
    table  = _stable(n, _interval(notional))
    rates  = _sell_rates(sell_base, table)[:n]
    rates  = [r for r in rates if r > 0]
    if not rates:
        return [], []
    qty_ea = coin_bal / len(rates)
    pairs  = sorted(zip(rates, [qty_ea] * len(rates)))
    p_out, q_out = zip(*pairs)
    return list(p_out), list(q_out)


def getMinProf(usd_bal, portfolio_val):
    x = usd_bal / portfolio_val if portfolio_val > 0 else 1.0
    a, b, c, d = 1.0, 0.3, 0.7, 5.0
    return max(c, min(d, (x - a) / (b - a) * (d - c) + c))


# ── Per-symbol simulation ─────────────────────────────────────────────
def run_symbol(symbol, prices):
    """
    Simulate the original trader_robinhood.py strategy on a single symbol.
    Returns result dict.
    """
    N    = len(prices)
    cash = float(INITIAL_CASH)

    holdings = 0.0
    avg_buy  = 0.0
    profit   = 0.0
    min_prof = 5.0

    # Active orders stored as sorted numpy arrays for O(log n) fill detection
    buy_p  = np.array([], dtype=float)   # ascending
    buy_q  = np.array([], dtype=float)
    sell_p = np.array([], dtype=float)   # ascending
    sell_q = np.array([], dtype=float)

    port_hist  = []
    n_buys     = 0
    n_sells    = 0
    halted     = False
    mode_counts = {"Buying": 0, "Recovery": 0, "Market": 0}

    def _place_initial(ask):
        nonlocal buy_p, buy_q, sell_p, sell_q
        bp, bq = make_buy_orders(INITIAL_CASH, ask)
        buy_p  = np.array(bp, dtype=float)
        buy_q  = np.array(bq, dtype=float)
        sell_p = np.array([], dtype=float)
        sell_q = np.array([], dtype=float)

    def _reset(price, cash_now, holdings_now, avg_now, min_p):
        nonlocal buy_p, buy_q, sell_p, sell_q
        coin = holdings_now
        avg  = avg_now
        mode = "Buying"

        new_sells_p, new_sells_q = [], []
        new_buys_p,  new_buys_q  = [], []

        if coin > 0 and avg > 0:
            thresh = avg * (1.0 + min_p / 100.0)
            if price > thresh:
                sell_base = price
                mode = "Market"
            else:
                sell_base = thresh
                mode = "Recovery"
            sp, sq = make_sell_orders(coin, sell_base)
            new_sells_p.extend(sp)
            new_sells_q.extend(sq)

        per_mkt = max(0.0, cash_now - RESERVE)
        buy_base = avg if (mode == "Market" and avg > 0 and avg < price) else price
        bp, bq = make_buy_orders(per_mkt, buy_base)
        new_buys_p.extend(bp)
        new_buys_q.extend(bq)

        buy_p  = np.array(new_buys_p,  dtype=float)
        buy_q  = np.array(new_buys_q,  dtype=float)
        sell_p = np.array(new_sells_p, dtype=float)
        sell_q = np.array(new_sells_q, dtype=float)
        return mode

    # Initial orders
    _place_initial(prices[0])

    for i, price in enumerate(prices):
        pv = cash + holdings * price

        if CIRCUIT_BRK and pv < INITIAL_CASH * (1 - CIRCUIT_BRK):
            port_hist.extend([pv] * (N - i))
            halted = True
            break

        # ── Fill buys (price <= buy_price) — bisect on ascending array ─
        if len(buy_p) > 0:
            # qualifying: buy_p >= price → index from bisect_left(buy_p, price)
            idx = bisect.bisect_left(buy_p, price)
            if idx < len(buy_p):
                fill_p = buy_p[idx:]
                fill_q = buy_q[idx:]
                # Process fills, respecting cash (highest price = right end, first priority)
                fill_mask = np.ones(len(fill_p), dtype=bool)
                running_cash = cash
                # iterate right-to-left (highest price first) for fill priority
                for j in range(len(fill_p) - 1, -1, -1):
                    cost = fill_p[j] * fill_q[j] * FEE_BUY
                    if running_cash >= cost:
                        running_cash -= cost
                        prev_h = holdings + sum(fill_q[k] for k in range(j + 1, len(fill_p)) if fill_mask[k])
                        new_q  = fill_q[j]
                        if prev_h > 0 and avg_buy > 0:
                            avg_buy = (avg_buy * prev_h + fill_p[j] * new_q) / (prev_h + new_q)
                        else:
                            avg_buy = fill_p[j]
                        holdings += new_q
                        cash      = running_cash
                        n_buys   += 1
                    else:
                        fill_mask[j] = False

                # Remove filled orders
                keep = np.ones(len(buy_p), dtype=bool)
                for j, filled in enumerate(fill_mask):
                    if filled:
                        keep[idx + j] = False
                buy_p = buy_p[keep]
                buy_q = buy_q[keep]

        # ── Fill sells (price >= sell_price) — bisect on ascending array ─
        if len(sell_p) > 0:
            # qualifying: sell_p <= price → index up to bisect_right(sell_p, price)
            idx = bisect.bisect_right(sell_p, price)
            if idx > 0:
                fill_p = sell_p[:idx]
                fill_q = sell_q[:idx]
                for j in range(len(fill_p)):
                    if holdings >= fill_q[j]:
                        proceeds   = fill_p[j] * fill_q[j] * FEE_SELL
                        cost_basis = avg_buy * fill_q[j] * FEE_BUY if avg_buy > 0 else 0
                        cash      += proceeds
                        holdings  -= fill_q[j]
                        profit    += proceeds - cost_basis
                        n_sells   += 1
                        if holdings <= 1e-12:
                            holdings = 0.0
                            avg_buy  = 0.0
                sell_p = sell_p[idx:]
                sell_q = sell_q[idx:]

        # ── Hourly reset ───────────────────────────────────────────────
        if i > 0 and i % INTERVAL_C == 0:
            pv       = cash + holdings * price
            min_prof = max(0.5, getMinProf(cash, pv))
            mode     = _reset(price, cash, holdings, avg_buy, min_prof)
            mode_counts[mode] += 1

        port_hist.append(cash + holdings * price)

    last_price  = prices[len(port_hist) - 1]
    final_value = port_hist[-1]
    return_pct  = (final_value - INITIAL_CASH) / INITIAL_CASH * 100
    bah_pct     = (last_price - prices[0]) / prices[0] * 100
    bah_val     = INITIAL_CASH * (1 + bah_pct / 100)

    arr  = np.array(port_hist)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / peak * 100
    max_dd = float(dd.min())

    total_resets = sum(mode_counts.values())

    return {
        "symbol":      symbol,
        "n_candles":   len(port_hist),
        "n_days":      len(port_hist) * 15 / 60 / 24,
        "price_open":  prices[0],
        "price_close": last_price,
        "price_chg":   bah_pct,
        "return_pct":  return_pct,
        "final_value": final_value,
        "bah_pct":     bah_pct,
        "bah_val":     bah_val,
        "alpha":       return_pct - bah_pct,
        "cash":        cash,
        "holdings":    holdings,
        "hold_val":    holdings * last_price,
        "avg_buy":     avg_buy,
        "profit":      profit,
        "n_buys":      n_buys,
        "n_sells":     n_sells,
        "max_dd":      max_dd,
        "halted":      halted,
        "port_hist":   port_hist,
        "prices":      prices[:len(port_hist)],
        "min_prof":    min_prof,
        "mode_buying":    mode_counts["Buying"]   / total_resets * 100 if total_resets else 0,
        "mode_recovery":  mode_counts["Recovery"] / total_resets * 100 if total_resets else 0,
        "mode_market":    mode_counts["Market"]   / total_resets * 100 if total_resets else 0,
    }


# ── Load all symbols ──────────────────────────────────────────────────
def load_all():
    results = []
    files   = sorted(f for f in os.listdir(DATA_DIR) if f.endswith("_usd_15m.csv"))
    print(f"  Found {len(files)} symbol files.\n")

    for fname in files:
        symbol = fname.replace("_usd_15m.csv", "").upper()
        path   = os.path.join(DATA_DIR, fname)
        df     = pd.read_csv(path, parse_dates=["date"]).dropna(subset=["close"])
        if len(df) < 50:
            print(f"  SKIP {symbol}: only {len(df)} rows")
            continue
        prices = df["close"].tolist()
        dates  = df["date"].tolist()

        res = run_symbol(symbol, prices)
        res["dates"] = dates[:res["n_candles"]]
        results.append(res)

        halt_s = " [HALTED]" if res["halted"] else ""
        print(f"  {symbol:<8}  {res['n_candles']:>5} candles  "
              f"ret={res['return_pct']:>+7.2f}%  bah={res['bah_pct']:>+7.2f}%  "
              f"alpha={res['alpha']:>+6.2f}pp  "
              f"dd={res['max_dd']:>6.1f}%  "
              f"buys={res['n_buys']:>5}  sells={res['n_sells']:>5}{halt_s}")

    return results


# ── Run ───────────────────────────────────────────────────────────────
print("=" * 90)
print("  ORIGINAL TRADER_ROBINHOOD.PY -- FULL UNIVERSE BACKTEST (15m)")
print(f"  Strategy: SWEEP={SWEEP}%  ORDER_SIZE=${ORDER_SIZE:.0f}  CYCLE={INTERVAL_C} candles (1h)")
print(f"  Capital:  ${INITIAL_CASH:,.0f} per symbol (independent)")
print(f"  Fee:      {(FEE_BUY-1)*100:.2f}% buy / {(1-FEE_SELL)*100:.2f}% sell")
print("=" * 90)
print()

results = load_all()

# Sort by return descending
results_sorted = sorted(results, key=lambda r: r["return_pct"], reverse=True)

# ═══════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════
hdr = "=" * 116
print(f"\n\n{hdr}")
print(f"  FULL UNIVERSE BACKTEST REPORT  |  ~{results[0]['n_days']:.0f} days  |  "
      f"${INITIAL_CASH:,.0f} per symbol  |  Ranked by Return")
print(f"{hdr}")
print(f"\n  {'#':>3}  {'Symbol':<8} {'Return%':>9} {'BAH%':>8} {'Alpha':>8} "
      f"{'FinalVal':>11} {'Cash':>10} {'HoldVal':>10} "
      f"{'MaxDD':>7} {'Buys':>6} {'Sells':>5} {'Halt':>5} {'PriceChg%':>10}")
print(f"  {'-'*112}")

for rank, r in enumerate(results_sorted, 1):
    h = "YES" if r["halted"] else "No"
    print(f"  {rank:>3}  {r['symbol']:<8} {r['return_pct']:>+8.2f}%  "
          f"{r['bah_pct']:>+7.2f}%  {r['alpha']:>+7.2f}pp  "
          f"${r['final_value']:>9,.2f}  ${r['cash']:>8,.2f}  ${r['hold_val']:>8,.2f}  "
          f"{r['max_dd']:>6.1f}%  {r['n_buys']:>6}  {r['n_sells']:>5}  {h:>5}  "
          f"{r['price_chg']:>+9.2f}%")

# Aggregate summary
n_pos  = sum(1 for r in results if r["return_pct"] > 0)
n_halt = sum(1 for r in results if r["halted"])
tot_i  = INITIAL_CASH * len(results)
tot_f  = sum(r["final_value"] for r in results)
avg_r  = sum(r["return_pct"] for r in results) / len(results)
avg_b  = sum(r["bah_pct"]    for r in results) / len(results)
avg_a  = sum(r["alpha"]       for r in results) / len(results)
avg_dd = sum(abs(r["max_dd"]) for r in results) / len(results)
best   = results_sorted[0]
worst  = results_sorted[-1]

print(f"\n{hdr}")
print(f"\n  AGGREGATE  ({len(results)} symbols)")
print(f"  Total initial:   ${tot_i:>14,.0f}")
print(f"  Total final:     ${tot_f:>14,.0f}  ({(tot_f-tot_i)/tot_i*100:>+.2f}%)")
print(f"  Avg return:      {avg_r:>+14.2f}%")
print(f"  Avg BAH:         {avg_b:>+14.2f}%")
print(f"  Avg alpha:       {avg_a:>+14.2f} pp")
print(f"  Avg max DD:      {avg_dd:>14.2f}%")
print(f"  Winners (>0%):   {n_pos:>14} / {len(results)}")
print(f"  Halted:          {n_halt:>14}")
print(f"  Best:            {best['symbol']:<8}  {best['return_pct']:>+.2f}%")
print(f"  Worst:           {worst['symbol']:<8}  {worst['return_pct']:>+.2f}%")

# Mode distribution
print(f"\n  SELL MODE DISTRIBUTION AT HOURLY RESETS (avg across symbols)")
avg_buying   = sum(r["mode_buying"]   for r in results) / len(results)
avg_recovery = sum(r["mode_recovery"] for r in results) / len(results)
avg_market   = sum(r["mode_market"]   for r in results) / len(results)
print(f"  Buying mode:     {avg_buying:>6.1f}%  (no holdings yet)")
print(f"  Recovery mode:   {avg_recovery:>6.1f}%  (underwater, waiting for recovery)")
print(f"  Market mode:     {avg_market:>6.1f}%  (profitable, selling at ask)")

# ── Save CSV ──────────────────────────────────────────────────────────
df_out = pd.DataFrame([{
    "symbol":      r["symbol"],
    "return_pct":  round(r["return_pct"],  2),
    "bah_pct":     round(r["bah_pct"],     2),
    "alpha_pp":    round(r["alpha"],        2),
    "final_value": round(r["final_value"], 2),
    "cash":        round(r["cash"],        2),
    "hold_val":    round(r["hold_val"],    2),
    "max_dd":      round(r["max_dd"],      2),
    "n_buys":      r["n_buys"],
    "n_sells":     r["n_sells"],
    "halted":      r["halted"],
    "price_open":  round(r["price_open"],  6),
    "price_close": round(r["price_close"], 6),
    "price_chg":   round(r["price_chg"],   2),
    "mode_buying_pct":   round(r["mode_buying"],   1),
    "mode_recovery_pct": round(r["mode_recovery"], 1),
    "mode_market_pct":   round(r["mode_market"],   1),
} for r in results_sorted])

csv_path = os.path.join(RESULTS_DIR, "backtest_original_full.csv")
df_out.to_csv(csv_path, index=False)
print(f"\n  CSV saved: research/results/backtest_original_full.csv")


# ═══════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════
print("\nGenerating charts...")

BG_DARK  = "#0f1117"
BG_PANEL = "#1a1d2e"
POS_CLR  = "#22c55e"
NEG_CLR  = "#ef4444"
BAH_CLR  = "#60a5fa"
PORT_CLR = "#f59e0b"

syms     = [r["symbol"]     for r in results_sorted]
rets     = [r["return_pct"] for r in results_sorted]
bahs     = [r["bah_pct"]    for r in results_sorted]
alphas   = [r["alpha"]      for r in results_sorted]
dds      = [abs(r["max_dd"]) for r in results_sorted]
bar_clrs = [POS_CLR if v >= 0 else NEG_CLR for v in rets]
bah_clrs = [POS_CLR if v >= 0 else NEG_CLR for v in bahs]

fig = plt.figure(figsize=(26, 20), facecolor=BG_DARK)
fig.suptitle(
    f"Original trader_robinhood.py  |  Full Universe 15m Backtest  |  "
    f"40 symbols  |  ${INITIAL_CASH:,.0f}/symbol  |  ~{results[0]['n_days']:.0f} days",
    color="white", fontsize=14, fontweight="bold", y=0.99,
)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35,
                       top=0.95, bottom=0.05, left=0.06, right=0.97)


def style_ax(ax, title):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color="white", fontsize=11, pad=6)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")


x = np.arange(len(syms))

# ─ [0, :] Strategy return vs BAH — full width ────────────────────────
ax0 = fig.add_subplot(gs[0, :])
style_ax(ax0, "Strategy Return vs Buy-and-Hold — Ranked by Strategy Return")
w = 0.38
ax0.bar(x - w/2, rets, w, color=bar_clrs, alpha=0.9, edgecolor="#111", label="Strategy")
ax0.bar(x + w/2, bahs, w, color=BAH_CLR,  alpha=0.45, edgecolor="#111", label="Buy & Hold")
ax0.axhline(0, color="white", lw=0.5, alpha=0.3)
ax0.axhline(avg_r, color=PORT_CLR, lw=1.0, ls="--", alpha=0.7, label=f"Avg strategy {avg_r:>+.1f}%")
ax0.axhline(avg_b, color=BAH_CLR,  lw=0.8, ls=":",  alpha=0.5, label=f"Avg BAH {avg_b:>+.1f}%")
for i, (ret, bah) in enumerate(zip(rets, bahs)):
    ax0.text(i - w/2, ret + (0.3 if ret >= 0 else -0.8),
             f"{ret:+.1f}%", ha="center", va="bottom" if ret >= 0 else "top",
             fontsize=6.5, color="white", fontweight="bold")
ax0.set_xticks(x)
ax0.set_xticklabels(syms, color="white", fontsize=8.5, rotation=45, ha="right")
ax0.set_ylabel("Return %", color="white", fontsize=9)
ax0.legend(fontsize=8, framealpha=0.35, loc="upper right")

# ─ [1, 0] Alpha (strategy - BAH) ─────────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
style_ax(ax1, "Alpha vs Buy-and-Hold (pp)")
alpha_clrs = [POS_CLR if a >= 0 else NEG_CLR for a in alphas]
ax1.bar(x, alphas, color=alpha_clrs, alpha=0.9, edgecolor="#111")
ax1.axhline(0,     color="white",   lw=0.5, alpha=0.3)
ax1.axhline(avg_a, color=PORT_CLR,  lw=1.0, ls="--", alpha=0.7, label=f"Avg {avg_a:>+.1f}pp")
for i, a in enumerate(alphas):
    ax1.text(i, a + (0.1 if a >= 0 else -0.3), f"{a:+.0f}", ha="center",
             va="bottom" if a >= 0 else "top", fontsize=6, color="white")
ax1.set_xticks(x)
ax1.set_xticklabels(syms, color="white", fontsize=7.5, rotation=45, ha="right")
ax1.set_ylabel("Alpha pp", color="white", fontsize=9)
ax1.legend(fontsize=8, framealpha=0.35)

# ─ [1, 1] Max drawdown ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
style_ax(ax2, "Max Drawdown from Peak")
dd_clrs = [NEG_CLR if d > 20 else PORT_CLR if d > 10 else POS_CLR for d in dds]
ax2.bar(x, [-d for d in dds], color=dd_clrs, alpha=0.88, edgecolor="#111")
ax2.axhline(-avg_dd, color="white", lw=0.8, ls="--", alpha=0.5, label=f"Avg -{avg_dd:.1f}%")
ax2.axhline(0, color="white", lw=0.3, alpha=0.2)
for i, d in enumerate(dds):
    ax2.text(i, -d - 0.1, f"{d:.0f}%", ha="center", va="top", fontsize=6, color="white")
ax2.set_xticks(x)
ax2.set_xticklabels(syms, color="white", fontsize=7.5, rotation=45, ha="right")
ax2.set_ylabel("Drawdown %", color="white", fontsize=9)
ax2.legend(fontsize=8, framealpha=0.35)

# ─ [2, 0] Return vs Drawdown scatter ─────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
style_ax(ax3, "Return vs Max Drawdown (risk/reward scatter)")
for r in results_sorted:
    clr = POS_CLR if r["return_pct"] >= 0 else NEG_CLR
    ax3.scatter(-r["max_dd"], r["return_pct"], color=clr, s=70, zorder=4, alpha=0.85)
    ax3.annotate(r["symbol"], (-r["max_dd"], r["return_pct"]),
                 textcoords="offset points", xytext=(4, 3),
                 fontsize=6.5, color="white", alpha=0.9)
ax3.axhline(0, color="white", lw=0.5, alpha=0.3, ls="--")
ax3.axvline(-avg_dd, color="grey", lw=0.5, alpha=0.3, ls="--")
ax3.set_xlabel("Max Drawdown %", color="white", fontsize=9)
ax3.set_ylabel("Return %", color="white", fontsize=9)

# ─ [2, 1] Aggregate stats text panel ─────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
ax4.set_facecolor(BG_PANEL)
ax4.axis("off")
for sp in ax4.spines.values():
    sp.set_edgecolor("#333")

top5   = results_sorted[:5]
bot5   = results_sorted[-5:]
halted = [r["symbol"] for r in results if r["halted"]]

summary = (
    f"  FULL UNIVERSE SUMMARY\n"
    f"  Symbols tested:   {len(results)}\n"
    f"  Capital/symbol:   ${INITIAL_CASH:,}\n"
    f"  Total capital:    ${tot_i:,}\n"
    f"  Total final:      ${tot_f:,.0f}\n"
    f"  Total P&L:        ${tot_f-tot_i:>+,.0f}  ({(tot_f-tot_i)/tot_i*100:>+.2f}%)\n"
    f"\n"
    f"  Avg strategy ret: {avg_r:>+.2f}%\n"
    f"  Avg buy-and-hold: {avg_b:>+.2f}%\n"
    f"  Avg alpha:        {avg_a:>+.2f} pp\n"
    f"  Avg max drawdown: {avg_dd:.2f}%\n"
    f"  Winners (>0%):    {n_pos} / {len(results)}\n"
    f"  Halted:           {n_halt}\n"
    f"\n"
    f"  TOP 5\n"
    + "".join(f"  {r['symbol']:<7} {r['return_pct']:>+.2f}%\n" for r in top5)
    + f"\n"
    f"  BOTTOM 5\n"
    + "".join(f"  {r['symbol']:<7} {r['return_pct']:>+.2f}%\n" for r in bot5)
    + (f"\n  HALTED: {', '.join(halted)}\n" if halted else "\n  No symbols halted\n")
    + f"\n"
    f"  Sell mode (avg):\n"
    f"  Buying:    {avg_buying:.1f}%\n"
    f"  Recovery:  {avg_recovery:.1f}%\n"
    f"  Market:    {avg_market:.1f}%\n"
)
ax4.text(0.04, 0.97, summary, transform=ax4.transAxes,
         fontsize=8.5, verticalalignment="top", color="white",
         fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#252840",
                   alpha=0.75, edgecolor="#444"))

out1 = os.path.join(RESULTS_DIR, "backtest_original_full.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"  Saved: research/results/backtest_original_full.png")
plt.close(fig)


# ── Figure 2: Portfolio curves — top 10 + bottom 10 ──────────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(22, 14), facecolor=BG_DARK,
                            gridspec_kw={"hspace": 0.45})
fig2.suptitle("Portfolio Value Curves — Top 10 & Bottom 10 by Return",
              color="white", fontsize=13, fontweight="bold", y=0.99)

cmap_g = plt.cm.Greens
cmap_r = plt.cm.Reds

for ax in axes2:
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

top10  = results_sorted[:10]
bot10  = results_sorted[-10:]

for ax2, group, cmap, title_sfx in [
    (axes2[0], top10, cmap_g, "Top 10"),
    (axes2[1], bot10, cmap_r, "Bottom 10"),
]:
    for j, r in enumerate(group):
        n   = r["n_candles"]
        ph  = np.array(r["port_hist"]) / INITIAL_CASH * 100 - 100  # % from start
        xs2 = np.linspace(0, r["n_days"], n)
        clr = cmap(0.4 + 0.5 * j / len(group))
        ax2.plot(xs2, ph, linewidth=1.2, color=clr, alpha=0.9,
                 label=f"{r['symbol']} {r['return_pct']:>+.1f}%")
    ax2.axhline(0, color="white", lw=0.6, ls="--", alpha=0.35)
    ax2.set_title(f"{title_sfx} — Portfolio % Change from $10,000 Start",
                  color="white", fontsize=11)
    ax2.set_xlabel("Days", color="white", fontsize=9)
    ax2.set_ylabel("% Return", color="white", fontsize=9)
    ax2.legend(fontsize=7.5, framealpha=0.35, ncol=2)

out2 = os.path.join(RESULTS_DIR, "backtest_original_full_curves.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"  Saved: research/results/backtest_original_full_curves.png")
plt.close(fig2)

print(f"\nDone. All output in research/results/")
