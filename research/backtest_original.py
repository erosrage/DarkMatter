"""
Backtest of Original trader_robinhood.py Strategy on 15m data.

Faithfully replicates the Bittrex/Robinhood grid logic:
  - Markets:    LTC, ETH, XRP  (original three)
  - Sweep:      20% range distributed across $10 chunks
  - Cycle:      hourly (4 × 15m candles): cancel, reprocess, re-place orders
  - avgBuy:     running weighted-average buy price per market
  - Sell modes: Market (ask > avgBuy*(1+minProf%)) or Recovery
  - minProf:    dynamic 0.7% – 5.0% based on cash/portfolio ratio
  - Fees:       0.26% buy-side cost / 0.26% sell-side reduction
  - Capital:    $30,000 shared, split equally across markets each cycle
"""

import os
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

# ── Config — mirrors trader_robinhood.py exactly ─────────────────────
MARKETS      = ['LTC', 'ETH', 'XRP']
SWEEP        = 20.0         # % range for order distribution
ORDER_SIZE   = 10.0         # USD per order chunk
RESERVE      = 0.0          # USD to hold back
INTERVAL_C   = 4            # candles per reset cycle (4 × 15m = 1 hr)
FEE_BUY      = 1.0026       # buy cost multiplier (0.26% fee)
FEE_SELL     = 0.9974       # sell revenue multiplier (0.26% fee)
INITIAL_CASH = 30_000       # total starting capital (shared)
CIRCUIT_BRK  = 0.30         # halt if portfolio drops 30%

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data", "15m")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Exact copies of helper functions from trader_robinhood.py ─────────
def getWavg(pairs):
    """Weighted average of (price, qty) pairs."""
    tot = sum(q for _, q in pairs)
    return sum(p * q for p, q in pairs) / tot if tot > 0 else 0.0


def getMinProf(usd_bal, portfolio_val):
    """Dynamic min-profit % clamped to [0.7, 5.0]."""
    x = usd_bal / portfolio_val if portfolio_val > 0 else 1.0
    a, b, c, d = 1.0, 0.3, 0.7, 5.0
    return max(c, min(d, (x - a) / (b - a) * (d - c) + c))


def _interval(balance):
    chunks = balance / ORDER_SIZE
    return SWEEP / chunks if chunks >= 1 else SWEEP


def _stable(n, spread_pct):
    return [i * spread_pct / 100.0 for i in range(1, int(n) + 1)]


def _sell_rates(base, table):
    rates = [base]
    for e in range(len(table) - 1):
        rates.append((1 + table[e]) * base)
    return rates


def _buy_rates(base, table):
    return [(1 - e) * base for e in table]


def make_buy_orders(usd_balance, ask):
    """Generate buy order list: [(price, qty), ...]  — from original buy()."""
    n = int(usd_balance / ORDER_SIZE)
    if n < 1:
        return []
    table = _stable(n, _interval(usd_balance))[:n]
    prices = _buy_rates(ask, table)
    return [(p, ORDER_SIZE / p) for p in prices if p > 0]


def make_sell_orders(coin_bal, sell_base):
    """Generate sell order list: [(price, qty), ...]  — from original sell()."""
    if coin_bal <= 0 or sell_base <= 0:
        return []
    notional = coin_bal * sell_base
    n = int(notional / ORDER_SIZE)
    if n < 1:
        return []
    table = _stable(n, _interval(notional))
    rates = _sell_rates(sell_base, table)[:n]
    qty_each = coin_bal / n
    return [(r, qty_each) for r in rates if r > 0]


# ── Data loading ──────────────────────────────────────────────────────
def load_data():
    frames = {}
    for m in MARKETS:
        path = os.path.join(DATA_DIR, f"{m.lower()}_usd_15m.csv")
        df = pd.read_csv(path, parse_dates=["date"]).dropna(subset=["close"])
        frames[m] = df.set_index("date")["close"]
    combined = pd.DataFrame(frames).dropna()
    print(f"  {len(combined)} common candles  "
          f"[{combined.index[0].date()} to {combined.index[-1].date()}]")
    return combined


# ── Main simulation ───────────────────────────────────────────────────
def simulate():
    print("\nLoading 15m price data...")
    combined = load_data()
    dates    = combined.index.tolist()
    N        = len(dates)

    cash = float(INITIAL_CASH)

    mkt = {m: {
        "holdings":  0.0,
        "avgBuy":    0.0,
        "profit":    0.0,
        "orders":    [],    # list of (side, price, qty)
        "buys_ev":   [],    # (candle_idx, fill_price)
        "sells_ev":  [],
        "mode_hist": [],    # (candle_idx, mode_str)
    } for m in MARKETS}

    min_prof  = 5.0
    port_hist = []
    halted    = False

    def pval(row_):
        return cash + sum(mkt[m]["holdings"] * float(row_[m]) for m in MARKETS)

    # ── Initial buy orders at candle 0 ────────────────────────────────
    row0    = combined.iloc[0]
    per_mkt = INITIAL_CASH / len(MARKETS)
    for m in MARKETS:
        mkt[m]["orders"] = [
            ("buy", p, q)
            for p, q in make_buy_orders(per_mkt, float(row0[m]))
            if p > 0 and q > 0
        ]

    print("Running simulation...")
    for i in range(N):
        row   = combined.iloc[i]
        pv    = pval(row)

        # Circuit breaker
        if CIRCUIT_BRK and pv < INITIAL_CASH * (1 - CIRCUIT_BRK):
            print(f"  Circuit breaker at candle {i}  port=${pv:,.0f}")
            port_hist.extend([pv] * (N - i))
            halted = True
            break

        # ── Fill orders ───────────────────────────────────────────────
        for m in MARKETS:
            st    = mkt[m]
            price = float(row[m])

            # Split and sort for deterministic fill priority
            buys  = sorted([o for o in st["orders"] if o[0] == "buy"],  key=lambda o: -o[1])
            sells = sorted([o for o in st["orders"] if o[0] == "sell"], key=lambda o:  o[1])

            filled  = []
            remain  = []

            for side, op, oq in buys:
                if price <= op:
                    cost = op * oq * FEE_BUY
                    if cash >= cost:
                        prev_h = st["holdings"]
                        cash          -= cost
                        st["holdings"] += oq
                        # Running weighted-average buy price
                        if prev_h > 0 and st["avgBuy"] > 0:
                            st["avgBuy"] = (st["avgBuy"] * prev_h + op * oq) / st["holdings"]
                        else:
                            st["avgBuy"] = op
                        st["buys_ev"].append((i, op))
                        filled.append((side, op, oq))
                    else:
                        remain.append((side, op, oq))
                else:
                    remain.append((side, op, oq))

            for side, op, oq in sells:
                if price >= op:
                    if st["holdings"] >= oq:
                        proceeds        = op * oq * FEE_SELL
                        cost_basis      = st["avgBuy"] * oq * FEE_BUY if st["avgBuy"] > 0 else 0
                        cash           += proceeds
                        st["holdings"] -= oq
                        st["profit"]   += proceeds - cost_basis
                        if st["holdings"] <= 1e-12:
                            st["holdings"] = 0.0
                            st["avgBuy"]   = 0.0
                        st["sells_ev"].append((i, op))
                        filled.append((side, op, oq))
                    else:
                        remain.append((side, op, oq))
                else:
                    remain.append((side, op, oq))

            st["orders"] = remain

        # ── Hourly cycle reset ─────────────────────────────────────────
        if i > 0 and i % INTERVAL_C == 0:
            pv        = pval(row)
            min_prof  = max(0.5, getMinProf(cash, pv))
            per_mkt   = max(0.0, cash - RESERVE) / len(MARKETS)

            for m in MARKETS:
                st    = mkt[m]
                price = float(row[m])
                coin  = st["holdings"]
                avg   = st["avgBuy"]

                st["orders"] = []   # cancel all pending

                # ── Sell orders ──────────────────────────────────────
                if coin > 0 and avg > 0:
                    thresh = avg * (1 + min_prof / 100.0)
                    if price > thresh:
                        sell_base = price   # Market mode
                        mode = "Market"
                    else:
                        sell_base = thresh  # Recovery mode
                        mode = "Recovery"
                    for p, q in make_sell_orders(coin, sell_base):
                        if p > 0 and q > 0:
                            st["orders"].append(("sell", p, q))
                else:
                    mode = "Buying"

                st["mode_hist"].append((i, mode))

                # ── Buy orders (mirrors original runMarket buy logic) ─
                # In Market mode, buy base = avgBuy (cheaper baseline)
                # Otherwise, buy base = current price
                if mode == "Market" and avg > 0 and avg < price:
                    buy_base = avg
                else:
                    buy_base = price

                for p, q in make_buy_orders(per_mkt, buy_base):
                    if p > 0 and q > 0:
                        st["orders"].append(("buy", p, q))

        port_hist.append(pval(row))

    last_i      = len(port_hist) - 1
    last_row    = combined.iloc[last_i]
    final_value = port_hist[-1]
    return_pct  = (final_value - INITIAL_CASH) / INITIAL_CASH * 100

    # ── Buy-and-hold benchmark ─────────────────────────────────────────
    bah_per = INITIAL_CASH / len(MARKETS)
    bah_val = sum(
        bah_per / float(combined.iloc[0][m]) * float(last_row[m])
        for m in MARKETS
    )
    bah_pct = (bah_val - INITIAL_CASH) / INITIAL_CASH * 100

    return {
        "dates":        dates[:len(port_hist)],
        "port_hist":    port_hist,
        "mkt":          mkt,
        "cash":         cash,
        "final_value":  final_value,
        "return_pct":   return_pct,
        "bah_val":      bah_val,
        "bah_pct":      bah_pct,
        "halted":       halted,
        "combined":     combined,
        "N":            N,
        "min_prof":     min_prof,
        "prices_first": {m: float(combined.iloc[0][m])    for m in MARKETS},
        "prices_last":  {m: float(last_row[m])            for m in MARKETS},
    }


# ── Run ───────────────────────────────────────────────────────────────
res = simulate()
mkt         = res["mkt"]
port_hist   = res["port_hist"]
dates       = res["dates"]
combined    = res["combined"]
n_candles   = len(port_hist)
n_days      = n_candles * 15 / 60 / 24

# ── Drawdown ──────────────────────────────────────────────────────────
arr  = np.array(port_hist)
peak = np.maximum.accumulate(arr)
dd   = (arr - peak) / peak * 100


# ═══════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════
hdr = "=" * 90
print(f"\n{hdr}")
print(f"  ORIGINAL TRADER_ROBINHOOD.PY  —  15m BACKTEST")
print(f"  Markets: {', '.join(MARKETS)}  |  SWEEP={SWEEP}%  "
      f"ORDER_SIZE=${ORDER_SIZE:.0f}  CYCLE={INTERVAL_C} candles (1h)")
print(f"  Capital: ${INITIAL_CASH:,.0f} total (${INITIAL_CASH/len(MARKETS):,.0f}/market)  "
      f"|  Fee: {(FEE_BUY-1)*100:.2f}% buy  {(1-FEE_SELL)*100:.2f}% sell")
print(f"{hdr}\n")

print(f"  Period:         {n_candles:,} candles  (~{n_days:.0f} days)")
print(f"  Start price:    " + "  ".join(f"{m}=${res['prices_first'][m]:>8,.3f}" for m in MARKETS))
print(f"  End   price:    " + "  ".join(f"{m}=${res['prices_last'][m]:>8,.3f}" for m in MARKETS))
print(f"\n  Strategy Return:  ${res['final_value']:>12,.2f}  ({res['return_pct']:>+.2f}%)")
print(f"  Buy-and-Hold:     ${res['bah_val']:>12,.2f}  ({res['bah_pct']:>+.2f}%)")
print(f"  Alpha:            {res['return_pct'] - res['bah_pct']:>+.2f} pp")
print(f"  Halted:           {'YES' if res['halted'] else 'No'}")
print(f"  Final cash:       ${res['cash']:>12,.2f}")
print(f"  Final minProf:    {res['min_prof']:.2f}%")
print(f"  Max drawdown:     {dd.min():.2f}%")

print(f"\n  {'Market':<8} {'Holdings':>12} {'HoldVal':>12} {'AvgBuy':>10} "
      f"{'Profit':>12} {'Buys':>6} {'Sells':>6}")
print(f"  {'-'*74}")

for m in MARKETS:
    st   = mkt[m]
    hval = st["holdings"] * res["prices_last"][m]
    print(f"  {m:<8} {st['holdings']:>12.6f}  ${hval:>10,.2f}  "
          f"${st['avgBuy']:>8,.4f}  ${st['profit']:>10,.2f}  "
          f"{len(st['buys_ev']):>6}  {len(st['sells_ev']):>6}")

total_profit = sum(mkt[m]["profit"] for m in MARKETS)
total_buys   = sum(len(mkt[m]["buys_ev"]) for m in MARKETS)
total_sells  = sum(len(mkt[m]["sells_ev"]) for m in MARKETS)
total_hold_v = sum(mkt[m]["holdings"] * res["prices_last"][m] for m in MARKETS)

print(f"\n  TOTAL     {'':>12}  ${total_hold_v:>10,.2f}  {'':>10}  "
      f"${total_profit:>10,.2f}  {total_buys:>6}  {total_sells:>6}")

# Mode breakdown
print(f"\n  SELL MODE DISTRIBUTION (at hourly resets)")
print(f"  {'Market':<8} {'Buying%':>9} {'Recovery%':>11} {'Market%':>9}")
print(f"  {'-'*42}")
for m in MARKETS:
    hist = mkt[m]["mode_hist"]
    if hist:
        buying    = sum(1 for _, mo in hist if mo == "Buying")    / len(hist) * 100
        recovery  = sum(1 for _, mo in hist if mo == "Recovery")  / len(hist) * 100
        market    = sum(1 for _, mo in hist if mo == "Market")    / len(hist) * 100
    else:
        buying = recovery = market = 0.0
    print(f"  {m:<8} {buying:>8.1f}%  {recovery:>10.1f}%  {market:>8.1f}%")
print()


# ═══════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════
print("Generating charts...")

BG_DARK  = "#0f1117"
BG_PANEL = "#1a1d2e"
PORT_CLR = "#f59e0b"
BAH_CLR  = "#60a5fa"
BUY_CLR  = "#22c55e"
SELL_CLR = "#ef4444"
PALETTE  = {"LTC": "#A0A0A0", "ETH": "#627EEA", "XRP": "#346AA9"}

xs          = np.arange(n_candles)
date_labels = [str(d)[:10] for d in dates]
tick_step   = max(1, n_candles // 8)
tick_pos    = list(range(0, n_candles, tick_step))
tick_lbl    = [date_labels[p] for p in tick_pos]

def fmt_k(v, _):
    return f"${v/1e3:.0f}k"

# ── Figure 1: Portfolio overview ──────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(18, 14), facecolor=BG_DARK,
                          gridspec_kw={"hspace": 0.45})
fig.suptitle(
    f"Original trader_robinhood.py  |  15m Backtest  |  "
    f"LTC / ETH / XRP  |  SWEEP={SWEEP}%  ORDER=${ORDER_SIZE:.0f}  "
    f"Strategy={res['return_pct']:>+.2f}%  vs BAH={res['bah_pct']:>+.2f}%",
    color="white", fontsize=13, fontweight="bold", y=0.99,
)

for ax in axes:
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

# ─ Portfolio value ────────────────────────────────────────────────────
ax0 = axes[0]
bah_hist = []
for i in range(n_candles):
    row = combined.iloc[i]
    bah_v = sum(
        (INITIAL_CASH / len(MARKETS)) / float(combined.iloc[0][m]) * float(row[m])
        for m in MARKETS
    )
    bah_hist.append(bah_v)

ax0.plot(xs, port_hist, color=PORT_CLR, linewidth=1.3, label=f"Strategy ({res['return_pct']:>+.1f}%)", zorder=3)
ax0.plot(xs, bah_hist,  color=BAH_CLR,  linewidth=1.0, ls="--", alpha=0.7,
         label=f"Buy & Hold ({res['bah_pct']:>+.1f}%)", zorder=2)
ax0.fill_between(xs, INITIAL_CASH, port_hist,
                 where=np.array(port_hist) >= INITIAL_CASH,
                 alpha=0.20, color=BUY_CLR)
ax0.fill_between(xs, INITIAL_CASH, port_hist,
                 where=np.array(port_hist) < INITIAL_CASH,
                 alpha=0.20, color=SELL_CLR)
ax0.axhline(INITIAL_CASH, color="white", lw=0.7, ls="--", alpha=0.4)
ax0.set_title("Portfolio Value — Strategy vs Buy & Hold", color="white", fontsize=11)
ax0.set_ylabel("USD", color="white", fontsize=9)
ax0.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax0.tick_params(colors="white", labelsize=7)
ax0.set_xticks(tick_pos)
ax0.set_xticklabels(tick_lbl, rotation=22, ha="right", fontsize=7, color="white")
ax0.legend(fontsize=8.5, framealpha=0.35)

# ─ Drawdown ───────────────────────────────────────────────────────────
ax1 = axes[1]
ax1.fill_between(xs, 0, dd, color=SELL_CLR, alpha=0.55)
ax1.plot(xs, dd, color=SELL_CLR, linewidth=0.9)
ax1.axhline(dd.min(), color=SELL_CLR, lw=0.6, ls="--", alpha=0.6,
            label=f"Max {dd.min():.1f}%")
ax1.axhline(0, color="white", lw=0.5, alpha=0.3)
ax1.set_title(f"Portfolio Drawdown from Peak  (worst: {dd.min():.1f}%)",
              color="white", fontsize=11)
ax1.set_ylabel("Drawdown %", color="white", fontsize=9)
ax1.tick_params(colors="white", labelsize=7)
ax1.set_xticks(tick_pos)
ax1.set_xticklabels(tick_lbl, rotation=22, ha="right", fontsize=7, color="white")
ax1.legend(fontsize=8, framealpha=0.35)

# ─ Normalised price chart ─────────────────────────────────────────────
ax2 = axes[2]
for m in MARKETS:
    p0  = float(combined.iloc[0][m])
    ser = [float(combined.iloc[min(j, len(combined)-1)][m]) / p0 * 100 - 100
           for j in range(n_candles)]
    buys_x  = [ev[0] for ev in mkt[m]["buys_ev"]  if ev[0] < n_candles]
    sells_x = [ev[0] for ev in mkt[m]["sells_ev"] if ev[0] < n_candles]
    buys_y  = [ser[x] for x in buys_x]
    sells_y = [ser[x] for x in sells_x]
    ax2.plot(range(n_candles), ser, color=PALETTE[m], linewidth=0.8, alpha=0.9, label=m)
    if buys_x:
        ax2.scatter(buys_x, buys_y, marker="^", color=BUY_CLR,
                    s=50, zorder=5, alpha=0.6, edgecolors="none")
    if sells_x:
        ax2.scatter(sells_x, sells_y, marker="v", color=SELL_CLR,
                    s=50, zorder=5, alpha=0.6, edgecolors="none")

ax2.axhline(0, color="white", lw=0.5, alpha=0.3)
ax2.set_title("Normalised Price — Buy (▲) / Sell (▼) Executions",
              color="white", fontsize=11)
ax2.set_ylabel("% from open", color="white", fontsize=9)
ax2.tick_params(colors="white", labelsize=7)
ax2.set_xticks(tick_pos)
ax2.set_xticklabels(tick_lbl, rotation=22, ha="right", fontsize=7, color="white")
ax2.legend(fontsize=9, framealpha=0.35)

out1 = os.path.join(RESULTS_DIR, "backtest_original_overview.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"  Saved: research/results/backtest_original_overview.png")
plt.close(fig)


# ── Figure 2: Per-market detail ───────────────────────────────────────
fig2, axs = plt.subplots(3, 3, figsize=(20, 15), facecolor=BG_DARK,
                          gridspec_kw={"hspace": 0.50, "wspace": 0.36})
fig2.suptitle(
    "Original trader_robinhood.py  —  Per-Market Detail  (15m, ~60 days)",
    color="white", fontsize=14, fontweight="bold", y=0.99,
)

for row_axs in axs:
    for ax in row_axs:
        ax.set_facecolor(BG_PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

for col, m in enumerate(MARKETS):
    st     = mkt[m]
    clr    = PALETTE[m]
    prices = [float(combined.iloc[min(j, len(combined)-1)][m]) for j in range(n_candles)]
    bev    = [(e[0], e[1]) for e in st["buys_ev"]  if e[0] < n_candles]
    sev    = [(e[0], e[1]) for e in st["sells_ev"] if e[0] < n_candles]

    # Price + trades
    ax = axs[0][col]
    ax.plot(xs, prices, color=clr, linewidth=0.9, alpha=0.9)
    if bev:
        ax.scatter([e[0] for e in bev], [e[1] for e in bev],
                   marker="^", color=BUY_CLR, s=60, zorder=5, alpha=0.7,
                   edgecolors="white", linewidths=0.3)
    if sev:
        ax.scatter([e[0] for e in sev], [e[1] for e in sev],
                   marker="v", color=SELL_CLR, s=60, zorder=5, alpha=0.7,
                   edgecolors="white", linewidths=0.3)
    ax.set_title(f"{m}  Buys={len(bev)}  Sells={len(sev)}", color="white", fontsize=11)
    ax.set_ylabel("Price USD", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xticks(tick_pos[::2])
    ax.set_xticklabels(tick_lbl[::2], rotation=22, ha="right", fontsize=7, color="white")

    # Sell mode timeline
    ax2b = axs[1][col]
    mode_map = {"Buying": 0, "Recovery": 1, "Market": 2}
    mode_c   = {"Buying": BAH_CLR, "Recovery": SELL_CLR, "Market": BUY_CLR}
    for ci, (mi, mo) in enumerate(st["mode_hist"]):
        nxt = st["mode_hist"][ci + 1][0] if ci + 1 < len(st["mode_hist"]) else n_candles
        ax2b.axvspan(mi, nxt, alpha=0.45, color=mode_c.get(mo, "grey"))
    # Legend patches
    from matplotlib.patches import Patch
    patches = [Patch(color=mode_c[k], label=k, alpha=0.6) for k in ["Buying", "Recovery", "Market"]]
    ax2b.legend(handles=patches, fontsize=7.5, framealpha=0.35, loc="upper left")
    ax2b.set_title(f"{m}  Sell Mode Over Time", color="white", fontsize=11)
    ax2b.tick_params(colors="white", labelsize=7)
    ax2b.set_xticks(tick_pos[::2])
    ax2b.set_xticklabels(tick_lbl[::2], rotation=22, ha="right", fontsize=7, color="white")
    ax2b.set_yticks([])

    # Stats panel
    ax3b = axs[2][col]
    ax3b.axis("off")
    p_first = res["prices_first"][m]
    p_last  = res["prices_last"][m]
    price_chg = (p_last - p_first) / p_first * 100
    hval = st["holdings"] * p_last
    stats = (
        f"  {m} SUMMARY\n"
        f"  Open price:   ${p_first:>10,.4f}\n"
        f"  Close price:  ${p_last:>10,.4f}\n"
        f"  Price Δ:      {price_chg:>+10.2f}%\n\n"
        f"  Buys:         {len(bev):>10}\n"
        f"  Sells:        {len(sev):>10}\n"
        f"  Holdings:     {st['holdings']:>10.4f}\n"
        f"  Hold value:   ${hval:>10,.2f}\n"
        f"  Avg buy:      ${st['avgBuy']:>10,.4f}\n"
        f"  Realised P&L: ${st['profit']:>10,.2f}\n\n"
        f"  Mode history: {len(st['mode_hist'])} resets\n"
    )
    if st["mode_hist"]:
        buying   = sum(1 for _, mo in st["mode_hist"] if mo == "Buying")
        recovery = sum(1 for _, mo in st["mode_hist"] if mo == "Recovery")
        market   = sum(1 for _, mo in st["mode_hist"] if mo == "Market")
        n = len(st["mode_hist"])
        stats += (
            f"  Buying:       {buying/n*100:>9.1f}%\n"
            f"  Recovery:     {recovery/n*100:>9.1f}%\n"
            f"  Market:       {market/n*100:>9.1f}%\n"
        )
    ax3b.text(0.04, 0.97, stats, transform=ax3b.transAxes,
              fontsize=8.5, verticalalignment="top", color="white",
              fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#252840",
                        alpha=0.7, edgecolor="#444"))

out2 = os.path.join(RESULTS_DIR, "backtest_original_permarket.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"  Saved: research/results/backtest_original_permarket.png")
plt.close(fig2)

print(f"\nDone. Charts saved to research/results/")
