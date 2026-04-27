"""
research/param_search_full.py
Full brute-force parameter search: all symbols × all available intervals × all param combos.

  - 80/20 temporal train/val split per symbol-interval pair
  - Fitness: best raw ROI on train data, halted runs excluded
  - Validation: winning train config re-run on held-out 20% of price history
  - Multiprocessing: one worker per (symbol, interval) pair
  - Resumable: skips pairs already present in the output CSV

Usage:
    python research/param_search_full.py           # full search (~hours)
    python research/param_search_full.py --quick   # reduced grid (~minutes)
"""

import os
import sys
import csv
import time
import itertools
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Backtest import ────────────────────────────────────────────────────────────
_BT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ArchivedTests")
if _BT_DIR not in sys.path:
    sys.path.insert(0, _BT_DIR)
from backtest import backtest as _backtest  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH SPACE  — edit here to widen or narrow the search
# ══════════════════════════════════════════════════════════════════════════════
QUICK_MODE = "--quick" in sys.argv

if QUICK_MODE:
    # ~108 combos/pair — fast sanity check
    SWEEP_VALS     = [0.05, 0.10, 0.20]
    STEPS_VALS     = [8, 20, 48]
    ORDER_PCT_VALS = [0.10, 0.25, 0.50]
    COUNTER_VALS   = [4, 8, 12]
    MA_VALS        = [None]                              # MA universally unhelpful — dropped
    ATR_CFGS       = [(None, None)]                      # ATR universally unhelpful — dropped
else:
    # ~1,050 combos/pair — full run (16x faster than original after dropping MA/ATR)
    SWEEP_VALS     = [0.05, 0.10, 0.15, 0.20, 0.25]
    STEPS_VALS     = [4, 8, 12, 20, 32, 48]
    ORDER_PCT_VALS = [0.05, 0.10, 0.20, 0.25, 0.50]
    COUNTER_VALS   = [2, 4, 6, 8, 10, 12, 16]
    MA_VALS        = [None]                              # MA universally unhelpful — dropped
    ATR_CFGS       = [(None, None)]                      # ATR universally unhelpful — dropped

# Fixed across all runs
INITIAL_CASH = 100_000
FIXED_PARAMS = dict(
    reset_interval  = None,
    reactive        = True,
    circuit_breaker = 0.20,
    min_sweep       = 0.005,
    max_sweep       = 0.50,
    initial_cash    = INITIAL_CASH,
)

TRAIN_RATIO = 0.80   # first 80% = train, last 20% = validation

# ══════════════════════════════════════════════════════════════════════════════
# DATA SOURCES  — add or remove intervals here
# ══════════════════════════════════════════════════════════════════════════════
_RESEARCH = os.path.dirname(os.path.abspath(__file__))

INTERVALS = {
    "2m":  (os.path.join(_RESEARCH, "data", "2m"),          "_usd_2m.csv"),
    "5m":  (os.path.join(_RESEARCH, "data", "5m"),          "_usd_5m.csv"),
    "15m": (os.path.join(_RESEARCH, "data", "15m"),         "_usd_15m.csv"),
    "1h":  (os.path.join(_RESEARCH, "data", "2y", "hourly"), "_usd_1h.csv"),
}

OUT_DIR    = os.path.join(_RESEARCH, "results")
CSV_PATH   = os.path.join(OUT_DIR, "param_search_full.csv")
CHART_PATH = os.path.join(OUT_DIR, "param_search_full.png")

CSV_FIELDS = [
    "symbol", "interval", "train_roi", "val_roi", "val_halted",
    "n_valid_combos", "n_train_candles", "n_val_candles",
    "sweep", "steps", "order_size_pct", "counter_multiplier",
    "ma_period", "atr_period", "atr_multiplier",
]

# ══════════════════════════════════════════════════════════════════════════════
# COMBO GENERATION  (module-level so worker processes share via re-import)
# ══════════════════════════════════════════════════════════════════════════════
def _build_combos():
    out = []
    for sweep, steps, order_pct, counter, ma, (atr_p, atr_m) in itertools.product(
        SWEEP_VALS, STEPS_VALS, ORDER_PCT_VALS, COUNTER_VALS, MA_VALS, ATR_CFGS
    ):
        out.append(dict(
            sweep              = sweep,
            steps              = steps,
            order_size_pct     = order_pct,
            counter_multiplier = counter,
            ma_period          = ma,
            atr_period         = atr_p,
            atr_multiplier     = atr_m if atr_m is not None else 2.0,
        ))
    return out

COMBOS = _build_combos()

# ══════════════════════════════════════════════════════════════════════════════
# WORKER  (must be importable at top level for multiprocessing on Windows)
# ══════════════════════════════════════════════════════════════════════════════
def _worker(args):
    """
    Run every combo on the training slice for one (symbol, interval) pair.
    Pick the best non-halted config by train ROI, then validate it on the
    held-out 20% slice.
    """
    symbol, interval, train_prices, val_prices = args

    best_roi   = -float("inf")
    best_combo = None
    n_valid    = 0

    for combo in COMBOS:
        try:
            r   = _backtest(train_prices, **combo, **FIXED_PARAMS)
            roi = (r["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
        except Exception:
            continue

        if r.get("halted"):
            continue

        n_valid += 1
        if roi > best_roi:
            best_roi   = roi
            best_combo = combo

    # Validate best config on holdout slice
    val_roi    = None
    val_halted = True
    if best_combo and len(val_prices) >= 20:
        try:
            r_v        = _backtest(val_prices, **best_combo, **FIXED_PARAMS)
            val_roi    = (r_v["final_value"] - INITIAL_CASH) / INITIAL_CASH * 100
            val_halted = bool(r_v.get("halted", False))
        except Exception:
            pass

    row = {
        "symbol"          : symbol,
        "interval"        : interval,
        "train_roi"       : round(best_roi, 4) if best_combo else None,
        "val_roi"         : round(val_roi, 4)  if val_roi is not None else None,
        "val_halted"      : val_halted,
        "n_valid_combos"  : n_valid,
        "n_train_candles" : len(train_prices),
        "n_val_candles"   : len(val_prices),
    }
    if best_combo:
        row.update({
            "sweep"             : best_combo["sweep"],
            "steps"             : best_combo["steps"],
            "order_size_pct"    : best_combo["order_size_pct"],
            "counter_multiplier": best_combo["counter_multiplier"],
            "ma_period"         : best_combo["ma_period"],
            "atr_period"        : best_combo["atr_period"],
            "atr_multiplier"    : best_combo["atr_multiplier"],
        })
    return row

# ══════════════════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════════════════
IVL_COLOR = {"2m": "#4C72B0", "5m": "#55A868", "15m": "#C44E52", "1h": "#8172B2"}
IVL_ORDER  = ["2m", "5m", "15m", "1h"]

def _make_chart(df):
    avail_ivl = [i for i in IVL_ORDER if i in df["interval"].unique()]
    symbols   = sorted(df["symbol"].unique())
    valid     = df.dropna(subset=["train_roi"])

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        "Grid Strategy — Full Parameter Search Results"
        f"  ({'Quick' if QUICK_MODE else 'Full'} mode, {len(COMBOS):,} combos/pair)",
        fontsize=15, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.50, wspace=0.35,
        height_ratios=[1.5, 1.0, 1.0],
    )

    # ── Panel 1: Grouped bar — best val ROI per symbol ────────────────────────
    ax1  = fig.add_subplot(gs[0, :])
    x    = np.arange(len(symbols))
    n_iv = len(avail_ivl)
    bw   = 0.75 / n_iv
    for k, ivl in enumerate(avail_ivl):
        sub  = valid[valid["interval"] == ivl].set_index("symbol")["val_roi"]
        vals = [sub.get(s, np.nan) for s in symbols]
        ax1.bar(
            x + k * bw - (n_iv - 1) * bw / 2, vals, bw * 0.90,
            label=ivl, color=IVL_COLOR.get(ivl, "grey"), alpha=0.85,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(symbols, rotation=45, ha="right", fontsize=8)
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set_ylabel("Best Validation ROI (%)")
    ax1.set_title("Best Validation ROI per Symbol per Interval", fontsize=12)
    ax1.legend(title="Interval", loc="upper right")
    ax1.yaxis.grid(True, alpha=0.35)

    # ── Panel 2: Train vs Val scatter ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for ivl in avail_ivl:
        sub = valid[valid["interval"] == ivl].dropna(subset=["val_roi"])
        ax2.scatter(sub["train_roi"], sub["val_roi"],
                    alpha=0.75, s=45, label=ivl,
                    color=IVL_COLOR.get(ivl, "grey"))
    all_vals = pd.concat([valid["train_roi"].dropna(), valid["val_roi"].dropna()])
    if len(all_vals):
        lo, hi = all_vals.min() * 1.1, all_vals.max() * 1.1
        ax2.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x (perfect)")
    ax2.axhline(0, color="red", lw=0.5, ls=":")
    ax2.axvline(0, color="red", lw=0.5, ls=":")
    ax2.set_xlabel("Train ROI (%)")
    ax2.set_ylabel("Validation ROI (%)")
    ax2.set_title("Train vs Validation ROI (overfitting check)", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 3: Heatmap — avg val ROI by sweep × counter_mult ───────────────
    ax3    = fig.add_subplot(gs[1, 1])
    hdf    = valid.dropna(subset=["sweep", "counter_multiplier", "val_roi"])
    if not hdf.empty:
        pivot = hdf.pivot_table(
            values="val_roi", index="sweep",
            columns="counter_multiplier", aggfunc="mean",
        ).sort_index(ascending=False)
        im = ax3.imshow(
            pivot.values, aspect="auto", cmap="RdYlGn",
            vmin=min(0, np.nanmin(pivot.values)),
            vmax=np.nanmax(pivot.values),
        )
        ax3.set_xticks(range(len(pivot.columns)))
        ax3.set_xticklabels([str(c) for c in pivot.columns], fontsize=8)
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels([f"{r:.2f}" for r in pivot.index], fontsize=8)
        ax3.set_xlabel("counter_multiplier")
        ax3.set_ylabel("sweep")
        ax3.set_title("Avg Val ROI: sweep × counter_multiplier", fontsize=11)
        plt.colorbar(im, ax=ax3, label="Val ROI (%)")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax3.text(j, i, f"{v:.1f}", ha="center", va="center",
                             fontsize=6.5, color="black")
    else:
        ax3.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax3.transAxes)

    # ── Panel 4: Box plot — val ROI distribution per interval ────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    box_data = [
        valid[valid["interval"] == ivl]["val_roi"].dropna().tolist()
        for ivl in avail_ivl
    ]
    if any(box_data):
        bp = ax4.boxplot(
            box_data, labels=avail_ivl, patch_artist=True,
            medianprops=dict(color="black", lw=1.5),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, ivl in zip(bp["boxes"], avail_ivl):
            patch.set_facecolor(IVL_COLOR.get(ivl, "grey"))
            patch.set_alpha(0.75)
    ax4.axhline(0, color="red", lw=0.8, ls="--")
    ax4.set_ylabel("Best Validation ROI (%)")
    ax4.set_title("Val ROI Distribution by Interval", fontsize=11)
    ax4.yaxis.grid(True, alpha=0.35)

    # ── Panel 5: Top-20 table by val ROI ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    top20 = valid.dropna(subset=["val_roi"]).sort_values("val_roi", ascending=False).head(20)
    rows  = []
    for _, r in top20.iterrows():
        ma_s  = str(int(r["ma_period"]))  if pd.notna(r.get("ma_period"))  and r.get("ma_period")  else "—"
        atr_s = str(int(r["atr_period"])) if pd.notna(r.get("atr_period")) and r.get("atr_period") else "—"
        rows.append([
            r["symbol"], r["interval"],
            f"{r['train_roi']:+.1f}%",
            f"{r['val_roi']:+.1f}%",
            f"{r.get('sweep',''):.2f}/{int(r.get('steps', 0)) if pd.notna(r.get('steps')) else '?'}",
            f"cm={r.get('counter_multiplier',''):.0f}",
            f"ma={ma_s} atr={atr_s}",
        ])
    cols = ["Symbol", "Ivl", "Train", "Val", "Swp/Steps", "CM", "MA/ATR"]
    tbl  = ax5.table(cellText=rows, colLabels=cols,
                     cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (row_i, col_i), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_i % 2 == 0:
            cell.set_facecolor("#EAF0F6")
    ax5.set_title("Top 20 by Validation ROI", fontsize=11)

    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {CHART_PATH}")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def _load_existing_pairs():
    """Return set of (symbol, interval) already in the CSV (for resumption)."""
    done = set()
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            for _, r in df.iterrows():
                done.add((r["symbol"], r["interval"]))
        except Exception:
            pass
    return done

def _discover_tasks(skip_pairs):
    """Find all (symbol, interval, train_prices, val_prices) from data dirs."""
    tasks = []
    for interval, (data_dir, suffix) in INTERVALS.items():
        if not os.path.isdir(data_dir):
            print(f"  [skip] {interval}: data dir not found ({data_dir})")
            continue
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(suffix):
                continue
            symbol = fname.split("_usd_")[0].upper()
            if (symbol, interval) in skip_pairs:
                print(f"  [resume] skipping {symbol} {interval} (already done)")
                continue
            path = os.path.join(data_dir, fname)
            try:
                df     = pd.read_csv(path, parse_dates=["date"])
                prices = df["close"].dropna().tolist()
                if len(prices) < 50:
                    continue
                split  = int(len(prices) * TRAIN_RATIO)
                tasks.append((symbol, interval, prices[:split], prices[split:]))
            except Exception as e:
                print(f"  [warn] {symbol} {interval}: {e}")
    return tasks

def _print_table(results):
    df    = pd.DataFrame(results)
    valid = df.dropna(subset=["train_roi"])
    if valid.empty:
        print("  No valid results to display.")
        return df

    ranked = valid.sort_values("val_roi", ascending=False, na_position="last")

    print(f"\n{'='*105}")
    print(f"  RESULTS — ranked by validation ROI")
    print(f"{'='*105}")
    header = (f"  {'Symbol':>8}  {'Ivl':<4}  {'TrainROI':>9}  {'ValROI':>9}  "
              f"{'Sweep':>6}  {'Steps':>5}  {'OrdPct':>6}  {'CM':>5}  {'MA':>5}  {'ATR':>5}")
    print(header)
    print(f"  {'-'*100}")

    for _, r in ranked.iterrows():
        t_s   = f"{r['train_roi']:>+9.2f}%" if pd.notna(r.get("train_roi")) else "      N/A"
        v_s   = f"{r['val_roi']:>+9.2f}%"   if pd.notna(r.get("val_roi"))   else "      N/A"
        ma_s  = str(int(r["ma_period"]))     if pd.notna(r.get("ma_period"))  and r.get("ma_period")  else " None"
        atr_s = str(int(r["atr_period"]))    if pd.notna(r.get("atr_period")) and r.get("atr_period") else " None"
        sw_s  = f"{r.get('sweep',''):>6}"    if pd.notna(r.get("sweep"))  else "     ?"
        st_s  = f"{int(r.get('steps',0)):>5}" if pd.notna(r.get("steps")) else "    ?"
        op_s  = f"{r.get('order_size_pct',''):>6}" if pd.notna(r.get("order_size_pct")) else "     ?"
        cm_s  = f"{r.get('counter_multiplier',''):>5}" if pd.notna(r.get("counter_multiplier")) else "    ?"
        print(f"  {r['symbol']:>8}  {r['interval']:<4}  {t_s}  {v_s}  "
              f"{sw_s}  {st_s}  {op_s}  {cm_s}  {ma_s:>5}  {atr_s:>5}")

    print(f"\n{'='*105}")
    print(f"  AGGREGATE BY INTERVAL")
    print(f"{'='*105}")
    for ivl in IVL_ORDER:
        sub = valid[valid["interval"] == ivl].dropna(subset=["val_roi"])
        if sub.empty:
            continue
        print(f"  {ivl:<4}  n={len(sub):>3}  "
              f"avg_train={sub['train_roi'].mean():>+7.2f}%  "
              f"avg_val={sub['val_roi'].mean():>+7.2f}%  "
              f"best_val={sub['val_roi'].max():>+7.2f}%  "
              f"worst_val={sub['val_roi'].min():>+7.2f}%")

    print(f"\n{'='*105}")
    print(f"  TOP CONFIGS BY VALIDATION ROI")
    print(f"{'='*105}")
    top = valid.dropna(subset=["val_roi"]).sort_values("val_roi", ascending=False).head(10)
    for _, r in top.iterrows():
        ma_s  = str(int(r["ma_period"]))     if pd.notna(r.get("ma_period"))  and r.get("ma_period")  else "None"
        atr_s = f"{int(r['atr_period'])}×{r['atr_multiplier']:.1f}" \
                if pd.notna(r.get("atr_period")) and r.get("atr_period") else "None"
        print(f"  {r['symbol']:>8} {r['interval']:<4}  "
              f"val={r['val_roi']:>+7.2f}%  train={r['train_roi']:>+7.2f}%  "
              f"sweep={r.get('sweep',''):.2f}  steps={int(r.get('steps',0)):>2}  "
              f"ord={r.get('order_size_pct',''):.2f}  cm={r.get('counter_multiplier',''):.0f}  "
              f"ma={ma_s}  atr={atr_s}")

    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Resume support
    skip  = _load_existing_pairs()
    tasks = _discover_tasks(skip)

    n_combos = len(COMBOS)
    n_tasks  = len(tasks)
    n_done   = len(skip)
    n_cores  = max(1, cpu_count() - 1)

    # Estimate runtime (50ms per backtest is conservative; reality varies by candle count)
    est_low  = n_combos * n_tasks * 0.03 / n_cores / 60
    est_high = n_combos * n_tasks * 0.10 / n_cores / 60

    print(f"\n{'='*72}")
    print(f"  FULL PARAMETER SEARCH {'(QUICK MODE)' if QUICK_MODE else ''}")
    print(f"{'='*72}")
    print(f"  Combos per pair       : {n_combos:,}")
    print(f"  Pairs to run          : {n_tasks}  ({n_done} already done — resuming)")
    print(f"  Total backtests       : {n_combos * n_tasks:,}")
    print(f"  CPU cores             : {n_cores}")
    print(f"  Estimated runtime     : {est_low:.0f}–{est_high:.0f} min")
    print(f"  Train / Val split     : {TRAIN_RATIO:.0%} / {1 - TRAIN_RATIO:.0%}")
    print(f"  Output CSV            : {CSV_PATH}")
    print(f"  Output chart          : {CHART_PATH}")
    print(f"{'='*72}\n")

    if n_tasks == 0:
        print("  Nothing to do — all pairs already in CSV. Generating chart from existing results.")
        df = pd.read_csv(CSV_PATH)
        _print_table(df.to_dict("records"))
        _make_chart(df)
        return

    # Initialise CSV (write header only if new file)
    write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    csv_file = open(CSV_PATH, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    results  = []
    t0       = time.time()

    with Pool(n_cores) as pool:
        for done_i, row in enumerate(
            pool.imap_unordered(_worker, tasks, chunksize=1), start=1
        ):
            results.append(row)
            writer.writerow(row)
            csv_file.flush()

            elapsed = time.time() - t0
            rate    = done_i / elapsed if elapsed > 0 else 0
            eta_s   = (n_tasks - done_i) / rate if rate > 0 else 0

            t_s = f"{row['train_roi']:>+7.2f}%" if row.get("train_roi") is not None else "    N/A"
            v_s = f"{row['val_roi']:>+7.2f}%"   if row.get("val_roi")   is not None else "    N/A"
            print(f"  [{done_i:>3}/{n_tasks}] {row['symbol']:>8} {row['interval']:<4}"
                  f"  train={t_s}  val={v_s}"
                  f"  eta={eta_s/60:>4.1f} min")

    csv_file.close()
    elapsed_total = time.time() - t0
    print(f"\nCompleted {n_tasks} pairs in {elapsed_total/60:.1f} min  "
          f"({n_combos * n_tasks / elapsed_total:,.0f} backtests/sec effective)")

    # Load full CSV (includes any previously resumed rows)
    all_df = pd.read_csv(CSV_PATH)
    df     = _print_table(all_df.to_dict("records"))
    _make_chart(all_df)
    print(f"\nDone.  CSV: {CSV_PATH}   Chart: {CHART_PATH}")


if __name__ == "__main__":
    main()
