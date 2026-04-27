"""
AERO Grid Trader -- Production
Based on Original trader_robinhood.py strategy (Bittrex/Robinhood port).

Strategy: hourly cancel-and-replace with weighted-average buy price tracking.
  - Every cycle: cancel all open orders, process fills, place fresh grid
  - Sell mode: 'Market' when ask > avgBuy*(1+minProf%), else 'Recovery'
  - Buy base:  avgBuy in Market mode (cheaper baseline), else current ask
  - Dynamic minProf: 0.7% – 5.0% based on cash / portfolio ratio
  - Fee model: 0.26% buy-side cost / 0.26% sell-side reduction (retained from original)

Backtest result (15m data, ~59 days, $10,000 starting capital):
  Return: +33.07%  |  Buy-and-Hold: +8.84%  |  Alpha: +24.22pp  |  MaxDD: -13.7%
  Buys: 5,339  |  Sells: 4,285  |  Not halted

Params (exact match to backtested strategy):
  SWEEP=20%  ORDER_SIZE=$10  INTERVAL=3600s (1h)  CIRCUIT_BREAKER=30%

Credentials: set RH_USERNAME and RH_PASSWORD environment variables.
"""

import asyncio
import math
import pickle
import os
import logging
from datetime import datetime

import robin_stocks.robinhood as rh

# ── Configuration — mirrors trader_robinhood.py for AERO ──────────────────────
SYMBOL          = "AERO"
SWEEP           = 20.0          # % range to distribute orders across
ORDER_SIZE      = 10.0          # USD per order chunk
RESERVE         = 0.0           # USD cash to hold back
INTERVAL        = 3600          # seconds between cycles (1 hour)
CIRCUIT_BREAKER = 0.30          # halt if portfolio drops >30% from starting value
MIN_PROF_INIT   = 5.0           # initial minimum profit % (dynamic, clamped 0.7–5.0)

# Robinhood precision for AERO
PRICE_DP = 6    # decimal places for price (AERO ~$0.50–$2.00)
QTY_DP   = 2    # decimal places for quantity

# ── Credentials ───────────────────────────────────────────────────────────────
RH_USERNAME = os.environ.get("RH_USERNAME", "")
RH_PASSWORD = os.environ.get("RH_PASSWORD", "")

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            os.path.join(_LOG_DIR, f"aero_trader_{datetime.now().strftime('%Y%m%d')}.log")
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("aero_trader")

# ── State ─────────────────────────────────────────────────────────────────────
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aero_state.pkl")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            s = pickle.load(f)
        s.setdefault("orders",          [])
        s.setdefault("avgBuy",          0.0)
        s.setdefault("prevBal",         0.0)
        s.setdefault("profit",          0.0)
        s.setdefault("minProf",         MIN_PROF_INIT)
        s.setdefault("cycles",          0)
        s.setdefault("starting_value",  None)
        log.info("Loaded existing state.")
        return s
    log.info("No state file found — starting fresh.")
    return {
        "orders":         [],
        "avgBuy":         0.0,
        "prevBal":        0.0,
        "profit":         0.0,
        "minProf":        MIN_PROF_INIT,
        "cycles":         0,
        "starting_value": None,
    }


def save_state(state):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)


# ── Original helper functions (exact copies from trader_robinhood.py) ─────────

def rp(p):
    return round(p, PRICE_DP)


def rq(q):
    return round(q, QTY_DP)


def getWavg(pairs):
    """Weighted average of [(price, qty), ...] — from original getWavg()."""
    tot = sum(q for _, q in pairs)
    return sum(p * q for p, q in pairs) / tot if tot > 0 else 0.0


def getMinProf(usd_bal, portfolio_val):
    """
    Dynamic minimum profit % — from original getMinProf().
    Returns value in [0.7, 5.0] as liquid ratio moves from 1.0 -> 0.3.
    """
    x = usd_bal / portfolio_val if portfolio_val > 0 else 1.0
    a, b, c, d = 1.0, 0.3, 0.7, 5.0
    return max(c, min(d, (x - a) / (b - a) * (d - c) + c))


def _interval(balance):
    chunks = balance / ORDER_SIZE
    return SWEEP / chunks if chunks >= 1 else SWEEP


def _stable(n, spread_pct):
    """Price-offset table [spread, 2*spread, ...] — from original getStable()."""
    return [i * spread_pct / 100.0 for i in range(1, int(n) + 1)]


def make_buy_orders(usd_balance, ask):
    """
    Generate buy order list [(price, qty), ...] — from original buy().
    Orders at descending prices below ask, each worth ORDER_SIZE USD.
    """
    n = int(usd_balance / ORDER_SIZE)
    if n < 1:
        return []
    table  = _stable(n, _interval(usd_balance))[:n]
    prices = [(1.0 - e) * ask for e in table]
    return [(rp(p), rq(ORDER_SIZE / p)) for p in prices if p > 0]


def make_sell_orders(coin_bal, sell_base):
    """
    Generate sell order list [(price, qty), ...] — from original sell().
    Qty split equally across ORDER_SIZE-sized chunks above sell_base.
    """
    if coin_bal <= 0 or sell_base <= 0:
        return []
    notional = coin_bal * sell_base
    n = int(notional / ORDER_SIZE)
    if n < 1:
        return []
    table = _stable(n, _interval(notional))
    rates = [sell_base]
    for e in range(len(table) - 1):
        rates.append((1.0 + table[e]) * sell_base)
    rates = rates[:n]
    qty_ea = rq(coin_bal / n)
    return [(rp(r), qty_ea) for r in rates if r > 0]


# ── Broker helpers ────────────────────────────────────────────────────────────

async def _relogin():
    log.warning("Re-authenticating session...")
    try:
        await asyncio.to_thread(rh.login, RH_USERNAME, RH_PASSWORD)
        log.info("Re-login successful.")
    except Exception as e:
        log.error(f"Re-login failed: {e}")


async def get_quote():
    """Return (bid, ask, mark) for AERO."""
    try:
        q    = await asyncio.to_thread(rh.crypto.get_crypto_quote, SYMBOL)
        bid  = float(q.get("bid_inclusive_of_sell_spread", q.get("bid_price",  0)) or 0)
        ask  = float(q.get("ask_inclusive_of_buy_spread",  q.get("ask_price",  0)) or 0)
        mark = float(q.get("mark_price", ask) or ask)
        return bid, ask, mark
    except Exception as e:
        log.error(f"get_quote: {e}")
        await _relogin()
        return None, None, None


async def get_buying_power():
    try:
        profile = await asyncio.to_thread(rh.profiles.load_account_profile)
        return float(profile.get("buying_power", 0) or 0)
    except Exception as e:
        log.warning(f"get_buying_power: {e}")
        return 0.0


async def get_coin_balance():
    """Available AERO coin quantity."""
    try:
        positions = await asyncio.to_thread(rh.crypto.get_crypto_positions)
        for pos in positions:
            if pos.get("currency", {}).get("code", "").upper() == SYMBOL:
                return float(pos.get("quantity_available", 0) or 0)
        return 0.0
    except Exception as e:
        log.warning(f"get_coin_balance: {e}")
        return 0.0


async def get_portfolio_value(mark, open_orders):
    """Approximate portfolio value: cash + reserved buy funds + coin holdings."""
    try:
        profile  = await asyncio.to_thread(rh.profiles.load_account_profile)
        cash     = float(profile.get("buying_power", 0) or 0)
        reserved = sum(o["price"] * o["qty"] for o in open_orders if o["side"] == "buy")
        coin_qty = await get_coin_balance()
        return cash + reserved + coin_qty * mark
    except Exception as e:
        log.warning(f"get_portfolio_value: {e}")
        return None


async def get_open_order_ids():
    try:
        orders = await asyncio.to_thread(rh.orders.get_all_open_crypto_orders)
        return {o["id"] for o in (orders or [])}
    except Exception as e:
        log.warning(f"get_open_order_ids: {e}")
        return set()


async def get_order_info(order_id):
    try:
        return await asyncio.to_thread(rh.orders.get_crypto_order_info, order_id)
    except Exception as e:
        log.warning(f"get_order_info({order_id[:8]}...): {e}")
        return None


async def cancel_order(order_id):
    try:
        await asyncio.to_thread(rh.orders.cancel_crypto_order, order_id)
    except Exception as e:
        log.warning(f"cancel_order({order_id[:8]}...): {e}")


async def place_buy(price, qty):
    log.info(f"  BUY  {qty:.{QTY_DP}f} {SYMBOL} @ ${price:.{PRICE_DP}f}")
    try:
        resp = await asyncio.to_thread(rh.orders.order_buy_crypto_limit, SYMBOL, qty, price)
        if not (resp and isinstance(resp, dict) and resp.get("id")):
            log.warning(f"  Buy rejected: {resp}")
            return None
        return resp
    except Exception as e:
        log.warning(f"  Buy failed: {e}")
        return None


async def place_sell(price, qty):
    log.info(f"  SELL {qty:.{QTY_DP}f} {SYMBOL} @ ${price:.{PRICE_DP}f}")
    try:
        resp = await asyncio.to_thread(rh.orders.order_sell_crypto_limit, SYMBOL, qty, price)
        if not (resp and isinstance(resp, dict) and resp.get("id")):
            log.warning(f"  Sell rejected: {resp}")
            return None
        return resp
    except Exception as e:
        log.warning(f"  Sell failed: {e}")
        return None


# ── Core cycle logic — mirrors original trader_robinhood.py ───────────────────

async def clear_orders(state):
    """Cancel all tracked open orders (mirrors original clearOrders)."""
    if not state["orders"]:
        return

    open_ids = await get_open_order_ids()
    remaining = []
    cancel_tasks = []
    for o in state["orders"]:
        if o["id"] in open_ids:
            log.info(f"  Cancelling {o['side']:4s} @ ${o['price']:.{PRICE_DP}f}  [{o['id'][:8]}...]")
            cancel_tasks.append(cancel_order(o["id"]))
        else:
            remaining.append(o)   # already gone — process in process_orders

    if cancel_tasks:
        await asyncio.gather(*cancel_tasks)

    # Keep only the orders not yet cancelled (will be processed next)
    state["orders"] = remaining


async def process_orders(state):
    """
    Fetch fill status of all disappeared orders; update avgBuy and profit.
    Mirrors original processOrders().
    """
    open_ids = await get_open_order_ids()

    # orders still in state["orders"] at this point are ones that disappeared
    # (weren't open when clear_orders ran — could be fills or self-cancellations)
    buys_to_wavg = [(state["avgBuy"], state["prevBal"])]   # seed with prior avg
    remaining    = []

    for o in state["orders"]:
        info = await get_order_info(o["id"])
        if info is None:
            remaining.append(o)
            continue

        order_state  = info.get("state", "unknown")
        qty_filled   = float(info.get("cumulative_quantity", 0) or 0)
        price_filled = float(info.get("average_price", 0) or info.get("price", 0) or 0)

        if order_state in ("filled", "partially_filled") and qty_filled > 0 and price_filled > 0:
            if o["side"] == "buy":
                buys_to_wavg.append((price_filled, qty_filled))
                log.info(f"  Filled BUY  {qty_filled:.{QTY_DP}f} {SYMBOL}"
                         f" @ ${price_filled:.{PRICE_DP}f}")
            elif o["side"] == "sell":
                # fee model retained from original (0.26% each side)
                cost = state["avgBuy"] * qty_filled * 1.0026
                sale = price_filled   * qty_filled * 0.9974
                pnl  = sale - cost
                state["profit"] += pnl
                log.info(f"  Filled SELL {qty_filled:.{QTY_DP}f} {SYMBOL}"
                         f" @ ${price_filled:.{PRICE_DP}f}  P&L ${pnl:.4f}")
        elif order_state not in ("filled", "cancelled", "failed", "rejected"):
            remaining.append(o)   # still pending somehow

    # Update weighted-average buy price
    state["avgBuy"]  = getWavg(buys_to_wavg)
    state["orders"]  = remaining   # clear processed orders


async def run_market(state, ask, buying_power):
    """
    Determine sell/buy orders for this cycle.
    Mirrors original runMarket() — returns (sell_orders, buy_orders).
    """
    coin_bal = await get_coin_balance()
    avg      = state["avgBuy"]
    min_prof = state["minProf"]

    sell_orders = []
    sell_type   = "Buying"
    sell_base_b = 0.0

    log.info(f"{SYMBOL}: holdings={coin_bal:.{QTY_DP}f}  ask=${ask:.{PRICE_DP}f}")

    if coin_bal > 0:
        sell_base_b = avg

        if avg > 0 and ask > avg * (1 + min_prof / 100.0):
            sell_base = ask
            sell_type = "Market"
        else:
            sell_base = avg * (1 + min_prof / 100.0) if avg > 0 else ask
            sell_type = "Recovery"

        log.info(f"  Holdings val: ${coin_bal * ask:,.4f}")
        log.info(f"  Avg buy:      ${avg:.{PRICE_DP}f}")
        log.info(f"  Realised P&L: ${state['profit']:.4f}")
        sell_orders = make_sell_orders(coin_bal, sell_base)
    else:
        state["avgBuy"] = 0.0

    # Buy base: use avgBuy in Market mode (cheaper baseline), else use ask
    if sell_type == "Market" and sell_base_b > 0 and sell_base_b < ask:
        buy_base = sell_base_b
    else:
        buy_base = ask

    per_mkt = max(0.0, buying_power - RESERVE)
    buy_orders = make_buy_orders(per_mkt, buy_base)

    state["prevBal"] = coin_bal
    log.info(f"  Mode:         {sell_type}  |  minProf={min_prof:.2f}%")
    log.info(f"  Placing {len(sell_orders)} sell + {len(buy_orders)} buy orders")

    return sell_type, sell_orders, buy_orders


async def run_cycle(state):
    cycle = state["cycles"] + 1
    log.info(f"\n== Cycle {cycle}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
             f"{'=' * 48}")

    # ── Fetch price ────────────────────────────────────────────────────────────
    bid, ask, mark = await get_quote()
    if ask is None or ask <= 0:
        log.error("Could not fetch valid price — skipping cycle.")
        return

    # ── Circuit breaker ────────────────────────────────────────────────────────
    portfolio_val = await get_portfolio_value(mark, state["orders"])
    if portfolio_val is not None:
        if state["starting_value"] is None:
            state["starting_value"] = portfolio_val
            log.info(f"Starting portfolio value: ${portfolio_val:,.4f}")
        else:
            pct_down = (1.0 - portfolio_val / state["starting_value"]) * 100
            log.info(f"Portfolio: ${portfolio_val:,.4f}"
                     f"  ({pct_down:>+.2f}% from ${state['starting_value']:,.4f})")
            if pct_down > CIRCUIT_BREAKER * 100:
                log.critical(
                    f"CIRCUIT BREAKER: portfolio is {pct_down:.1f}% below start "
                    f"(limit {CIRCUIT_BREAKER*100:.0f}%) -- HALTING."
                )
                save_state(state)
                return

    # ── Cancel all open orders from last cycle ─────────────────────────────────
    log.info("Cancelling open orders...")
    await clear_orders(state)

    # ── Process fills from last cycle ──────────────────────────────────────────
    log.info("Processing fills...")
    await process_orders(state)

    # ── Update dynamic minProf ─────────────────────────────────────────────────
    buying_power   = await get_buying_power()
    usd_avail      = buying_power
    if portfolio_val is not None:
        state["minProf"] = max(0.5, getMinProf(usd_avail, portfolio_val))

    log.info(f"Cash (buying power): ${usd_avail:,.4f}")
    log.info(f"Min profit threshold: {state['minProf']:.2f}%")
    log.info(f"Cum. realised P&L:   ${state['profit']:.4f}")

    # ── Generate new orders ────────────────────────────────────────────────────
    sell_type, sell_orders, buy_orders = await run_market(state, ask, usd_avail)

    # ── Place sells first (free up coin positions before re-buying) ────────────
    new_orders = []
    for price, qty in sell_orders:
        if qty <= 0 or price <= 0:
            continue
        resp = await place_sell(price, qty)
        if resp and isinstance(resp, dict) and resp.get("id"):
            new_orders.append({"id": resp["id"], "side": "sell", "price": price, "qty": qty})
        await asyncio.sleep(0.3)

    # ── Place buys ─────────────────────────────────────────────────────────────
    for price, qty in buy_orders:
        if qty <= 0 or price <= 0:
            continue
        resp = await place_buy(price, qty)
        if resp and isinstance(resp, dict) and resp.get("id"):
            new_orders.append({"id": resp["id"], "side": "buy", "price": price, "qty": qty})
        await asyncio.sleep(0.3)

    state["orders"] = new_orders
    state["cycles"] = cycle
    save_state(state)

    log.info(f"Cycle complete. {len(new_orders)} orders tracked "
             f"({sum(1 for o in new_orders if o['side']=='sell')} sells, "
             f"{sum(1 for o in new_orders if o['side']=='buy')} buys).")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    if not RH_USERNAME or not RH_PASSWORD:
        raise EnvironmentError(
            "Set RH_USERNAME and RH_PASSWORD environment variables before running."
        )

    log.info("Logging in to Robinhood...")
    await asyncio.to_thread(rh.login, RH_USERNAME, RH_PASSWORD)

    log.info(
        f"{SYMBOL} Grid Trader started | "
        f"sweep={SWEEP}% order_size=${ORDER_SIZE:.0f} "
        f"interval={INTERVAL}s circuit_breaker={CIRCUIT_BREAKER:.0%} "
        f"min_prof_init={MIN_PROF_INIT:.1f}%"
    )

    state = load_state()

    while True:
        loop_start = asyncio.get_event_loop().time()
        try:
            await run_cycle(state)
        except Exception as e:
            log.exception(f"Unhandled error in run_cycle: {e}")
            save_state(state)

        elapsed  = asyncio.get_event_loop().time() - loop_start
        wait     = max(0, INTERVAL - elapsed)
        log.info(f"Next cycle in {wait / 60:.1f} min\n")
        await asyncio.sleep(wait)


if __name__ == "__main__":
    asyncio.run(main())
