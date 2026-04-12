"""
ETC Grid Trader -- Production
Optimised params from 3-round 2m backtest (research/backtest_2m_optimised.py).

Strategy: reactive counter-orders, no periodic reset.
  - On startup: place buy grid limited by available cash (closest levels first)
  - On buy fill: immediately place a counter-sell at +7.5% (step_pct x cm)
  - On sell fill: immediately place a counter-buy at -7.5% if cash available
  - No periodic cancel-and-replace; orders persist until filled

Optimised params (backtest result: +25.00% over 34 days on $100k):
  sweep=0.15  steps=16  order_pct=0.50  counter_mult=8.0

Credentials: set RH_USERNAME and RH_PASSWORD environment variables.
"""

import asyncio
import math
import pickle
import os
import logging
from datetime import datetime

import robin_stocks.robinhood as rh

# ── Configuration ─────────────────────────────────────────────────────────────
SYMBOL         = "ETC"
SWEEP          = 0.15       # grid width each side (15%)
STEPS          = 16         # buy levels (and sell levels) in the initial grid
ORDER_SIZE_PCT = 0.50       # fraction of available buying power per order level
COUNTER_MULT   = 8.0        # counter-order gap multiplier (step_pct x COUNTER_MULT)
INTERVAL       = 120        # seconds between cycles (2 minutes — matches backtest cadence)
CIRCUIT_BREAKER = 0.20      # halt if portfolio drops >20% from starting value

# step_pct = SWEEP / STEPS = 0.15 / 16 = 0.9375% per step
# counter distance = step_pct * COUNTER_MULT = 7.5% above/below fill price

# ── Credentials ───────────────────────────────────────────────────────────────
RH_USERNAME = os.environ.get("RH_USERNAME", "")
RH_PASSWORD = os.environ.get("RH_PASSWORD", "")

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"logs/etc_trader_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("etc_trader")

# ── State ─────────────────────────────────────────────────────────────────────
STATE_FILE = "etc_state.pkl"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            s = pickle.load(f)
        s.setdefault("open_orders", [])
        s.setdefault("starting_value", None)
        s.setdefault("cycles", 0)
        return s
    return {"cycles": 0, "starting_value": None, "open_orders": []}

def save_state(state):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)

# ── Grid ──────────────────────────────────────────────────────────────────────
STEP_PCT     = SWEEP / STEPS   # 0.009375
QTY_DECIMALS = 2

def price_dp(price):
    return max(2, math.ceil(-math.log10(price * STEP_PCT)) + 1)

def round_price(p, ref):
    return round(p, price_dp(ref))

def round_qty(q):
    return round(q, QTY_DECIMALS)

def make_buy_levels(price):
    """Return buy price levels from closest to furthest."""
    dp = price_dp(price)
    return [round(price * (1 - SWEEP * i / STEPS), dp) for i in range(1, STEPS + 1)]

# ── Broker helpers ────────────────────────────────────────────────────────────

async def get_price():
    quote = await asyncio.to_thread(rh.crypto.get_crypto_quote, SYMBOL)
    return float(quote["mark_price"])

async def get_buying_power():
    try:
        profile = await asyncio.to_thread(rh.profiles.load_account_profile)
        return float(profile.get("buying_power", 0))
    except Exception as e:
        log.warning(f"Could not fetch buying power: {e}")
        return 0.0

async def get_portfolio_value(price, open_orders):
    try:
        profile   = await asyncio.to_thread(rh.profiles.load_account_profile)
        cash      = float(profile.get("buying_power", 0))
        reserved  = sum(o["price"] * o["qty"] for o in open_orders if o["side"] == "buy")
        positions = await asyncio.to_thread(rh.crypto.get_crypto_positions)
        qty = 0.0
        for pos in positions:
            if pos.get("currency", {}).get("code", "").upper() == SYMBOL:
                qty = float(pos.get("quantity", 0))
        return cash + reserved + qty * price
    except Exception as e:
        log.warning(f"Could not fetch portfolio value: {e}")
        return None

async def get_open_order_ids():
    try:
        orders = await asyncio.to_thread(rh.orders.get_all_open_crypto_orders)
        return {o["id"] for o in orders}
    except Exception as e:
        log.warning(f"Could not fetch open orders: {e}")
        return set()

async def get_order_state(order_id):
    try:
        info = await asyncio.to_thread(rh.orders.get_crypto_order_info, order_id)
        return info.get("state", "unknown")
    except Exception as e:
        log.warning(f"Could not check order state {order_id}: {e}")
        return "unknown"

async def place_buy(level, qty):
    log.info(f"  BUY  {qty:.6f} {SYMBOL} @ ${level:.4f}")
    try:
        resp = await asyncio.to_thread(rh.orders.order_buy_crypto_limit, SYMBOL, qty, level)
        return resp
    except Exception as e:
        log.warning(f"  Buy order failed: {e}")
        return None

async def place_sell(level, qty):
    log.info(f"  SELL {qty:.6f} {SYMBOL} @ ${level:.4f}")
    try:
        resp = await asyncio.to_thread(rh.orders.order_sell_crypto_limit, SYMBOL, qty, level)
        return resp
    except Exception as e:
        log.warning(f"  Sell order failed: {e}")
        return None

# ── Cycle ─────────────────────────────────────────────────────────────────────

async def run_cycle(state):
    cycle = state["cycles"] + 1
    log.info(f"-- Cycle {cycle} ({SYMBOL}) ----------------------------------------")

    try:
        price = await get_price()
    except Exception as e:
        log.error(f"Failed to fetch price: {e}")
        return

    log.info(f"{SYMBOL} price: ${price:.4f}")

    # ── Circuit breaker ────────────────────────────────────────────────────────
    portfolio_val = await get_portfolio_value(price, state["open_orders"])
    if portfolio_val is not None:
        if state["starting_value"] is None:
            state["starting_value"] = portfolio_val
            log.info(f"Starting portfolio value: ${portfolio_val:,.2f}")
        elif portfolio_val < state["starting_value"] * (1 - CIRCUIT_BREAKER):
            pct_down = (1 - portfolio_val / state["starting_value"]) * 100
            log.critical(
                f"CIRCUIT BREAKER: portfolio ${portfolio_val:,.2f} is {pct_down:.1f}% "
                f"below start ${state['starting_value']:,.2f} -- HALTING."
            )
            return

    # ── Detect fills ───────────────────────────────────────────────────────────
    open_ids = await get_open_order_ids()
    tracked  = state["open_orders"]

    if not tracked:
        # ── Initial grid placement ─────────────────────────────────────────────
        log.info("No tracked orders -- placing initial buy grid.")
        buying_power = await get_buying_power()
        levels       = make_buy_levels(price)
        order_usd    = buying_power * ORDER_SIZE_PCT
        new_orders   = []
        remaining    = buying_power

        if order_usd > 0:
            max_levels = int(remaining / order_usd)
            levels = levels[:max_levels]

        for level in levels:
            qty  = round_qty(order_usd / level)
            resp = await place_buy(level, qty)
            if resp and isinstance(resp, dict) and resp.get("id"):
                new_orders.append({
                    "id": resp["id"], "side": "buy",
                    "price": level, "qty": qty, "step_pct": STEP_PCT,
                })
            remaining -= order_usd
            await asyncio.sleep(0.5)

        state["open_orders"] = new_orders
        log.info(f"Initial grid: {len(new_orders)} buy orders placed.")

    else:
        # ── Reactive re-entry on fills ─────────────────────────────────────────
        disappeared = [o for o in tracked if o["id"] not in open_ids]
        still_open  = [o for o in tracked if o["id"] in open_ids]

        filled = []
        for o in disappeared:
            order_state = await get_order_state(o["id"])
            if order_state in ("filled", "partially_filled"):
                filled.append(o)
            else:
                log.info(f"  Order {o['id'][:8]}... was {order_state} -- skipping counter-order")

        if filled:
            log.info(f"Detected {len(filled)} filled order(s) -- placing counter-orders.")
            buying_power  = await get_buying_power()
            counter_tasks = []
            counter_meta  = []

            for order in filled:
                sp = order["step_pct"]
                if order["side"] == "buy":
                    counter_price = round_price(order["price"] * (1 + sp * COUNTER_MULT), order["price"])
                    counter_tasks.append(place_sell(counter_price, order["qty"]))
                    counter_meta.append({
                        "side": "sell", "price": counter_price,
                        "qty": order["qty"], "step_pct": sp,
                    })
                    log.info(f"  fill BUY @ {order['price']:.4f} -> counter SELL @ {counter_price:.4f}")
                else:
                    counter_price = round_price(order["price"] * (1 - sp * COUNTER_MULT), order["price"])
                    order_usd     = buying_power * ORDER_SIZE_PCT
                    if buying_power >= order_usd and order_usd > 0:
                        qty = round_qty(order_usd / counter_price)
                        counter_tasks.append(place_buy(counter_price, qty))
                        counter_meta.append({
                            "side": "buy", "price": counter_price,
                            "qty": qty, "step_pct": sp,
                        })
                        buying_power -= order_usd
                        log.info(f"  fill SELL @ {order['price']:.4f} -> counter BUY @ {counter_price:.4f}")
                    else:
                        log.info(f"  fill SELL @ {order['price']:.4f} -> counter BUY skipped (insufficient cash)")

            results = await asyncio.gather(*counter_tasks, return_exceptions=True)
            new_orders = list(still_open)
            for resp, meta in zip(results, counter_meta):
                if resp and isinstance(resp, dict) and resp.get("id"):
                    new_orders.append({**meta, "id": resp["id"]})
            state["open_orders"] = new_orders
        else:
            log.info(f"No fills detected. {len(still_open)} orders still open.")

    state["cycles"] = cycle
    save_state(state)
    log.info(f"State saved -- {len(state['open_orders'])} tracked orders.")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    if not RH_USERNAME or not RH_PASSWORD:
        raise EnvironmentError("Set RH_USERNAME and RH_PASSWORD environment variables.")

    log.info("Logging in to Robinhood...")
    rh.login(RH_USERNAME, RH_PASSWORD)

    log.info(
        f"{SYMBOL} Grid Trader started | "
        f"sweep={SWEEP} steps={STEPS} order_pct={ORDER_SIZE_PCT} "
        f"cm={COUNTER_MULT} circuit_breaker={CIRCUIT_BREAKER:.0%}"
    )

    state = load_state()

    while True:
        loop_start = asyncio.get_event_loop().time()
        await run_cycle(state)
        elapsed = asyncio.get_event_loop().time() - loop_start
        wait    = max(0, INTERVAL - elapsed)
        log.info(f"Next cycle in {wait / 60:.1f} min\n")
        await asyncio.sleep(wait)


if __name__ == "__main__":
    asyncio.run(main())
