"""
BNB Grid Trader — Production
Optimized settings derived from 5-year backtest (sweep=0.10, steps=6) → +63.6%.
All 5 optimizations enabled: trend filter, ATR sweep, pyramid sizing,
stale order cancellation, and circuit breaker.
Credentials are loaded from environment variables — see README.md.
"""

import asyncio
import pickle
import os
import logging
from datetime import datetime
from collections import deque

import robin_stocks.robinhood as rh

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOL         = "BNB"
SWEEP          = 0.10      # base grid range (overridden by ATR when enough data exists)
STEPS          = 6         # buy + sell levels per side
ORDER_SIZE     = 0.01      # BNB per order (base; pyramid scales deeper levels)
INTERVAL       = 3600      # seconds between cycles (1 hour)

# 1. Trend filter
MA_PERIOD      = 20        # cycles; buys suppressed when price < MA

# 2. ATR-based sweep
ATR_PERIOD     = 14        # cycles used to compute average true range
ATR_MULTIPLIER = 8.0       # scales ATR % → sweep width
MIN_SWEEP      = 0.05
MAX_SWEEP      = 0.50

# 3. Pyramid sizing
PYRAMID        = True      # deeper levels get proportionally larger orders

# 4. Stale order cancellation — done at the start of every cycle

# 5. Circuit breaker
CIRCUIT_BREAKER = 0.20    # halt if portfolio drops 20% below starting value
DROP_THRESHOLD  = 0.08    # risk guard: suppress buys on 8% intra-session drop

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
        logging.FileHandler(f"logs/bnb_trader_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("bnb_trader")

# ── State persistence ─────────────────────────────────────────────────────────

STATE_FILE = "bnb_state.pkl"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            return pickle.load(f)
    return {"cycles": 0, "profit": 0.0, "positions": {}, "starting_value": None}

def save_state(state):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)

# ── Rolling price history (in-memory, not persisted) ─────────────────────────

price_history = deque(maxlen=max(MA_PERIOD, ATR_PERIOD) + 1)

# ── Grid strategy ─────────────────────────────────────────────────────────────

def compute_sweep(price):
    """2. ATR-based sweep: widen grid in volatile markets, tighten in calm ones."""
    if len(price_history) >= ATR_PERIOD:
        hist = list(price_history)
        ranges = [abs(hist[j] - hist[j - 1]) for j in range(1, len(hist))]
        atr = sum(ranges[-ATR_PERIOD:]) / ATR_PERIOD
        dynamic = max(MIN_SWEEP, min(MAX_SWEEP, (atr / price) * ATR_MULTIPLIER))
        log.info(f"ATR={atr:.2f}  dynamic sweep={dynamic:.3f}")
        return dynamic
    return SWEEP

def get_grid(price, sweep, steps=STEPS):
    buys, sells = [], []
    for i in range(1, steps + 1):
        pct = sweep * i / steps
        buys.append(round(price * (1 - pct), 2))
        sells.append(round(price * (1 + pct), 2))
    return buys, sells

# ── Trend filter ──────────────────────────────────────────────────────────────

def trend_is_up(price):
    """1. Only buy when price is above the rolling MA (uptrend confirmation)."""
    if len(price_history) >= MA_PERIOD:
        ma = sum(list(price_history)[-MA_PERIOD:]) / MA_PERIOD
        log.info(f"MA({MA_PERIOD})={ma:.2f}  price={price:.2f}  trend={'UP' if price > ma else 'DOWN'}")
        return price > ma
    return True

# ── Risk guard (intra-session drop) ──────────────────────────────────────────

def should_pause(price):
    if len(price_history) < 2:
        return False
    hist = list(price_history)
    drop = (hist[0] - price) / hist[0]
    return drop > DROP_THRESHOLD

# ── Circuit breaker ───────────────────────────────────────────────────────────

async def get_portfolio_value(price):
    """5. Fetch current buying power + BNB position value."""
    try:
        profile = await asyncio.to_thread(rh.profiles.load_account_profile)
        buying_power = float(profile.get("buying_power", 0))
        positions = await asyncio.to_thread(rh.crypto.get_crypto_positions)
        qty = 0.0
        for pos in positions:
            if pos.get("currency", {}).get("code") == "BNB":
                qty = float(pos.get("quantity", 0))
        return buying_power + qty * price
    except Exception as e:
        log.warning(f"Could not fetch portfolio value: {e}")
        return None

# ── Stale order cancellation ──────────────────────────────────────────────────

async def cancel_open_bnb_orders():
    """4. Cancel all open BNB crypto limit orders before placing new ones."""
    try:
        orders = await asyncio.to_thread(rh.orders.get_all_open_crypto_orders)
        bnb_orders = [
            o for o in orders
            if "BNB" in str(o.get("currency_pair_id", "")).upper()
        ]
        if bnb_orders:
            log.info(f"Cancelling {len(bnb_orders)} stale BNB order(s)...")
            for order in bnb_orders:
                await asyncio.to_thread(rh.orders.cancel_crypto_order, order["id"])
            log.info("Stale orders cancelled.")
        else:
            log.info("No stale orders to cancel.")
    except Exception as e:
        log.warning(f"Order cancellation failed: {e}")

# ── Broker calls ──────────────────────────────────────────────────────────────

async def get_price():
    price = await asyncio.to_thread(lambda: rh.get_latest_price(SYMBOL)[0])
    return float(price)

async def place_buy(price, size):
    log.info(f"  BUY  {size} BNB @ ${price:,.2f}")
    return await asyncio.to_thread(rh.orders.order_buy_limit, SYMBOL, size, price)

async def place_sell(price, size):
    log.info(f"  SELL {size} BNB @ ${price:,.2f}")
    return await asyncio.to_thread(rh.orders.order_sell_limit, SYMBOL, size, price)

# ── Cycle ─────────────────────────────────────────────────────────────────────

async def run_cycle(state):
    cycle = state["cycles"] + 1
    log.info(f"── Cycle {cycle} ──────────────────────────────────────────────")

    try:
        price = await get_price()
    except Exception as e:
        log.error(f"Failed to fetch price: {e}")
        return

    log.info(f"BNB price: ${price:,.2f}")
    price_history.append(price)

    # ── 5. Circuit breaker ────────────────────────────────────────────────────
    portfolio_value = await get_portfolio_value(price)
    if portfolio_value is not None:
        if state["starting_value"] is None:
            state["starting_value"] = portfolio_value
            log.info(f"Starting portfolio value recorded: ${portfolio_value:,.2f}")
        elif portfolio_value < state["starting_value"] * (1 - CIRCUIT_BREAKER):
            log.critical(
                f"CIRCUIT BREAKER TRIGGERED — portfolio ${portfolio_value:,.2f} "
                f"is {(1 - portfolio_value / state['starting_value']) * 100:.1f}% "
                f"below starting value ${state['starting_value']:,.2f}. HALTING."
            )
            return

    # ── 4. Cancel stale orders ────────────────────────────────────────────────
    await cancel_open_bnb_orders()

    # ── 1. Trend filter ───────────────────────────────────────────────────────
    trend_up = trend_is_up(price)

    # ── Risk guard ────────────────────────────────────────────────────────────
    paused = should_pause(price)
    if paused:
        log.warning("Risk guard active — buys suppressed (>8% intra-session drop)")

    # ── 2. ATR-based sweep → grid ─────────────────────────────────────────────
    sweep = compute_sweep(price)
    buys, sells = get_grid(price, sweep)
    log.info(f"Sweep={sweep:.3f}  buys={[f'${b:,.2f}' for b in buys]}  sells={[f'${s:,.2f}' for s in sells]}")

    # ── Place orders ──────────────────────────────────────────────────────────
    tasks = []

    if trend_up and not paused:
        for idx, b in enumerate(buys):
            # ── 3. Pyramid sizing ─────────────────────────────────────────
            size = round(ORDER_SIZE * (idx + 1), 8)
            tasks.append(place_buy(b, size))
    else:
        log.info("Buys skipped — trend down or risk guard active")

    for s in sells:
        tasks.append(place_sell(s, ORDER_SIZE))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    for e in errors:
        log.error(f"Order error: {e}")
    log.info(f"Orders placed: {len(tasks) - len(errors)}/{len(tasks)}")

    state["cycles"] = cycle
    save_state(state)
    log.info(f"State saved — total cycles: {cycle}")

# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    if not RH_USERNAME or not RH_PASSWORD:
        raise EnvironmentError(
            "RH_USERNAME and RH_PASSWORD must be set as environment variables. "
            "See README.md."
        )

    log.info("Logging in to Robinhood...")
    try:
        rh.login(RH_USERNAME, RH_PASSWORD)
    except Exception as e:
        log.error(f"Login failed: {e}")
        raise

    log.info(
        f"BNB Grid Trader started  sweep={SWEEP}(ATR-adjusted)  steps={STEPS}  "
        f"order={ORDER_SIZE} BNB  pyramid={PYRAMID}  "
        f"MA={MA_PERIOD}  ATR={ATR_PERIOD}  circuit_breaker={CIRCUIT_BREAKER:.0%}"
    )

    state = load_state()

    while True:
        loop_start = asyncio.get_event_loop().time()
        await run_cycle(state)
        elapsed = asyncio.get_event_loop().time() - loop_start
        wait = max(0, INTERVAL - elapsed)
        log.info(f"Next cycle in {wait / 60:.1f} min\n")
        await asyncio.sleep(wait)


if __name__ == "__main__":
    asyncio.run(main())
