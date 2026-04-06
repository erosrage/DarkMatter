import asyncio
from state import load_state, save_state
from strategy import get_grid, dynamic_sweep
from broker import login, get_price, get_portfolio_value, get_open_orders, cancel_order, buy_limit, sell_limit
from risk import trend_is_up, portfolio_ok

# Markets and their per-market grid defaults (used before ATR history accumulates)
# sweep: fallback grid width on each side, steps: number of grid levels
MARKETS = {
    #           sweep   steps   ma_period (None = trend filter off)
    "ETH": {"sweep": 0.40, "steps": 2, "ma_period": 168},  # 2y hourly: +80.6%
    "BNB": {"sweep": 0.25, "steps": 2, "ma_period": None}, # 2y hourly: +8.8% (MA filter blocks profitable buy)
    "SOL": {"sweep": 0.30, "steps": 2, "ma_period": 168},  # 2y hourly: +3.2%
}

INTERVAL        = 3600   # seconds between cycles
ORDER_SIZE_USD  = 10     # USD per grid level (base); pyramid doubles this per depth
RESET_INTERVAL  = 24     # cycles between full grid resets (1 reset per day at 1hr cycles)
PRICE_CACHE_LEN = 750    # candles kept per market (~1 month at 1h; enough for any MA period)

price_cache = {m: [] for m in MARKETS}


async def _cancel_market_orders(symbol, state):
    """Cancel all tracked open orders for a market and clear them from state."""
    order_ids = state["open_orders"].get(symbol, [])
    if not order_ids:
        return
    await asyncio.gather(
        *[cancel_order(oid) for oid in order_ids],
        return_exceptions=True,
    )
    state["open_orders"][symbol] = []


async def run_market(symbol, state, is_reset_cycle, portfolio_val):
    price = await get_price(symbol)

    # ── Update price cache (every cycle, even between resets) ─────────────────
    cache = price_cache[symbol]
    cache.append(price)
    if len(cache) > PRICE_CACHE_LEN:
        cache.pop(0)

    # ── 5. Circuit breaker ─────────────────────────────────────────────────────
    if portfolio_val is not None and not portfolio_ok(portfolio_val, state["initial_portfolio"]):
        print(f"[{symbol}] Circuit breaker triggered — halting all orders.")
        return

    # ── 2. Grid reset cadence — let orders fill between resets ────────────────
    if not is_reset_cycle:
        return

    # ── Cancel stale orders before placing fresh grid ─────────────────────────
    await _cancel_market_orders(symbol, state)

    # ── 2. ATR-based sweep (falls back to per-market default) ─────────────────
    cfg = MARKETS[symbol]
    sweep = dynamic_sweep(price, cache, default=cfg["sweep"])

    buys, sells = get_grid(price, sweep=sweep, steps=cfg["steps"])

    # ── 1. Trend filter — suppress buys in downtrends (per-market, None = off) ──
    ma_period   = cfg.get("ma_period")
    buy_allowed = trend_is_up(cache, ma_period) if ma_period else True

    new_order_ids = []

    async def place_buy(idx, level):
        # ── 3. Pyramid sizing: deeper levels get larger orders ─────────────────
        # ── 6. Dollar-based sizing: qty = USD_value / price ───────────────────
        qty = round((ORDER_SIZE_USD * (idx + 1)) / level, 6)
        resp = await buy_limit(symbol, qty, level)
        if resp and isinstance(resp, dict) and "id" in resp:
            new_order_ids.append(resp["id"])

    async def place_sell(level):
        qty = round(ORDER_SIZE_USD / level, 6)
        resp = await sell_limit(symbol, qty, level)
        if resp and isinstance(resp, dict) and "id" in resp:
            new_order_ids.append(resp["id"])

    tasks = []
    if buy_allowed:
        tasks += [place_buy(idx, b) for idx, b in enumerate(buys)]
    tasks += [place_sell(s) for s in sells]

    await asyncio.gather(*tasks)

    # ── 4. Persist open order IDs ──────────────────────────────────────────────
    state["open_orders"][symbol] = new_order_ids


async def run_cycle(state):
    cycle = state["cycles"]
    is_reset = (cycle % RESET_INTERVAL == 0)

    # Refresh portfolio value for circuit breaker
    portfolio_val = None
    try:
        portfolio_val = await get_portfolio_value()
        if state["initial_portfolio"] is None:
            state["initial_portfolio"] = portfolio_val
    except Exception:
        pass

    await asyncio.gather(*[
        run_market(m, state, is_reset, portfolio_val) for m in MARKETS
    ])

    state["cycles"] += 1
    save_state(state)


async def main():
    login()
    state = load_state()

    while True:
        start = asyncio.get_event_loop().time()
        await run_cycle(state)
        elapsed = asyncio.get_event_loop().time() - start
        await asyncio.sleep(max(0, INTERVAL - elapsed))


if __name__ == "__main__":
    asyncio.run(main())
