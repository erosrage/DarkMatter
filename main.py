import asyncio
from state import load_state, save_state
from strategy import get_grid, dynamic_sweep
from broker import login, get_price, get_portfolio_value, get_open_orders, cancel_order, buy_limit, sell_limit
from risk import trend_is_up, portfolio_ok

# Markets and their per-market grid defaults (used before ATR history accumulates)
# sweep: fallback grid width on each side, steps: number of grid levels
MARKETS = {
    #            sweep  steps  ma_period  counter_mult  order_size_pct
    "ETH": {"sweep": 0.10, "steps": 4,  "ma_period": 336, "counter_mult": 3.0, "order_size_pct": 0.15},
    "BNB": {"sweep": 0.06, "steps": 12, "ma_period": None, "counter_mult": 3.0, "order_size_pct": 0.15},
    "SOL": {"sweep": 0.20, "steps": 4,  "ma_period": None, "counter_mult": 3.0, "order_size_pct": 0.15},
}

INTERVAL = 3600  # seconds between cycles
RESET_INTERVAL  = 24     # reset grids every 24 cycles (~daily at 1h intervals)
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


async def run_market(symbol, state, is_reset_cycle, open_order_ids, portfolio_val):
    price = await get_price(symbol)

    # ── Update price cache (every cycle, even between resets) ─────────────────
    cache = price_cache[symbol]
    cache.append(price)
    if len(cache) > PRICE_CACHE_LEN:
        cache.pop(0)

    # ── Circuit breaker ────────────────────────────────────────────────────────
    if portfolio_val is not None and not portfolio_ok(portfolio_val, state["initial_portfolio"]):
        print(f"[{symbol}] Circuit breaker triggered — halting all orders.")
        return

    cfg = MARKETS[symbol]
    ma_period   = cfg.get("ma_period")
    buy_allowed = trend_is_up(cache, ma_period) if ma_period else True

    # ── Full grid reset every RESET_INTERVAL cycles ────────────────────────────
    if is_reset_cycle:
        await _cancel_market_orders(symbol, state)

        sweep    = dynamic_sweep(price, cache, default=cfg["sweep"])
        step_pct = sweep / cfg["steps"]
        buys, sells = get_grid(price, sweep=sweep, steps=cfg["steps"])

        new_orders = []

        order_size_usd = portfolio_val * cfg["order_size_pct"] if portfolio_val else 0

        async def place_buy(idx, level):
            qty  = round(order_size_usd / level, 6)
            resp = await buy_limit(symbol, qty, level)
            if resp and isinstance(resp, dict) and "id" in resp:
                new_orders.append({"id": resp["id"], "side": "buy",
                                   "price": level, "qty": qty, "step_pct": step_pct})

        async def place_sell(level):
            qty  = round(order_size_usd / level, 6)
            resp = await sell_limit(symbol, qty, level)
            if resp and isinstance(resp, dict) and "id" in resp:
                new_orders.append({"id": resp["id"], "side": "sell",
                                   "price": level, "qty": qty, "step_pct": step_pct})

        tasks = []
        if buy_allowed:
            tasks += [place_buy(idx, b) for idx, b in enumerate(buys)]
        tasks += [place_sell(s) for s in sells]

        await asyncio.gather(*tasks)
        state["open_orders"][symbol] = new_orders
        return

    # ── Reactive re-entry: detect fills, place counter-orders ─────────────────
    tracked = state["open_orders"].get(symbol, [])
    if not tracked:
        return

    filled      = [o for o in tracked if o["id"] not in open_order_ids]
    still_open  = [o for o in tracked if o["id"] in open_order_ids]

    if not filled:
        return

    counter_tasks = []
    counter_meta  = []

    cm = cfg.get("counter_mult", 1.0)
    for order in filled:
        sp = order["step_pct"]
        if order["side"] == "buy":
            counter_price = round(order["price"] * (1 + sp * cm), 8)
            qty           = order["qty"]   # sell same qty we bought
            counter_tasks.append(sell_limit(symbol, qty, counter_price))
            counter_meta.append({"side": "sell", "price": counter_price,
                                  "qty": qty, "step_pct": sp})
        else:  # sell filled → counter buy (only in uptrend)
            if buy_allowed:
                counter_price = round(order["price"] * (1 - sp * cm), 8)
                order_usd     = portfolio_val * cfg["order_size_pct"] if portfolio_val else 0
                qty           = round(order_usd / counter_price, 6)
                counter_tasks.append(buy_limit(symbol, qty, counter_price))
                counter_meta.append({"side": "buy", "price": counter_price,
                                      "qty": qty, "step_pct": sp})

    results = await asyncio.gather(*counter_tasks, return_exceptions=True)

    new_orders = list(still_open)
    for resp, meta in zip(results, counter_meta):
        if resp and isinstance(resp, dict) and "id" in resp:
            new_orders.append({**meta, "id": resp["id"]})

    state["open_orders"][symbol] = new_orders


async def run_cycle(state):
    cycle = state["cycles"]
    is_reset = (RESET_INTERVAL is not None and cycle % RESET_INTERVAL == 0) or cycle == 0

    # Refresh portfolio value for circuit breaker
    portfolio_val = None
    try:
        portfolio_val = await get_portfolio_value()
        if state["initial_portfolio"] is None:
            state["initial_portfolio"] = portfolio_val
    except Exception:
        pass

    # Fetch open orders once per cycle for fill detection
    open_order_ids = set()
    try:
        open_orders = await get_open_orders()
        open_order_ids = {o["id"] for o in open_orders}
    except Exception:
        pass

    await asyncio.gather(*[
        run_market(m, state, is_reset, open_order_ids, portfolio_val) for m in MARKETS
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
