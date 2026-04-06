import asyncio
from state import load_state, save_state
from strategy import get_sd_levels, compute_atr
from broker import login, get_price, get_portfolio_value, cancel_order, buy_limit, sell_limit
from risk import portfolio_ok

# Per-market config:
#   window        — rolling candles for mean/σ (168h = 1 week)
#   steps         — number of buy/sell levels on each side
#   std_multiplier— σ spacing between levels
MARKETS = {
    "BNB":  {"window": 336, "steps": 2, "std_multiplier": 0.5},  # 2y hourly baseline: +60.4%
    "BCH":  {"window": 720, "steps": 4, "std_multiplier": 1.5},  # 2y hourly baseline: +47.4%
    "PAXG": {"window": 336, "steps": 2, "std_multiplier": 1.0},  # 2y hourly baseline: +32.3%
    "LTC":  {"window": 336, "steps": 6, "std_multiplier": 0.5},  # 2y hourly baseline: +27.8%
    "AAVE": {"window": 168, "steps": 6, "std_multiplier": 2.0},  # 2y hourly baseline: +24.2%
}

INTERVAL        = 3600   # seconds between cycles
ORDER_SIZE_USD  = 10     # USD per grid level (base); pyramid scales this per depth
RESET_INTERVAL  = 24     # cycles between full grid resets (1 reset per day)
PRICE_CACHE_LEN = 750    # candles kept per market (~1 month at 1h)

price_cache = {m: [] for m in MARKETS}


async def _cancel_market_orders(symbol, state):
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

    # Update price cache every cycle for accurate mean/σ
    cache = price_cache[symbol]
    cache.append(price)
    if len(cache) > PRICE_CACHE_LEN:
        cache.pop(0)

    # Circuit breaker
    if portfolio_val is not None and not portfolio_ok(portfolio_val, state["initial_portfolio"]):
        print(f"[{symbol}] Circuit breaker triggered — halting.")
        return

    if not is_reset_cycle:
        return

    await _cancel_market_orders(symbol, state)

    cfg = MARKETS[symbol]

    # SD-based levels — dynamically sized to recent volatility
    buys, sells = get_sd_levels(
        cache,
        steps          = cfg["steps"],
        std_multiplier = cfg["std_multiplier"],
        window         = cfg["window"],
    )

    new_order_ids = []

    async def place_buy(idx, level):
        qty  = round((ORDER_SIZE_USD * (idx + 1)) / level, 6)  # pyramid sizing
        resp = await buy_limit(symbol, qty, level)
        if resp and isinstance(resp, dict) and "id" in resp:
            new_order_ids.append(resp["id"])

    async def place_sell(level):
        qty  = round(ORDER_SIZE_USD / level, 6)
        resp = await sell_limit(symbol, qty, level)
        if resp and isinstance(resp, dict) and "id" in resp:
            new_order_ids.append(resp["id"])

    tasks = [place_buy(idx, b) for idx, b in enumerate(buys)]
    tasks += [place_sell(s) for s in sells]

    await asyncio.gather(*tasks)
    state["open_orders"][symbol] = new_order_ids


async def run_cycle(state):
    cycle    = state["cycles"]
    is_reset = (cycle % RESET_INTERVAL == 0)

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
        start   = asyncio.get_event_loop().time()
        await run_cycle(state)
        elapsed = asyncio.get_event_loop().time() - start
        await asyncio.sleep(max(0, INTERVAL - elapsed))


if __name__ == "__main__":
    asyncio.run(main())
