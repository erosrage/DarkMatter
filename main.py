import asyncio
from state import load_state, save_state
from strategy import get_grid
from broker import login, get_price, buy_limit, sell_limit
from risk import should_pause

# Markets and their backtested optimal grid parameters
# sweep: price range on each side, steps: number of grid levels
MARKETS = {
    "ETH": {"sweep": 0.40, "steps": 2},   # 5y backtest: +37%
    "BNB": {"sweep": 0.10, "steps": 6},   # 5y backtest: +64%
    "SOL": {"sweep": 0.30, "steps": 16},  # 5y backtest: +165%
}

INTERVAL = 3600
ORDER_SIZE = 0.01

price_cache = {m: [] for m in MARKETS}

async def run_market(symbol, state):
    price = await get_price(symbol)

    price_cache[symbol].append(price)
    if len(price_cache[symbol]) > 10:
        price_cache[symbol].pop(0)

    pause = should_pause(price_cache[symbol])

    cfg = MARKETS[symbol]
    buys, sells = get_grid(price, sweep=cfg["sweep"], steps=cfg["steps"])

    tasks = []

    if not pause:
        for b in buys:
            tasks.append(buy_limit(symbol, ORDER_SIZE, b))

    for s in sells:
        tasks.append(sell_limit(symbol, ORDER_SIZE, s))

    await asyncio.gather(*tasks)

async def run_cycle():
    state = load_state()

    await asyncio.gather(*[
        run_market(m, state) for m in MARKETS.keys()
    ])

    state["cycles"] += 1
    save_state(state)

async def main():
    login("username", "password")

    while True:
        start = asyncio.get_event_loop().time()

        await run_cycle()

        elapsed = asyncio.get_event_loop().time() - start
        await asyncio.sleep(max(0, INTERVAL - elapsed))

if __name__ == "__main__":
    asyncio.run(main())
