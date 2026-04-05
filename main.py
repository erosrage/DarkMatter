import asyncio
from state import load_state, save_state
from strategy import get_grid
from broker import login, get_price, buy_limit, sell_limit
from risk import should_pause

MARKETS = ["ETH", "BTC", "DOGE"]
INTERVAL = 3600
ORDER_SIZE = 0.01

price_cache = {m: [] for m in MARKETS}

async def run_market(symbol, state):
    price = await get_price(symbol)

    price_cache[symbol].append(price)
    if len(price_cache[symbol]) > 10:
        price_cache[symbol].pop(0)

    pause = should_pause(price_cache[symbol])

    buys, sells = get_grid(price)

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
        run_market(m, state) for m in MARKETS
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
