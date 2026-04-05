import robin_stocks.robinhood as rh
import asyncio

def login(username, password):
    rh.login(username, password)

async def get_price(symbol):
    return float(await asyncio.to_thread(
        lambda: rh.get_latest_price(symbol)[0]
    ))

async def buy_limit(symbol, qty, price):
    return await asyncio.to_thread(
        rh.orders.order_buy_limit, symbol, qty, price
    )

async def sell_limit(symbol, qty, price):
    return await asyncio.to_thread(
        rh.orders.order_sell_limit, symbol, qty, price
    )
