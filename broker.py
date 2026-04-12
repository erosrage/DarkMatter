import os
import robin_stocks.robinhood as rh
import asyncio

def login():
    username = os.environ.get("RH_USERNAME")
    password = os.environ.get("RH_PASSWORD")
    if not username or not password:
        raise RuntimeError("RH_USERNAME and RH_PASSWORD environment variables must be set.")
    rh.login(username, password)

async def get_price(symbol):
    return float(await asyncio.to_thread(
        lambda: rh.get_latest_price(symbol)[0]
    ))

async def get_portfolio_value():
    profile = await asyncio.to_thread(rh.profiles.load_portfolio_profile)
    return float(profile["total_equity"])

async def get_open_orders():
    return await asyncio.to_thread(rh.orders.get_all_open_crypto_orders)

async def cancel_order(order_id):
    return await asyncio.to_thread(rh.orders.cancel_crypto_order, order_id)

async def buy_limit(symbol, qty, price):
    return await asyncio.to_thread(
        rh.orders.order_buy_limit, symbol, qty, price
    )

async def sell_limit(symbol, qty, price):
    return await asyncio.to_thread(
        rh.orders.order_sell_limit, symbol, qty, price
    )
