"""
Robinhood Grid Trader — Refactored from Original (og.md)

Original: Bittrex Python 2 grid trader (og.md)
Refactored: Robinhood API via robin_stocks, Python 3

Original settings preserved:
  markets:    LTC, ETH, XRP  (formerly 'BTC-LTC', 'BTC-ETH', 'BTC-XRP' on Bittrex)
  reserve:    $0.00           (USD cash to hold back; formerly 0.0 BTC)
  sweep:      20.0%           (range over which orders are distributed; unchanged)
  orderSize:  $10.00          (per-chunk order size; formerly 0.001 BTC ≈ $30 at the time)
  interval:   3600 seconds    (run once per hour; unchanged)

Original logic preserved:
  - Weighted-average buy price tracking (getWavg)
  - Dynamic minimum-profit threshold (getMinProf)
  - Stable price-offset table for order levels (getStable)
  - Sell-side: 'Market' mode when ask > avgBuy*(1+minProf%), else 'Recovery' mode
  - Buy-side: descending buy rates below current ask
  - Fee model: 0.26% buy cost overhead / 0.26% sell revenue reduction (Bittrex model retained)
  - State persisted to rh_state.pkl (replaces state.pkl)

Key changes from original:
  - Python 2 → Python 3
  - bittrex library → robin_stocks.robinhood
  - BTC-quoted pairs → USD-quoted (Robinhood crypto is always vs USD)
  - BTC reserve/balance → USD reserve/balance
  - print "..." → logging
  - Orders tracked with symbol & side in state (Robinhood lacks o['Exchange'] field)

Credentials: set RH_USERNAME and RH_PASSWORD environment variables.
"""

import robin_stocks.robinhood as rh
from datetime import datetime
import time
import sys
import pickle
import os
import logging

# ── Credentials ────────────────────────────────────────────────────────────────
RH_USERNAME = os.environ.get("RH_USERNAME", "")
RH_PASSWORD = os.environ.get("RH_PASSWORD", "")

# ── Original settings (preserved from og.md) ──────────────────────────────────
MARKETS    = ['LTC', 'ETH', 'XRP']   # formerly 'BTC-LTC', 'BTC-ETH', 'BTC-XRP'
RESERVE    = 0.0                      # USD to hold back (formerly BTC reserve = 0.0)
SWEEP      = 20.0                     # % range to distribute orders (unchanged)
ORDER_SIZE = 10.0                     # USD per order chunk (formerly 0.001 BTC)
INTERVAL   = 3600.0                   # seconds between runs (unchanged)

# ── Paths ──────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(_DIR, "rh_state.pkl")

# ── Logging ────────────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(_DIR), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            os.path.join(_LOG_DIR, f"rh_trader_{datetime.now().strftime('%Y%m%d')}.log")
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("rh_trader")


# ── State initialisation (mirrors original logic) ─────────────────────────────

def _load_state():
    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            s = pickle.load(f)
        count = sum(1 for m in MARKETS if m in s)
        if count == len(MARKETS):
            log.info("Loaded state.")
            return s
        else:
            log.error("Failed to load state: market keys missing.")
            sys.exit(1)
    else:
        s = {
            "orders": [],
            "cycles": 0,
            "minProf": 5.0,
        }
        for m in MARKETS:
            s[m] = {"avgBuy": 0.0, "prevBal": 0.0, "profit": 0.0}
        log.info("First run, generated blank state.")
        _save_state(s)
        return s

def _save_state(s):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(s, f, pickle.HIGHEST_PROTOCOL)

state = _load_state()

if "minProf" not in state:
    state["minProf"] = 5.0


# ── Original helper functions (preserved) ─────────────────────────────────────

def getWavg(orders):
    """Weighted average of (price, volume) pairs — from original getWavg()."""
    tot = sum(o[1] for o in orders)
    if not tot > 0.0:
        return 0.0
    return sum(o[1] * o[0] for o in orders) / tot


def getProfit(market, price, vol):
    """
    Realised profit on a sell, preserving original fee model:
      cost = avgBuy * vol * 1.0026   (0.26% Bittrex buy fee)
      sale = price  * vol * 0.9974   (0.26% Bittrex sell fee)
    Robinhood charges no explicit commission but has a spread; the 0.26% model
    is retained from the original for strategy consistency.
    """
    cost = state[market]["avgBuy"] * vol * 1.0026
    sale = price * vol * 0.9974
    return sale - cost


def getMinProf(usd_bal, portfolio_val):
    """
    Dynamic minimum profit % — from original getMinProf(btcBal, pVal).
    x = liquid_usd / portfolio_value  (formerly btcBal / pVal)
    Returns a value in [0.7, 5.0]% as liquid ratio moves from 1.0 → 0.3.
    """
    x = usd_bal / portfolio_val if portfolio_val > 0 else 1.0
    a, b, c, d = 1.0, 0.3, 0.7, 5.0
    return (x - a) / (b - a) * (d - c) + c


def getInterval(balance, sweep):
    """Step fraction between levels = sweep / number_of_chunks."""
    chunks = balance / ORDER_SIZE
    if chunks < 1:
        return sweep
    return sweep / chunks


def getStable(chunks, spread):
    """Build price-offset table [spread, 2*spread, ...] — from original getStable()."""
    table = []
    for ii in range(1, int(chunks) + 1):
        table.append(ii * spread / 100.0)
    return table


def sat(f):
    """8-decimal string format — from original sat()."""
    return "{0:.8f}".format(f)


def getSellRates(base, table):
    """[base, base*(1+t[0]), base*(1+t[1]), ...] — from original."""
    rates = [base]
    for e in range(len(table) - 1):
        rates.append((1 + table[e]) * base)
    return rates


def getBuyRates(base, table):
    """[base*(1-t[0]), base*(1-t[1]), ...] — from original."""
    return [(1 - e) * base for e in table]


# ── Order-placement helpers ────────────────────────────────────────────────────

def buy(market, table, balance):
    """
    Generate buy orders for a market.
    From original buy(market, table, balance) — balance is USD here.
    Returns list of (price, qty) tuples.
    """
    buyChunks = int(balance / ORDER_SIZE)
    if buyChunks > len(table):
        buyChunks = len(table)
    if buyChunks < 1:
        return []
    amounts = [ORDER_SIZE / table[e] for e in range(buyChunks)]  # coin qty
    return [(table[e], amounts[e]) for e in range(buyChunks)]


def sell(market, table, coin_balance):
    """
    Generate sell orders for existing coin holdings.
    From original sell(market, table, balance) — balance is coin qty here.
    Returns list of (price, qty) tuples.
    """
    if not table or table[0] <= 0:
        return []
    sellChunks = int(coin_balance * table[0] / ORDER_SIZE)
    if sellChunks > len(table):
        sellChunks = len(table)
    if sellChunks < 1:
        return []
    amt = coin_balance / sellChunks  # equal coin qty per chunk
    return [(table[e], amt) for e in range(sellChunks)]


# ── Robinhood API wrappers ─────────────────────────────────────────────────────

def get_quote(symbol):
    """Return (bid, ask) for a crypto symbol."""
    try:
        q = rh.crypto.get_crypto_quote(symbol)
        bid = float(q.get("bid_inclusive_of_sell_spread", q.get("bid_price", 0)) or 0)
        ask = float(q.get("ask_inclusive_of_buy_spread", q.get("ask_price", 0)) or 0)
        return bid, ask
    except Exception as e:
        log.error(f"get_quote({symbol}): {e}")
        return None, None


def get_coin_balance(symbol):
    """Available coin quantity on Robinhood."""
    try:
        positions = rh.crypto.get_crypto_positions()
        for pos in positions:
            code = pos.get("currency", {}).get("code", "").upper()
            if code == symbol.upper():
                return float(pos.get("quantity_available", 0) or 0)
        return 0.0
    except Exception as e:
        log.warning(f"get_coin_balance({symbol}): {e}")
        return 0.0


def get_usd_buying_power():
    """Available USD buying power."""
    try:
        profile = rh.profiles.load_account_profile()
        return float(profile.get("buying_power", 0) or 0)
    except Exception as e:
        log.warning(f"get_usd_buying_power: {e}")
        return 0.0


def get_usd_portfolio_cash():
    """Total USD cash (including amounts held for open orders)."""
    try:
        profile = rh.profiles.load_account_profile()
        return float(profile.get("portfolio_cash", profile.get("buying_power", 0)) or 0)
    except Exception as e:
        log.warning(f"get_usd_portfolio_cash: {e}")
        return 0.0


def cancel_order(order_id):
    """Cancel an open order."""
    try:
        rh.orders.cancel_crypto_order(order_id)
    except Exception as e:
        log.warning(f"cancel_order({order_id[:8]}...): {e}")


# ── Core logic — mirrors original clearOrders / processOrders / runMarket ──────

def clearOrders(market):
    """
    Cancel all tracked open orders for `market`.
    From original clearOrders(market, orders) — now uses state['orders'] directly.
    """
    still_open = rh.orders.get_all_open_crypto_orders() or []
    open_ids = {o["id"] for o in still_open if isinstance(o, dict)}

    remaining = []
    for o in state["orders"]:
        if o["symbol"] == market and o["id"] in open_ids:
            log.info(f"  Cancelling {market} order {o['id'][:8]}...")
            cancel_order(o["id"])
        else:
            remaining.append(o)
    state["orders"] = remaining


def processOrders():
    """
    Fetch status of all tracked orders; update avgBuy from filled buys and
    accumulate profit from filled sells. Mirrors original processOrders().
    """
    buys = {m: [(state[m]["avgBuy"], state[m]["prevBal"])] for m in MARKETS}

    remaining = []
    for o in state["orders"]:
        order_id = o.get("id")
        if not order_id:
            continue
        try:
            info = rh.orders.get_crypto_order_info(order_id)
        except Exception as e:
            log.warning(f"Could not fetch order {order_id[:8]}...: {e}")
            remaining.append(o)
            continue

        order_state   = info.get("state", "unknown")
        qty_filled    = float(info.get("cumulative_quantity", 0) or 0)
        price_filled  = float(info.get("average_price", 0) or info.get("price", 0) or 0)
        symbol        = o.get("symbol")
        side          = o.get("side", info.get("side", ""))

        if order_state in ("filled", "partially_filled") and qty_filled > 0 and price_filled > 0:
            if side == "buy" and symbol in MARKETS:
                buys[symbol].append((price_filled, qty_filled))
                log.info(f"  Filled BUY  {symbol} {qty_filled:.6f} @ ${price_filled:.6f}")
            elif side == "sell" and symbol in MARKETS:
                pnl = getProfit(symbol, price_filled, qty_filled)
                state[symbol]["profit"] += pnl
                log.info(f"  Filled SELL {symbol} {qty_filled:.6f} @ ${price_filled:.6f}  P&L ${pnl:.4f}")
        elif order_state not in ("filled", "cancelled", "failed", "rejected"):
            remaining.append(o)  # still open / pending

    state["orders"] = remaining

    for m in MARKETS:
        state[m]["avgBuy"] = getWavg(buys[m])


def runMarket(market, usd_balance):
    """
    Run grid logic for a single market.
    Mirrors original runMarket(market, btcBal, coinBal) but in USD terms.

    Returns (coin_value_usd, buy_orders, sell_orders).
    buy_orders / sell_orders: list of (price, qty) tuples.
    """
    bid, ask = get_quote(market)
    if bid is None or ask is None:
        return (0.0, [], [])

    coinBal = get_coin_balance(market)

    sells     = []
    sellType  = "Buying"
    sellBaseB = 0.0

    log.info(f"{market:5s}: holdings={coinBal:.6f}  ask=${ask:.6f}  bid=${bid:.6f}")

    if coinBal > 0:
        sellBaseB = state[market]["avgBuy"]

        if sellBaseB > 0 and ask > sellBaseB * (1 + state["minProf"] / 100.0):
            sellBase = ask
            sellType = "Market"
        else:
            sellBase = sellBaseB * (1 + state["minProf"] / 100.0) if sellBaseB > 0 else ask
            sellType = "Recovery"

        log.info(f"  Value:        ${coinBal * bid:,.2f}")
        log.info(f"  Avg Buy:      ${sellBaseB:.6f}")
        log.info(f"  Profit:       ${state[market]['profit']:.4f}")
        if state["cycles"] > 0:
            log.info(f"  Profit/Cycle: ${state[market]['profit'] / state['cycles']:.4f}")

        chunks  = int(coinBal * sellBase / ORDER_SIZE)
        sTable  = getStable(chunks, getInterval(coinBal * sellBase, SWEEP))
        sells   = sell(market, getSellRates(sellBase, sTable), coinBal)
    else:
        state[market]["avgBuy"] = 0.0

    # Buy base price — mirrors original logic
    if sellType == "Market" and sellBaseB > 0 and sellBaseB < ask:
        buyBase = sellBaseB
    else:
        buyBase = ask

    log.info(f"  State:        {sellType}")
    log.info(f"  {'='*28}")

    chunks   = int(usd_balance / ORDER_SIZE)
    sTable   = getStable(chunks, getInterval(usd_balance, SWEEP))
    buys     = buy(market, getBuyRates(buyBase, sTable), usd_balance)

    state[market]["prevBal"] = coinBal

    return (coinBal * bid, buys, sells)


# ── Main loop — mirrors original main() ───────────────────────────────────────

def main():
    try:
        log.info("Clearing open orders...")
        for m in MARKETS:
            clearOrders(m)

        usd_avail = get_usd_buying_power()
        usd_total = get_usd_portfolio_cash()

        processOrders()

        log.info("")
        log.info(str(datetime.now()))
        log.info("=" * 28)

        # Allocate buying power equally across markets (minus reserve)
        usd_for_trading = max(0.0, usd_avail - RESERVE)
        per_market_usd  = usd_for_trading / len(MARKETS) if MARKETS else 0.0

        portfolio_value = usd_total  # start with cash, add coin values below
        all_buys  = []
        all_sells = []

        for m in MARKETS:
            coin_val, mbuys, msells = runMarket(m, per_market_usd)
            portfolio_value += coin_val
            all_buys.extend((m, b) for b in mbuys)
            all_sells.extend((m, s) for s in msells)

        log.info(f"USD Cash:      ${usd_total:>12,.2f}")
        log.info(f"Portfolio:     ${portfolio_value:>12,.2f}")

        total_profit = sum(state[m]["profit"] for m in MARKETS)
        log.info(f"Profit:        ${total_profit:>12,.4f}")
        if state["cycles"] > 0:
            log.info(f"Profit/Cycle:  ${total_profit / state['cycles']:>12,.4f}")

        # Dynamic minimum-profit threshold — mirrors original getMinProf(btcBal, pVal)
        state["minProf"] = max(0.5, getMinProf(usd_avail, portfolio_value))
        log.info(f"Min Prof:      {state['minProf']:>12.2f}%")
        log.info(f"{'='*28}\n")
        log.info(f"Placing {len(all_sells) + len(all_buys)} orders...")

        # Sells first — free up positions before re-buying
        for market, (price, qty) in all_sells:
            qty   = round(qty, 6)
            price = round(price, 6)
            if qty <= 0 or price <= 0:
                continue
            try:
                resp = rh.orders.order_sell_crypto_limit(market, qty, price)
                if resp and isinstance(resp, dict) and resp.get("id"):
                    state["orders"].append({
                        "id": resp["id"], "symbol": market,
                        "side": "sell", "price": price, "qty": qty,
                    })
                    log.info(f"  SELL {qty:.6f} {market} @ ${price:.6f}")
                else:
                    log.warning(f"  Sell rejected {market}: {resp}")
            except Exception as e:
                log.warning(f"  Sell failed {market}: {e}")

        for market, (price, qty) in all_buys:
            qty   = round(qty, 6)
            price = round(price, 6)
            if qty <= 0 or price <= 0:
                continue
            try:
                resp = rh.orders.order_buy_crypto_limit(market, qty, price)
                if resp and isinstance(resp, dict) and resp.get("id"):
                    state["orders"].append({
                        "id": resp["id"], "symbol": market,
                        "side": "buy", "price": price, "qty": qty,
                    })
                    log.info(f"  BUY  {qty:.6f} {market} @ ${price:.6f}")
                else:
                    log.warning(f"  Buy rejected {market}: {resp}")
            except Exception as e:
                log.warning(f"  Buy failed {market}: {e}")

        state["cycles"] += 1
        _save_state(state)
        log.info("Done.\n")

    except Exception as e:
        log.exception(f"Error in main(): {e}")
        main()  # retry on exception (mirrors original behaviour)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not RH_USERNAME or not RH_PASSWORD:
        raise EnvironmentError(
            "Set RH_USERNAME and RH_PASSWORD environment variables before running."
        )

    log.info("Logging in to Robinhood...")
    rh.login(RH_USERNAME, RH_PASSWORD)
    log.info(
        f"RH Grid Trader started | markets={MARKETS} | "
        f"sweep={SWEEP}% orderSize=${ORDER_SIZE} reserve=${RESERVE} interval={INTERVAL}s"
    )

    start_time = time.time()
    while True:
        sTime = time.time()
        main()
        elapsed = time.time() - sTime
        log.info(f"Time: {elapsed:.2f} sec.")
        sleep_time = INTERVAL - ((time.time() - start_time) % INTERVAL)
        time.sleep(sleep_time)
