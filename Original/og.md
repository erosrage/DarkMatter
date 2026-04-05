import bittrex
from datetime import datetime
import time
import sys
import pickle
import os.path

apiKey = {
  "key": "key goes here",
  "secret": "secret goes here"
}

markets = ['BTC-LTC', 'BTC-ETH', 'BTC-XRP']
reserve = 0.0 #How much btc to reserve
sweep = 20.0 #percent range to distribute orders
orderSize = 0.001 #order size in btc
interval = 3600.0 #run once hour

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

state = False
if os.path.isfile('state.pkl'):
    state = load_obj('state')
    count = 0
    for m in markets:
        if m in state:
            count += 1

    if count == len(markets):
        print "Loaded state."
    else:
        print "Failed to load state."
        exit()
else:
    state = {}
    state['orders'] = []
    state['cycles'] = 0
    state['minProf'] = 5.0
    for m in markets:
        state[m] = {}
        state[m]['avgBuy'] = 0.0
        state[m]['prevBal'] = 0.0
        state[m]['profit'] = 0.0

    print "First run, generated blank state."
    save_obj(state, 'state')

if not 'minProf' in state:
    state['minProf'] = 5.0


def getWavg(orders):
    tot = 0.0
    for o in orders:
        tot += o[1]

    if not tot > 0.0:
        return 0.0

    avg = 0.0
    for o in orders:
        avg += o[1]*o[0]

    return avg/tot

def getProfit(market, price, vol):
    cost = state[market]['avgBuy']*vol*1.0026
    sale = price*vol*0.9974
    return(sale-cost)

def processOrders():
    buys = {}
    for m in markets:
        buys[m] = [(state[m]['avgBuy'], state[m]['prevBal'])]

    for ii in range(len(state['orders'])):
        o = state['orders'].pop()
        res = b.get_order(o['uuid'])
        while not res['success']:
            print res['message']
            res = b.get_order(o['uuid'])

        o = res['result']
        vol = o['Quantity']-o['QuantityRemaining']
        if vol > 0.0:
            if o['Type'] == 'LIMIT_BUY':
                buys[o['Exchange']].append((o['PricePerUnit'], vol))
            if o['Type'] == 'LIMIT_SELL':
                state[o['Exchange']]['profit'] += getProfit(o['Exchange'], o['PricePerUnit'], vol)

    for m in markets:
        state[m]['avgBuy'] = getWavg(buys[m])

def getMinProf(btcBal, pVal):
    x = btcBal/pVal
    a = 1.0
    b = 0.3
    c = 0.7
    d = 5.0
    return (x-a)/(b-a)*(d-c)+c

def getInterval(balance, sweep):
    chunks = balance/orderSize
    return sweep/chunks

def getStable(chunks, spread):
    table = []
    for ii in range(1,chunks+1):
        table.append(ii*spread/100.0)
    return table

b = bittrex.Bittrex(apiKey['key'], apiKey['secret'])

def sat(f):
    return "{0:.8f}".format(f)

def getSellRates(base, table):
    rates = [base]
    for e in range(len(table)-1):
        rates.append((1+table[e])*base)
    return rates

def getBuyRates(base, table):
    rates = []
    for e in table:
        rates.append((1-e)*base)
    return rates

def clearOrders(market, orders):
    for o in orders:
        if o['Exchange'] == market:
            res = b.cancel(o['OrderUuid'])
            while not res['success'] and not res['message'] == 'ORDER_NOT_OPEN':
                print res['message']
                res = b.cancel(o['OrderUuid'])

def buy(market, table, balance):
    amounts = []

    buyChunks = int(balance/orderSize)
    if buyChunks > len(table):
        buyChunks = len(table)

    if buyChunks < 1:
        return []

    for e in range(buyChunks):
        amounts.append(orderSize/table[e])

    orders = []
    for e in range(buyChunks):
        orders.append((table[e], amounts[e]))

    return orders

def sell(market, table, balance):
    amounts = []
    sellChunks = int(balance/(orderSize/table[0]))
    if sellChunks > len(table):
        sellChunks = len(table)

    if sellChunks < 1:
        return []

    amt = balance/sellChunks

    for e in range(sellChunks):
        amounts.append(amt)

    orders = []
    for e in range(sellChunks):
        orders.append((table[e], amounts[e]))

    return orders

def runMarket(market, btcBal, coinBal):
    ob = b.get_orderbook(market, 'both', depth=1)
    while not ob['success']:
        print ob
        ob = b.get_orderbook(market, 'both', depth=1)

    if coinBal['Available'] == None:
        coinBal = 0.0
    else:
        coinBal = coinBal['Available']

    sellBaseA = ob['result']['sell'][0]['Rate']-0.00000001

    sells = []

    sellType = "Buying"
    sellBaseB = 0.0

    print market.split('-')[1]+':         ', sat(coinBal)
    if coinBal > 0:
        sellBaseB = state[market]['avgBuy']
        if sellBaseA > sellBaseB*(1+(state['minProf']/100.0)):
            sellBase = sellBaseA
            sellType = "Market"
        else:
            sellBase = sellBaseB*(1+(state['minProf']/100.0))
            sellType = "Recovery"

        print "Value:       ", sat(coinBal*ob['result']['buy'][0]['Rate'])
        print "Avg Buy:     ", sat(sellBaseB)
        print "Profit:      ", sat(state[market]['profit'])
        if state['cycles'] > 0.0:
            print "Profit/Cycle:", sat(state[market]['profit']/state['cycles'])

        chunks = int(coinBal*sellBase/orderSize)
        sTable = getStable(chunks, getInterval(coinBal*sellBase, sweep))
        sellRates = getSellRates(sellBase, sTable)
        sells = sell(market, sellRates, coinBal)
    else:
        state[market]['avgBuy'] = 0.0

    if sellType == "Market":
        if sellBaseB < ob['result']['sell'][0]['Rate'] and sellBaseB > 0.0:
            buyBase = sellBaseB
        else:
            buyBase = ob['result']['sell'][0]['Rate']
    else:
        buyBase = ob['result']['sell'][0]['Rate']

    print "State:       ", sellType
    print "============================"

    chunks = int(btcBal/orderSize)
    sTable = getStable(chunks, getInterval(btcBal, sweep))
    buyRates = getBuyRates(buyBase, sTable)
    buys = buy(market, buyRates, btcBal)
    state[market]['prevBal'] = coinBal

    return((coinBal*ob['result']['buy'][0]['Rate'], buys, sells))

def getBalance(balances, coin):
    for bal in balances:
        if bal['Currency'] == coin:
            return bal

def main():
    try:
        print "Clearing orders..."
        orders = b.get_open_orders('')
        while not orders['success']:
            print orders
            orders = b.get_open_orders('')

        if 'result' in orders:
            orders = orders['result']

            for m in markets:
                clearOrders(m, orders)

        reserved = 1
        balances = []
        while reserved > 0.0:
            balances = b.get_balances()
            while not balances['success']:
                print balances
                balances = b.get_balances()

            balances = balances['result']
            btcBal = getBalance(balances, 'BTC')
            reserved = btcBal['Balance']-btcBal['Available']

        processOrders()

        print "Done.\n"
        print str(datetime.now())
        print "============================"

        btcTot = btcBal['Balance']
        btcBal = btcBal['Available']
        btcBal = (btcBal-reserve)/(len(markets))

        value = btcTot
        buys = []
        sells = []
        for m in markets:
            r = runMarket(m, btcBal, getBalance(balances, m.split('-')[1]))
            value += r[0]
            for buy in r[1]:
                buys.append((m, buy))
            for sell in r[2]:
                sells.append((m, sell))

        print "BTC:         ", sat(btcTot)
        print "Portfolio:   ", sat(value)
        profit = 0
        for m in markets:
            profit += state[m]['profit']
        print "Profit:      ", sat(profit)
        if state['cycles'] > 0:
            print "Profit/Cycle:", sat(profit/state['cycles'])

        state['minProf'] = getMinProf(btcTot/value)
              "Portfolio:   "
        print "Min Prof:    ", state['minProf']
        print "============================\n"
        print "Placing", len(sells)+len(buys), "orders..."

        for s in sells:
            market = s[0]
            o = s[1]
            res = b.sell_limit(market, o[1], o[0])
            while not res['success']:
                print res['message']
                print market, o[1], o[0]
                res = b.sell_limit(market, o[1], o[0])

            state['orders'].append(res['result'])

        for s in buys:
            market = s[0]
            o = s[1]
            res = b.buy_limit(market, o[1], o[0])
            while not res['success']:
                print res['message']
                print market, 0[1], o[0]
                res = b.buy_limit(market, o[1], o[0])

            state['orders'].append(res['result'])

        state['cycles'] += 1
        save_obj(state, 'state')
        print "Done.\n"


    except Exception, e:
        print e
        print sys.exc_info()[0]
        main()

starttime=time.time()
while True:
    sTime = time.time()
    main()
    print "Time: {0:.2f} sec.".format(time.time()-sTime)
    time.sleep(interval - ((time.time() - starttime) % interval))