def get_grid(price, sweep=0.2, steps=8):
    buys, sells = [], []

    for i in range(1, steps + 1):
        pct = sweep * i / steps
        buys.append(price * (1 - pct))
        sells.append(price * (1 + pct))

    return buys, sells
