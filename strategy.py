def get_grid(price, sweep=0.2, steps=8):
    buys, sells = [], []

    for i in range(1, steps + 1):
        pct = sweep * i / steps
        buys.append(price * (1 - pct))
        sells.append(price * (1 + pct))

    return buys, sells


def compute_atr(price_series, period=14):
    """Average True Range over the last `period` candles."""
    if len(price_series) < period + 1:
        return None
    ranges = [abs(price_series[i] - price_series[i - 1]) for i in range(-period, 0)]
    return sum(ranges) / period


def dynamic_sweep(price, price_series, atr_multiplier=8.0, min_sweep=0.05, max_sweep=0.50, default=0.20):
    """Scale grid width by ATR. Falls back to `default` if not enough history."""
    atr = compute_atr(price_series)
    if atr is None or price <= 0:
        return default
    return max(min_sweep, min(max_sweep, (atr / price) * atr_multiplier))
