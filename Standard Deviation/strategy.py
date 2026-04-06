import statistics


def get_sd_levels(price_series, steps=3, std_multiplier=1.0, window=168):
    """
    Generate buy/sell price levels at mean ± N*σ of the recent price window.

    price_series  : recent price history (list of floats, newest last)
    steps         : number of levels on each side (e.g. 3 → 1σ, 2σ, 3σ)
    std_multiplier: spacing between levels in σ units (1.0 = 1σ per step)
    window        : how many candles to use for mean/σ calculation

    Returns (buys, sells) — both sorted nearest-to-price first.
    """
    lookback = price_series[-window:] if len(price_series) >= window else price_series

    if len(lookback) < 2:
        return [], []

    mean = statistics.mean(lookback)
    std  = statistics.stdev(lookback)

    if std == 0:
        return [], []

    buys  = [mean - std_multiplier * (i + 1) * std for i in range(steps)]
    sells = [mean + std_multiplier * (i + 1) * std for i in range(steps)]

    # Drop any buy levels that would be at or below zero
    buys = [b for b in buys if b > 0]

    return buys, sells


def compute_atr(price_series, period=48):
    """Average True Range over the last `period` candles."""
    if len(price_series) < period + 1:
        return None
    ranges = [abs(price_series[i] - price_series[i - 1]) for i in range(-period, 0)]
    return sum(ranges) / period
