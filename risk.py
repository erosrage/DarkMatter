def trend_is_up(price_series, ma_period=20):
    """Returns True when the latest price is above the MA — uptrend confirmed.
    Returns True (allow buys) when there is not enough history yet."""
    if len(price_series) < ma_period:
        return True
    ma = sum(price_series[-ma_period:]) / ma_period
    return price_series[-1] > ma


def portfolio_ok(current_value, initial_value, threshold=0.20):
    """Returns False if the portfolio has dropped more than `threshold` from
    its initial value — triggers the circuit breaker."""
    if initial_value is None or initial_value <= 0:
        return True
    drop = (initial_value - current_value) / initial_value
    return drop <= threshold
