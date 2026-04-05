def should_pause(price_series, drop_threshold=0.08):
    if len(price_series) < 2:
        return False

    recent = price_series[-1]
    past = price_series[0]

    drop = (past - recent) / past
    return drop > drop_threshold
