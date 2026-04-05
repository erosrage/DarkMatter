# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the live trading bot
python main.py

# Run backtesting on historical data
python backtest.py

# Download/refresh historical price data
python datagen.py
```

There are no build steps, test runner, or linter configured. Dependencies are managed manually; required packages are `robin_stocks`, `pandas`, `matplotlib`, `seaborn`, and `yfinance`.

## Architecture

This is a **cryptocurrency grid trading bot** for Robinhood. It runs an infinite async loop, executing a grid trading strategy across multiple markets every hour.

### Component Overview

**[main.py](main.py)** — Orchestrator. Configures `MARKETS`, `INTERVAL`, and `ORDER_SIZE`, authenticates with Robinhood, and runs `run_cycle()` on a timer. Each cycle runs all markets concurrently via `asyncio`.

**[broker.py](broker.py)** — Robinhood API wrapper around `robin_stocks`. Provides `login()`, `get_price()`, `buy_limit()`, and `sell_limit()`.

**[strategy.py](strategy.py)** — Core grid logic. `get_grid(price, sweep=0.2, steps=8)` generates symmetric buy/sell price levels: 8 levels spaced from 2.5% to 20% above/below current price.

**[risk.py](risk.py)** — Risk guard. `should_pause(price_series, drop_threshold=0.08)` compares first vs. last of the last 10 prices; if drop exceeds 8%, all BUY orders are suppressed for that cycle (sells still execute).

**[state.py](state.py)** — Pickle-based persistence. `load_state()` / `save_state(state)` persist cycle count, positions, and profit to `state.pkl` in the working directory.

**[backtest.py](backtest.py)** — Simulation engine. `backtest(prices, sweep, steps, order_size)` replays the grid strategy on CSV price data. `plot_trades()` visualizes buy/sell points on a price chart.

**[datagen.py](datagen.py)** — Data utility. Downloads 2 years of daily close prices for 60+ Robinhood-listed cryptos via `yfinance` into `<symbol>_usd_2y.csv` files.

### Live Trading Data Flow

```
main.py
  └─ run_cycle() [async, every 3600s]
       └─ per market (ETH, BTC, DOGE) in parallel:
            ├─ broker.get_price()          → current price
            ├─ risk.should_pause()         → check last-10 prices for >8% drop
            ├─ strategy.get_grid()         → compute 8 buy + 8 sell levels
            ├─ broker.sell_limit() × 8     → always placed
            └─ broker.buy_limit() × 8      → skipped if paused
  └─ state.save_state()                   → persist to state.pkl
```

### Key Configuration (all hardcoded in source)

| Setting | Location | Default |
|---|---|---|
| Markets | `main.py` `MARKETS` | `["ETH", "BTC", "DOGE"]` |
| Loop interval | `main.py` `INTERVAL` | `3600` seconds |
| Order size | `main.py` `ORDER_SIZE` | `0.01` |
| Grid sweep | `strategy.py` | `±20%` |
| Grid steps | `strategy.py` | `8` levels |
| Crash threshold | `risk.py` | `8%` drop |
| Broker credentials | `main.py` `login()` | hardcoded strings |

Robinhood credentials are currently hardcoded as placeholder strings in `main.py` and must be replaced before running live.
