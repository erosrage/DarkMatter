# ETH Grid Trader — Production

Grid trading bot for ETH/USD on Robinhood. Optimized settings from a 2-year backtest:
- **Sweep:** ±40% price range
- **Steps:** 2 levels (1 buy + 1 sell at 20%, another at 40%)
- **Interval:** Every hour
- **Risk guard:** Buys suppressed if ETH drops >8% over the last 10 readings

---

## 1. Prerequisites

```bash
pip install robin_stocks pandas
```

---

## 2. API Credentials

**Never hardcode your username or password in the script.**

The bot reads credentials from environment variables. Set them before running:

### Windows (Command Prompt)
```cmd
set RH_USERNAME=your_robinhood_email@example.com
set RH_PASSWORD=your_robinhood_password
```

### Windows (PowerShell)
```powershell
$env:RH_USERNAME = "your_robinhood_email@example.com"
$env:RH_PASSWORD = "your_robinhood_password"
```

### Mac / Linux
```bash
export RH_USERNAME="your_robinhood_email@example.com"
export RH_PASSWORD="your_robinhood_password"
```

> **Note:** Robinhood uses SMS/email MFA. The first login will prompt you to enter a one-time code in the terminal. After that, `robin_stocks` caches a token locally so subsequent runs won't prompt again.

---

## 3. Configuration

All tuneable settings are at the top of `eth_trader.py`:

| Variable | Default | Description |
|---|---|---|
| `SWEEP` | `0.40` | Grid range on each side (40%) |
| `STEPS` | `2` | Number of buy/sell levels |
| `ORDER_SIZE` | `0.01` | ETH per order |
| `INTERVAL` | `3600` | Seconds between cycles (1 hour) |
| `DROP_THRESHOLD` | `0.08` | % drop that triggers the risk pause |

**Adjust `ORDER_SIZE`** to control how much ETH is bought/sold per level. At 2 steps, each cycle can place up to 2 buy orders and 2 sell orders.

---

## 4. Running the Bot

From the `PROD/` directory:

```bash
python eth_trader.py
```

The bot will:
1. Log in to Robinhood
2. Fetch the current ETH price
3. Place grid orders every hour
4. Save state to `eth_state.pkl` after each cycle
5. Write logs to `logs/eth_trader_YYYYMMDD.log`

To stop the bot, press `Ctrl+C`.

---

## 5. Monitoring

### Live log output
The bot prints to both the terminal and a daily log file:

```
logs/eth_trader_20260405.log
```

Sample output:
```
2026-04-05 14:00:01  INFO     ── Cycle 1 ──────────────────────────────────────
2026-04-05 14:00:02  INFO     ETH price: $1,823.45
2026-04-05 14:00:02  INFO     Grid  buys : ['$1,458.76', '$1,094.07']
2026-04-05 14:00:02  INFO     Grid  sells: ['$2,188.14', '$2,552.83']
2026-04-05 14:00:02  INFO       BUY  0.01 ETH @ $1,458.76
2026-04-05 14:00:02  INFO       BUY  0.01 ETH @ $1,094.07
2026-04-05 14:00:03  INFO       SELL 0.01 ETH @ $2,188.14
2026-04-05 14:00:03  INFO       SELL 0.01 ETH @ $2,552.83
2026-04-05 14:00:03  INFO     Orders placed: 4/4
2026-04-05 14:00:03  INFO     State saved — total cycles: 1
2026-04-05 14:00:03  INFO     Next cycle in 60.0 min
```

### Check open orders
Log in to Robinhood and check **Account → Orders** to see pending limit orders.

### Check state file
The `eth_state.pkl` file persists cycle count and position data. It is created automatically on first run.

---

## 6. Running 24/7 (Always-On Deployment)

To keep the bot running after you close the terminal:

### Windows — Task Scheduler
1. Open **Task Scheduler** → Create Basic Task
2. Set trigger: **At startup** (or a specific time)
3. Action: `python C:\...\PROD\eth_trader.py`
4. Set environment variables via a wrapper `.bat` file:

```bat
:: run_trader.bat
set RH_USERNAME=your_email@example.com
set RH_PASSWORD=your_password
python "C:\Users\Michael\OneDrive\Desktop\Projects\TradingBot\trading_bot\PROD\eth_trader.py"
```

Then point Task Scheduler at `run_trader.bat` instead of the Python script directly.

### Mac / Linux — screen or nohup
```bash
# Keep running after terminal closes
nohup python eth_trader.py &

# Or use screen for easy re-attach
screen -S eth_trader
python eth_trader.py
# Detach: Ctrl+A then D
# Re-attach: screen -r eth_trader
```

### VPS / Cloud
Deploy to any Linux VPS (e.g. AWS EC2 t3.micro, DigitalOcean Droplet) and use `screen` or a `systemd` service. The bot uses minimal CPU and memory.

---

## 7. Safety Notes

- The risk guard **suppresses all buy orders** (but still places sells) if ETH drops more than 8% over the last 10 hourly readings.
- Orders are **limit orders** — they will only fill if the price reaches the grid level.
- Robinhood may reject orders if you have insufficient buying power. Monitor your account balance.
- Do not run multiple instances simultaneously — they will place duplicate orders.
