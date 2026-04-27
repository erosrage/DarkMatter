"""
Download 5-minute OHLCV data for all Robinhood-listed cryptos via yfinance.
yfinance caps 5m intervals at 60 days of history.
Output: research/data/5m/<symbol>_usd_5m.csv  (columns: date, close)
Run from the project root: python research/datagen_5m.py
"""

import os
import yfinance as yf
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "5m")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = [
    "AAVE-USD", "ADA-USD", "AERO-USD", "ARB-USD", "AVAX-USD", "AVNT-USD",
    "BCH-USD", "BNB-USD", "BONK-USD", "BTC-USD", "CRV-USD", "DOGE-USD",
    "DOT-USD", "EIGEN-USD", "ENA-USD", "ETC-USD", "ETH-USD", "FLOKI-USD",
    "HBAR-USD", "LDO-USD", "LINK-USD", "LTC-USD", "ONDO-USD", "OP-USD",
    "PAXG-USD", "PNUT-USD", "PYTH-USD", "RENDER-USD", "SEI-USD", "SHIB-USD",
    "SNX-USD", "SOL-USD", "TON-USD", "TRUMP-USD", "USDC-USD", "W-USD",
    "WIF-USD", "XLM-USD", "XRP-USD", "XTZ-USD",
]

ok, skipped = [], []

for ticker in TICKERS:
    symbol   = ticker.replace("-USD", "").lower()
    out_path = os.path.join(OUT_DIR, f"{symbol}_usd_5m.csv")

    try:
        df = yf.download(ticker, period="60d", interval="5m", auto_adjust=True, progress=False)
        if df.empty or len(df) < 100:
            print(f"  SKIP {ticker}: insufficient data ({len(df)} rows)")
            skipped.append(ticker)
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"].dropna()
        out   = pd.DataFrame({"date": close.index, "close": close.values})
        out.to_csv(out_path, index=False)
        print(f"  OK    {ticker}: {len(out):,} rows  [{out['date'].iloc[0]} to {out['date'].iloc[-1]}]")
        ok.append(symbol)

    except Exception as e:
        print(f"  ERR   {ticker}: {e}")
        skipped.append(ticker)

print(f"\nDone. {len(ok)} symbols saved, {len(skipped)} skipped.")
print("Saved:", ok)
