import yfinance as yf
import pandas as pd
import time

# All tradable cryptos on Robinhood (US) as of April 2026
# Source: https://robinhood.com/us/en/support/articles/coin-availability/
# Note: Some newer/obscure tickers may not be available on yfinance
ROBINHOOD_CRYPTOS = {
    "BTC-USD":     "Bitcoin",
    "ETH-USD":     "Ethereum",
    "DOGE-USD":    "Dogecoin",
    "LTC-USD":     "Litecoin",
    "SHIB-USD":    "Shiba Inu",
    "AAVE-USD":    "Aave",
    "AERO-USD":    "Aerodrome Finance",
    "ARB-USD":     "Arbitrum",
    "ASTER-USD":   "Aster",
    "AVAX-USD":    "Avalanche",
    "AVNT-USD":    "Avantis",
    "BCH-USD":     "Bitcoin Cash",
    "BNB-USD":     "BNB",
    "BONK-USD":    "BONK",
    "ADA-USD":     "Cardano",
    "MEW-USD":     "cat in a dogs world",
    "LINK-USD":    "Chainlink",
    "COMP-USD":    "Compound",
    "CRV-USD":     "Curve DAO",
    "WIF-USD":     "Dogwifhat",
    "EIGEN-USD":   "EigenCloud",
    "ENA-USD":     "Ethena",
    "ETC-USD":     "Ethereum Classic",
    "FLOKI-USD":   "Floki",
    "HBAR-USD":    "Hedera",
    "HYPE-USD":    "Hyperliquid",
    "ZRO-USD":     "LayerZero",
    "LDO-USD":     "Lido DAO",
    "LIT-USD":     "Lighter",
    "MNT-USD":     "Mantle",
    "SYRUP-USD":   "Maple Finance",
    "MOODENG-USD": "Moo Deng",
    "TRUMP-USD":   "OFFICIAL TRUMP",
    "ONDO-USD":    "Ondo",
    "XCN-USD":     "Onyxcoin",
    "OP-USD":      "Optimism",
    "PAXG-USD":    "Pax Gold",
    "PNUT-USD":    "Peanut the Squirrel",
    "PEPE-USD":    "Pepecoin",
    "DOT-USD":     "Polkadot",
    "POPCAT-USD":  "Popcat",
    "PENGU-USD":   "Pudgy Penguins",
    "PYTH-USD":    "Pyth Network",
    "RENDER-USD":  "Render",
    "SEI-USD":     "SEI",
    "SKY-USD":     "Sky",
    "SOL-USD":     "Solana",
    "SUI-USD":     "SUI",
    "XLM-USD":     "Stellar Lumens",
    "SNX-USD":     "Synthetix",
    "XTZ-USD":     "Tezos",
    "GRT-USD":     "The Graph",
    "TON-USD":     "Toncoin",
    "UNI-USD":     "Uniswap",
    "USDC-USD":    "USD Coin",
    "VIRTUAL-USD": "Virtuals Protocol",
    "W-USD":       "Wormhole",
    "XRP-USD":     "XRP",
    # Tickers below are unlikely to exist on yfinance but included for completeness
    "CC-USD":      "Canton Coin",
    "SKR-USD":     "Seeker",
    "XPL-USD":     "Plasma",
    "WLFI-USD":    "World Liberty Financial",
    "ZORA-USD":    "Zora",
}


def download(symbol: str, name: str, period: str = "2y") -> bool:
    """Download OHLCV data for a single crypto. Returns True on success."""
    try:
        print(f"  Downloading {name} ({symbol})...", end=" ", flush=True)
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)

        if df.empty:
            print("SKIP (no data)")
            return False

        df = df.reset_index()[["Date", "Close"]]
        df.columns = ["date", "close"]

        filename = f"{symbol.replace('-', '_').lower()}_{period}.csv"
        df.to_csv(filename, index=False)
        print(f"OK  -> {filename}  ({len(df)} rows)")
        return True

    except Exception as e:
        print(f"ERROR ({e})")
        return False


def download_all(period: str = "2y", delay: float = 0.3) -> None:
    """Download all Robinhood-listed cryptos and print a summary."""
    total = len(ROBINHOOD_CRYPTOS)
    succeeded, failed = [], []

    print(f"\n{'='*60}")
    print(f" Robinhood Crypto Downloader — {total} assets, period={period}")
    print(f"{'='*60}\n")

    for symbol, name in ROBINHOOD_CRYPTOS.items():
        ok = download(symbol, name, period)
        (succeeded if ok else failed).append(symbol)
        time.sleep(delay)   # be polite to yfinance

    print(f"\n{'='*60}")
    print(f" Done: {len(succeeded)}/{total} downloaded successfully")
    if failed:
        print(f" Skipped/failed ({len(failed)}): {', '.join(failed)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    download_all(period="2y")