import ccxt
import pandas as pd

# Constants
SYMBOL = 'BTC/USDT'
TIMEFRAMES = ['5m', '30m', '1h', '1d']
DATE_FROM = '2013-01-01T00:00:00Z'
DATA_DIR = '../data'


def fetch_data(timeframe):
    # Initialize the exchange
    print('Initializing the exchange...')
    exchange = ccxt.binance()
    since = exchange.parse8601(DATE_FROM)

    # Initialize the list to store the candles
    print(f'Fetching data for the {timeframe} timeframe...')
    all_candles = []

    # Fetch the data
    while True:
        candles = exchange.fetch_ohlcv(SYMBOL, timeframe, since)
        if not candles:
            break
        since = candles[-1][0] + 1
        all_candles.extend(candles)
        print(f'Fetched {len(all_candles)} candles')

    # Convert to DataFrame
    print('Converting to DataFrame...')
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(all_candles, columns=columns)

    # Convert timestamp to datetime
    print('Converting timestamp to datetime...')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Define the CSV file name
    # Note: This will overwrite the file if it already exists
    print('Saving the data to a CSV file...')
    csv_file = f'{DATA_DIR}/raw/bitcoin_history_{timeframe}.csv'
    df.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")


def main():
    # Fetch data for each timeframe
    for timeframe in TIMEFRAMES:
        fetch_data(timeframe)


if __name__ == "__main__":
    main()
