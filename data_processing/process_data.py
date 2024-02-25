import json
import os

import pandas as pd

RAW_DATA_PATH = '../data/raw/'
PROCESSED_DATA_PATH = '../data/processed/'


def add_technical_indicators(df):
    # Simple Moving Average (SMA)
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()

    # Exponential Moving Average (EMA)
    df['EMA_7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['EMA_30'] = df['close'].ewm(span=30, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


def add_future_labels(df, steps=1):
    # Shift the data to create future labels for each feature we want to predict
    future_labels = df[['close', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI']].shift(-steps)
    df = pd.concat([df, future_labels.add_suffix('_future')], axis=1)

    # Drop the last 'steps' rows which will have NaN values because of the shift
    df = df[:-steps]

    return df


def preprocess_data(df):
    # Ensure the timestamp is a datetime type
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df.sort_values('timestamp', inplace=True)

    # Add technical indicators
    df = add_technical_indicators(df)

    # Add future labels
    df = add_future_labels(df)

    # Drop any rows with NaN values resulting from the technical indicators
    df.dropna(inplace=True)

    return df


def generate_metadata(df):
    # Generate metadata for the dataset
    metadata = {
        'num_rows': int(len(df)),
        'num_columns': int(len(df.columns)),
        'columns': df.columns.tolist(),
        'num_missing_values': int(df.isnull().sum().sum()),
        'features': ['open', 'high', 'low', 'close', 'volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI'],
        'targets': ['close_future', 'SMA_7_future', 'SMA_30_future', 'EMA_7_future', 'EMA_30_future', 'RSI_future']
    }

    return metadata


def main():
    # Load every dataset in the raw data folder
    # Process each dataset and save it to the processed data folder
    # Note: This will overwrite any existing processed datasets with the same name
    for file in os.listdir(RAW_DATA_PATH):
        if file.endswith('.csv'):
            print(f"Processing {file}...")

            # Load the dataset
            df = pd.read_csv(RAW_DATA_PATH + file)

            # Preprocess the dataset
            df = preprocess_data(df)

            # Save the processed dataset
            df.to_csv(PROCESSED_DATA_PATH + file, index=False)

            # Generate metadata for the dataset
            metadata = generate_metadata(df)

            # Save the metadata
            with open(PROCESSED_DATA_PATH + file.replace('.csv', '_metadata.json'), 'w') as metadata_file:
                json.dump(metadata, metadata_file, indent=4)

            print(f"Processed dataset saved to {PROCESSED_DATA_PATH + file}")


if __name__ == "__main__":
    main()
