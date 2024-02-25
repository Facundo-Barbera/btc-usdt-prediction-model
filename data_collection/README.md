# Data Collection and Storage for BTC-USDT Prediction Model
This document provides a detailed explanation of the data collection and storage processes involved in the BTC-USDT prediction model project.

## Data Collection

Data collection is the first step in the process of making a prediction model. 
In this project, the data collection process involves obtaining historical data for BTC-USDT.

### Data Sources

To obtain the data, the library `ccxt` is used, specifically the binance API.

### Libraries Used

Two main libraries are used in the data collection process:

1. `ccxt`: This library is used to interact with the Binance API and fetch the required data.
2. `pandas`: Using dataframes from the `pandas` library, the fetched data is stored in CSV files.

#### Installation

To install the `ccxt` library, run the following command:

```bash
pip install ccxt
```

To install the `pandas` library, run the following command:

```bash
pip install pandas
```

Or alternatively, you can install both libraries at once by running the following command:

```bash
pip install ccxt pandas
```

These libraries are essential for the data collection and storage processes.

## Data Storage

The data obtained from the data collection process is stored in CSV files.
This is done using the `pandas` library.

## Credits

Author: [Facundo Bautista Barbera](https://github.com/Facundo-Barbera)