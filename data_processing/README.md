# Data Processing for BTC-USDT Prediction Model
This document provides a detailed explanation of the data processing involved in the BTC-USDT prediction model project.

## Data Processing

Data processing is a crucial step in the process of making a prediction model.
In this project, the data processing involves transforming the raw historical data for BTC-USDT into a format suitable for machine learning models.

### Data Processing Steps

The data processing involves several steps:

1. Loading the raw data from CSV files.
2. Ensuring the timestamp is a datetime type.
3. Sorting the data by timestamp.
4. Adding technical indicators to the data, such as Simple Moving Average (SMA), Exponential Moving Average (EMA), and Relative Strength Index (RSI).
5. Adding future labels to the data by shifting the data.
6. Dropping any rows with NaN values resulting from the technical indicators.
7. Saving the transformed dataset to a new CSV file.

### Libraries Used

The main library used in the data processing is `pandas`. 
Using dataframes from the `pandas` library, the raw data is transformed and stored in new CSV files.

#### Installation

To install the `pandas` library, run the following command:

```bash
pip install pandas
```

This library is essential for the data processing process.

## Credits

Author: [Facundo Bautista Barbera](https://github.com/Facundo-Barbera)