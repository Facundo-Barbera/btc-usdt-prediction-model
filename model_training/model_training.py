
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Constants
DATASET_PATH = '../data/processed/bitcoin_history_1d.csv'
DATASET_METADATA_PATH = '../data/processed/bitcoin_history_1d_metadata.json'
TRAINING_PROGRESS_PATH = '../models/training_progress.json'

BEST_MODEL_PATH = '../models/best_model.keras'
FEATURE_SCALER_PATH = '../models/feature_scaler.json'
TARGET_SCALER_PATH = '../models/target_scaler.json'
MODEL_METADATA_PATH = '../models/model_metadata.json'

EPOCHS = 100
BATCH_SIZE = 32


# Function to save training progress
def save_training_progress(current_epoch, current_logs):
    training_progress = {'epoch': current_epoch, 'logs': current_logs}

    with open(TRAINING_PROGRESS_PATH, 'w') as progress_file:
        json.dump(training_progress, progress_file)


# Function to resume training from a saved state
def fetch_training_progress():
    initial_epoch = 0  # Default value

    if os.path.exists(TRAINING_PROGRESS_PATH):
        with open(TRAINING_PROGRESS_PATH) as progress_file:
            saved_progress = json.load(progress_file)
        initial_epoch = saved_progress['epoch'] + 1
        print(f"Resuming training from epoch {initial_epoch}")

    return initial_epoch


# Function to load and preprocess data
def load_and_preprocess_data():
    dataset = pd.read_csv(DATASET_PATH)
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])

    return dataset


# Function to split the dataset into features and targets
def split_features_and_targets(dataset):
    # Load metadata
    with open(DATASET_METADATA_PATH) as metadata_file:
        metadata = json.load(metadata_file)

    # Metadata includes features and targets
    feature_columns = dataset[metadata['features']]
    target_columns = dataset[metadata['targets']]

    return feature_columns, target_columns


# Function to scale features and targets
def scale_features_and_targets(feature_columns, target_columns):
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(feature_columns)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_targets = target_scaler.fit_transform(target_columns)

    return feature_scaler, scaled_features, target_scaler, scaled_targets


# Function to split the dataset into training and testing sets
def split_training_and_testing_sets(scaled_features, scaled_targets):
    x_train, x_test, y_train, y_test = train_test_split(scaled_features, scaled_targets,
                                                        test_size=0.2, random_state=42)

    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    return x_train, x_test, y_train, y_test


# Function to build the model
def build_model(x_train, y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        LSTM(50),
        Dense(y_train.shape[1])
    ])

    model.compile(optimizer='adam', loss='mse')

    model_callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, mode='min'),
        ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True),
        LambdaCallback(on_epoch_end=lambda epoch, logs: save_training_progress(epoch, logs))
    ]

    return model, model_callbacks


# Function to train the model
def train_model(model, x_train, y_train, x_test, y_test, model_callbacks):
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test),
              callbacks=model_callbacks, initial_epoch=fetch_training_progress())

    return model


# Function to create model metadata
def create_model_metadata():
    metadata = {
        'features': ['open', 'high', 'low', 'close', 'volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI'],
        'targets': ['close_future', 'SMA_7_future', 'SMA_30_future', 'EMA_7_future', 'EMA_30_future', 'RSI_future']
    }

    with open(MODEL_METADATA_PATH, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)


# Main function
def main():
    dataset = load_and_preprocess_data()
    feature_columns, target_columns = split_features_and_targets(dataset)
    feature_scaler, scaled_features, target_scaler, scaled_targets = scale_features_and_targets(feature_columns,
                                                                                                target_columns)
    x_train, x_test, y_train, y_test = split_training_and_testing_sets(scaled_features, scaled_targets)
    model, model_callbacks = build_model(x_train, y_train)
    trained_model = train_model(model, x_train, y_train, x_test, y_test, model_callbacks)

    # Save scalers
    with open(FEATURE_SCALER_PATH, 'w') as f:
        json.dump({'min_': feature_scaler.min_.tolist(), 'scale_': feature_scaler.scale_.tolist()}, f)

    with open(TARGET_SCALER_PATH, 'w') as f:
        json.dump({'min_': target_scaler.min_.tolist(), 'scale_': target_scaler.scale_.tolist()}, f)

    # Save model metadata
    create_model_metadata()

    # Model evaluation
    predictions = trained_model.predict(x_test)
    mse_score = mean_squared_error(y_test, predictions, multioutput='uniform_average')
    rmse_score = np.sqrt(mse_score)

    print(f"RMSE: {rmse_score}")
    print(f"MSE: {mse_score}")
    print(f"Model trained and evaluated successfully!")


if __name__ == '__main__':
    main()
