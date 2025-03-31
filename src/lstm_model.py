import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from math import sqrt

# ----------------------------- CONFIG -----------------------------
CSV_PATH = 'LSTM.csv'       # CSV file with preprocessed raceway data
TIME_COLUMN = 'time'        # Column for time axis
INJECTION_RATES = [100, 120, 140, 150, 160, 180, 200]
N_LAGS = 5
EPOCHS = 50
# ------------------------------------------------------------------


def create_lag_features(data, n_lags):
    """
    Creates lag features for each column in the dataset.

    Parameters:
        data (pd.DataFrame): Scaled dataframe
        n_lags (int): Number of timesteps for lagging

    Returns:
        pd.DataFrame: DataFrame with lag features
    """
    df_lagged = data.copy()
    for i in range(1, n_lags + 1):
        for col in data.columns:
            df_lagged[f"{col}_lag_{i}"] = data[col].shift(i)
    return df_lagged.dropna()


def train_lstm_model():
    """
    Trains an LSTM model on raceway size data and evaluates its performance.
    """
    # Load dataset
    df = pd.read_csv(CSV_PATH)

    # Define input features (depth up, depth down, height for all injection rates)
    depth_up_cols = [f"Depth_up_{rate}" for rate in INJECTION_RATES]
    depth_down_cols = [f"Depth_down_{rate}" for rate in INJECTION_RATES]
    height_cols = [f"Height_{rate}" for rate in INJECTION_RATES]
    features = depth_up_cols + depth_down_cols + height_cols

    # Scale features
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    # Create lag features
    lagged_df = create_lag_features(scaled_df, N_LAGS)

    # Join back time column for train/test split
    lagged_df[TIME_COLUMN] = df[TIME_COLUMN].iloc[N_LAGS:].values

    # Train-test split
    train_df = lagged_df[lagged_df[TIME_COLUMN] < 2.8]
    test_df = lagged_df[lagged_df[TIME_COLUMN] >= 2.8]

    # Separate inputs and outputs
    X_train = train_df.drop(features + [TIME_COLUMN], axis=1).values
    Y_train = train_df[features].values

    X_test = test_df.drop(features + [TIME_COLUMN], axis=1).values
    Y_test = test_df[features].values

    # Reshape for LSTM: [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], N_LAGS, X_train.shape[1] // N_LAGS))
    X_test = X_test.reshape((X_test.shape[0], N_LAGS, X_test.shape[1] // N_LAGS))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(len(features)))  # Output layer (multi-target)
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    print("Training LSTM model...")
    model.fit(X_train, Y_train, epochs=EPOCHS, verbose=1)

    # Predict and inverse scale
    predictions = model.predict(X_test)
    predictions_inverse = scaler.inverse_transform(predictions)
    Y_test_inverse = scaler.inverse_transform(Y_test)

    # Evaluation metrics
    mae = mean_absolute_error(Y_test_inverse, predictions_inverse)
    rmse = sqrt(mean_squared_error(Y_test_inverse, predictions_inverse))
    r2 = r2_score(Y_test_inverse, predictions_inverse)

    print(f"\n📊 LSTM Model Performance:")
    print(f"MAE  = {mae:.3f}")
    print(f"RMSE = {rmse:.3f}")
    print(f"R²   = {r2:.3f}")


if __name__ == "__main__":
    train_lstm_model()

