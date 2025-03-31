import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from math import sqrt

# ----------------------------- CONFIG -----------------------------
CSV_PATH = 'LSTM.csv'
TIME_COLUMN = 'time'
INJECTION_RATES = [100, 120, 140, 150, 160, 180, 200]
N_LAGS = 5
EPOCHS = 50
# ------------------------------------------------------------------


def create_lag_features(data, n_lags):
    """
    Creates lag features for each column in the dataset.
    """
    df_lagged = data.copy()
    for i in range(1, n_lags + 1):
        for col in data.columns:
            df_lagged[f"{col}_lag_{i}"] = data[col].shift(i)
    return df_lagged.dropna()


def train_cnn_lstm_model():
    """
    Trains a CNN-LSTM model to predict raceway size and evaluates its performance.
    """
    # Load dataset
    df = pd.read_csv(CSV_PATH)

    # Define input features
    depth_up_cols = [f"Depth_up_{rate}" for rate in INJECTION_RATES]
    depth_down_cols = [f"Depth_down_{rate}" for rate in INJECTION_RATES]
    height_cols = [f"Height_{rate}" for rate in INJECTION_RATES]
    features = depth_up_cols + depth_down_cols + height_cols

    # Normalize
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    # Lag features
    lagged_df = create_lag_features(scaled_df, N_LAGS)
    lagged_df[TIME_COLUMN] = df[TIME_COLUMN].iloc[N_LAGS:].values

    # Train-test split
    train_df = lagged_df[lagged_df[TIME_COLUMN] < 2.8]
    test_df = lagged_df[lagged_df[TIME_COLUMN] >= 2.8]

    # Prepare inputs/outputs
    X_train = train_df.drop(features + [TIME_COLUMN], axis=1).values
    Y_train = train_df[features].values
    X_test = test_df.drop(features + [TIME_COLUMN], axis=1).values
    Y_test = test_df[features].values

    # Reshape for CNN-LSTM: [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], N_LAGS, X_train.shape[1] // N_LAGS))
    X_test = X_test.reshape((X_test.shape[0], N_LAGS, X_test.shape[1] // N_LAGS))

    # Build model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(len(features)))
    model.compile(optimizer='adam', loss='mse')

    print("Training CNN-LSTM model...")
    model.fit(X_train, Y_train, epochs=EPOCHS, verbose=1)

    # Predict and inverse transform
    predictions = model.predict(X_test)
    predictions_inverse = scaler.inverse_transform(predictions)
    Y_test_inverse = scaler.inverse_transform(Y_test)

    # Evaluation
    mae = mean_absolute_error(Y_test_inverse, predictions_inverse)
    rmse = sqrt(mean_squared_error(Y_test_inverse, predictions_inverse))
    r2 = r2_score(Y_test_inverse, predictions_inverse)

    print(f"\n📊 CNN-LSTM Model Performance:")
    print(f"MAE  = {mae:.3f}")
    print(f"RMSE = {rmse:.3f}")
    print(f"R²   = {r2:.3f}")


if __name__ == "__main__":
    train_cnn_lstm_model()

