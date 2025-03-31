import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Prints evaluation metrics for regression.

    Parameters:
        y_true (ndarray): True target values
        y_pred (ndarray): Predicted values
        model_name (str): Name of the model (for print header)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 {model_name} Performance:")
    print(f"MAE  = {mae:.3f}")
    print(f"RMSE = {rmse:.3f}")
    print(f"R²   = {r2:.3f}")

    return mae, rmse, r2


def plot_predictions(y_true, y_pred, title="Predicted vs Actual", sample_idx=0, figsize=(10, 5)):
    """
    Plots predicted vs. actual values for a specific test sample.

    Parameters:
        y_true (ndarray): True values
        y_pred (ndarray): Predicted values
        title (str): Plot title
        sample_idx (int): Index of sample to plot
    """
    plt.figure(figsize=figsize)
    plt.plot(y_true[sample_idx], label="Actual", marker='o')
    plt.plot(y_pred[sample_idx], label="Predicted", marker='x')
    plt.title(f"{title} - Sample {sample_idx}")
    plt.xlabel("Feature Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss(history, title="Training Loss", figsize=(8, 5)):
    """
    Plots training and validation loss over epochs.

    Parameters:
        history: Keras History object from model.fit()
        title (str): Plot title
    """
    plt.figure(figsize=figsize)
    plt.plot(history.history['loss'], label="Training Loss")
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

