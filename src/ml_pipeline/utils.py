"""Shared paths and helpers used by every model in the pipeline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed=42):
    """Seed Python/NumPy/TensorFlow RNGs so pipeline runs are reproducible."""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def to_series(X):
    """Reshape a (samples, timesteps) matrix into (samples, timesteps, 1) for Conv1D/LSTM input."""
    return X.reshape((X.shape[0], X.shape[1], 1))


def evaluate(name, y_true, y_pred):
    """Compute MAE/RMSE/MAPE for a model's predictions and print a one-line summary."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        if nonzero.any()
        else float("nan")
    )

    print(f"[{name}] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")
    return {"model": name, "mae": mae, "rmse": rmse, "mape": mape}


def plot_predictions(name, y_true, y_pred, filename):
    """Plot actual vs. predicted values (with a legend) and save to output/plots/."""
    plt.figure(figsize=(20, 5))
    plt.plot(np.asarray(y_true).ravel(), color="red", label="Actual")
    plt.plot(np.asarray(y_pred).ravel(), color="blue", label="Predicted")
    plt.title(name)
    plt.legend()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()
