import numpy as np

def wsmape(y_true, y_pred, epsilon=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + epsilon
    return np.mean(2.0 * np.abs(y_true - y_pred) / denom)
