"""
solver.py
---------
Loads a pre-trained surrogate (Gaussian Process) model and uses differential evolution
to find a geometry (width, height, spacing) that achieves a target transmission/frequency.
"""

from pathlib import Path
import joblib
import numpy as np
from scipy.optimize import differential_evolution

# ---- Paths and project configuration ----
BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"
GP_PATH = MODEL_DIR / "gpr_active.pkl"  # Should have been trained and saved already

BOUNDS = np.array([
    [2., 6.],   # width
    [5., 14.],  # height
    [5., 14.],  # spacing
])


# ---- Load model bundle once, keep in module globals ----
try:
    _bundle = joblib.load(GP_PATH)
except Exception as e:
    print(f"[Error] Could not load surrogate model from {GP_PATH}: {e}")
    raise

_gprs = _bundle["gprs"]
_xs   = _bundle["Xs"]     # StandardScaler for X
_ys   = _bundle["ys"]     # StandardScaler for y


def _predict(x):
    """
    Given a 1-D array x = [w, h, s], returns predicted (max_trans_val, max_trans_freq).
    Handles scaling and unscaling using fitted scalers.
    """
    try:
        x_scaled = _xs.transform([x])
    except Exception as e:
        print(f"[Error] Scaler error: {e}")
        raise

    preds = []
    for gp in _gprs:
        y_pred = gp.predict(x_scaled)[0]
        preds.append(y_pred)
    try:
        out = _ys.inverse_transform([preds])[0]
    except Exception as e:
        print(f"[Error] Error in inverse_transform: {e}")
        raise
    return out


def inverse_design(target_val, target_freq, w_val=1.0, w_freq=1.0,
                   popsize=20, seed=1):
    """
    Finds optimal geometry (width, height, spacing) to hit target transmission & frequency.
    Returns geometry, predictions, and prediction uncertainties.
    """
    def obj(x):
        try:
            val_hat, freq_hat = _predict(x)
        except Exception as e:
            print("[Error] Prediction error in obj:", e)
            val_hat, freq_hat = 0, 0
        return w_val * (val_hat - target_val) ** 2 + w_freq * (freq_hat - target_freq) ** 2

    try:
        result = differential_evolution(
            obj,
            bounds=BOUNDS,
            strategy="best1bin",
            popsize=popsize,
            seed=seed,
            tol=1e-4,
            polish=True,
            disp=False
        )
    except Exception as e:
        print(f"[Error] Optimization failed: {e}")
        raise

    w, h, s = result.x
    try:
        x_scaled = _xs.transform([result.x])
    except Exception as e:
        print("[Error] Error scaling optimal x:", e)
        x_scaled = np.zeros((1, 3))

    final = []
    for gp in _gprs:
        mean, std = gp.predict(x_scaled, return_std=True)
        final.append((mean[0], std[0]))

    means = [m for m, s in final]
    stds  = [s for m, s in final]

    try:
        val_opt, freq_opt = _ys.inverse_transform([means])[0]
        std_scaled = np.array(stds) * _ys.scale_
        val_std, freq_std = std_scaled
    except Exception as e:
        print("[Error] Error in final prediction scaling:", e)
        val_opt, freq_opt, val_std, freq_std = 0, 0, 0, 0

    return {
        "width":   round(w, 3),
        "height":  round(h, 3),
        "spacing": round(s, 3),
        "T_val":   round(val_opt, 4),
        "T_freq":  round(freq_opt, 4),
        "val_std": round(val_std, 4),
        "freq_std": round(freq_std, 4),
        "score":   round(result.fun, 6)
    }


if __name__ == "__main__":
    res = inverse_design(0.85, 9.6, w_val=1.0, w_freq=1.0)
    print("Best geometry found:")
    for k, v in res.items():
        print(f"  {k}: {v}")
