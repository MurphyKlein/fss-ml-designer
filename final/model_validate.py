"""
model_validate.py
-----------------
Loads best_surrogate.pkl (trained earlier), runs outlier pruning again,
and checks 5-fold RMSE to make sure model serialization works.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error

# ---- Paths and config ----
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
CSV_PATH = DATA_DIR / "results.csv"
BEST_PKL = MODEL_DIR / "best_surrogate.pkl"

INPUT_COLS = ["width", "height", "spacing"]
TARGET = "max_trans_freq"
CONTAM = 0.05
N_SPLITS = 5
SEED = 42

# ---- Helper: RMSE ----
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ---- Data loading and outlier filtering ----
try:
    df = pd.read_csv(CSV_PATH)[INPUT_COLS + [TARGET]].dropna().reset_index(drop=True)
except Exception as err:
    print(f"[Error] Could not read {CSV_PATH}: {err}")
    exit(1)

try:
    iso = IsolationForest(
        contamination=CONTAM,
        n_estimators=300,
        random_state=SEED
    ).fit(df[INPUT_COLS + [TARGET]])
    mask = iso.predict(df[INPUT_COLS + [TARGET]]) == 1
except Exception as err:
    print("[Warning] IsolationForest failed:", err)
    mask = np.ones(len(df), dtype=bool)

removed = np.sum(~mask)
print(f"Validation: removed {removed}/{len(df)} rows via IsolationForest ({100*CONTAM:.1f}% expected)\n")

df = df[mask].reset_index(drop=True)
X = df[INPUT_COLS].to_numpy(float)
y = df[TARGET].to_numpy(float)

# ---- Model loading ----
try:
    model = joblib.load(BEST_PKL)
    print(f"Loaded model from {BEST_PKL.name}\n")
except Exception as err:
    print(f"[Error] Could not load model from {BEST_PKL}: {err}")
    exit(2)

# ---- Cross-validated RMSE (same settings as training) ----
print(f"Evaluating 5-fold RMSE with {len(X)} samples...")

try:
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        neg_mse = cross_val_score(
            model, X, y,
            scoring="neg_mean_squared_error",
            cv=cv, n_jobs=-1
        )
    final_rmse = np.sqrt(-neg_mse.mean())
    print(f"\n5-fold RMSE (GHz): {final_rmse:.4f}")
except Exception as err:
    print("[Error] during cross-validation:", err)
    exit(3)

# ---- Optionally print all folds ----
# print("All fold RMSEs:")
# for i, mse in enumerate(-neg_mse):
#     print(f"  Fold {i+1}: {np.sqrt(mse):.4f} GHz")
