"""
model_benchmark.py
------------------
Try a bunch of regressors on max_trans_freq (GHz) using 5-fold CV.
Shows a quick PCA plot of outliers (IsolationForest).
Dumps the best *serialisable* model to models/best_surrogate.pkl.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel as C, DotProduct, WhiteKernel
from sklearn.decomposition import PCA
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# ---- Paths and config ----
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "results.csv"
BEST_PKL = MODEL_DIR / "best_surrogate.pkl"

INPUT_COLS = ["width", "height", "spacing"]
TARGET = "max_trans_freq"
CONTAM = 0.05
SEED = 42
N_SPLITS = 5
RIDGE_ALPHA = 1.0
POLY_DEG = [1, 2, 3, 4]

# ---- Helpers ----
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmse_cv(model, X, y):
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    neg_mse = cross_val_score(
        model, X, y, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1)
    return np.sqrt(-neg_mse.mean())

def rmse_cv_interp(X, y):
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rms = []
    for tr, te in cv.split(X):
        interp = LinearNDInterpolator(X[tr], y[tr], rescale=False)
        near = NearestNDInterpolator(X[tr], y[tr])
        yp = interp(X[te])
        missing = np.isnan(yp)
        if np.any(missing):
            yp[missing] = near(X[te][missing])
        rms.append(rmse(y[te], yp))
    return float(np.mean(rms))

# ---- Data loading and outlier filtering ----
try:
    df = pd.read_csv(CSV_PATH)[INPUT_COLS + [TARGET]].dropna().reset_index(drop=True)
except Exception as e:
    print(f"[Error] Could not read input data from {CSV_PATH}: {e}")
    exit(1)

iso = IsolationForest(
    contamination=CONTAM,
    n_estimators=300,
    random_state=SEED
)
iso.fit(df[INPUT_COLS + [TARGET]])
mask = iso.predict(df[INPUT_COLS + [TARGET]]) == 1
print(f"Removed {np.sum(~mask)}/{len(df)} rows (IsolationForest @ {int(100*CONTAM)}%)\n")

# ---- PCA outlier visualization ----
try:
    p2 = PCA(n_components=2, random_state=SEED).fit_transform(df[INPUT_COLS + [TARGET]])
    plt.figure(figsize=(5, 4))
    plt.scatter(p2[mask, 0], p2[mask, 1], c="green", s=18, label="inliers")
    plt.scatter(p2[~mask, 0], p2[~mask, 1], c="red", s=32, label="outliers")
    plt.title("Outlier map (PCA 1-2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("[Warning] PCA plot failed (maybe not enough samples?):", e)

df = df[mask].reset_index(drop=True)
X = df[INPUT_COLS].to_numpy(float)
y = df[TARGET].to_numpy(float)

# ---- Model zoo ----
MODELS = {}

for deg in POLY_DEG:
    key = f"poly_deg_{deg}"
    MODELS[key] = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=deg, include_bias=False),
        Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
    )

MODELS["linear_reg"] = make_pipeline(StandardScaler(), LinearRegression())
MODELS["SVR_poly"] = make_pipeline(
    StandardScaler(),
    SVR(kernel="poly", degree=3, C=10, epsilon=0.01)
)
MODELS["random_forest"] = make_pipeline(
    RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
)
MODELS["GB_trees"] = make_pipeline(
    StandardScaler(),
    GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=SEED
    )
)
MODELS["GP_RQ_Lin"] = make_pipeline(
    StandardScaler(),
    GaussianProcessRegressor(
        kernel=C(1.0, (1e-4, 1e4)) *
               RationalQuadratic(1.0, 1.0, length_scale_bounds=(1e-4, 1e4), alpha_bounds=(1e-2, 1e4))
             + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-4, 1e4))
             + WhiteKernel(1e-2, (1e-8, 1e4)),
        n_restarts_optimizer=30,
        random_state=SEED
    )
)

# ---- Evaluate and compare all models ----
results = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("Model Results (RMSE, lower is better):")
    for name, model in MODELS.items():
        try:
            score = rmse_cv(model, X, y)
        except Exception as e:
            print(f"[Error] Evaluating model {name}: {e}")
            score = np.nan
        results.append((name, score, model))
        print(f"  {name:<14s}: {score:.4f} GHz")

# Baseline: ND interpolator
try:
    interp_score = rmse_cv_interp(X, y)
    print(f"Linear ND interp : {interp_score:.4f} GHz\n")
    results.append(("Linear ND interp", interp_score, None))
except Exception as e:
    print("[Warning] Linear ND interpolation failed:", e)

# ---- Save the best serialisable model ----
results_serial = [r for r in results if r[2] is not None and np.isfinite(r[1])]
if not results_serial:
    print("[Error] No valid models to save!")
    exit(2)

best_name, best_rmse, best_model = min(results_serial, key=lambda t: t[1])

try:
    joblib.dump(best_model, BEST_PKL)
    print(f"\nBest model: {best_name}  (RMSE {best_rmse:.4f} GHz)")
    print(f"Model saved to: {BEST_PKL}\n")
except Exception as e:
    print("[Error] Failed to save best model:", e)

# ---- Ranked summary ----
print("Ranked summary:")
for idx, (n, s, _) in enumerate(sorted(results_serial, key=lambda t: t[1])):
    print(f"{idx+1:>2d}. {n:<14s}  {s:.4f} GHz")
