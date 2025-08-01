"""
inverse_design.py
-----------------
Given a target transmission value and frequency, finds an optimal geometry using a
Gaussian Process surrogate model.

Usage:
    python inverse_design.py --target_val 0.85 --target_freq 9.6
"""

import argparse
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RationalQuadratic, ConstantKernel as C, DotProduct, WhiteKernel
)

# ---- Paths and project config ----
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
CSV_PATH = DATA_DIR / "results.csv"
GP_PATH = MODEL_DIR / "gpr_active.pkl"
CONTAM = 0.05
SEED = 42
N_SPLITS = 5

BOUNDS = np.array([
    [2.0, 6.0],     # width bounds
    [5.0, 14.0],    # height bounds
    [5.0, 14.0],    # spacing bounds
])

INPUT_COLS = ["width", "height", "spacing"]
TARGET = ["max_trans_val", "max_trans_freq"]


# ---- Train the surrogate model if not already present ----
def train_quick_gp():
    print("⚠ Surrogate model gpr_active.pkl not found — training a new one...")

    try:
        df = pd.read_csv(CSV_PATH)[INPUT_COLS + TARGET].dropna().reset_index()
    except Exception as e:
        print(f"[Error] Could not load data from {CSV_PATH}: {e}")
        raise

    # Outlier removal
    iso = IsolationForest(
        contamination=CONTAM,
        n_estimators=300,
        random_state=SEED
    ).fit(df[INPUT_COLS + TARGET])
    mask = iso.predict(df[INPUT_COLS + TARGET]) == 1
    print(f"[Info] Removed {np.sum(~mask)}/{len(df)} rows (IsolationForest)")

    df = df[mask].reset_index(drop=True)
    X = df[INPUT_COLS].to_numpy(float)
    Y = df[TARGET].to_numpy(float)

    xs, ys = StandardScaler(), StandardScaler()
    Xn = xs.fit_transform(X)
    Yn = ys.fit_transform(Y)

    # Kernel: rational quadratic + dot product + white noise
    kernel = (
        C(1.0, (1e-4, 1e4))
        * RationalQuadratic(1.0, 1.0, length_scale_bounds=(1e-4, 1e4), alpha_bounds=(1e-4, 1e4))
        + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-4, 1e4))
        + WhiteKernel(1e-2, (1e-8, 1e4))
    )

    # Cross-validate and print RMSEs for each target
    rmses = []
    for i, col in enumerate(TARGET):
        print(f"[Info] Training GPR for target {col}...")
        model = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(kernel, n_restarts_optimizer=30, random_state=SEED)
        )
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        neg_mse = cross_val_score(
            model, X, Y[:, i],
            scoring="neg_mean_squared_error",
            cv=cv, n_jobs=-1
        )
        this_rmse = np.sqrt(-neg_mse.mean())
        rmses.append(this_rmse)
        print(f"  RMSE for target {col}: {this_rmse:.4f}")

    # Fit GPR models on all data (for deployment)
    gprs = []
    for i in range(Yn.shape[1]):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            random_state=SEED,
            n_restarts_optimizer=30
        )
        gp.fit(Xn, Yn[:, i])
        gprs.append(gp)

    # Save all to disk for future runs
    try:
        joblib.dump({"gprs": gprs, "Xs": xs, "ys": ys}, GP_PATH)
        print(f"[Info] Surrogate model saved to {GP_PATH}")
    except Exception as e:
        print(f"[Error] Could not save model: {e}")
        raise

    return gprs, xs, ys


# ---- Load (or train) the model ----
if GP_PATH.exists():
    try:
        model_bundle = joblib.load(GP_PATH)
        gprs, xs, ys = model_bundle["gprs"], model_bundle["Xs"], model_bundle["ys"]
    except Exception as e:
        print(f"[Error] Loading model from {GP_PATH}: {e}")
        raise
else:
    gprs, xs, ys = train_quick_gp()


# ---- The objective function for the optimizer ----
def obj(x, targ_val, targ_freq, w_val=1.0, w_freq=1.0):
    w, h, s = x
    feats = np.array([[w, h, s]])
    feats_n = xs.transform(feats)
    preds = [g.predict(feats_n) for g in gprs]
    val_hat, freq_hat = ys.inverse_transform(np.column_stack(preds))[0]
    return w_val * (val_hat - targ_val) ** 2 + w_freq * (freq_hat - targ_freq) ** 2


# ---- Command-line interface parsing ----
parser = argparse.ArgumentParser()
parser.add_argument("--target_val", type=float, required=True,
    help="Target max-transmission value (normalized 0–1)")
parser.add_argument("--target_freq", type=float, required=True,
    help="Target resonance frequency (GHz)")
parser.add_argument("--w_val", type=float, default=1.0,
    help="Weight on transmission value term")
parser.add_argument("--w_freq", type=float, default=1.0,
    help="Weight on frequency term")
args = parser.parse_args()


# ---- Run the optimizer ----
try:
    result = differential_evolution(
        func=obj,
        bounds=BOUNDS,
        args=(args.target_val, args.target_freq, args.w_val, args.w_freq),
        strategy='best1bin', popsize=20, tol=1e-4, seed=1, polish=True, disp=False
    )
except Exception as e:
    print(f"[Error] Optimization failed: {e}")
    raise

w_opt, h_opt, s_opt = result.x
feats_opt = xs.transform([[w_opt, h_opt, s_opt]])
results = [g.predict(feats_opt, return_std=True) for g in gprs]
means = np.array([r[0][0] for r in results])
stds = np.array([r[1][0] for r in results]) * ys.scale_
val_opt, freq_opt = ys.inverse_transform(means.reshape(1, -1))[0]
val_std, freq_std = stds

# ---- Output block ----
print("\nOptimal Geometry:")
print(f"  Width   = {w_opt:.3f} mm")
print(f"  Height  = {h_opt:.3f} mm")
print(f"  Spacing = {s_opt:.3f} mm")

print("\nPredicted Outputs:")
print(f"  Transmission Value  = {val_opt:.3f} ± {val_std:.3f} (target {args.target_val:.3f})")
print(f"  Resonance Frequency = {freq_opt:.3f} ± {freq_std:.3f} GHz (target {args.target_freq:.3f})")

print(f"\nObjective Score: {result.fun:.6f}")
