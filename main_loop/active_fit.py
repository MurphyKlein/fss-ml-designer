"""
active_fit.py
-------------
Active learning driver: maintains the loop for sampling, surrogate fitting,
and uncertainty-driven candidate selection for FSS optimization.
Manages convergence by monitoring RMSE drop history.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import qmc
import sys

# ---- Project path setup ----
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ---- Project folders and files ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
if not DATA_DIR.exists():
    DATA_DIR.mkdir()
MODEL_DIR = BASE_DIR / "models"
if not MODEL_DIR.exists():
    MODEL_DIR.mkdir()
RESULTS_CSV = DATA_DIR / "results.csv"

# ---- Convergence criteria & history tracking ----
RMSE_DROP_TOL = 0.02       # Stop if RMSE drops less than 2%
CONSECUTIVE_DROPS = 3      # ...for this many steps
HISTORY_FILE = MODEL_DIR / "rmse_history.npy"

INIT_SAMPLES = 60
BATCH_SIZE = 10
MAX_SAMPLES = 200
RANDOM_SEED = 42

BOUNDS = {
    "width": (2.0, 6.0),
    "height": (5.0, 14.0),
    "spacing": (5.0, 14.0),
}


def is_valid(pt):
    """Return True if geometry is admissible. Placeholder for user checks."""
    # For now, everything is valid; update if you want physical limits!
    return True


rng = np.random.default_rng(RANDOM_SEED)

# ---- Try to load prior RMSE history, or start fresh ----
try:
    if HISTORY_FILE.exists():
        _history = list(np.load(HISTORY_FILE))
    else:
        _history = []
except Exception as e:
    print("[Warning] Couldn't load RMSE history, starting fresh. Details:", e)
    _history = []


def latin_hypercube(n):
    """
    Latin hypercube sampling of n geometries in the design bounds.
    Returns array shape (n, 3).
    """
    mins = np.array([v[0] for v in BOUNDS.values()])
    maxs = np.array([v[1] for v in BOUNDS.values()])
    sampler = qmc.LatinHypercube(d=3, seed=RANDOM_SEED)
    pts = []
    while len(pts) < n:
        cand = mins + sampler.random(n) * (maxs - mins)
        for row in cand:
            if is_valid(row):
                pts.append(row)
            if len(pts) == n:
                break
    return np.array(pts)


def has_k_small_drops(hist, tol, k):
    """
    Check if last k drops in RMSE were all 'small' (i.e., below tol fraction).
    """
    if len(hist) < k + 1:
        return False
    for i in range(-k, 0):
        prev, curr = hist[i - 1], hist[i]
        if (prev - curr) / prev >= tol:
            return False
    return True


def step():
    """
    Main active learning step.
    Seeds initial geometries if needed, then fits surrogates and selects a new batch.
    Returns a string status: 'need_sim', 'converged', or 'error'.
    """
    # Seed if needed
    if not RESULTS_CSV.exists():
        pts = latin_hypercube(INIT_SAMPLES)
        seed_df = pd.DataFrame(pts, columns=["width", "height", "spacing"])
        seed_df["area"] = seed_df["width"] * seed_df["height"] / 2
        seed_df["perim"] = seed_df["width"] + 2 * np.sqrt(
            (seed_df["width"] / 2) ** 2 + seed_df["height"] ** 2
        )
        seed_df.to_csv(RESULTS_CSV, index=False)
        print(f"Seeded {INIT_SAMPLES} geometries to {RESULTS_CSV} — run HFSS next.")
        return "need_sim"

    # Load data so far
    try:
        df = pd.read_csv(RESULTS_CSV)
    except Exception as e:
        print(f"[Error] Failed to read {RESULTS_CSV}: {e}")
        return "error"

    # If results are incomplete, stop here and wait for new simulations
    if df.shape[1] < 9 or df.iloc[:, 5:].isna().any().any():
        print("[Info] Not enough results yet, waiting for more simulations...")
        return "need_sim"
    if len(df) < 2:
        print("[Info] Need at least 2 completed simulations before training.")
        return "need_sim"

    # Extract features/targets
    X = df[["width", "height", "spacing", "area", "perim"]].values
    y = df.iloc[:, 5:].values
    xs, ys = StandardScaler(), StandardScaler()
    Xn = xs.fit_transform(X)
    yn = ys.fit_transform(y)

    # Build kernel
    kernel = (
        C(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-6, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    )

    rmses_scaled = []
    gprs = []
    # Fit one GPR per output column
    for i in range(yn.shape[1]):
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        rmse_s = np.sqrt(
            -cross_val_score(
                gpr, Xn, yn[:, i],
                cv=min(5, len(Xn)),
                scoring="neg_mean_squared_error"
            ).mean()
        )
        rmses_scaled.append(rmse_s)
        gpr.fit(Xn, yn[:, i])
        gprs.append(gpr)

    # Convert RMSEs back to original units (per target column)
    rmses_unscaled = np.array(rmses_scaled) * ys.scale_
    rmse_scaled = np.mean(rmses_scaled)
    rmse_unscaled = np.mean(rmses_unscaled)

    # Track convergence history for stopping
    _history.append(rmse_scaled)
    try:
        np.save(HISTORY_FILE, np.array(_history))
    except Exception as e:
        print("[Warning] Failed to save history:", e)

    print(
        f"[{len(df):>3}] RMSE (scaled)={rmse_scaled:.4f} | "
        f"RMSE (orig)={rmse_unscaled:.4f} | "
        f"Per-col orig RMSEs={np.round(rmses_unscaled, 4)}"
    )

    # Check for convergence
    if has_k_small_drops(_history, RMSE_DROP_TOL, CONSECUTIVE_DROPS) or len(df) >= MAX_SAMPLES:
        print(
            f"Converged! (last {CONSECUTIVE_DROPS} drops < {RMSE_DROP_TOL:.0%}) — "
            f"model saved as gpr_active.pkl"
        )
        try:
            joblib.dump({"gprs": gprs, "Xs": xs, "ys": ys},
                        MODEL_DIR / "gpr_active.pkl")
        except Exception as e:
            print("[Error] Failed to save model:", e)
            return "error"
        return "converged"

    # ---- Select new candidates (batch) ----
    mins = np.array([v[0] for v in BOUNDS.values()])
    maxs = np.array([v[1] for v in BOUNDS.values()])
    CAND_POOL = 5000

    # Generate candidate geometries randomly within bounds
    raw = mins + rng.random((CAND_POOL, 3)) * (maxs - mins)
    cand = np.array([row for row in raw if is_valid(row)])

    # Compute derived features (area, perim)
    widths, heights = cand[:, 0], cand[:, 1]
    areas = widths * heights / 2
    perims = widths + 2 * np.sqrt((widths / 2) ** 2 + heights ** 2)
    cand = np.hstack([cand, areas[:, None], perims[:, None]])

    # Use the GPRs to estimate standard deviations at each candidate
    Xsamp = xs.transform(cand)
    stds = np.column_stack([
        gpr.predict(Xsamp, return_std=True)[1] for gpr in gprs
    ])
    finalstd = stds.sum(axis=1)
    sorted_indices = np.argsort(-finalstd)  # descending order

    # Exploit-explore tradeoff
    epsilon = 0.6
    n_exploit = int(BATCH_SIZE * (1 - epsilon))
    n_explore = BATCH_SIZE - n_exploit

    idx_sigma = sorted_indices[:n_exploit]
    pts_sigma = cand[idx_sigma]

    mask = np.ones(len(cand), dtype=bool)
    mask[idx_sigma] = False
    idx_random = rng.choice(np.where(mask)[0], size=n_explore, replace=False)
    pts_random = cand[idx_random]

    # Combine all new points
    new_pts = np.vstack([pts_sigma, pts_random])
    df_new = pd.DataFrame(new_pts, columns=["width", "height", "spacing", "area", "perim"])
    df = pd.concat([df, df_new], ignore_index=True).reset_index(drop=True)
    try:
        df.to_csv(RESULTS_CSV, index=False)
    except Exception as e:
        print("[Error] Failed to save updated CSV:", e)
        return "error"

    print(f"Added {BATCH_SIZE} new geometries ({n_exploit} hi-σ, {n_explore} random) — run HFSS next.")
    return "need_sim"
