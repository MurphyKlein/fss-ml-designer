"""
scatter_with_fit.py
-------------------
Plots each input feature against an output column and overlays
the best-fit line with R². Choose your own output column up top!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Settings ----
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "results.csv"
OUTPUT_COL = "min_refl_freq"  # <-- Choose target column!
FEATURES = ["width", "height", "spacing", "area", "perim"]
POINT_SIZE = 25
ALPHA_DATA = 0.65   # Transparency for scatter points
ALPHA_LINE = 1.0    # Line opacity


# ---- Load data ----
try:
    df = pd.read_csv(CSV_PATH)
except Exception as err:
    print(f"[Error] Could not read the CSV file at {CSV_PATH}: {err}")
    exit(1)

if OUTPUT_COL not in df.columns:
    print(f"[Error] Output column {OUTPUT_COL!r} not found in the CSV! Check your data.")
    exit(2)


# ---- Loop through features and plot ----
for feat in FEATURES:
    if feat not in df.columns:
        print(f"[Warning] Feature {feat!r} not found in the CSV – skipping.")
        continue

    # Get the x and y data for the plot
    x = df[feat].to_numpy()
    y = df[OUTPUT_COL].to_numpy()

    # ---- Least-squares line fit (polyfit degree 1) ----
    try:
        m, b = np.polyfit(x, y, deg=1)
    except Exception as e:
        print(f"[Warning] Could not fit line for feature {feat}: {e}")
        continue

    y_fit = m * x + b

    # Compute R² (coefficient of determination)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    print(f"{OUTPUT_COL} vs {feat}:  y = {m:.3g}x + {b:.3g},   R^2 = {r2:.3f}")

    # ---- Plotting ----
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=POINT_SIZE, alpha=ALPHA_DATA, label="Data")

    # It's a bit inefficient, but let's sort x for the fit line
    idx_sort = np.argsort(x)
    x_sorted = x[idx_sort]
    y_line = m * x_sorted + b

    plt.plot(x_sorted, y_line, linewidth=2, alpha=ALPHA_LINE,
             label=f"Fit: y = {m:.3g}x + {b:.3g}")

    plt.xlabel(feat)
    plt.ylabel(OUTPUT_COL)
    plt.title(f"{OUTPUT_COL} vs {feat}\n$R^2$ = {r2:.3f}")
    plt.grid(True, linewidth=0.4, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
