"""
simulate_batch.py
-----------------
Runs HFSS for all geometries in results.csv that still need output.
Saves updated results after batch.
"""

import sys
from pathlib import Path
import pandas as pd

# ---- Project import setup ----
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ---- HFSS/PyAEDT import ----
try:
    from ansys.aedt.core import Hfss
except ImportError:
    print("[Error] Could not import Hfss from ansys.aedt.core. Make sure AEDT Python modules are installed.")
    raise

# ---- Paths and directories ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
if not DATA_DIR.exists():
    DATA_DIR.mkdir()
RESULTS_CSV = DATA_DIR / "results.csv"
EXPORT_DIR = DATA_DIR / "exports"
if not EXPORT_DIR.exists():
    EXPORT_DIR.mkdir()

# ---- HFSS project settings ----
AEDT_FILE = r"C:/Users/mdklein/ANSYS/New Project Default/FSS.aedt"
DESIGN_NAME = "FSS Large"
SETUP_NAME = "Setup1"
SWEEP_NAME = "Sweep1"  # or "LastAdaptive"

OUTPUT_COLS = [
    "min_refl_val", "min_refl_freq",
    "max_trans_val", "max_trans_freq"
]


def run_batch():
    """
    Runs HFSS on all geometries with missing output, updates CSV, returns count.
    """
    if not RESULTS_CSV.exists():
        print("[Warning] results.csv not found. Nothing to simulate!")
        return 0

    try:
        df = pd.read_csv(RESULTS_CSV)
    except Exception as e:
        print(f"[Error] Failed to read {RESULTS_CSV}: {e}")
        return 0

    # Ensure output columns exist (if missing, create as NA)
    for col in OUTPUT_COLS:
        if col not in df.columns:
            print(f"[Info] Output column '{col}' missing, adding...")
            df[col] = pd.NA

    df[OUTPUT_COLS] = df[OUTPUT_COLS].astype("float64")

    # Find the rows that need to be simulated (at least one missing output)
    pending = df[df[OUTPUT_COLS].isna().any(axis=1)].copy()
    if pending.empty:
        print("[Info] No pending geometries — HFSS run skipped.")
        return 0

    print(f"[Info] HFSS batch: {len(pending)} geometries to solve")

    try:
        with Hfss(
            project=AEDT_FILE,
            design=DESIGN_NAME,
            version="2023.2",
            non_graphical=False,   # Set to True for server/batch runs
            new_desktop=True,
            close_on_exit=True
        ) as app:

            app.active_setup = SETUP_NAME

            for idx, row in pending.iterrows():
                w = row["width"]
                h = row["height"]
                s = row["spacing"]
                print(f"   • idx {idx}: w={w} h={h} s={s}")

                app["$codew"] = f"{w}mm"
                app["$codeh"] = f"{h}mm"
                app["$codes"] = f"{s}mm"

                ok_solve = app.analyze_setup(SETUP_NAME)
                if not ok_solve:
                    print("     ✖ HFSS solve failed for this geometry.")
                    continue

                ts_file = EXPORT_DIR / f"FSS_w{w}_h{h}_s{s}.s20p"
                ok_export = app.export_touchstone(
                    setup=SETUP_NAME,
                    sweep=SWEEP_NAME,
                    output_file=str(ts_file),
                    renormalization=False,
                    gamma_impedance_comments=True
                )
                if not ok_export or not ts_file.exists():
                    print("     ✖ Touchstone export failed!")
                    continue

                try:
                    from data_analysis.touchanalysis import analysis
                    vec = analysis(str(ts_file))
                except Exception as e:
                    print(f"     ✖ Data analysis failed: {e}")
                    continue

                df.loc[idx, OUTPUT_COLS] = vec
                print("     ✔ done")

            app.save_project()

    except Exception as e:
        print(f"[Error] HFSS batch run failed: {e}")
        return 0

    try:
        df.to_csv(RESULTS_CSV, index=False)
        print("[Info] Batch finished and CSV updated.")
    except Exception as e:
        print(f"[Error] Failed to save updated CSV: {e}")
        return 0

    return len(pending)


if __name__ == "__main__":
    n = run_batch()
    print(f"[Info] Simulated {n} geometries in this batch.")
