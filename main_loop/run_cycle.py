"""
run_cycle.py
------------
Master loop: alternates between active_fit.step() and simulate_batch.run_batch().
Continues until surrogate model converges.
"""

import sys
from pathlib import Path

# ---- Project import setup ----
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ---- Import main workflow modules ----
try:
    from main_loop.active_fit import step
    from main_loop.simulate_batch import run_batch
except ImportError as e:
    print("[Error] Could not import step/run_batch from main_loop! Check your PYTHONPATH and directory structure.")
    raise


def main():
    """
    Master loop for alternating between active learning/model update and HFSS simulation batch.
    Keeps running until surrogate model converges (as reported by active_fit.step()).
    """
    print("=== FSS Active Learning Loop Starting ===")
    cycle_count = 1

    while True:
        print(f"\n=== Cycle {cycle_count} ===")

        # Update/train the surrogate model
        try:
            status = step()
        except Exception as e:
            print(f"[Error] during step(): {e}")
            break

        if status == "converged":
            print("🎉 Finished – surrogate model ready!")
            break
        elif status == "error":
            print("[Error] Something went wrong in step(), exiting loop.")
            break
        elif status == "need_sim":
            print("[Info] New geometries require simulation.")

        # Run a batch of simulations
        try:
            n_sims = run_batch()
            print(f"[Info] Simulated {n_sims} geometries in this batch.")
        except Exception as e:
            print(f"[Error] during run_batch(): {e}")
            break

        cycle_count += 1

    print("\n=== Loop Complete ===")


if __name__ == "__main__":
    main()
