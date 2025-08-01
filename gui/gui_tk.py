"""
inverse_design_gui.py
---------------------
Tkinter GUI for interactive inverse FSS design.
Lets user input target values and weights, runs optimisation,
and displays the predicted geometry and uncertainty.
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import sys

# ---- Project import setup ----
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ---- Import solver backend ----
try:
    from reverse.reverse_gui import inverse_design
except ImportError as e:
    print("[Error] Couldn't import inverse_design from reverse.reverse_gui!")
    raise

# ---- Tkinter root window ----
root = tk.Tk()
root.title("FSS Inverse Designer")

# ---- Input frame and controls ----
frm = ttk.Frame(root, padding=12)
frm.grid(row=0, column=0)

def lbl(label, row):
    ttk.Label(frm, text=label).grid(row=row, column=0, sticky="e", pady=2)

def ent(var, row):
    entry = ttk.Entry(frm, textvariable=var, width=10)
    entry.grid(row=row, column=1, pady=2)
    return entry

# ---- User input variables (defaults) ----
var_val = tk.DoubleVar(value=0.85)
var_freq = tk.DoubleVar(value=9.6)
var_wv = tk.DoubleVar(value=1.0)
var_wf = tk.DoubleVar(value=1.0)

# Lay out inputs (explicit, not looped, for clarity)
lbl("Target T_val (0-1):", 0)
ent(var_val, 0)
lbl("Target freq (GHz):", 1)
ent(var_freq, 1)
lbl("Weight on T_val:", 2)
ent(var_wv, 2)
lbl("Weight on freq:", 3)
ent(var_wf, 3)

# ---- Output text area ----
out = tk.Text(root, width=46, height=10, state="disabled", bg="#f6f6f6")
out.grid(row=1, column=0, padx=10, pady=(0, 10))

def write(txt):
    """Write text to output box and make it read-only."""
    out.configure(state="normal")
    out.delete(1.0, "end")
    out.insert("end", txt)
    out.configure(state="disabled")

# ---- Optimisation logic (runs in background thread) ----
def run_opt():
    try:
        val = var_val.get()
        freq = var_freq.get()
        wv = var_wv.get()
        wf = var_wf.get()
        print(f"[Info] Starting optimization: T_val={val}, freq={freq}, w_val={wv}, w_freq={wf}")
        res = inverse_design(val, freq, wv, wf)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to optimize:\n{e}")
        print("[Error] Optimization failed:", e)
        return

    # Build a human-readable message (handles missing fields)
    msg = [
        "Optimal Geometry",
        f"  Width   = {res.get('width', '???')} mm",
        f"  Height  = {res.get('height', '???')} mm",
        f"  Spacing = {res.get('spacing', '???')} mm",
        "",
        "Predicted Outputs",
    ]
    if "val_std" in res and "freq_std" in res:
        msg.append(f"  T_val  = {res.get('T_val', '???'):.3f} ± {res.get('val_std', 0):.3f}")
        msg.append(f"  T_freq = {res.get('T_freq', '???'):.3f} ± {res.get('freq_std', 0):.3f} GHz")
    else:
        msg.append(f"  T_val  = {res.get('T_val', '???'):.3f}")
        msg.append(f"  T_freq = {res.get('T_freq', '???'):.3f} GHz")
    msg.append("")
    msg.append(f"Objective Score = {res.get('score', '???')}")
    write("\n".join(msg))

def start_thread():
    """User feedback before thread starts, runs optimizer in background."""
    write("Optimising... please wait.")
    threading.Thread(target=run_opt, daemon=True).start()

# ---- Run button ----
ttk.Button(frm, text="Run Optimisation", command=start_thread)\
    .grid(row=4, column=0, columnspan=2, pady=8)

# ---- Mainloop (blocks until window closes) ----
root.mainloop()
