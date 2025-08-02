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
    from optimization.reverse_gui import inverse_design
except ImportError as e:
    print("[Error] Couldn't import inverse_design from reverse.reverse_gui!")
    raise

# Tk root‑window
root = tk.Tk()
root.title("FSS Inverse Designer")

# Make the single column/row of the toplevel stretchy
root.columnconfigure(0, weight=1)     # stretch in X
root.rowconfigure(1, weight=1)        # stretch the Text widget in Y

# ---- Input frame (row 0) ----
frm = ttk.Frame(root, padding=12)
frm.grid(row=0, column=0, sticky="ew")        # stretch horizontally
frm.columnconfigure(1, weight=1)              # col‑1 (entries) expands

def lbl(text, row):
    ttk.Label(frm, text=text).grid(row=row, column=0, sticky="e", pady=2)

def ent(var, row):
    e = ttk.Entry(frm, textvariable=var)
    e.grid(row=row, column=1, sticky="ew", pady=2)   # expand with window
    return e

# ---- User‑input variables & widgets ----
var_val  = tk.DoubleVar(value=0.85)
var_freq = tk.DoubleVar(value=9.6)
var_wv   = tk.DoubleVar(value=1.0)
var_wf   = tk.DoubleVar(value=1.0)

lbl("Target T_val (0‑1):", 0); ent(var_val, 0)
lbl("Target freq (GHz):", 1); ent(var_freq, 1)
lbl("Weight on T_val:",    2); ent(var_wv,   2)
lbl("Weight on freq:",     3); ent(var_wf,   3)

# ---- Run button (still inside frm) ----
ttk.Button(frm, text="Run Optimisation", command=lambda: threading.Thread(target=run_opt, daemon=True).start())\
    .grid(row=4, column=0, columnspan=2, pady=(8, 0), sticky="ew")

# ---- Output text area (row 1) ----
out = tk.Text(root, width=46, height=10, state="disabled", bg="#f6f6f6", wrap="word")
out.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")  # expand both ways


# ---- Helper functions ----
def write(txt: str):
    out.configure(state="normal")
    out.delete(1.0, "end")
    out.insert("end", txt)
    out.configure(state="disabled")

def run_opt():
    """
    Run the inverse design. Output the geometry.
    """
    try:
        res = inverse_design(var_val.get(), var_freq.get(), var_wv.get(), var_wf.get())
    except Exception as e:
        messagebox.showerror("Error", f"Failed to optimise:\n{e}")
        return

    msg = [
        "Optimal Geometry",
        f"  Width   = {res.get('width',  '???')} mm",
        f"  Height  = {res.get('height', '???')} mm",
        f"  Spacing = {res.get('spacing','???')} mm",
        "",
        "Predicted Outputs"
    ]
    if {"val_std", "freq_std"} <= res.keys():
        msg.append(f"  T_val  = {res['T_val']:.3f} ± {res['val_std']:.3f}")
        msg.append(f"  T_freq = {res['T_freq']:.3f} ± {res['freq_std']:.3f} GHz")
    else:
        msg.append(f"  T_val  = {res.get('T_val',  '???'):.3f}")
        msg.append(f"  T_freq = {res.get('T_freq', '???'):.3f} GHz")
    msg += ["", f"Objective Score = {res.get('score','???')}"]

    write("\n".join(msg))

# -- Start GUI --
root.mainloop()
