#!/usr/bin/env python3
"""
Plotting script for VFI outputs produced by src/vfi_dump.cpp.

Usage:
  python3 scripts/plot_vfi.py output_dir

Produces PNGs in the output_dir:
  - value_function.png
  - policy_function.png
  - convergence.png
  - euler_residuals.png

Depends on: matplotlib, pandas, numpy
  pip install -r requirements.txt
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: plot_vfi.py <output_dir>")
    sys.exit(1)

outdir = sys.argv[1]

final_csv = os.path.join(outdir, "vfi_final.csv")
euler_csv = os.path.join(outdir, "vfi_euler.csv")
snap_glob = os.path.join(outdir, "vfi_iter_*.csv")

if not os.path.exists(final_csv):
    print("Cannot find", final_csv)
    sys.exit(1)

df = pd.read_csv(final_csv)
i = df["i"].to_numpy()
K = df["K"].to_numpy()
V = df["V"].to_numpy()
Kp = df["Kp"].to_numpy()

# Policy function plot K' vs K
plt.figure(figsize=(8,6))
plt.plot(K, Kp, label="Policy K'(K)")
plt.plot(K, K, label="45-degree (K'=K)", linestyle='--')
plt.xlabel("K")
plt.ylabel("K'")
plt.title("Policy function")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "policy_function.png"), dpi=200)
print("Saved", os.path.join(outdir, "policy_function.png"))

# Value function plot
plt.figure(figsize=(8,6))
plt.plot(K, V, label="Value V(K)")
plt.xlabel("K")
plt.ylabel("V")
plt.title("Value function")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "value_function.png"), dpi=200)
print("Saved", os.path.join(outdir, "value_function.png"))

# Convergence: overlay a few V snapshots if available
snap_files = sorted(glob.glob(snap_glob))
if snap_files:
    plt.figure(figsize=(8,6))
    # pick up to 12 snapshots spanning the run
    sample = snap_files[::max(1, len(snap_files)//12)]
    for path in sample:
        sdf = pd.read_csv(path)
        it_str = os.path.basename(path).replace("vfi_iter_", "").replace(".csv", "")
        plt.plot(sdf["K"], sdf["V"], alpha=0.6, label=f"iter {it_str}")
    # final
    final_snap = os.path.join(outdir, "vfi_iter_final.csv")
    if os.path.exists(final_snap):
        sdf = pd.read_csv(final_snap)
        plt.plot(sdf["K"], sdf["V"], color='k', linewidth=2.0, label="final")
    plt.xlabel("K")
    plt.ylabel("V")
    plt.title("VFI convergence (V over iterations)")
    plt.legend(fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "convergence.png"), dpi=200)
    print("Saved", os.path.join(outdir, "convergence.png"))
else:
    print("No snapshots found to plot convergence (expected files vfi_iter_*.csv).")

# Euler residuals plot
if os.path.exists(euler_csv):
    edf = pd.read_csv(euler_csv)
    # resid may contain NaNs
    resid = edf["euler_resid"].to_numpy()
    mask = np.isfinite(resid)
    plt.figure(figsize=(8,6))
    plt.plot(edf["K"][mask], np.abs(resid[mask]), label="|Euler residual|")
    plt.yscale('log')
    plt.xlabel("K")
    plt.ylabel("|resid| (log scale)")
    plt.title("Euler residuals across grid (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "euler_residuals.png"), dpi=200)
    print("Saved", os.path.join(outdir, "euler_residuals.png"))
else:
    print("No euler residuals file found at", euler_csv)