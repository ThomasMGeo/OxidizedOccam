#!/usr/bin/env python3
"""Plot MT forward-model response from OxidizedOccam CSV output.

Produces forward_response.png with observed data, starting-model
response, and final inverted-model response plotted as apparent
resistivity and phase vs. period.
"""

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

example_dir = os.path.dirname(os.path.abspath(__file__))

# Load starting model response
start_csv = os.path.join(example_dir, "forward.csv")
if not os.path.exists(start_csv):
    sys.exit(f"Run 'occam1d forward' first to generate {start_csv}")
df_start = pd.read_csv(start_csv)

# Find final iteration CSV
iter_csvs = sorted(glob.glob(os.path.join(example_dir, "iter_????.csv")))
df_final = None
if iter_csvs:
    df_final = pd.read_csv(iter_csvs[-1])
    final_iter = int(re.search(r"iter_(\d+)", iter_csvs[-1]).group(1))

# Separate RhoXY (type 103) and PhiXY (type 104)
rho_start = df_start[df_start["type"] == 103].sort_values("freq_hz")
phi_start = df_start[df_start["type"] == 104].sort_values("freq_hz")

fig, (ax_rho, ax_phi) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Apparent resistivity
period = 1.0 / rho_start["freq_hz"].values
ax_rho.loglog(period, rho_start["observed"].values, "ko", ms=5, label="Observed", zorder=3)
ax_rho.loglog(period, rho_start["computed"].values, "s", color="0.65", ms=4,
              label="Starting model", zorder=1)
if df_final is not None:
    rho_final = df_final[df_final["type"] == 103].sort_values("freq_hz")
    if not rho_final.empty:
        period_f = 1.0 / rho_final["freq_hz"].values
        ax_rho.loglog(period_f, rho_final["computed"].values, "r^", ms=4,
                      label=f"Iter {final_iter}", zorder=2)

ax_rho.set_ylabel("Apparent Resistivity (ohm-m)", fontsize=11)
ax_rho.legend(fontsize=9)
ax_rho.grid(True, which="both", alpha=0.3)
ax_rho.set_title("MT RhoXY", fontsize=12)

# Phase
period_phi = 1.0 / phi_start["freq_hz"].values
ax_phi.semilogx(period_phi, phi_start["observed"].values, "ko", ms=5, label="Observed", zorder=3)
ax_phi.semilogx(period_phi, phi_start["computed"].values, "s", color="0.65", ms=4,
                label="Starting model", zorder=1)
if df_final is not None:
    phi_final = df_final[df_final["type"] == 104].sort_values("freq_hz")
    if not phi_final.empty:
        period_f = 1.0 / phi_final["freq_hz"].values
        ax_phi.semilogx(period_f, phi_final["computed"].values, "r^", ms=4,
                        label=f"Iter {final_iter}", zorder=2)

ax_phi.set_ylabel("Phase (degrees)", fontsize=11)
ax_phi.set_xlabel("Period (s)", fontsize=11)
ax_phi.legend(fontsize=9)
ax_phi.grid(True, which="both", alpha=0.3)
ax_phi.set_title("MT PhiXY", fontsize=12)
ax_phi.set_ylim(0, 90)

rms_start = np.sqrt(np.mean(df_start["residual"].values ** 2))
title = f"MT Forward Response  (starting RMS = {rms_start:.2f}"
if df_final is not None:
    rms_final = np.sqrt(np.mean(df_final["residual"].values ** 2))
    title += f",  final RMS = {rms_final:.2f}"
title += ")"
fig.suptitle(title, fontsize=13, y=1.01)
fig.tight_layout()

out = os.path.join(example_dir, "forward_response.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
