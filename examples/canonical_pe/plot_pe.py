#!/usr/bin/env python3
"""Plot polarization ellipse parameters for the tilted-transmitter example.

With a 20° off-inline, 20° dip transmitter the horizontal E and B fields
are genuinely elliptically polarized, so PEmin and PBmin are non-zero.

Produces pe_response.png with PEmax/PEmin and PBmax/PBmin vs offset.
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
    sys.exit(f"Run 'occam1d forward --startup startup' first")
df = pd.read_csv(start_csv)

# Find final iteration CSV
iter_csvs = sorted(glob.glob(os.path.join(example_dir, "iter_????.csv")))
df_final = None
final_iter = None
if iter_csvs:
    df_final = pd.read_csv(iter_csvs[-1])
    final_iter = int(re.search(r"iter_(\d+)", iter_csvs[-1]).group(1))

freqs = sorted(df["freq_hz"].unique())

# PE types: (code, name, row_label)
pe_types = [
    (41, "PEmax", "|PEmax| (E-field major axis)"),
    (42, "PEmin", "|PEmin| (E-field minor axis)"),
    (43, "PBmax", "|PBmax| (B-field major axis)"),
    (44, "PBmin", "|PBmin| (B-field minor axis)"),
]

fig, axes = plt.subplots(len(pe_types), len(freqs),
                         figsize=(5.5 * len(freqs), 3.2 * len(pe_types)),
                         squeeze=False, sharex=True)

for i, (code, name, ylabel) in enumerate(pe_types):
    for j, freq in enumerate(freqs):
        ax = axes[i, j]
        sub = df[(df["type"] == code) &
                 (df["freq_hz"] == freq)].sort_values("offset_m")
        if sub.empty:
            ax.set_visible(False)
            continue

        offset_km = sub["offset_m"].values / 1000.0
        obs = np.abs(sub["observed"].values)
        comp = np.abs(sub["computed"].values)

        ax.semilogy(offset_km, obs, "ko", ms=3, label="Observed", zorder=3)
        ax.semilogy(offset_km, comp, "-", color="0.65", lw=1.2,
                    label="Starting model", zorder=1)

        if df_final is not None:
            sub_f = df_final[(df_final["type"] == code) &
                             (df_final["freq_hz"] == freq)].sort_values("offset_m")
            if not sub_f.empty:
                ax.semilogy(sub_f["offset_m"].values / 1000.0,
                            np.abs(sub_f["computed"].values),
                            "r-", lw=1.6, label=f"Iter {final_iter}", zorder=2)

        if i == 0:
            ax.set_title(f"{freq} Hz", fontsize=11)
        if j == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if i == len(pe_types) - 1:
            ax.set_xlabel("Offset (km)", fontsize=10)
        if i == 0 and j == len(freqs) - 1:
            ax.legend(fontsize=8, loc="upper right")

        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

# Also compute ellipticity ratio (PEmin/PEmax) if data available
pe_max_sub = df[df["type"] == 41]
pe_min_sub = df[df["type"] == 42]

rms_start = np.sqrt(np.mean(df[df["type"].isin([41, 42, 43, 44])]["residual"].values ** 2))
title = f"Polarization Ellipse — Tilted Tx (az=70°, dip=20°)  (PE RMS = {rms_start:.2f}"
if df_final is not None:
    pe_final = df_final[df_final["type"].isin([41, 42, 43, 44])]
    if not pe_final.empty:
        rms_final = np.sqrt(np.mean(pe_final["residual"].values ** 2))
        title += f",  final = {rms_final:.2f}"
title += ")"
fig.suptitle(title, fontsize=12, y=1.01)
fig.tight_layout()

out = os.path.join(example_dir, "pe_response.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
