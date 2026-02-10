#!/usr/bin/env python3
"""Plot forward-model response for the tilted-transmitter PE example.

Produces forward_response.png showing observed data, starting-model
response, and final inverted-model response vs Tx-Rx offset for each
data type and frequency.
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
final_iter = None
if iter_csvs:
    df_final = pd.read_csv(iter_csvs[-1])
    final_iter = int(re.search(r"iter_(\d+)", iter_csvs[-1]).group(1))

# Unique frequencies and field components
freqs = sorted(df_start["freq_hz"].unique())
types = sorted(df_start["type_name"].unique())

fig, axes = plt.subplots(len(types), len(freqs),
                         figsize=(5 * len(freqs), 2.8 * len(types)),
                         squeeze=False, sharex=True)

for i, typ in enumerate(types):
    for j, freq in enumerate(freqs):
        ax = axes[i, j]
        sub = df_start[(df_start["type_name"] == typ) &
                       (df_start["freq_hz"] == freq)].sort_values("offset_m")
        if sub.empty:
            ax.set_visible(False)
            continue

        offset_km = sub["offset_m"].values / 1000.0
        obs = sub["observed"].values
        comp_start = sub["computed"].values

        ax.semilogy(offset_km, np.abs(obs), "ko", ms=3,
                    label="Observed", zorder=3)
        ax.semilogy(offset_km, np.abs(comp_start), "-", color="0.65", lw=1,
                    label="Starting model", zorder=1)

        if df_final is not None:
            sub_f = df_final[(df_final["type_name"] == typ) &
                             (df_final["freq_hz"] == freq)].sort_values("offset_m")
            if not sub_f.empty:
                ax.semilogy(sub_f["offset_m"].values / 1000.0,
                            np.abs(sub_f["computed"].values),
                            "r-", lw=1.4, label=f"Iter {final_iter}", zorder=2)

        if i == 0:
            ax.set_title(f"{freq} Hz", fontsize=11)
        if j == 0:
            ax.set_ylabel(f"|{typ}|", fontsize=10)
        if i == len(types) - 1:
            ax.set_xlabel("Offset (km)", fontsize=10)
        if i == 0 and j == len(freqs) - 1:
            ax.legend(fontsize=7, loc="upper right")

        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

rms_start = np.sqrt(np.mean(df_start["residual"].values ** 2))
title = f"Tilted Transmitter (az=70, dip=20)  (starting RMS = {rms_start:.2f}"
if df_final is not None:
    rms_final = np.sqrt(np.mean(df_final["residual"].values ** 2))
    title += f",  final RMS = {rms_final:.2f}"
title += ")"
fig.suptitle(title, fontsize=13, y=1.01)
fig.tight_layout()

out = os.path.join(example_dir, "forward_response.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
