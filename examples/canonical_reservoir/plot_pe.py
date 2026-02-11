#!/usr/bin/env python3
"""Plot polarization ellipse (PEmax) forward response.

Reads pe_forward.csv (starting model) and the final iteration CSV
from the original inversion, reconstructs PEmax from EyReal/EyImag,
and plots PEmax vs Tx-Rx offset for each frequency.

Produces pe_response.png.
"""

import glob
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

example_dir = os.path.dirname(os.path.abspath(__file__))

# ── Load PE forward CSV (starting model with PEmax data) ────────────
pe_csv = os.path.join(example_dir, "pe_forward.csv")
if not os.path.exists(pe_csv):
    sys.exit("Run 'occam1d forward --startup startup_pe --output pe_forward' first")
df_pe = pd.read_csv(pe_csv)

# ── Reconstruct PEmax from final inversion iteration ────────────────
iter_csvs = sorted(glob.glob(os.path.join(example_dir, "iter_????.csv")))
df_final_pe = None
final_iter = None
if iter_csvs:
    df_final = pd.read_csv(iter_csvs[-1])
    final_iter = int(re.search(r"iter_(\d+)", iter_csvs[-1]).group(1))

    ey_real = df_final[df_final["type"] == 3].set_index(["freq_hz", "tx_idx", "rx_idx"])
    ey_imag = df_final[df_final["type"] == 4].set_index(["freq_hz", "tx_idx", "rx_idx"])

    common = ey_real.index.intersection(ey_imag.index)
    rows = []
    for key in common:
        r = ey_real.loc[key]
        im = ey_imag.loc[key]
        pe_comp = math.sqrt(r["computed"] ** 2 + im["computed"] ** 2)
        pe_obs = math.sqrt(r["observed"] ** 2 + im["observed"] ** 2)
        rows.append(
            {
                "freq_hz": key[0],
                "offset_m": r["offset_m"],
                "pe_computed": pe_comp,
                "pe_observed": pe_obs,
            }
        )
    df_final_pe = pd.DataFrame(rows)

# ── Filter to PEmax rows from PE forward CSV ────────────────────────
df_pmax = df_pe[df_pe["type"] == 41].copy()
freqs = sorted(df_pmax["freq_hz"].unique())

# ── Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, len(freqs), figsize=(5.5 * len(freqs), 4.2), squeeze=False, sharey=True
)

for j, freq in enumerate(freqs):
    ax = axes[0, j]
    sub = df_pmax[df_pmax["freq_hz"] == freq].sort_values("offset_m")
    if sub.empty:
        continue

    offset_km = sub["offset_m"].values / 1000.0
    obs = np.abs(sub["observed"].values)
    comp = np.abs(sub["computed"].values)

    ax.semilogy(offset_km, obs, "ko", ms=4, label="Observed", zorder=3)
    ax.semilogy(offset_km, comp, "-", color="0.65", lw=1.2, label="Starting model", zorder=1)

    if df_final_pe is not None:
        sub_f = df_final_pe[df_final_pe["freq_hz"] == freq].sort_values("offset_m")
        if not sub_f.empty:
            ax.semilogy(
                sub_f["offset_m"].values / 1000.0,
                np.abs(sub_f["pe_computed"].values),
                "r-",
                lw=1.6,
                label=f"Iter {final_iter}",
                zorder=2,
            )

    ax.set_title(f"{freq} Hz", fontsize=11)
    ax.set_xlabel("Offset (km)", fontsize=10)
    if j == 0:
        ax.set_ylabel("|PEmax|  (V/m per Am)", fontsize=10)
    if j == len(freqs) - 1:
        ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)

rms_start = np.sqrt(np.mean(df_pmax["residual"].values ** 2))
title = f"Polarization Ellipse Major Axis  (starting RMS = {rms_start:.2f}"
if final_iter is not None:
    title += f",  final iter = {final_iter}"
title += ")"
fig.suptitle(title, fontsize=12, y=1.02)
fig.tight_layout()

out = os.path.join(example_dir, "pe_response.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
