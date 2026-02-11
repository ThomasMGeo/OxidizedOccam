#!/usr/bin/env python3
"""Plot inversion convergence for the tilted-transmitter PE example.

Produces inversion_models.png showing the resistivity model at each
iteration plus an RMS convergence curve.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt

from oxoccam_plot import (
    TRUE_MODELS,
    build_profile,
    make_step_profile,
    read_iter_file,
    read_model_file,
)

example_dir = os.path.dirname(os.path.abspath(__file__))

# ── locate files ────────────────────────────────────────────────────
model_path = os.path.join(example_dir, "model")
if not os.path.exists(model_path):
    sys.exit("model file not found — run generate.py first")

depths, base_resis, free_mask = read_model_file(model_path)

iter_files = sorted(glob.glob(os.path.join(example_dir, "iter_????.iter")))
if not iter_files:
    sys.exit("No iter_NNNN.iter files found — run inversion first")

iterations = []
for fp in iter_files:
    it, misfit, params, _bounds = read_iter_file(fp)
    iterations.append((it, misfit, params))

n_iters = len(iterations)

# ── plot ────────────────────────────────────────────────────────────
fig, (ax_model, ax_rms) = plt.subplots(
    1, 2, figsize=(11, 7), gridspec_kw={"width_ratios": [3, 1.2]}
)

cmap = plt.cm.Purples
for idx, (it, _misfit, params) in enumerate(iterations):
    if it == 0:
        continue
    frac = idx / max(n_iters - 1, 1)
    color = cmap(0.20 + 0.65 * frac)
    lw = 0.6 + 2.0 * frac
    alpha = 0.4 + 0.6 * frac
    final_it = iterations[-1][0]
    show_label = (it == 1) or (it % 5 == 0) or (it == final_it)
    pd_, pr = build_profile(depths, base_resis, free_mask, params)
    ax_model.plot(
        pr, pd_, color=color, lw=lw, alpha=alpha, label=f"Iter {it}" if show_label else None
    )

true = TRUE_MODELS["canonical_reservoir"]
td, tr = make_step_profile(true.depths, true.resistivities, true.bottom_extension)
ax_model.plot(tr, td, "r--", lw=2.0, label="True model", zorder=10)

ax_model.set_xscale("log")
ax_model.set_xlabel("Resistivity (ohm-m)", fontsize=11)
ax_model.set_ylabel("Depth (m)", fontsize=11)
ax_model.set_ylim(4500, 800)
ax_model.set_xlim(0.1, 300)
ax_model.legend(fontsize=9, loc="lower left")
ax_model.set_title("Resistivity Models", fontsize=12)
ax_model.grid(True, which="both", alpha=0.3)

its = [t[0] for t in iterations]
rms = [t[1] for t in iterations]
ax_rms.plot(its, rms, "ko-", ms=4, lw=1.2)
ax_rms.axhline(1.0, color="r", ls="--", lw=1, label="Target RMS")
ax_rms.set_xlabel("Iteration", fontsize=11)
ax_rms.set_ylabel("RMS Misfit", fontsize=11)
ax_rms.set_title("Convergence", fontsize=12)
ax_rms.legend(fontsize=9)
ax_rms.set_ylim(bottom=0)
ax_rms.grid(True, alpha=0.3)

fig.suptitle("OxidizedOccam — Tilted Transmitter PE Example", fontsize=14, y=1.01)
fig.tight_layout()

out = os.path.join(example_dir, "inversion_models.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
