#!/usr/bin/env python3
"""Plot inversion convergence from OxidizedOccam iteration files.

Produces inversion_models.png showing the resistivity model at each
iteration, with later iterations drawn darker and thicker so the
final model stands out.  The true reservoir model is shown for
reference.
"""

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

example_dir = os.path.dirname(__file__)

# ── helpers ──────────────────────────────────────────────────────────

def read_model_file(path):
    """Return (depths, resistivities, is_free) arrays from model file."""
    depths, resis, free = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Format") or line.startswith("#") or line.startswith("!"):
                continue
            parts = line.split("!")  # strip inline comment
            tokens = parts[0].split()
            if len(tokens) < 2:
                continue
            d = float(tokens[0].replace("d", "e").replace("D", "E"))
            r = float(tokens[1].replace("d", "e").replace("D", "E"))
            depths.append(d)
            resis.append(r)
            free.append(r == -1)
    return np.array(depths), np.array(resis), np.array(free)


def read_iter_file(path):
    """Return (iteration, misfit, params) from an iteration file."""
    iteration, misfit = 0, 0.0
    params = []
    in_params = False
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            low = stripped.lower()
            if low.startswith("iteration:"):
                iteration = int(stripped.split(":")[1].strip())
            elif low.startswith("misfit value:"):
                misfit = float(stripped.split(":")[1].strip().replace("d", "e").replace("D", "E"))
            elif low.startswith("param count:"):
                in_params = True
            elif in_params:
                for tok in stripped.split():
                    params.append(float(tok.replace("d", "e").replace("D", "E")))
    return iteration, misfit, np.array(params)


def build_profile(depths, base_resis, free_mask, log10_params):
    """Merge free-parameter values into full resistivity profile.

    Returns (plot_depths, plot_resis) suitable for a step plot.
    """
    rho = base_resis.copy()
    j = 0
    for i in range(len(rho)):
        if free_mask[i]:
            rho[i] = 10.0 ** log10_params[j]
            j += 1
        else:
            rho[i] = abs(rho[i])  # fixed layers stored as positive
    # Build step-profile for plotting (duplicate each depth)
    plot_d, plot_r = [], []
    for i in range(len(depths)):
        d_top = depths[i]
        d_bot = depths[i + 1] if i + 1 < len(depths) else depths[i] + 500
        plot_d.extend([d_top, d_bot])
        plot_r.extend([rho[i], rho[i]])
    return np.array(plot_d), np.array(plot_r)


# ── true model ───────────────────────────────────────────────────────

TRUE_DEPTHS = [0, 1000, 2000, 2100, 4500]   # seafloor=0, reservoir 2-2.1 km
TRUE_RESIS  = [0.3, 1.0, 100.0, 1.0, 1.0]  # sea, sediment, reservoir, sediment


def true_profile():
    d, r = [], []
    for i in range(len(TRUE_DEPTHS)):
        d_top = TRUE_DEPTHS[i]
        d_bot = TRUE_DEPTHS[i + 1] if i + 1 < len(TRUE_DEPTHS) else TRUE_DEPTHS[i] + 500
        d.extend([d_top, d_bot])
        r.extend([TRUE_RESIS[i], TRUE_RESIS[i]])
    return np.array(d), np.array(r)


# ── locate files ─────────────────────────────────────────────────────

# Find model file (referenced inside iter file)
sample_iter = os.path.join(example_dir, "iter_0000.iter")
if not os.path.exists(sample_iter):
    sys.exit("Run 'occam1d invert' first to generate iteration files.")

# Parse model file path from the iter file
model_path = None
with open(sample_iter) as f:
    for line in f:
        if line.strip().lower().startswith("model file:"):
            model_path = line.split(":", 1)[1].strip()
            break

# Model path is relative to where inversion was run (the Fortran example dir)
fortran_dir = os.path.join(os.path.dirname(__file__),
    "../../Occam1DCSEM_matlab/Examples/Synthetic/CanonicalReservoir/Canonical_RealImag_BxEyEz")
model_full = os.path.join(fortran_dir, model_path)
if not os.path.exists(model_full):
    sys.exit(f"Cannot find model file: {model_full}")

depths, base_resis, free_mask = read_model_file(model_full)

# Collect iteration files
iter_files = sorted(glob.glob(os.path.join(example_dir, "iter_????.iter")))
if not iter_files:
    sys.exit("No iter_NNNN.iter files found.")

iterations = []
for fp in iter_files:
    it, misfit, params = read_iter_file(fp)
    iterations.append((it, misfit, params))

n_iters = len(iterations)

# ── plot ─────────────────────────────────────────────────────────────

fig, (ax_model, ax_rms) = plt.subplots(1, 2, figsize=(11, 7),
                                        gridspec_kw={"width_ratios": [3, 1.2]})

# Colour map: early iterations light, late iterations dark
cmap = plt.cm.Blues
for idx, (it, misfit, params) in enumerate(iterations):
    if it == 0:
        continue  # skip flat starting model

    frac = idx / max(n_iters - 1, 1)  # 0..1
    color = cmap(0.20 + 0.65 * frac)  # lighter palette
    lw = 0.6 + 2.0 * frac             # thin → thick
    alpha = 0.4 + 0.6 * frac          # transparent → opaque

    final_it = iterations[-1][0]
    show_label = (it == 1) or (it % 5 == 0) or (it == final_it)
    pd, pr = build_profile(depths, base_resis, free_mask, params)
    ax_model.plot(pr, pd, color=color, lw=lw, alpha=alpha,
                  label=f"Iter {it}" if show_label else None)

# True model
td, tr = true_profile()
ax_model.plot(tr, td, "r--", lw=2.0, label="True model", zorder=10)

ax_model.set_xscale("log")
ax_model.set_xlabel("Resistivity (ohm-m)", fontsize=11)
ax_model.set_ylabel("Depth (m)", fontsize=11)
ax_model.set_ylim(4500, 800)
ax_model.set_xlim(0.1, 300)
ax_model.legend(fontsize=9, loc="lower left")
ax_model.set_title("Resistivity Models", fontsize=12)
ax_model.grid(True, which="both", alpha=0.3)

# RMS convergence curve
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

fig.suptitle("OxidizedOccam 1D Inversion — Canonical Reservoir", fontsize=14, y=1.01)
fig.tight_layout()

out = os.path.join(example_dir, "inversion_models.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
