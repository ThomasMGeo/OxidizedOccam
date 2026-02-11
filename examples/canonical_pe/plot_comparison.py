#!/usr/bin/env python3
"""Compare fields-only vs PE-only vs fields+PE inversions for the tilted-transmitter example.

Produces comparison.png with:
  Left:  Resistivity models (fields-only vs PE-only vs fields+PE vs true)
  Right: RMS convergence curves
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
depths, base_resis, free_mask = read_model_file(model_path)

# Fields + PE inversion
pe_iters = sorted(glob.glob(os.path.join(example_dir, "iter_????.iter")))
pe_data = []
for fp in pe_iters:
    it, misfit, params, _bounds = read_iter_file(fp)
    pe_data.append((it, misfit, params))

# Fields-only inversion
fo_iters = sorted(glob.glob(os.path.join(example_dir, "fo_iter_????.iter")))
fo_data = []
for fp in fo_iters:
    it, misfit, params, _bounds = read_iter_file(fp)
    fo_data.append((it, misfit, params))

# PE-only inversion
po_iters = sorted(glob.glob(os.path.join(example_dir, "po_iter_????.iter")))
po_data = []
for fp in po_iters:
    it, misfit, params, _bounds = read_iter_file(fp)
    po_data.append((it, misfit, params))

if not pe_data and not fo_data and not po_data:
    sys.exit("No iteration files found")

# ── plot ────────────────────────────────────────────────────────────
fig, (ax_model, ax_rms) = plt.subplots(
    1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [3, 1.5]}
)

# True model
true = TRUE_MODELS["canonical_reservoir"]
td, tr = make_step_profile(true.depths, true.resistivities, true.bottom_extension)
ax_model.plot(tr, td, "r--", lw=2.5, label="True model", zorder=10)

# Fields+PE final model
if pe_data:
    final_pe = pe_data[-1]
    pd_, pr = build_profile(depths, base_resis, free_mask, final_pe[2])
    ax_model.plot(
        pr,
        pd_,
        color="#7B2FBE",
        lw=2.0,
        label=f"Fields + PE (iter {final_pe[0]}, RMS {final_pe[1]:.2f})",
        zorder=5,
    )

# Fields-only final model
if fo_data:
    final_fo = fo_data[-1]
    fd_, fr = build_profile(depths, base_resis, free_mask, final_fo[2])
    ax_model.plot(
        fr,
        fd_,
        color="#2196F3",
        lw=2.0,
        ls="-",
        label=f"Fields only (iter {final_fo[0]}, RMS {final_fo[1]:.2f})",
        zorder=6,
    )

# PE-only final model
if po_data:
    final_po = po_data[-1]
    pod_, por = build_profile(depths, base_resis, free_mask, final_po[2])
    ax_model.plot(
        por,
        pod_,
        color="#FF9800",
        lw=2.0,
        ls="-",
        label=f"PE only (iter {final_po[0]}, RMS {final_po[1]:.2f})",
        zorder=7,
    )

ax_model.set_xscale("log")
ax_model.set_xlabel("Resistivity (ohm-m)", fontsize=11)
ax_model.set_ylabel("Depth (m)", fontsize=11)
ax_model.set_ylim(4500, 800)
ax_model.set_xlim(0.1, 300)
ax_model.legend(fontsize=9, loc="lower left")
ax_model.set_title("Resistivity Models", fontsize=12)
ax_model.grid(True, which="both", alpha=0.3)

# RMS convergence
if pe_data:
    its_pe = [t[0] for t in pe_data]
    rms_pe = [t[1] for t in pe_data]
    ax_rms.plot(its_pe, rms_pe, "o-", color="#7B2FBE", ms=2, lw=1.2, label="Fields + PE")

if fo_data:
    its_fo = [t[0] for t in fo_data]
    rms_fo = [t[1] for t in fo_data]
    ax_rms.plot(its_fo, rms_fo, "s-", color="#2196F3", ms=4, lw=1.2, label="Fields only")

if po_data:
    its_po = [t[0] for t in po_data]
    rms_po = [t[1] for t in po_data]
    ax_rms.plot(its_po, rms_po, "^-", color="#FF9800", ms=4, lw=1.2, label="PE only")

ax_rms.axhline(1.0, color="r", ls="--", lw=1, label="Target RMS")
ax_rms.set_xlabel("Iteration", fontsize=11)
ax_rms.set_ylabel("RMS Misfit", fontsize=11)
ax_rms.set_title("Convergence", fontsize=12)
ax_rms.legend(fontsize=9)
ax_rms.set_ylim(bottom=0)
ax_rms.grid(True, alpha=0.3)

fig.suptitle(
    "Tilted Tx (az=70\u00b0, dip=20\u00b0) \u2014 Fields Only vs PE Only vs Fields + PE",
    fontsize=14,
    y=1.01,
)
fig.tight_layout()

out = os.path.join(example_dir, "comparison.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
