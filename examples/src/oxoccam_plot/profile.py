"""Resistivity-profile construction and true-model presets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrueModel:
    """A known true resistivity model used as a reference in plots."""

    depths: list[float]
    resistivities: list[float]
    bottom_extension: float = 500.0


# ── presets ──────────────────────────────────────────────────────────

TRUE_MODELS: dict[str, TrueModel] = {
    "canonical_reservoir": TrueModel(
        depths=[0, 1000, 2000, 2100, 4500],
        resistivities=[0.3, 1.0, 100.0, 1.0, 1.0],
    ),
    "canonical_mt": TrueModel(
        depths=[0, 1000, 2000, 2100, 4500],
        resistivities=[0.3, 1.0, 100.0, 1.0, 100.0],
        bottom_extension=1000.0,
    ),
}


def make_step_profile(
    depths: list[float] | NDArray,
    resistivities: list[float] | NDArray,
    bottom_extension: float = 500.0,
) -> tuple[NDArray, NDArray]:
    """Build (plot_depths, plot_resistivities) for a step-style profile plot.

    Each layer spans from its depth to the next layer's depth (or
    ``depth[-1] + bottom_extension`` for the deepest layer).
    """
    plot_d: list[float] = []
    plot_r: list[float] = []
    n = len(depths)
    for i in range(n):
        d_top = float(depths[i])
        d_bot = float(depths[i + 1]) if i + 1 < n else d_top + bottom_extension
        plot_d.extend([d_top, d_bot])
        plot_r.extend([float(resistivities[i])] * 2)
    return np.array(plot_d), np.array(plot_r)


def build_profile(
    depths: NDArray,
    base_resis: NDArray,
    free_mask: NDArray,
    log10_params: NDArray,
    bottom_extension: float = 500.0,
) -> tuple[NDArray, NDArray]:
    """Merge free-parameter values into a full resistivity profile.

    Returns ``(plot_depths, plot_resistivities)`` suitable for a step plot.
    """
    rho = base_resis.copy()
    j = 0
    for i in range(len(rho)):
        if free_mask[i]:
            rho[i] = 10.0 ** log10_params[j]
            j += 1
        else:
            rho[i] = abs(rho[i])
    return make_step_profile(depths, rho, bottom_extension)
