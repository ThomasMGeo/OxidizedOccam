"""I/O helpers for reading OxidizedOccam model and iteration files."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _fortran_float(s: str) -> float:
    """Parse a float that may use Fortran 'd'/'D' exponent notation."""
    return float(s.replace("d", "e").replace("D", "E"))


def read_model_file(path: str) -> tuple[NDArray, NDArray, NDArray]:
    """Return (depths, resistivities, is_free) arrays from a model file.

    Handles both numeric resistivities and "?" placeholders (free parameters).
    """
    depths: list[float] = []
    resis: list[float] = []
    free: list[bool] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("Format", "#", "!")):
                continue
            tokens = line.split("!")[0].split()
            if len(tokens) < 2:
                continue
            d = _fortran_float(tokens[0])
            r_str = tokens[1].strip()
            r = -1.0 if r_str == "?" else _fortran_float(r_str)
            depths.append(d)
            resis.append(r)
            free.append(r < 0)
    return np.array(depths), np.array(resis), np.array(free)


def read_iter_file(
    path: str,
) -> tuple[int, float, NDArray, tuple[float, float] | None]:
    """Return (iteration, misfit, params, bounds) from an iteration file.

    *bounds* is ``(lo, hi)`` in log10(ohm-m) if the file contains a
    ``Model Bounds:`` line, otherwise ``None``.
    """
    iteration, misfit = 0, 0.0
    params: list[float] = []
    bounds: tuple[float, float] | None = None
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
                misfit = _fortran_float(stripped.split(":")[1].strip())
            elif low.startswith(("model bounds:", "model limits:")):
                parts = stripped.split(":")[1].strip().split(",")
                if len(parts) == 2:
                    bounds = (float(parts[0]), float(parts[1]))
            elif low.startswith("param count:"):
                in_params = True
            elif in_params:
                for tok in stripped.split():
                    params.append(_fortran_float(tok))
    return iteration, misfit, np.array(params), bounds
