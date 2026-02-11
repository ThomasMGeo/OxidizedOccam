"""Tests for the oxoccam_plot shared package."""

import os

import numpy as np

from oxoccam_plot import (
    build_profile,
    make_step_profile,
    read_iter_file,
    read_model_file,
)

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir)


def test_read_model_file():
    depths, resis, free = read_model_file(os.path.join(EXAMPLES, "canonical_reservoir/model"))
    assert len(depths) == 77
    # First two layers: air and sea (fixed)
    assert depths[0] == -100000.0
    assert resis[0] == 1e12
    assert not free[0]
    assert depths[1] == 0.0
    assert resis[1] == 0.3
    assert not free[1]
    # Third layer onward: free parameters (resistivity = -1)
    assert depths[2] == 1000.0
    assert free[2]
    assert np.sum(free) == 75


def test_read_iter_file():
    # Unbounded iter file
    it, misfit, params, bounds = read_iter_file(
        os.path.join(EXAMPLES, "canonical_reservoir/iter_0000.iter")
    )
    assert it == 0
    assert abs(misfit - 16.71064) < 0.001
    assert len(params) == 75
    assert bounds is None

    # Bounded iter file
    it_b, misfit_b, params_b, bounds_b = read_iter_file(
        os.path.join(EXAMPLES, "canonical_reservoir_bounded/iter_0000.iter")
    )
    assert it_b == 0
    assert abs(misfit_b - 16.62341) < 0.001
    assert len(params_b) == 75
    assert bounds_b == (0.0, 2.0)


def test_make_step_profile():
    depths = [0, 100, 500]
    resis = [10.0, 1.0, 100.0]
    pd, pr = make_step_profile(depths, resis, bottom_extension=200.0)
    # 3 layers → 6 points (top/bottom pair per layer)
    assert len(pd) == 6
    assert len(pr) == 6
    np.testing.assert_array_equal(pd, [0, 100, 100, 500, 500, 700])
    np.testing.assert_array_equal(pr, [10, 10, 1, 1, 100, 100])


def test_build_profile():
    depths = np.array([0.0, 100.0, 200.0, 300.0])
    base_resis = np.array([0.3, -1.0, -1.0, 1.0])
    free_mask = np.array([False, True, True, False])
    log10_params = np.array([0.0, 2.0])  # 1 ohm-m, 100 ohm-m
    pd, pr = build_profile(depths, base_resis, free_mask, log10_params)
    assert len(pd) == 8  # 4 layers x 2
    # Fixed layers keep their absolute value
    assert pr[0] == 0.3
    assert pr[1] == 0.3
    # Free layers get 10**param
    assert pr[2] == 1.0
    assert pr[4] == 100.0
    # Last fixed layer
    assert pr[6] == 1.0
