"""Regression: crossflow both-unmixed series must not truncate on the n=1 term alone."""

from __future__ import annotations

import numpy as np
import pytest

from heat_exchanger.epsilon_ntu import epsilon_ntu


def test_crossflow_unmixed_cr1_smooth_near_ntu6():
    # Previously tol_it=1e-4 skipped the whole correction when |term_1| < tol, while
    # later terms were still O(1e-3), producing a spurious step in eps(NTU, 1).
    ntu = np.linspace(5.5, 6.5, 401)
    eps = np.array([epsilon_ntu(x, 1.0, exchanger_type="cross_flow", flow_type="unmixed") for x in ntu])
    d2 = np.diff(eps, n=2)
    assert np.all(np.isfinite(eps))
    assert np.max(np.abs(d2)) < 0.01


@pytest.mark.parametrize("cr", [0.5, 1.0])
def test_crossflow_unmixed_monotone_in_ntu(cr: float):
    ntu = np.linspace(0.2, 8.0, 200)
    eps = np.array([epsilon_ntu(x, cr, exchanger_type="cross_flow", flow_type="unmixed") for x in ntu])
    assert np.all(np.diff(eps) >= -1e-9)
