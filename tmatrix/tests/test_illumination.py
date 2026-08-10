"""Illumination direction sets: Lebedev rule exactness and scheme selection."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst_tmatrix.quadrature import (fibonacci_directions,
                                    illumination_directions,
                                    lebedev_directions,
                                    verify_spherical_rule)


def test_builtin_lebedev_rules_are_exact():
    """The built-in 6-, 14-, and 26-point rules integrate all spherical
    harmonics up to their design precision (3, 5, 7) to machine accuracy."""
    for n_min, prec in [(1, 3), (7, 5), (15, 7)]:
        th, ph, w, npts = lebedev_directions(n_min)
        assert abs(np.sum(w) - 1.0) < 1e-14
        assert verify_spherical_rule(th, ph, w, prec) < 1e-13


def test_lebedev_selects_smallest_sufficient_rule():
    assert lebedev_directions(1)[3] == 6
    assert lebedev_directions(7)[3] == 14
    assert lebedev_directions(15)[3] == 26


def test_lebedev_unavailable_count_raises():
    try:
        lebedev_directions(1000)
    except ValueError as e:
        assert "fibonacci" in str(e)
    else:
        raise AssertionError("expected ValueError for unavailable rule size")


def test_scheme_selection():
    th, ph = illumination_directions("fibonacci", 23)
    assert len(th) == 23
    assert np.all(np.sin(th) > 1e-6)          # pole-free by construction
    th, ph = illumination_directions("lebedev", 15)
    assert len(th) == 26                      # smallest sufficient rule
    try:
        illumination_directions("unknown", 5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown scheme")
