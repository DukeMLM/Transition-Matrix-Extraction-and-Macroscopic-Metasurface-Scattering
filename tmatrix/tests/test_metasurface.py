"""Offline tests for cst_tmatrix.postprocess.metasurface (no CST required).

Two groups:
- Plumbing: trivial limits, scaling with the formula's own parameters.
- Physics, via the EXACT Mie sphere reference (already independently
  validated: tmatrix_solve.sphere_reference_T against passivity_check;
  mie.efficiencies against the analytic PEC/lossless limits in
  tests/test_vswf.py). This is what caught a real sign error in
  specular_s_parameters that a real CST periodic-boundary comparison's
  magnitude-only agreement did NOT catch -- see
  cst_tmatrix/postprocess/metasurface.py's module docstring for the full
  story. It does NOT substitute for the CST comparison (which validates
  the isolated-scatterer approximation itself, not just the formula's
  internal consistency) -- but it is strictly stronger than "plumbing
  only," and should have existed from the start.
"""
from __future__ import annotations

import numpy as np
import pytest

from cst_tmatrix.vswf import n_modes
from cst_tmatrix.tmatrix_solve import sphere_reference_T, passivity_check, \
    _diag_to_full
from cst_tmatrix.mie import sphere_tmatrix_diagonal, efficiencies
from cst_tmatrix.postprocess.metasurface import (
    forward_scattering_amplitude, specular_s_parameters)


def test_zero_scatterer_is_transparent():
    """T = 0 (no scatterer at all): the array must be perfectly transparent
    regardless of period, wavenumber, or angle -- S21 = 1, S11 = 0 exactly."""
    lmax = 3
    N = n_modes(lmax)
    T = np.zeros((N, N), dtype=complex)
    S11, S21 = specular_s_parameters(T, lmax, k=2.5, period_area=9.0,
                                     theta_i=0.3, phi_i=1.1, pol_i="theta")
    assert S11 == 0
    assert S21 == pytest.approx(1.0)


def test_scaling_with_period_area():
    """The scattering correction to S21 enters as 1/A; halving the area
    must exactly double the perturbation (S21 - 1) for a fixed T, angle,
    and k -- a direct consequence of the formula's own structure, so any
    deviation here is a bug in the implementation, not a physics question."""
    lmax = 2
    N = n_modes(lmax)
    rng = np.random.default_rng(5)
    T = 0.01 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    kwargs = dict(k=1.7, theta_i=0.4, phi_i=0.9, pol_i="phi")
    _, S21_a = specular_s_parameters(T, lmax, period_area=4.0, **kwargs)
    _, S21_2a = specular_s_parameters(T, lmax, period_area=8.0, **kwargs)
    ratio = (S21_a - 1.0) / (S21_2a - 1.0)
    assert ratio == pytest.approx(2.0, rel=1e-10)


def test_forward_amplitude_at_default_direction_is_incidence_direction():
    lmax = 2
    N = n_modes(lmax)
    rng = np.random.default_rng(6)
    T = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    F1 = forward_scattering_amplitude(T, lmax, 0.5, 1.2, "theta")
    F2 = forward_scattering_amplitude(T, lmax, 0.5, 1.2, "theta",
                                      theta_eval=0.5, phi_eval=1.2)
    assert F1 == F2


def test_rejects_normal_incidence_singularity():
    lmax = 2
    N = n_modes(lmax)
    T = np.eye(N, dtype=complex)
    with pytest.raises(ValueError):
        specular_s_parameters(T, lmax, k=1.0, period_area=1.0,
                              theta_i=0.0, phi_i=0.0)


@pytest.mark.parametrize("x", [0.1, 0.3, 0.6, 1.0])
def test_forward_amplitude_matches_optical_theorem(x):
    """Im(F_package)/k must equal -k/(4 pi) * sigma_ext (mie.efficiencies'
    independently-validated Qext) for an EXACT Mie sphere -- pins the sign
    that switching this package's e^{+j omega t} convention against the
    "standard" e^{-i omega t} optical-theorem literature introduces. This
    is the check that caught the original bug (a missing minus sign that a
    real CST comparison's magnitude-only agreement did not catch)."""
    lmax = 5
    r = 1.0e-6
    k = x / r
    m = 2.0 - 0.3j                      # loss: negative Im(m), this package's convention
    T = sphere_reference_T(lmax, x, m)
    Qext, _ = efficiencies(x, m)
    sigma_ext = Qext * np.pi * r ** 2

    F_th, _ = forward_scattering_amplitude(T, lmax, np.radians(5.0), 0.0, "theta")
    expected_Im_f_standard = k / (4 * np.pi) * sigma_ext
    assert (-F_th.imag / k) == pytest.approx(expected_Im_f_standard, rel=1e-6)


@pytest.mark.parametrize("x", [0.1, 0.4, 0.8, 1.3])
def test_forward_amplitude_treats_magnetic_and_electric_symmetrically(x):
    """Swap the Mie a_l (electric/N-type) and b_l (magnetic/M-type)
    coefficients into each other's slot before building the full T-matrix.
    Qext = (2/x^2) sum (2l+1) Re(a_l + b_l) is symmetric under this swap
    (same total extinguished power), so if evaluate_farfield/
    forward_scattering_amplitude treats M-type and N-type content
    identically, the optical-theorem match must survive the swap exactly
    -- including at x=0.1, where the swap makes the dipole essentially
    pure M-type (magnetic/electric ratio ~500) instead of pure N-type.
    This is what a spherically-symmetric-only test suite would miss: a
    real design can be magnetic-dipole-dominated (e.g. a ring resonator),
    which an isotropic dielectric Mie sphere at modest x never naturally
    is."""
    lmax = 5
    r = 1.0e-6
    k = x / r
    m = 2.0 - 0.3j
    TM, TN = sphere_tmatrix_diagonal(lmax, x, m)
    T_swapped = _diag_to_full(lmax, TN, TM)         # electric -> M slot, magnetic -> N slot
    Qext, _ = efficiencies(x, m)
    sigma_ext = Qext * np.pi * r ** 2

    F_th, _ = forward_scattering_amplitude(T_swapped, lmax, np.radians(5.0), 0.0, "theta")
    expected_Im_f_standard = k / (4 * np.pi) * sigma_ext
    assert (-F_th.imag / k) == pytest.approx(expected_Im_f_standard, rel=1e-6)


def test_sphere_respects_unitarity_bound_at_weak_scattering():
    """For an exact, independently-passive Mie sphere (passivity_check on
    the T-matrix itself, not this module) at small size parameter (leading-
    order / weak-scattering regime), |S11|^2 + |S21|^2 must sit at or just
    below 1 -- energy conservation, to the accuracy this leading-order
    (no inter-element coupling) formula is expected to hold at weak
    scattering. This is what a sign error violates immediately and
    unboundedly (see the module docstring); a correct formula stays close
    to and below 1 here regardless of how the isolated-scatterer
    approximation later degrades at stronger scattering."""
    lmax = 5
    r = 1.0e-6
    m = 2.0 - 0.3j
    period_area = (2.0e-6) ** 2
    theta_i = np.radians(5.0)
    for x in (0.1, 0.2):
        k = x / r
        T = sphere_reference_T(lmax, x, m)
        maxsv, _ = passivity_check(T)
        assert maxsv == pytest.approx(1.0, abs=1e-4)   # sanity: reference IS passive
        S11, S21 = specular_s_parameters(T, lmax, k, period_area, theta_i,
                                         0.0, pol_i="theta")
        total = abs(S11) ** 2 + abs(S21) ** 2
        assert 0.9 < total <= 1.0 + 1e-6
