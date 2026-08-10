"""Formal regression tests for the VSWF / T-matrix math core.

Run:  python -m pytest tests/test_vswf.py -q
Every convention choice in cst_tmatrix.vswf is locked by these tests.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst_tmatrix import vswf
from cst_tmatrix.quadrature import (fibonacci_directions, gauss_legendre_sphere,
                                    quadrature_for_monitor, unit_vectors,
                                    spherical_basis)
from cst_tmatrix.tmatrix_solve import (solve_tmatrix, passivity_check,
                                       sphere_reference_T)

LMAX = 5
K = 1.0
RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
def sphere_grid():
    th, ph, w, _ = gauss_legendre_sphere(2 * LMAX + 6)
    return th, ph, w


def test_harmonics_match_scipy(sphere_grid):
    from scipy.special import sph_harm
    th, ph, _ = sphere_grid
    P = vswf._legendre_norm(LMAX, np.cos(th), np.sin(th))
    for n in range(1, LMAX + 1):
        for m in range(-n, n + 1):
            Y_mine = vswf._P_signed(P, n, m, th.size) * np.exp(1j * m * ph)
            assert np.max(np.abs(Y_mine - sph_harm(m, n, ph, th))) < 1e-10


def test_tau_matches_finite_difference(sphere_grid):
    th, _, _ = sphere_grid
    _, tau = vswf.pi_tau(LMAX, th)
    dth = 1e-6
    for n in range(1, LMAX + 1):
        Pp = vswf._legendre_norm(n, np.cos(th + dth), np.sin(th + dth))
        Pm = vswf._legendre_norm(n, np.cos(th - dth), np.sin(th - dth))
        for m in range(-n, n + 1):
            fd = (vswf._P_signed(Pp, n, m, th.size)
                  - vswf._P_signed(Pm, n, m, th.size)) / (2 * dth)
            assert np.max(np.abs(tau[n, m + LMAX] - fd)) < 1e-5


def test_projection_round_trip(sphere_grid):
    th, ph, w = sphere_grid
    R = 2.0 / K
    pts = R * unit_vectors(th, ph)
    c0 = (RNG.standard_normal(vswf.n_modes(LMAX))
          + 1j * RNG.standard_normal(vswf.n_modes(LMAX)))
    for kind in ("regular", "outgoing"):
        E, ZH = vswf.evaluate_EH(c0, LMAX, kind, K, pts)
        c = vswf.project_surface_field(E, LMAX, kind, K, R, th, ph, w)
        assert np.max(np.abs(c - c0)) < 1e-10
        c = vswf.project_surface_field(E, LMAX, kind, K, R, th, ph, w, H=ZH)
        assert np.max(np.abs(c - c0)) < 1e-10


def test_plane_wave_expansion(sphere_grid):
    th, ph, w = sphere_grid
    R = 2.0 / K
    pts = R * unit_vectors(th, ph)
    for (ti, pii, pol) in [(0.7, 1.1, "theta"), (0.7, 1.1, "phi"),
                           (2.2, 4.9, "theta"), (2.2, 4.9, "phi")]:
        E, ZH = vswf.plane_wave_field(ti, pii, pol, K, pts)
        a_num = vswf.project_surface_field(E, LMAX, "regular", K, R,
                                           th, ph, w, H=ZH)
        a_ana = vswf.plane_wave_coefficients(ti, pii, pol, LMAX)
        assert np.max(np.abs(a_num - a_ana)) < 1e-9


def test_plane_wave_reconstruction():
    lmax = 12
    r_test = 1.2 / K
    tp = np.repeat(np.linspace(0.3, 2.8, 7), 7)
    pp = np.tile(np.linspace(0, 6, 7), 7)
    pts = r_test * unit_vectors(tp, pp)
    a = vswf.plane_wave_coefficients(0.7, 1.1, "theta", lmax)
    E_rec = vswf.evaluate_field(a, lmax, "regular", K, pts)
    E_ref, _ = vswf.plane_wave_field(0.7, 1.1, "theta", K, pts)
    assert np.max(np.abs(E_rec - E_ref)) < 1e-8


def test_separation_and_mie_extraction():
    """Full synthetic pipeline: total E/H on monitor sphere with random
    per-run source phases -> separate -> lstsq -> Mie T-matrix."""
    X, m_rel = 2.0, 1.5 + 0j
    R_mon = 2 * X / K
    T_true = sphere_reference_T(LMAX, X, m_rel)
    n_dirs = int(np.ceil(0.75 * vswf.n_modes(LMAX)))
    th_i, ph_i = fibonacci_directions(n_dirs)
    th, ph, w, _ = quadrature_for_monitor(LMAX, K, R_mon)
    pts = R_mon * unit_vectors(th, ph)

    A_cols, F_cols = [], []
    for i in range(n_dirs):
        for pol in ("theta", "phi"):
            a_nom = vswf.plane_wave_coefficients(th_i[i], ph_i[i], pol, LMAX)
            c_k = np.exp(1j * 2 * np.pi * RNG.random())   # unknown source phase
            f_real = T_true @ (c_k * a_nom)
            E_i, H_i = vswf.plane_wave_field(th_i[i], ph_i[i], pol, K, pts)
            E_s, H_s = vswf.evaluate_EH(f_real, LMAX, "outgoing", K, pts)
            a_rec, f_rec = vswf.separate_surface_field(
                c_k * E_i + E_s, c_k * H_i + H_s, LMAX, K, R_mon, th, ph, w)
            A_cols.append(a_rec)
            F_cols.append(f_rec)
    T_rec, diag = solve_tmatrix(np.array(A_cols).T, np.array(F_cols).T)
    assert np.max(np.abs(T_rec - T_true)) < 1e-12
    assert diag["cond"] < 3
    smax, uni = passivity_check(T_rec)
    assert uni < 1e-10          # lossless sphere -> unitary S


def test_farfield_route():
    X = 2.0
    T_true = sphere_reference_T(LMAX, X, None)     # PEC
    th, ph, w, _ = gauss_legendre_sphere(LMAX, oversample=1.5)
    _, th_hat, ph_hat = spherical_basis(th, ph)
    kr_far = 1e7
    pts_far = (kr_far / K) * unit_vectors(th, ph)
    a = vswf.plane_wave_coefficients(0.7, 1.1, "theta", LMAX)
    f = T_true @ a
    E_far = vswf.evaluate_field(f, LMAX, "outgoing", K, pts_far)
    F_th = np.einsum("ij,ij->i", E_far, th_hat) * kr_far * np.exp(1j * kr_far)
    F_ph = np.einsum("ij,ij->i", E_far, ph_hat) * kr_far * np.exp(1j * kr_far)
    f_rec = vswf.project_farfield(F_th, F_ph, LMAX, K, th, ph, w)
    assert np.max(np.abs(f_rec - f)) < 1e-5 * np.max(np.abs(f))


def test_evaluate_farfield_matches_numerical_limit():
    """evaluate_farfield (analytic asymptotic form) must reproduce the same
    kr -> infinity limit as evaluate_field (the route test_farfield_route
    already validates project_farfield against) -- the exact-inverse
    relationship checked in test_evaluate_farfield_is_project_farfield_inverse
    below only proves internal consistency between the two NEW asymptotic
    routes; this ties evaluate_farfield to the independently-locked exact
    field evaluator."""
    rng = np.random.default_rng(3)
    N = vswf.n_modes(LMAX)
    f = rng.normal(size=N) + 1j * rng.normal(size=N)
    th, ph, w, _ = gauss_legendre_sphere(LMAX, oversample=1.5)
    _, th_hat, ph_hat = spherical_basis(th, ph)
    kr_far = 1e7
    pts_far = (kr_far / K) * unit_vectors(th, ph)
    E_far = vswf.evaluate_field(f, LMAX, "outgoing", K, pts_far)
    F_th_num = np.einsum("ij,ij->i", E_far, th_hat) * kr_far * np.exp(1j * kr_far)
    F_ph_num = np.einsum("ij,ij->i", E_far, ph_hat) * kr_far * np.exp(1j * kr_far)
    F_th, F_ph = vswf.evaluate_farfield(f, LMAX, th, ph)
    assert np.max(np.abs(F_th - F_th_num)) < 1e-5 * np.max(np.abs(F_th))
    assert np.max(np.abs(F_ph - F_ph_num)) < 1e-5 * np.max(np.abs(F_ph))


def test_evaluate_farfield_is_project_farfield_inverse():
    rng = np.random.default_rng(4)
    N = vswf.n_modes(LMAX)
    f = rng.normal(size=N) + 1j * rng.normal(size=N)
    th, ph, w, _ = gauss_legendre_sphere(LMAX, oversample=1.5)
    F_th, F_ph = vswf.evaluate_farfield(f, LMAX, th, ph)
    f_rec = vswf.project_farfield(F_th, F_ph, LMAX, K, th, ph, w)
    assert np.max(np.abs(f_rec - f)) < 1e-10 * np.max(np.abs(f))


def test_mie_efficiency_consistency():
    from cst_tmatrix.mie import efficiencies
    qe, qs = efficiencies(x=3.0, m=1.5 + 0j)
    assert abs(qe - qs) / qe < 1e-12          # lossless: Qext == Qsca
    qe, qs = efficiencies(x=50.0, m=None)
    assert abs(qe - 2.0) < 0.05               # PEC extinction paradox


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
