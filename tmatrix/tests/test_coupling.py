"""Offline validation of Foldy-Lax multiple-scattering coupling (no CST).

Tolerances here are set from MEASURED behaviour of the numerically
constructed translation operator (see coupling.py module docstring for the
measurement tables), not guessed.
"""
from __future__ import annotations

import numpy as np
import pytest

from cst_tmatrix import vswf, mie
from cst_tmatrix.quadrature import gauss_legendre_sphere, unit_vectors
from cst_tmatrix.tmatrix_solve import reciprocity_check
from cst_tmatrix.postprocess.coupling import (
    translation_operator, plane_wave_coefficients_at, coupling_matrix,
    foldy_lax_solve, single_scattering_solve, check_pair_separation,
    effective_site_tmatrix)

K = 1.0
R_CIRC = 1.0
LMAX = mie.wiscombe_lmax(K * R_CIRC)          # 6 for k*r = 1


def mie_T(lmax=LMAX, eps=12.0, k=K, r=R_CIRC):
    """Diagonal Mie T-matrix for a sphere -- a physically real T with the
    correct decay of high-order coefficients."""
    pol, ns, _ = vswf.mode_list(lmax)
    T_M, T_N = mie.sphere_tmatrix_diagonal(lmax, k * r, m=np.sqrt(eps))
    return np.diag(np.where(pol == 0, T_M[ns - 1], T_N[ns - 1]))


def compact_coeffs(lmax=LMAX, seed=0):
    """Random outgoing coefficients with physically decaying high-l content."""
    rng = np.random.default_rng(seed)
    N = vswf.n_modes(lmax)
    _, ns, _ = vswf.mode_list(lmax)
    f = rng.normal(size=N) + 1j * rng.normal(size=N)
    return f / (2.0 ** ns)


# --------------------------------------------------------------------------
# translation operator
# --------------------------------------------------------------------------

def test_translated_expansion_reproduces_the_original_field():
    """The core identity: an outgoing field about the origin, re-expanded in
    regular modes about a displaced centre, must reproduce the same field
    inside the receiving particle's own radius."""
    d = np.array([9.0, 12.0, 0.0])                    # |d| = 15
    C = translation_operator(LMAX, K, d, r_circ=R_CIRC)
    f = compact_coeffs()
    c = C @ f
    th, ph, _w, _ = gauss_legendre_sphere(LMAX + 4)
    loc = R_CIRC * unit_vectors(th, ph)
    E_direct, _ = vswf.evaluate_EH(f, LMAX, "outgoing", K, loc + d)
    E_trans, _ = vswf.evaluate_EH(c, LMAX, "regular", K, loc)
    err = np.linalg.norm(E_trans - E_direct) / np.linalg.norm(E_direct)
    assert err < 1e-4, f"{err:.3e}"


def test_translation_operator_is_independent_of_projection_radius():
    """The translation coefficients are purely geometric, so the constructed
    operator must not depend on the sphere used to build it.  This is the
    check on the numerical construction itself."""
    d = np.array([9.0, 12.0, 0.0])
    C_ref = translation_operator(LMAX, K, d, r_circ=1.0)
    for rho in (0.5, 0.75, 1.5):
        C = translation_operator(LMAX, K, d, r_circ=rho)
        rel = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)
        assert rel < 1e-5, f"rho={rho}: {rel:.3e}"


def test_translation_rejects_radius_beyond_the_separation():
    """rho >= |d| would enclose the source singularity."""
    d = np.array([3.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        translation_operator(LMAX, K, d, r_circ=3.0)
    with pytest.raises(ValueError):
        translation_operator(LMAX, K, d, r_circ=5.0)


def test_accuracy_degrades_as_spheres_approach_and_the_check_says_so():
    """Documents the real limit of the method: the warning must fire exactly
    where the measured accuracy actually deteriorates."""
    f = compact_coeffs()
    th, ph, _w, _ = gauss_legendre_sphere(LMAX + 4)
    loc = R_CIRC * unit_vectors(th, ph)

    def field_err(dist):
        d = np.array([dist, 0.0, 0.0])
        c = translation_operator(LMAX, K, d, r_circ=R_CIRC) @ f
        E_direct, _ = vswf.evaluate_EH(f, LMAX, "outgoing", K, loc + d)
        E_trans, _ = vswf.evaluate_EH(c, LMAX, "regular", K, loc)
        return np.linalg.norm(E_trans - E_direct) / np.linalg.norm(E_direct)

    far, near = field_err(15.0), field_err(2.2)
    assert far < near                     # closer is worse
    assert check_pair_separation(15.0, R_CIRC, R_CIRC) is None
    assert check_pair_separation(2.2, R_CIRC, R_CIRC) is not None
    assert "OVERLAP" in check_pair_separation(1.5, R_CIRC, R_CIRC)


# --------------------------------------------------------------------------
# incident field bookkeeping
# --------------------------------------------------------------------------

def test_plane_wave_phase_at_a_shifted_site():
    """The translation-eigenfunction shortcut must agree with a direct
    projection of the plane wave about the shifted centre."""
    th, ph, w, _ = gauss_legendre_sphere(LMAX + 6)
    rho = 1.0
    loc = rho * unit_vectors(th, ph)
    for r_site in ([3.0, 0, 0], [0, 0, 5.0], [2.0, -3.0, 1.5]):
        r_site = np.array(r_site, float)
        a_shift = plane_wave_coefficients_at(0.7, 1.1, "theta", LMAX, K,
                                             r_site)
        E, H = vswf.plane_wave_field(0.7, 1.1, "theta", K, loc + r_site)
        a_proj = vswf.project_surface_field(E, LMAX, "regular", K, rho,
                                            th, ph, w, H=H)
        rel = np.linalg.norm(a_shift - a_proj) / np.linalg.norm(a_proj)
        assert rel < 1e-10, f"{r_site}: {rel:.3e}"


# --------------------------------------------------------------------------
# Foldy-Lax
# --------------------------------------------------------------------------

def dimer(sep=15.0):
    pos = np.array([[0.0, 0.0, 0.0], [sep, 0.0, 0.0]])
    T = mie_T()
    a = [plane_wave_coefficients_at(0.7, 1.1, "theta", LMAX, K, p)
         for p in pos]
    return pos, [T, T], a


def test_foldy_lax_residual_is_machine_precision():
    pos, Ts, a = dimer()
    _b, info = foldy_lax_solve(Ts, pos, a, K, LMAX, R_CIRC, warn=False)
    assert info["residual"] < 1e-10


def test_zero_tmatrix_neighbour_leaves_the_other_site_isolated():
    """A neighbour that does not scatter cannot influence anything -- an
    exact limit, so this is a sharp check on the assembly."""
    pos, Ts, a = dimer()
    N = vswf.n_modes(LMAX)
    Ts = [Ts[0], np.zeros((N, N), dtype=complex)]
    b, _ = foldy_lax_solve(Ts, pos, a, K, LMAX, R_CIRC, warn=False)
    b_iso = single_scattering_solve(Ts, a)
    assert np.allclose(b[0], b_iso[0], rtol=1e-12, atol=1e-14)
    assert np.allclose(b[1], 0.0, atol=1e-14)


def test_coupling_vanishes_as_sites_separate():
    """Foldy-Lax must approach the isolated (Born) result at large
    separation, and the approach must be monotone in distance."""
    def rel_diff(sep):
        pos, Ts, a = dimer(sep)
        b, _ = foldy_lax_solve(Ts, pos, a, K, LMAX, R_CIRC, warn=False)
        b0 = single_scattering_solve(Ts, a)
        return (np.linalg.norm(np.concatenate(b) - np.concatenate(b0))
                / np.linalg.norm(np.concatenate(b0)))

    d_near, d_mid, d_far = rel_diff(8.0), rel_diff(30.0), rel_diff(200.0)
    assert d_near > d_mid > d_far
    assert d_far < 1e-2


def test_coupling_is_not_negligible_at_realistic_spacing():
    """Guards against a silently-inert implementation: at a spacing typical
    of a metasurface lattice the correction must be a real effect, otherwise
    these tests would pass even if C were zero."""
    pos, Ts, a = dimer(6.0)
    b, _ = foldy_lax_solve(Ts, pos, a, K, LMAX, R_CIRC, warn=False)
    b0 = single_scattering_solve(Ts, a)
    rel = (np.linalg.norm(np.concatenate(b) - np.concatenate(b0))
           / np.linalg.norm(np.concatenate(b0)))
    assert rel > 1e-3


def test_identical_sites_under_normal_incidence_respond_identically():
    """Symmetry check with physical content: for incidence along +z the two
    sites of an x-separated dimer are equivalent, so their outgoing
    coefficients must match after removing the incident phase (which is
    equal here, both sites being at z = 0)."""
    sep = 12.0
    pos = np.array([[-sep / 2, 0.0, 0.0], [sep / 2, 0.0, 0.0]])
    T = mie_T()
    a = [plane_wave_coefficients_at(0.0, 0.0, "theta", LMAX, K, p)
         for p in pos]
    b, _ = foldy_lax_solve([T, T], pos, a, K, LMAX, R_CIRC, warn=False)
    # the geometry is mirror-symmetric about x=0 under normal incidence;
    # |b| must be equal site to site
    assert np.allclose(np.abs(b[0]), np.abs(b[1]), rtol=1e-8, atol=1e-12)


def test_precomputed_coupling_matrix_is_reusable():
    """Reusing C across incidences must give bitwise-comparable results --
    this is what makes a sweep over many illuminations affordable."""
    pos, Ts, _a = dimer()
    C = coupling_matrix(LMAX, K, pos, R_CIRC, warn=False)
    for th, ph in ((0.7, 1.1), (1.4, -0.3)):
        a = [plane_wave_coefficients_at(th, ph, "theta", LMAX, K, p)
             for p in pos]
        b1, _ = foldy_lax_solve(Ts, pos, a, K, LMAX, R_CIRC, C=C, warn=False)
        b2, _ = foldy_lax_solve(Ts, pos, a, K, LMAX, R_CIRC, warn=False)
        assert np.allclose(np.concatenate(b1), np.concatenate(b2), rtol=1e-10)


def test_effective_site_tmatrix_reduces_to_T_without_coupling():
    T = mie_T()
    N = T.shape[0]
    assert np.allclose(effective_site_tmatrix(T, np.zeros((N, N))), T)


def test_effective_site_tmatrix_solves_its_defining_equation():
    """T_eff = [I - T Omega]^-1 T  <=>  T_eff = T + T Omega T_eff."""
    T = mie_T()
    N = T.shape[0]
    rng = np.random.default_rng(2)
    Om = 0.01 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    Teff = effective_site_tmatrix(T, Om)
    assert np.allclose(Teff, T + T @ Om @ Teff, rtol=1e-8, atol=1e-12)
