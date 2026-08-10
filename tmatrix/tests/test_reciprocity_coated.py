"""Lock the reciprocity diagnostic and the coated-sphere Mie reference.

Reciprocity: the index relation used by tmatrix_solve.reciprocity_check,
    T[(p',n',m'),(p,n,m)] = (-1)^{m+m'} T[(p,n,-m),(p',n',-m')],
is verified here against Saxon's amplitude reciprocity theorem
    e2 . F(k2 <- k1; e1) = e1 . F(-k1 <- -k2; e2),
which follows from Lorentz reciprocity and is convention-independent (real
polarization vectors).  Both sides are built ONLY from primitives locked by
test_vswf.py (plane_wave_coefficients + the VSWF farfield asymptotics), so
this pins the sign factor the same way the live CST run pinned the Mie
conjugation: with physics, not literature.

Coated sphere: mie_ab_coated is locked by its two exact limits
(vacuum shell -> homogeneous core sphere; index-matched core -> homogeneous
outer sphere) and by passivity of the resulting T for a lossy shell.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst_tmatrix import vswf
from cst_tmatrix.mie import (mie_ab, mie_ab_coated,
                             coated_sphere_tmatrix_diagonal)
from cst_tmatrix.quadrature import spherical_basis
from cst_tmatrix.tmatrix_solve import (reciprocity_check, sphere_reference_T,
                                       coated_sphere_reference_T,
                                       passivity_check)

LMAX = 3
N = vswf.n_modes(LMAX)
HALF = N // 2


# ---------------------------------------------------------------------------
# Saxon amplitude machinery (from locked primitives only)
# ---------------------------------------------------------------------------

def _farfield_F(coeffs, theta, phi):
    """F(theta,phi) Cartesian 3-vector from outgoing coefficients:
    E_sca -> F e^{-jkr}/(kr); z_n -> j^{n+1} e^{-jkr}/kr,
    (kr z_n)'/kr -> j^n e^{-jkr}/kr."""
    th = np.atleast_1d(float(theta))
    ph = np.atleast_1d(float(phi))
    _, th_hat, ph_hat = spherical_basis(th, ph)
    pi, tau = vswf.pi_tau(LMAX, th)
    F = np.zeros(3, dtype=complex)
    for n in range(1, LMAX + 1):
        gam = 1.0 / np.sqrt(n * (n + 1))
        for m in range(-n, n + 1):
            idx = vswf.block_index(n, m)
            ang = np.exp(1j * m * ph[0])
            piv = pi[n, m + LMAX][0] * ang
            tav = tau[n, m + LMAX][0] * ang
            C = gam * (1j * piv * th_hat[0] - tav * ph_hat[0])
            B = gam * (tav * th_hat[0] + 1j * piv * ph_hat[0])
            F += coeffs[idx] * (1j) ** (n + 1) * C
            F += coeffs[HALF + idx] * (1j) ** n * B
    return F


def _pw_coeffs_vec(theta, phi, e_vec):
    """Plane-wave coefficients for an arbitrary real polarization vector
    (perpendicular to k_hat), by linearity on the theta/phi basis."""
    th = np.atleast_1d(float(theta))
    ph = np.atleast_1d(float(phi))
    _, th_hat, ph_hat = spherical_basis(th, ph)
    return (float(e_vec @ th_hat[0])
            * vswf.plane_wave_coefficients(theta, phi, "theta", LMAX)
            + float(e_vec @ ph_hat[0])
            * vswf.plane_wave_coefficients(theta, phi, "phi", LMAX))


def _saxon_err(T, rng, npairs=8):
    def rand_dir_pol():
        th = np.arccos(rng.uniform(-0.98, 0.98))
        ph = rng.uniform(0, 2 * np.pi)
        _, th_hat, ph_hat = spherical_basis(np.atleast_1d(th),
                                            np.atleast_1d(ph))
        a = rng.normal(size=2)
        a /= np.hypot(*a)
        return th, ph, a[0] * th_hat[0] + a[1] * ph_hat[0]

    errs = []
    for _ in range(npairs):
        t1, p1, e1 = rand_dir_pol()
        t2, p2, e2 = rand_dir_pol()
        A12 = e2 @ _farfield_F(T @ _pw_coeffs_vec(t1, p1, e1), t2, p2)
        A21 = e1 @ _farfield_F(
            T @ _pw_coeffs_vec(np.pi - t2, (p2 + np.pi) % (2 * np.pi), e2),
            np.pi - t1, (p1 + np.pi) % (2 * np.pi))
        errs.append(abs(A12 - A21) / max(abs(A12), abs(A21), 1e-300))
    return max(errs)


def _reciprocity_map(T):
    pol, ns, ms = vswf.mode_list(LMAX)
    neg = np.array([(0 if p == 0 else HALF) + vswf.block_index(n, -m)
                    for p, n, m in zip(pol, ns, ms)])
    sign = (-1.0) ** (ms[:, None] + ms[None, :])
    return sign * T[np.ix_(neg, neg)].T


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reciprocity_relation_pinned_by_saxon():
    """A random T symmetrized under the index relation satisfies amplitude
    reciprocity at machine precision; the raw random T does not."""
    rng = np.random.default_rng(7)
    T_rand = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    assert _saxon_err(T_rand, rng) > 0.1                    # violates
    T_sym = 0.5 * (T_rand + _reciprocity_map(T_rand))
    assert _saxon_err(T_sym, rng) < 1e-12                   # satisfies
    # and reciprocity_check agrees with the map used here
    assert reciprocity_check(T_sym, LMAX) < 1e-14
    assert reciprocity_check(T_rand, LMAX) > 0.1


def test_reciprocity_of_mie_sphere():
    T = sphere_reference_T(LMAX, 0.9, 2.0)
    assert reciprocity_check(T, LMAX) < 1e-14


def test_coated_vacuum_shell_reduces_to_core():
    """m_shell = 1: the shell is background -> homogeneous sphere of the core."""
    l = np.arange(1, LMAX + 1)
    for m1 in (2.0, 1.8 + 0.3j):            # B&H convention here (raw mie_ab*)
        a_ref, b_ref = mie_ab(l, 0.8, m1)
        a_c, b_c = mie_ab_coated(l, 0.8, 1.7, m1, 1.0)
        assert np.max(np.abs(a_c - a_ref)) < 1e-12
        assert np.max(np.abs(b_c - b_ref)) < 1e-12


def test_coated_matched_core_reduces_to_outer():
    """m_core = m_shell: homogeneous sphere of the OUTER radius."""
    l = np.arange(1, LMAX + 1)
    for m in (2.0, 1.6 + 0.2j):             # B&H convention here
        a_ref, b_ref = mie_ab(l, 1.7, m)
        a_c, b_c = mie_ab_coated(l, 0.8, 1.7, m, m)
        assert np.max(np.abs(a_c - a_ref)) < 1e-12
        assert np.max(np.abs(b_c - b_ref)) < 1e-12


def test_coated_tmatrix_passivity_and_reciprocity():
    """Lossy coated sphere in the PACKAGE convention (loss = negative imag):
    S = I + 2T must be subunitary, T reciprocal; lossless case unitary."""
    T_lossy = coated_sphere_reference_T(LMAX, 0.8, 1.5, 2.0, 1.8 - 0.25j)
    smax, _ = passivity_check(T_lossy)
    assert smax < 1.0 - 1e-6                # strictly dissipative
    assert reciprocity_check(T_lossy, LMAX) < 1e-14
    T_ll = coated_sphere_reference_T(LMAX, 0.8, 1.5, 2.0, 1.8)
    _, uni = passivity_check(T_ll)
    assert uni < 1e-10                      # lossless -> unitary S
