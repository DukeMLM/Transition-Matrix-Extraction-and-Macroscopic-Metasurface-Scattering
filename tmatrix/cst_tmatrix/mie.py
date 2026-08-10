"""Analytic Mie theory for homogeneous and PEC spheres.

Used ONLY for validation: the T-matrix of a sphere is diagonal with entries
given by the Mie coefficients, so every extracted sphere T-matrix can be
checked against these closed forms.

The Bohren & Huffman closed forms below (mie_ab, mie_ab_pec) are written in
the literature convention e^{-i omega t} / h_l^(1).  The package convention
(vswf.py, and CST itself) is e^{+j omega t} / h_l^(2) — the two are related
by complex conjugation, so sphere_tmatrix_diagonal conjugates on the way
out:
    T^{MM}_l = conj(-b_l)   (magnetic / TE, M-modes)
    T^{NN}_l = conj(-a_l)   (electric / TM, N-modes)
For lossy spheres pass m in the package convention (loss = NEGATIVE
imaginary part); it is conjugated into the B&H convention internally.

This conjugation is NOT testable offline (synthetic tests are self-
consistent under conjugation of the reference); it was pinned by a live CST
run: the measured scattered coefficients of an eps=4 sphere matched
conj(T_BH) @ a to 1.6e-3 while T_BH itself was off by O(2).
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def _psi(l, z):
    """Riccati-Bessel psi_l(z) = z j_l(z) and derivative, complex-safe."""
    z = np.asarray(z, dtype=complex)
    jl = _sph_jn_complex(l, z)
    jlp = _sph_jn_complex(l, z, derivative=True)
    return z * jl, jl + z * jlp


def _sph_jn_complex(l, z, derivative=False):
    """spherical_jn supports complex arguments via ascending recurrence on
    j_l = sqrt(pi/2z) J_{l+1/2}; scipy's spherical_jn accepts complex z."""
    return spherical_jn(l, z, derivative=derivative)


def _xi(l, x):
    """Riccati-Hankel xi_l(x) = x h_l^(1)(x) and derivative (real argument)."""
    jl = spherical_jn(l, x)
    yl = spherical_yn(l, x)
    jlp = spherical_jn(l, x, derivative=True)
    ylp = spherical_yn(l, x, derivative=True)
    hl = jl + 1j * yl
    hlp = jlp + 1j * ylp
    return x * hl, hl + x * hlp


def mie_ab(l: np.ndarray, x: float, m: complex):
    """Bohren & Huffman Mie coefficients a_l (electric), b_l (magnetic).

    Parameters
    ----------
    l : array of multipole orders (int, >=1)
    x : size parameter k*a of the sphere in the background medium
    m : refractive-index contrast n_sphere / n_background (complex)
    """
    l = np.asarray(l)
    psi_x, dpsi_x = _psi(l, x)
    psi_mx, dpsi_mx = _psi(l, m * x)
    xi_x, dxi_x = _xi(l, x)

    a = (m * psi_mx * dpsi_x - psi_x * dpsi_mx) / (
        m * psi_mx * dxi_x - xi_x * dpsi_mx)
    b = (psi_mx * dpsi_x - m * psi_x * dpsi_mx) / (
        psi_mx * dxi_x - m * xi_x * dpsi_mx)
    return a, b


def _chi(l, z):
    """Riccati-Bessel chi_l(z) = -z y_l(z) and derivative, complex-safe."""
    z = np.asarray(z, dtype=complex)
    yl = spherical_yn(l, z)
    ylp = spherical_yn(l, z, derivative=True)
    return -z * yl, -(yl + z * ylp)


def mie_ab_coated(l: np.ndarray, x: float, y: float, m1: complex, m2: complex):
    """Bohren & Huffman coated-sphere Mie coefficients a_l, b_l (B&H sec. 8.1,
    e^{-i omega t} convention — conjugated by the *_tmatrix_diagonal wrapper).

    x : size parameter k*a of the CORE
    y : size parameter k*b of the SHELL (outer), y >= x
    m1, m2 : core / shell refractive-index contrast vs the background,
        in the B&H convention (loss = positive imaginary part)
    """
    l = np.asarray(l)
    psi_m1x, dpsi_m1x = _psi(l, m1 * x)
    psi_m2x, dpsi_m2x = _psi(l, m2 * x)
    chi_m2x, dchi_m2x = _chi(l, m2 * x)
    psi_m2y, dpsi_m2y = _psi(l, m2 * y)
    chi_m2y, dchi_m2y = _chi(l, m2 * y)
    psi_y, dpsi_y = _psi(l, y)
    xi_y, dxi_y = _xi(l, y)

    A = (m2 * psi_m2x * dpsi_m1x - m1 * dpsi_m2x * psi_m1x) / (
        m2 * chi_m2x * dpsi_m1x - m1 * dchi_m2x * psi_m1x)
    B = (m2 * psi_m1x * dpsi_m2x - m1 * psi_m2x * dpsi_m1x) / (
        m2 * dchi_m2x * psi_m1x - m1 * dpsi_m1x * chi_m2x)

    a = (psi_y * (dpsi_m2y - A * dchi_m2y)
         - m2 * dpsi_y * (psi_m2y - A * chi_m2y)) / (
        xi_y * (dpsi_m2y - A * dchi_m2y)
        - m2 * dxi_y * (psi_m2y - A * chi_m2y))
    b = (m2 * psi_y * (dpsi_m2y - B * dchi_m2y)
         - dpsi_y * (psi_m2y - B * chi_m2y)) / (
        m2 * xi_y * (dpsi_m2y - B * dchi_m2y)
        - dxi_y * (psi_m2y - B * chi_m2y))
    return a, b


def mie_ab_pec(l: np.ndarray, x: float):
    """PEC sphere Mie coefficients."""
    l = np.asarray(l)
    psi_x, dpsi_x = _psi(l, x)
    xi_x, dxi_x = _xi(l, x)
    a = psi_x / xi_x
    b = dpsi_x / dxi_x
    return a, b


def sphere_tmatrix_diagonal(lmax: int, x: float, m: complex | None = None):
    """Diagonal of the sphere T-matrix in the PACKAGE convention
    (e^{+j omega t}, h^(2)): (T_M, T_N) each of shape (lmax,), entry index
    l-1 for order l. m=None means PEC; lossy m has negative imaginary part
    in this convention (conjugated into B&H internally)."""
    l = np.arange(1, lmax + 1)
    if m is None:
        a, b = mie_ab_pec(l, x)
    else:
        a, b = mie_ab(l, x, np.conj(m))
    # conjugate: B&H e^{-i omega t}/h^(1)  ->  package e^{+j omega t}/h^(2)
    return -np.conj(b), -np.conj(a)


def coated_sphere_tmatrix_diagonal(lmax: int, x_core: float, x_shell: float,
                                   m_core: complex, m_shell: complex):
    """Diagonal of the coated (core-shell) sphere T-matrix in the PACKAGE
    convention (e^{+j omega t}, h^(2)): (T_M, T_N) each of shape (lmax,).

    x_core / x_shell : size parameters k*a (core) and k*b (outer shell).
    m_core / m_shell : refractive-index contrasts in the package convention
        (loss = NEGATIVE imaginary part); conjugated into B&H internally.
    """
    l = np.arange(1, lmax + 1)
    a, b = mie_ab_coated(l, x_core, x_shell,
                         np.conj(m_core), np.conj(m_shell))
    # conjugate: B&H e^{-i omega t}/h^(1)  ->  package e^{+j omega t}/h^(2)
    return -np.conj(b), -np.conj(a)


def efficiencies(x: float, m: complex | None = None, lmax: int | None = None):
    """Extinction and scattering efficiencies Q_ext, Q_sca (sanity checks).
    m in the package convention (loss = negative imaginary part)."""
    if lmax is None:
        lmax = wiscombe_lmax(x)
    l = np.arange(1, lmax + 1)
    if m is None:
        a, b = mie_ab_pec(l, x)
    else:
        a, b = mie_ab(l, x, np.conj(m))
    pref = (2 * l + 1)
    q_ext = 2 / x**2 * np.sum(pref * (a + b).real)
    q_sca = 2 / x**2 * np.sum(pref * (np.abs(a) ** 2 + np.abs(b) ** 2))
    return q_ext, q_sca


def wiscombe_lmax(x: float, margin: int = 0) -> int:
    """Wiscombe truncation criterion for size parameter x = k*R of the
    smallest sphere circumscribing the scatterer."""
    if x <= 8:
        n = x + 4 * x ** (1 / 3) + 1
    elif x < 4200:
        n = x + 4.05 * x ** (1 / 3) + 2
    else:
        n = x + 4 * x ** (1 / 3) + 2
    return int(np.ceil(n)) + margin
