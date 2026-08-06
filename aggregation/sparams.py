"""Deliverable 2: S-parameters from aggregated T-matrix scattering coefficients.

For a periodic array (unit-cell area A_cell, subwavelength pitch: only the
0th diffraction order propagates), a normally incident plane wave
E = e_hat exp(ikz) produces per-cell outgoing coefficients f, whose far field
is F(r_hat) (vswf.far_field_amplitude).  Summing the lattice of spherical
waves gives the 0th-order plane-wave amplitudes (physics convention e^{-iwt}):

    S21(co)  = 1 + (2 pi i / (k A_cell cos th)) e_hat* . F(+z_hat)
    S11(co)  =     (2 pi i / (k A_cell cos th)) e_ref* . F(-z_hat)

(matches the manual's Eqs. with j -> i).  Cross-polarized terms use the
orthogonal polarization vector.  For a finite array of N cells, A_cell ->
N A_cell and F is the phase-referenced sum of per-cell far fields.

Single-scatterer cross sections (used for validation):
    sigma_ext = (4 pi / k) Im[e_hat* . F(k_inc_hat)]
    sigma_sca = int |F|^2 dOmega
    sigma_abs = sigma_ext - sigma_sca  >= 0  for a passive scatterer.
"""
import numpy as np

from vswf import far_field_amplitude, sphere_quadrature


def sparams_normal(k, A_cell, modes, f, e_co=(1, 0, 0), e_cross=(0, 1, 0)):
    """S-parameters for normal incidence (+z propagation), one unit cell.

    f: per-cell outgoing coefficients (already includes array coupling).
    Returns dict with complex S21/S11 co- and cross-polarized.
    """
    pref = 2j * np.pi / (k * A_cell)
    F = far_field_amplitude(k, f, modes, np.array([[0, 0, 1.0], [0, 0, -1.0]]))
    e_co = np.asarray(e_co, dtype=complex)
    e_cross = np.asarray(e_cross, dtype=complex)
    return {
        "S21_co": 1.0 + pref * np.vdot(e_co, F[0]),
        "S21_cross": pref * np.vdot(e_cross, F[0]),
        "S11_co": pref * np.vdot(e_co, F[1]),
        "S11_cross": pref * np.vdot(e_cross, F[1]),
    }


def energy_balance(S):
    """Returns (R, T, A) power terms; A = 1 - R - T must be in [0, 1]."""
    R = abs(S["S11_co"]) ** 2 + abs(S["S11_cross"]) ** 2
    T = abs(S["S21_co"]) ** 2 + abs(S["S21_cross"]) ** 2
    return R, T, 1.0 - R - T


def cross_sections(k, modes, f, k_hat, e_hat, n_theta=40, n_phi=80):
    """(sigma_ext, sigma_sca, sigma_abs) for a single scatterer, unit E0."""
    k_hat = np.asarray(k_hat, dtype=float)
    F_fwd = far_field_amplitude(k, f, modes, k_hat[None, :])[0]
    sigma_ext = 4 * np.pi / k * np.imag(np.vdot(e_hat, F_fwd))
    TH, PH, W = sphere_quadrature(n_theta, n_phi)
    rhat = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                     np.cos(TH)], axis=1)
    F = far_field_amplitude(k, f, modes, rhat)
    sigma_sca = float(np.sum(W * np.sum(np.abs(F) ** 2, axis=1)))
    return sigma_ext, sigma_sca, sigma_ext - sigma_sca
