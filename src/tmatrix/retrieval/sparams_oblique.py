"""Oblique-incidence specular Jones blocks (retrieval deliverable 2).

Implements the par. 2 forward map of INVERSE_TMATRIX_FROM_FLOQUET.md for an
arbitrary incidence direction k_hat (either hemisphere, k_hat_z != 0), in the
e^{-i omega t} convention of the aggregation package.  For a unit-amplitude
incident plane wave E = e_b exp(i k k_hat . r) on the infinite square array
(cell area A), the specular (0th-order) Jones blocks are

    S21_ab = delta_ab + (2 pi i / (k A cos th)) e_a^(t)* . F_b(k_hat_t)
    S11_ab =            (2 pi i / (k A cos th)) e_a^(r)* . F_b(k_hat_r)

with k_hat_t = k_hat, k_hat_r = k_hat - 2 (k_hat . z_hat) z_hat (same k_par,
flipped k_z), cos th = |k_hat_z|, and F_b = vswf.far_field_amplitude of the
per-cell outgoing coefficients f_b under incident polarization e_b.
Row a = RECEIVE polarization, column b = INCIDENT polarization, in both
blocks.

NORMATIVE index order: 0 = TE, 1 = TM, in rows and columns of every 2x2
block returned by this module.

NORMATIVE polarization basis (doc par. 2), evaluated SEPARATELY for k_hat_t
and k_hat_r.  Parametrized by (theta, phi, direction) with

    k_hat = (sin th cos ph, sin th sin ph, direction * cos th),
    direction in {+1, -1},  0 <= th < pi/2:

    e_TE(k_hat) = (z_hat x k_hat)/|z_hat x k_hat| = (-sin ph, cos ph, 0)
    e_TM(k_hat) = e_TE(k_hat) x k_hat
                = (k_hat_z cos ph, k_hat_z sin ph, -sin th)

Both expressions are smooth in theta at fixed phi, so the theta -> 0 limit is
taken by continuity along fixed phi: at exact theta = 0 they evaluate to
e_TE = (-sin ph, cos ph, 0), e_TM = (direction cos ph, direction sin ph, 0),
using the phi carried by the caller.  The reflected basis is obtained by the
same formulas with direction -> -direction (k_hat_r has k_z of the opposite
sign but the same in-plane part).

theta -> 0 mapping to sparams.sparams_normal (derived here, asserted to
<= 1e-12 in test_sparams_oblique.py test (e)).  At theta = 0, phi = 0,
direction = +1: e_TE^t = e_TE^r = y_hat, e_TM^t = +x_hat, but
e_TM^r = -x_hat (k_hat_r = -z_hat flips k_hat_z, hence the reflected-TM
sign).  With

    Sn_x = sparams_normal(k, A, modes, f_x, e_co=x_hat, e_cross=y_hat)
           (f_x from incidence e = x_hat = e_TM), and
    Sn_y = sparams_normal(k, A, modes, f_y, e_co=y_hat, e_cross=x_hat)
           (f_y from incidence e = y_hat = e_TE),

the exact mapping is

    S21[0,0] = +Sn_y[S21_co]      S21[0,1] = +Sn_x[S21_cross]
    S21[1,0] = +Sn_y[S21_cross]   S21[1,1] = +Sn_x[S21_co]
    S11[0,0] = +Sn_y[S11_co]      S11[0,1] = +Sn_x[S11_cross]
    S11[1,0] = -Sn_y[S11_cross]   S11[1,1] = -Sn_x[S11_co]

i.e. every S21 entry and the TE receive row of S11 map with + sign; the TM
receive row of S11 (row index 1) carries the -1 from e_TM^r = -x_hat.
"""

import numpy as np

from tmatrix.aggregation.vswf import plane_wave_coeffs, far_field_amplitude
from tmatrix.aggregation.aggregate import solve_periodic

Z_HAT = np.array([0.0, 0.0, 1.0])


def _check_direction(direction):
    d = float(direction)
    if d not in (1.0, -1.0):
        raise ValueError("direction must be +1 or -1")
    return d


def khat_from_angles(theta, phi, direction=+1):
    """Unit propagation direction
    k_hat = (sin th cos ph, sin th sin ph, direction cos th)."""
    d = _check_direction(direction)
    st, ct = np.sin(theta), np.cos(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), d * ct])


def pol_basis(theta, phi, direction=+1):
    """Normative (e_TE, e_TM) for k_hat(theta, phi, direction).

    e_TE = (z_hat x k_hat)/|z_hat x k_hat| = (-sin ph, cos ph, 0),
    e_TM = e_TE x k_hat = (k_z cos ph, k_z sin ph, -sin th),
    k_z = direction cos th.  The theta -> 0 limit is by continuity along
    fixed phi (the formulas are already that limit at theta = 0).  For the
    reflected wave k_hat_r pass direction -> -direction (same theta, phi).
    Returns (e_TE, e_TM), each a real (3,) unit vector.
    """
    d = _check_direction(direction)
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    kz = d * ct
    e_te = np.array([-sp, cp, 0.0])
    e_tm = np.array([kz * cp, kz * sp, -st])
    return e_te, e_tm


def jones_blocks(k, A_cell, modes, f_pair, theta, phi, direction=+1):
    """Specular Jones blocks from per-cell outgoing coefficients.

    f_pair : (f_TE, f_TM) -- per-cell outgoing VSWF coefficients under
        incident polarization e_TE and e_TM respectively (normative order:
        index 0 = TE, 1 = TM).  These must already include the array
        coupling (aggregate.solve_periodic) for the same (theta, phi,
        direction).
    Returns (S11, S21), each (2, 2) complex, rows = receive polarization,
    columns = incident polarization, index order 0 = TE, 1 = TM (normative).
    """
    d = _check_direction(direction)
    khat_t = khat_from_angles(theta, phi, d)
    khat_r = khat_t - 2.0 * khat_t[2] * Z_HAT
    cos_th = abs(khat_t[2])
    if cos_th < 1e-12:
        raise ValueError("grazing incidence (cos th ~ 0) is not supported")
    pref = 2j * np.pi / (k * A_cell * cos_th)
    e_t = pol_basis(theta, phi, d)            # (e_TE, e_TM) at k_hat_t
    e_r = pol_basis(theta, phi, -d)           # (e_TE, e_TM) at k_hat_r
    S11 = np.zeros((2, 2), dtype=complex)
    S21 = np.zeros((2, 2), dtype=complex)
    rhat = np.stack([khat_t, khat_r])
    for b in range(2):
        F = far_field_amplitude(k, f_pair[b], modes, rhat)
        for a in range(2):
            S21[a, b] = (1.0 if a == b else 0.0) + pref * np.vdot(e_t[a], F[0])
            S11[a, b] = pref * np.vdot(e_r[a], F[1])
    return S11, S21


def sparams_oblique(k, A_cell, modes, T, C, theta, phi, direction=+1):
    """Full par. 2 forward map: T-matrix + Bloch lattice sum -> Jones blocks.

    For each incident polarization e_b in the normative order (0 = TE,
    1 = TM), builds a_inc,b = plane_wave_coeffs(k_hat, e_b), solves the
    periodic Foldy-Lax system f_b = T (I - C T)^{-1} a_inc,b
    (aggregate.solve_periodic), and projects the specular orders.

    C must be the Bloch lattice sum at THIS (k, k_par):
    C = lattice_sum_C_bloch(k, pitch, modes, r0, quad,
                            k_par = k sin th (cos ph, sin ph)) --
    note k_par does not depend on `direction` (k_hat and k_hat_r share it).

    Returns dict with:
        "S11", "S21" : (2, 2) complex Jones blocks (rows = receive, columns
                       = incident polarization; 0 = TE, 1 = TM, normative)
        "f"          : (2, modes.n) per-cell outgoing coefficients per
                       incident polarization
        "a_inc"      : (2, modes.n) incident regular coefficients
    """
    d = _check_direction(direction)
    khat = khat_from_angles(theta, phi, d)
    e_b = pol_basis(theta, phi, d)
    a_inc = np.empty((2, modes.n), dtype=complex)
    f = np.empty((2, modes.n), dtype=complex)
    for b in range(2):
        a_inc[b] = plane_wave_coeffs(khat, e_b[b], modes)
        _, f[b] = solve_periodic(T, C, a_inc[b])
    S11, S21 = jones_blocks(k, A_cell, modes, f, theta, phi, d)
    return {"S11": S11, "S21": S21, "f": f, "a_inc": a_inc}


def energy_balance_oblique(S11, S21):
    """Per incident polarization b (columns; 0 = TE, 1 = TM):
    R_b = sum_a |S11[a,b]|^2, T_b = sum_a |S21[a,b]|^2, A_b = 1 - R_b - T_b.

    Valid when only the specular order propagates (subwavelength lattice):
    incident, reflected and transmitted orders share the same |cos th|, so no
    obliquity factors appear in the ratios.  Returns (R, T, A), each (2,).
    """
    R = (np.abs(S11) ** 2).sum(axis=0)
    T = (np.abs(S21) ** 2).sum(axis=0)
    return R, T, 1.0 - R - T
