"""Stage-2-lite: ground-plane coupling via exact PEC image theory.

A PEC mirror at z = -h below the scatterer (center at origin) is handled by
image theory: the image of an outgoing multipole field is an outgoing field
at the image center (0, 0, -2h) with per-mode parity signs s_nu:

    E_im(c_im + rho) = D . V_nu(sigma rho) = s_nu V_nu(rho),
    D = diag(-1, -1, +1),  sigma = diag(1, 1, -1)

(D flips tangential E so that E_tan = 0 on the mirror plane; s_nu is
determined numerically once per basis and verified diagonal).

Single scatterer over mirror:   a = a_inc_tot + R_m f,  f = T a,
    R_m = A((0, 0, -2h)) diag(s)   =>   f = (I - T R_m)^{-1} T a_inc_tot
which is exactly the manual's T_coupled = (I - T R)^{-1} T with the image
reflection operator playing the role of R.

Periodic array over mirror:
    a = a_inc_tot + [C + C_im diag(s)] f
with C the in-plane lattice sum (R != 0) and C_im the image-lattice sum over
ALL R (including R = 0, the cell's own image at distance 2h).

Incident field (incidence from +z going DOWN, E = x_hat e^{-ikz}, PEC at
z = -h):  ground reflection r_g = -e^{2ikh}, and
    a_inc_tot = a_pw(-z_hat, x_hat) + r_g a_pw(+z_hat, x_hat).

Reflection referenced at z = 0:
    S11 = r_g + (2 pi i/(k A)) [e*.F_dir(+z) + e^{2ikh} e*.F_im(+z)]
where F_im is the far field of diag(s) f (image sheet at z = -2h).

Validity caveat: the image center distance 2h must exceed the projection
radius r0, and re-expansion accuracy degrades when 2h is smaller than the
circumscribing radius of the physical cell (Rayleigh-hypothesis regime).
The demo geometry (2h = 0.7 um vs r_circ = 0.72 um) sits at that boundary;
results are qualitative there.
"""
import numpy as np

from tmatrix.aggregation.vswf import ModeBasis, vswf_fields, plane_wave_coeffs, far_field_amplitude
from tmatrix.aggregation.translate import translation_matrix, lattice_sum_C


def mirror_parity_signs(modes, k=1.0, rng_seed=7):
    """Diagonal signs s_nu with D.V_nu(sigma rho) = s_nu V_nu(rho).

    Computed numerically and verified to be a consistent +/-1 per mode.
    """
    rng = np.random.default_rng(rng_seed)
    pts = rng.normal(size=(8, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= rng.uniform(1.0, 2.0, size=(8, 1))
    E, _ = vswf_fields(k, modes, pts, "outgoing")
    sig = np.array([1.0, 1.0, -1.0])
    D = np.array([-1.0, -1.0, 1.0])
    Em, _ = vswf_fields(k, modes, pts * sig, "outgoing")
    s = np.empty(modes.n)
    for i in range(modes.n):
        mask = np.abs(E[i]) > 1e-3 * np.abs(E[i]).max()
        vals = (D * Em[i])[mask] / E[i][mask]
        s_i = np.mean(vals).real
        assert np.allclose(vals, round(s_i), atol=1e-9), \
            f"mirror map not diagonal for mode {i}"
        s[i] = round(s_i)
    return s


def solve_single_mirror(T, R_m, a_inc_tot):
    """f = (I - T R_m)^{-1} T a_inc_tot (manual's T_coupled applied)."""
    n = T.shape[0]
    f = np.linalg.solve(np.eye(n, dtype=complex) - T @ R_m, T @ a_inc_tot)
    return f


def sparams_mirror_periodic(k, h, A_cell, modes, T, C, C_im, s,
                            e_co=(1, 0, 0), e_cross=(0, 1, 0)):
    """S11 (co, cross) for the periodic array over a PEC mirror at z = -h.

    C: in-plane lattice sum (R != 0); C_im: image-lattice sum (all R,
    z-offset -2h).  Returns dict with S11 terms and absorption.
    """
    e_co = np.asarray(e_co, dtype=complex)
    e_cross = np.asarray(e_cross, dtype=complex)
    r_g = -np.exp(2j * k * h)
    a_inc = plane_wave_coeffs([0, 0, -1], e_co, modes) \
        + r_g * plane_wave_coeffs([0, 0, 1], e_co, modes)
    n = modes.n
    Ctot = C + C_im * s[None, :]          # C_im @ diag(s)
    a = np.linalg.solve(np.eye(n, dtype=complex) - Ctot @ T, a_inc)
    f = T @ a
    pref = 2j * np.pi / (k * A_cell)
    up = np.array([[0, 0, 1.0]])
    F_dir = far_field_amplitude(k, f, modes, up)[0]
    F_im = far_field_amplitude(k, s * f, modes, up)[0]
    ph = np.exp(2j * k * h)
    S11_co = r_g + pref * (np.vdot(e_co, F_dir) + ph * np.vdot(e_co, F_im))
    S11_x = pref * (np.vdot(e_cross, F_dir) + ph * np.vdot(e_cross, F_im))
    R = abs(S11_co) ** 2 + abs(S11_x) ** 2
    return {"S11_co": S11_co, "S11_cross": S11_x, "R": R, "A": 1.0 - R}


def image_lattice_sum(k, pitch, modes, h, r0, quad, kRc=(10.0, 14.0, 20.0),
                      r_max_factor=3.5, projector=None):
    """C_im = sum_{all R} A((R_x, R_y, -2h)) via shells with z-offset."""
    from tmatrix.aggregation.translate import square_lattice_shells, translation_shells_offset, \
        assemble_shell_sum
    kRc = np.asarray(kRc, dtype=float)
    Rcs = kRc / k
    r_max = r_max_factor * Rcs.max()
    radii, angles = square_lattice_shells(pitch, r_max)
    radii = np.concatenate([[0.0], radii])          # own image at R = 0
    angles = [np.array([0.0])] + angles
    A_s = translation_shells_offset(k, radii, -2 * h, modes, r0, quad,
                                    projector=projector)
    return assemble_shell_sum(A_s, radii, angles, modes, Rcs)
