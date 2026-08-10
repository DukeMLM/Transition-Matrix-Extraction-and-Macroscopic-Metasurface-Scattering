"""End-to-end synthetic validation: fake 'CST' data from Mie theory, run the
full extraction pipeline (near-field route AND farfield route), compare the
recovered T-matrix against the analytic Mie T-matrix.

This is the exact pipeline that will run on real CST exports — only the
field source differs.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst_tmatrix import vswf
from cst_tmatrix.quadrature import (fibonacci_directions, gauss_legendre_sphere,
                                    unit_vectors, spherical_basis)
from cst_tmatrix.tmatrix_solve import solve_tmatrix, passivity_check, sphere_reference_T

# --- scenario: dielectric sphere ------------------------------------------
X = 2.0            # size parameter k*a
M_REL = 1.5 + 0j   # refractive index contrast (lossless -> unitarity check)
LMAX = 6
K = 1.0            # wavenumber (units absorb into radii)
A_SPH = X / K
R_MON = 2.0 * A_SPH          # monitor sphere radius (clear of scatterer)

T_true = sphere_reference_T(LMAX, X, M_REL)

# --- illuminations ---------------------------------------------------------
N = vswf.n_modes(LMAX)
n_dirs = int(np.ceil(0.75 * N))      # x2 pols -> 1.5x overdetermined
th_i, ph_i = fibonacci_directions(n_dirs)

# --- monitor grid ----------------------------------------------------------
th, ph, w, _ = gauss_legendre_sphere(LMAX, oversample=1.2)
pts = R_MON * unit_vectors(th, ph)

# --- farfield grid (same quadrature reused) --------------------------------
A_cols, F_near_cols, F_far_cols = [], [], []
for i in range(n_dirs):
    for pol in ("theta", "phi"):
        a = vswf.plane_wave_coefficients(th_i[i], ph_i[i], pol, LMAX)
        f = T_true @ a
        # synthetic 'CST' total near field on the monitor sphere:
        E_inc, _ = vswf.plane_wave_field(th_i[i], ph_i[i], pol, K, pts)
        E_sca = vswf.evaluate_field(f, LMAX, "outgoing", K, pts)
        E_tot = E_inc + E_sca
        # pipeline near-field route: subtract incident, project
        f_rec = vswf.project_surface_field(E_tot - E_inc, LMAX, "outgoing",
                                           K, R_MON, th, ph, w)
        # pipeline farfield route: exact asymptotic pattern F(th,ph)
        # F = sum f_nm j^{n+1} C_nm + g_nm j^n B_nm  -> synthesize directly
        kr_far = 1e7
        pts_far = (kr_far / K) * unit_vectors(th, ph)
        E_far = vswf.evaluate_field(f, LMAX, "outgoing", K, pts_far)
        _, th_hat, ph_hat = spherical_basis(th, ph)
        F_th = np.einsum("ij,ij->i", E_far, th_hat) * kr_far * np.exp(1j * kr_far)
        F_ph = np.einsum("ij,ij->i", E_far, ph_hat) * kr_far * np.exp(1j * kr_far)
        f_rec_far = vswf.project_farfield(F_th, F_ph, LMAX, K, th, ph, w)

        A_cols.append(a)
        F_near_cols.append(f_rec)
        F_far_cols.append(f_rec_far)

A = np.array(A_cols).T
F_near = np.array(F_near_cols).T
F_far = np.array(F_far_cols).T

for label, F in [("near-field", F_near), ("farfield ", F_far)]:
    T_rec, diag = solve_tmatrix(A, F)
    err = np.max(np.abs(T_rec - T_true))
    smax, uni = passivity_check(T_rec)
    print(f"{label}: max|T_rec - T_mie| = {err:.3e}   "
          f"lstsq residual = {diag['residual']:.1e}   cond(A) = {diag['cond']:.1f}")
    print(f"           S-matrix max singular value = {smax:.6f}  "
          f"unitarity error = {uni:.2e} (lossless -> ~0)")

# PEC sphere too
T_pec = sphere_reference_T(LMAX, X, None)
F_cols = []
for i in range(n_dirs):
    for pol in ("theta", "phi"):
        a = vswf.plane_wave_coefficients(th_i[i], ph_i[i], pol, LMAX)
        f = T_pec @ a
        E_sca = vswf.evaluate_field(f, LMAX, "outgoing", K, pts)
        F_cols.append(vswf.project_surface_field(E_sca, LMAX, "outgoing",
                                                 K, R_MON, th, ph, w))
T_rec, diag = solve_tmatrix(A, np.array(F_cols).T)
print(f"PEC sphere: max|T_rec - T_mie| = {np.max(np.abs(T_rec - T_pec)):.3e}")
