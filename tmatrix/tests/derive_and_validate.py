"""Offline derivation/validation of the VSWF machinery.

Stages:
  1. Y_nm, tau, pi against scipy sph_harm + finite differences.
  2. Projection round-trip: random coefficients -> field on sphere -> project.
  3. Plane-wave expansion: numeric ground truth by projection (E+H), then
     comparison against the analytic formula (reports per-(n,pol) ratios so
     any sign error is immediately visible and fixable).
  4. Farfield projection round-trip against evaluate_field at large kr.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.special import sph_harm
from cst_tmatrix import vswf
from cst_tmatrix.quadrature import gauss_legendre_sphere, unit_vectors

rng = np.random.default_rng(7)
LMAX = 5
K = 1.0
R = 2.0 / K          # kr = 2: plane-wave content above n~10 negligible

th, ph, w, _ = gauss_legendre_sphere(2 * LMAX + 6)   # heavy oversampling
pts = R * unit_vectors(th, ph)

print("=== Stage 1: Y / tau / pi vs scipy ===")
P = vswf._legendre_norm(LMAX, np.cos(th), np.sin(th))
pi_t, tau_t = vswf.pi_tau(LMAX, th)
ok = True
for n in range(1, LMAX + 1):
    for m in range(-n, n + 1):
        Y_mine = vswf._P_signed(P, n, m, th.size) * np.exp(1j * m * ph)
        Y_ref = sph_harm(m, n, ph, th)
        err = np.max(np.abs(Y_mine - Y_ref))
        if err > 1e-10:
            ok = False
            print(f"  Y mismatch n={n} m={m}: {err:.2e}")
        # tau by finite difference on theta (interior nodes only)
        dth = 1e-6
        Pp = vswf._legendre_norm(n, np.cos(th + dth), np.sin(th + dth))
        Pm = vswf._legendre_norm(n, np.cos(th - dth), np.sin(th - dth))
        tau_fd = (vswf._P_signed(Pp, n, m, th.size)
                  - vswf._P_signed(Pm, n, m, th.size)) / (2 * dth)
        err_t = np.max(np.abs(tau_t[n, m + LMAX] - tau_fd))
        if err_t > 1e-5:
            ok = False
            print(f"  tau mismatch n={n} m={m}: {err_t:.2e}")
print("  PASS" if ok else "  FAIL")

print("=== Stage 2: projection round-trip (regular, E-only / E+H) ===")
c0 = rng.standard_normal(vswf.n_modes(LMAX)) + 1j * rng.standard_normal(vswf.n_modes(LMAX))
E, ZH = vswf.evaluate_EH(c0, LMAX, "regular", K, pts)
c_E = vswf.project_surface_field(E, LMAX, "regular", K, R, th, ph, w)
c_EH = vswf.project_surface_field(E, LMAX, "regular", K, R, th, ph, w, H=ZH)
print(f"  E-only : {np.max(np.abs(c_E - c0)):.2e}")
print(f"  E + H  : {np.max(np.abs(c_EH - c0)):.2e}")

print("=== Stage 2b: projection round-trip (outgoing) ===")
E, ZH = vswf.evaluate_EH(c0, LMAX, "outgoing", K, pts)
c_o = vswf.project_surface_field(E, LMAX, "outgoing", K, R, th, ph, w)
print(f"  E-only : {np.max(np.abs(c_o - c0)):.2e}")

print("=== Stage 3: plane-wave expansion ===")
for (ti, pi_i, pol) in [(0.7, 1.1, "theta"), (0.7, 1.1, "phi"),
                        (2.2, 4.9, "theta"), (2.2, 4.9, "phi")]:
    Epw, ZHpw = vswf.plane_wave_field(ti, pi_i, pol, K, pts)
    a_num = vswf.project_surface_field(Epw, LMAX, "regular", K, R, th, ph, w,
                                       H=ZHpw)
    a_ana = vswf.plane_wave_coefficients(ti, pi_i, pol, LMAX)
    # report worst absolute mismatch and per-(n, block) ratio table
    err = np.max(np.abs(a_num - a_ana))
    scale = np.max(np.abs(a_num))
    print(f"  dir=({ti},{pi_i}) pol={pol}: |a_num|max={scale:.3f} "
          f"max|a_num - a_ana|={err:.2e}")
    if err > 1e-6 * scale:
        half = vswf.n_modes(LMAX) // 2
        for n in range(1, LMAX + 1):
            for blk, name in [(0, "M"), (half, "N")]:
                iv = [blk + vswf.block_index(n, m) for m in range(-n, n + 1)]
                num = a_num[iv]
                ana = a_ana[iv]
                big = np.abs(num) > 1e-8
                if big.any():
                    ratio = num[big] / np.where(np.abs(ana[big]) < 1e-300, np.nan, ana[big])
                    print(f"    n={n} {name}: ratio num/ana = {np.round(ratio, 6)}")

print("=== Stage 4: field reconstruction from analytic PW coefficients ===")
# evaluate the truncated expansion INSIDE the convergence sphere and compare
r_test = 1.2 / K
tp, pp = np.linspace(0.3, 2.8, 7), np.linspace(0, 6, 7)
test_pts = r_test * unit_vectors(np.repeat(tp, 7), np.tile(pp, 7))
a_ana = vswf.plane_wave_coefficients(0.7, 1.1, "theta", 12)
E_rec = vswf.evaluate_field(a_ana, 12, "regular", K, test_pts)
E_ref, _ = vswf.plane_wave_field(0.7, 1.1, "theta", K, test_pts)
print(f"  max|E_rec - E_pw| = {np.max(np.abs(E_rec - E_ref)):.2e} (want <1e-6)")

print("=== Stage 5: farfield projection consistency ===")
c1 = rng.standard_normal(vswf.n_modes(LMAX)) + 1j * rng.standard_normal(vswf.n_modes(LMAX))
kr_far = 4000.0
pts_far = (kr_far / K) * unit_vectors(th, ph)
E_far = vswf.evaluate_field(c1, LMAX, "outgoing", K, pts_far)
r_hat, th_hat, ph_hat = __import__("cst_tmatrix.quadrature", fromlist=["spherical_basis"]).spherical_basis(th, ph)
F_th = np.einsum("ij,ij->i", E_far, th_hat) * kr_far * np.exp(1j * kr_far)
F_ph = np.einsum("ij,ij->i", E_far, ph_hat) * kr_far * np.exp(1j * kr_far)
c_far = vswf.project_farfield(F_th, F_ph, LMAX, K, th, ph, w)
print(f"  max|c_far - c| = {np.max(np.abs(c_far - c1)):.2e} (finite-kr error ~1/kr = {1/kr_far:.1e})")
