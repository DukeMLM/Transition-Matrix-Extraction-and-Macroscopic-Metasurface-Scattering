"""Validation ladder for supercell.py (manual sections 6.5.5, 6.6, 7.5, 8, 6.4).

    python test_supercell.py            # full ladder, Ewald coupling
    python test_supercell.py --taper    # add the tapered-sum comparison (slow)

Coupling.  The manual's Eq. (48) blocks are computed two ways: the repository's
own Gaussian-taper + Richardson sum (`supercell.block_lattice_sums`) and Ewald
summation through treams (`ewald_supercell.block_lattice_sums_ewald`).  The
ladder below runs on the Ewald blocks, because the tapered blocks do NOT
converge for shifted sub-lattices -- section [taper] measures by how much and
shows the two agreeing where the taper is converged.  This is the choice manual
section 6.5.3 anticipates ("a validated Ewald or reciprocal-space method is an
alternative").

Checks, in the manual's numbering:

  6.5.5-1  M = 1 reproduces translate.lattice_sum_C / aggregate.solve_periodic
  6.5.5-2  a 2x2 supercell of four identical atoms at base pitch p reproduces
           the one-atom primitive lattice of pitch p (coupling and complex S)
  6.5.5-3  permutation invariance of the basis atoms
  6.5.5-4  needs CST -- see cst_supercell/
  6.5.5-5  power balance and reciprocity
  6.5.5-6  pair-block convergence (Ewald eta bracket; taper set, r0, quadrature
           and real-space truncation for the tapered sum)
  6.6      translation validity, Rg round trip, plane-wave phase identity
  7.5/8    output channels, background test S = S_bg, the checkerboard
           selection rule
  6.4      finite cluster T^O: far field vs the multi-center sum, L_C
           convergence, origin-shift covariance
"""
import argparse
import sys

import numpy as np

from tmat_io import TMatrixData
from vswf import (ModeBasis, RegularProjector, far_field_amplitude,
                  plane_wave_coeffs)
from translate import lattice_sum_C, make_quad
from aggregate import solve_periodic, build_finite_system, solve_finite
from sparams import sparams_normal
import supercell as sc
import ewald_supercell as ew

H5_A = "../test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5"
H5_B = "../test/2x2/saw_gold_wl17p30um_10to34THz.tmat.h5"
PITCH = 8.0
R0 = 3.0
A_RADIUS, B_RADIUS = 2.87712, 3.59639      # circumscribing radii (um)

_fails = []


def check(name, value, tol, unit=""):
    ok = np.isfinite(value) and value <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<58s} {value:11.3e} "
          f"(tol {tol:.0e}){unit}")
    if not ok:
        _fails.append(name)
    return ok


def info(name, value, unit=""):
    print(f"  [info] {name:<58s} {value:11.3e}{unit}")


def truncate(data, lmax):
    sel = np.nonzero(data.modes.l <= lmax)[0]
    return sel, ModeBasis(data.modes.l[sel], data.modes.m[sel],
                          data.modes.pol[sel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taper", action="store_true",
                    help="also run the tapered-sum comparison (minutes)")
    ap.add_argument("--lmax", type=int, default=3)
    ap.add_argument("--index", type=int, default=12)
    args = ap.parse_args()

    dA, dB = TMatrixData(H5_A), TMatrixData(H5_B)
    i = args.index
    k, lam = dA.k_at(i), dA.wavelength_um[i]
    sel, mb = truncate(dA, args.lmax)
    TA, TB = dA.T[i][np.ix_(sel, sel)], dB.T[i][np.ix_(sel, sel)]
    quad = make_quad(20, 40)
    proj = RegularProjector(mb, quad)
    p, P = PITCH, 2 * PITCH
    A1, A2 = (P, 0.0), (0.0, P)

    print(f"lambda = {lam:.3f} um (index {i}), k = {k:.5f} rad/um, "
          f"lmax = {args.lmax} ({mb.n} modes), base pitch {p} um, "
          f"supercell {P} um")

    # ---------------------------------------------------------------- 6.5.5-1
    print("\n[6.5.5-1] one-atom reduction (M = 1)")
    C_taper = lattice_sum_C(k, p, mb, R0, quad, projector=proj)
    C_ew = ew.block_lattice_sums_ewald(k, (p, 0), (0, p), [(0, 0)], mb)[0, 0]
    check("Ewald C_p vs the repo's tapered lattice_sum_C (rel)",
          np.abs(C_ew - C_taper).max() / np.abs(C_taper).max(), 1e-4)
    _, eta_info = ew.converged_W(k, (p, 0), (0, p), [(0, 0)], mb,
                                 return_info=True)
    check("Ewald eta bracket spread, one-atom cell",
          max(eta_info["eta_deviations"].values()), 1e-8)

    a_inc1 = sc.incident_blocks([0, 0, 1], [1, 0, 0], mb, [(0, 0)], k)
    _, f1 = sc.solve_supercell(C_ew[None, None], [TA], a_inc1)
    _, f_ref = solve_periodic(TA, C_ew,
                              plane_wave_coeffs([0, 0, 1], [1, 0, 0], mb))
    check("solve_supercell(M=1) vs aggregate.solve_periodic (rel)",
          np.linalg.norm(f1[0] - f_ref) / np.linalg.norm(f_ref), 1e-14)
    S1 = sc.zeroth_order(sc.floquet_smatrix(k, (p, 0), (0, p), [(0, 0)], mb,
                                            f1))
    S_ref = sparams_normal(k, p * p, mb, f_ref)
    check("S21 vs sparams.sparams_normal", abs(S1["S21_co"] - S_ref["S21_co"]),
          1e-14)
    check("S11 vs sparams.sparams_normal", abs(S1["S11_co"] - S_ref["S11_co"]),
          1e-14)

    # ---------------------------------------------------------------- 6.5.5-2
    print("\n[6.5.5-2] identical-atom 2x2 supercell == one-atom lattice")
    rho4 = np.array([(0.0, 0.0), (p, 0.0), (0.0, p), (p, p)])
    W4 = ew.block_lattice_sums_ewald(k, A1, A2, rho4, mb)
    for s in range(4):
        check(f"sum_t W[{s},t] vs C_p (rel)",
              np.abs(W4[s].sum(axis=0) - C_ew).max() / np.abs(C_ew).max(),
              1e-13)
    a_inc4 = sc.incident_blocks([0, 0, 1], [1, 0, 0], mb, rho4, k)
    _, f4 = sc.solve_supercell(W4, [TA] * 4, a_inc4)
    check("all four f_s identical (rel spread)",
          np.abs(f4 - f4[0]).max() / np.abs(f4[0]).max(), 1e-11)
    check("f_s vs the one-atom f (rel)",
          np.linalg.norm(f4[0] - f_ref) / np.linalg.norm(f_ref), 1e-11)
    res4 = sc.floquet_smatrix(k, A1, A2, rho4, mb, f4)
    S4 = sc.zeroth_order(res4)
    check("supercell S21 vs one-atom S21", abs(S4["S21_co"] - S_ref["S21_co"]),
          1e-11)
    check("supercell S11 vs one-atom S11", abs(S4["S11_co"] - S_ref["S11_co"]),
          1e-11)
    spur = max((r["power"] for r in res4["trans"] + res4["refl"]
                if r["n"] != (0, 0)), default=0.0)
    check("power in the orders the folding invents", spur, 1e-20)

    # ---------------------------------------------------------------- 6.5.5-3
    print("\n[6.5.5-3] permutation invariance (heterogeneous a,b;b,a)")
    rho_cb = np.array([(-p / 2, -p / 2), (p / 2, -p / 2),
                       (-p / 2, p / 2), (p / 2, p / 2)])
    T_cb = [TA, TB, TB, TA]
    Wcb = ew.block_lattice_sums_ewald(k, A1, A2, rho_cb, mb)
    a_cb = sc.incident_blocks([0, 0, 1], [1, 0, 0], mb, rho_cb, k)
    _, f_cb = sc.solve_supercell(Wcb, T_cb, a_cb)
    r_cb = sc.floquet_smatrix(k, A1, A2, rho_cb, mb, f_cb)
    S_cb = sc.zeroth_order(r_cb)

    perm = [2, 0, 3, 1]
    Wp = ew.block_lattice_sums_ewald(k, A1, A2, rho_cb[perm], mb)
    a_p = sc.incident_blocks([0, 0, 1], [1, 0, 0], mb, rho_cb[perm], k)
    _, f_p = sc.solve_supercell(Wp, [T_cb[j] for j in perm], a_p)
    S_p = sc.zeroth_order(sc.floquet_smatrix(k, A1, A2, rho_cb[perm], mb, f_p))
    check("S21 under basis-atom relabelling",
          abs(S_cb["S21_co"] - S_p["S21_co"]), 1e-12)
    check("S11 under basis-atom relabelling",
          abs(S_cb["S11_co"] - S_p["S11_co"]), 1e-12)
    check("per-atom f under relabelling (rel)",
          np.abs(f_cb[perm] - f_p).max() / np.abs(f_cb).max(), 1e-12)
    Wt = ew.block_lattice_sums_ewald(k, A1, A2, rho_cb + [P, 0], mb)
    check("W invariant under a whole-cell lattice shift (rel)",
          np.abs(Wt - Wcb).max() / np.abs(Wcb).max(), 1e-12)

    # ---------------------------------------------------------------- 6.5.5-5
    print("\n[6.5.5-5] power balance and reciprocity")
    check("absorption A = 1 - R - T >= 0 (reported as -A)", -r_cb["A"], 0.0)
    print(f"         R = {r_cb['R']:.5f}  T = {r_cb['T']:.5f}  "
          f"A = {r_cb['A']:.5f}   open orders "
          f"{[o['n'] for o in r_cb['orders']]}")
    xpol = max(abs(S_cb["S21_cross"]), abs(S_cb["S11_cross"]))
    check("cross-polarized |S| at normal incidence (C4v about an A site)",
          xpol, 1e-10)
    a_back = sc.incident_blocks([0, 0, -1], [1, 0, 0], mb, rho_cb, k)
    _, f_back = sc.solve_supercell(Wcb, T_cb, a_back)
    E_b = sc.floquet_amplitudes(k, P * P, mb, f_back, rho_cb,
                                [o for o in r_cb["orders"] if o["n"] == (0, 0)],
                                -1)[0]
    S12 = 1.0 + np.vdot(np.array([1.0, 0, 0]), E_b)
    check("|S21 - S12| (reciprocity of the array)",
          abs(S_cb["S21_co"] - S12), 5e-2)
    print(f"         S21 = {S_cb['S21_co']:.5f}   S12 = {S12:.5f}")

    # ---------------------------------------------------------------- 6.5.5-6
    print("\n[6.5.5-6] pair-block convergence (Ewald split)")
    _, eta4 = ew.converged_W(k, A1, A2, rho_cb, mb, return_info=True)
    check("eta bracket spread over the 16 blocks",
          max(eta4["eta_deviations"].values()), 1e-8)
    for lm in (2, 4):
        _, mbx = truncate(dA, lm)
        Wx = ew.block_lattice_sums_ewald(k, A1, A2, rho_cb, mbx)
        Cx = ew.block_lattice_sums_ewald(k, (p, 0), (0, p), [(0, 0)], mbx)[0, 0]
        check(f"sub-lattice identity still exact at lmax {lm}",
              np.abs(Wx.sum(axis=1) - Cx[None]).max() / np.abs(Cx).max(), 1e-12)

    # ------------------------------------------------------------------- 6.6
    print("\n[6.6] translation validity and the regular addition matrix")
    check("circumscribing-sphere margin (a_A + a_B) - d_min  [<= 0]",
          A_RADIUS + B_RADIUS - p, 0.0, " um")
    check("A-A / B-B margin (2 a) - d, d = p sqrt2  [<= 0]",
          2 * B_RADIUS - p * np.sqrt(2), 0.0, " um")
    # Rg is not band limited: a target truncated at lmax cannot absorb the
    # l -> l' mixing of a translation, so the identities are tested with a
    # source basis large enough that the truncation is not the error.
    LB = 12
    mb_big = ModeBasis.standard(LB)
    quad_big = make_quad(2 * LB + 8, 4 * LB + 16)
    proj_big = RegularProjector(mb_big, quad_big)
    Rg_f = sc.regular_translation(k, [p, 0, 0], mb_big, mb_big, R0, quad_big,
                                  projector=proj_big)
    Rg_b = sc.regular_translation(k, [-p, 0, 0], mb_big, mb_big, R0, quad_big,
                                  projector=proj_big)
    blk = (Rg_f @ Rg_b)[:mb.n, :mb.n]
    check(f"Rg(d) Rg(-d) = I on the lmax-{args.lmax} block (L={LB}, rel)",
          np.abs(blk - np.eye(mb.n)).max(), 1e-6)
    rho_t = np.array([2.3, -1.7, 0.0])
    proj_small = RegularProjector(mb, quad_big)
    Rg_t = sc.regular_translation(k, -rho_t, mb_big, mb, R0, quad_big,
                                  projector=proj_small)
    a_big = plane_wave_coeffs([0, 0, 1], [1, 0, 0], mb_big)
    want = np.exp(1j * k * rho_t[2]) * plane_wave_coeffs([0, 0, 1], [1, 0, 0],
                                                         mb)
    check("Rg(-rho) a_pw == e^{i k.rho} a_pw (rel)",
          np.linalg.norm(Rg_t @ a_big - want) / np.linalg.norm(want), 1e-8)

    # ----------------------------------------------------------------- 7.5/8
    print("\n[7.5 / 8] output channels")
    b1, b2 = sc.reciprocal_vectors(A1, A2)
    check("reciprocal basis b_i . a_j = 2 pi delta_ij",
          max(abs(b1 @ A1 - 2 * np.pi), abs(b1 @ A2), abs(b2 @ A1),
              abs(b2 @ A2 - 2 * np.pi)), 1e-9)
    Z = np.zeros_like(TA)
    _, f0 = sc.solve_supercell(Wcb, [Z] * 4, a_cb)
    r0res = sc.floquet_smatrix(k, A1, A2, rho_cb, mb, f0)
    S0 = sc.zeroth_order(r0res)
    check("T = 0 -> S21 = 1 exactly", abs(S0["S21_co"] - 1.0), 1e-15)
    check("T = 0 -> S11 = 0 exactly", abs(S0["S11_co"]), 1e-15)
    check("T = 0 -> total propagating power = 1",
          abs(r0res["T"] + r0res["R"] - 1.0), 1e-13)

    ih = int(np.argmin(np.abs(dA.wavelength_um - 9.5)))
    kh = dA.k_at(ih)
    Wh = ew.block_lattice_sums_ewald(kh, A1, A2, rho_cb, mb)
    Th = [dA.T[ih][np.ix_(sel, sel)], dB.T[ih][np.ix_(sel, sel)],
          dB.T[ih][np.ix_(sel, sel)], dA.T[ih][np.ix_(sel, sel)]]
    _, fh = sc.solve_supercell(
        Wh, Th, sc.incident_blocks([0, 0, 1], [1, 0, 0], mb, rho_cb, kh))
    rh = sc.floquet_smatrix(kh, A1, A2, rho_cb, mb, fh)
    odd = [r for r in rh["trans"] + rh["refl"] if sum(r["n"]) % 2]
    even = [r for r in rh["trans"] + rh["refl"]
            if sum(r["n"]) % 2 == 0 and r["n"] != (0, 0)]
    print(f"         at lambda = {dA.wavelength_um[ih]:.3f} um: "
          f"{len(rh['orders'])} open orders, {len(odd) // 2} of them odd")
    check("odd (n1+n2) orders extinguished by the a,b;b,a symmetry",
          max((r["power"] for r in odd), default=0.0), 1e-18)
    info("largest even-order power (real diffraction)",
         max((r["power"] for r in even), default=0.0))
    check("power balance above the Rayleigh onset (reported as -A)",
          -rh["A"], 0.0)

    # ------------------------------------------------------------------- 6.4
    print("\n[6.4] finite cluster single-origin T-matrix")
    pos = np.column_stack([rho_cb, np.zeros(4)])
    quad_c = make_quad(32, 64)
    proj_c = RegularProjector(mb, quad_c)
    U = sc.finite_interaction(k, pos, mb, R0, quad_c, projector=proj_c)
    M, _ = build_finite_system(k, T_cb, pos, mb, R0, quad_c)
    _, f_sites = solve_finite(M, T_cb, a_cb)
    rhat = np.array([[0, 0, 1.0], [0, 0, -1.0], [0.6, 0.0, 0.8],
                     [0.3, 0.5, -0.812]])
    rhat /= np.linalg.norm(rhat, axis=1)[:, None]
    F_multi = np.zeros((len(rhat), 3), dtype=complex)
    for s in range(4):
        F_multi += np.exp(-1j * k * (rhat @ pos[s]))[:, None] * \
            far_field_amplitude(k, f_sites[s], mb, rhat)
    prev = None
    for LC in (8, 10, 12, 14):
        TO, mbc = sc.cluster_T(k, pos, T_cb, mb, LC, R0, quad_c, U=U,
                               projector=proj_c)
        F_c = far_field_amplitude(k, TO @ plane_wave_coeffs([0, 0, 1],
                                                            [1, 0, 0], mbc),
                                  mbc, rhat)
        rel = np.abs(F_c - F_multi).max() / np.abs(F_multi).max()
        check(f"L_C = {LC:2d}: cluster far field vs multi-center sum (rel)",
              rel, 3e-2 if LC < 12 else 3e-3)
        prev = (TO, mbc)
    TO, mbc = prev
    check("S_sph = I + 2 T^O passivity, max singular value",
          np.linalg.svd(np.eye(mbc.n) + 2 * TO, compute_uv=False).max(),
          1.05)
    O = np.array([0.4, -0.3, 0.0])
    TO_s, mbc_s = sc.cluster_T(k, pos, T_cb, mb, 14, R0, quad_c, U=U,
                               origin=O, projector=proj_c)
    a_s = plane_wave_coeffs([0, 0, 1], [1, 0, 0], mbc_s) * \
        np.exp(1j * k * O[2])
    F_s = far_field_amplitude(k, TO_s @ a_s, mbc_s, rhat)
    F_s = F_s * np.exp(-1j * k * (rhat @ O))[:, None]
    check("origin shift O: far field covariant to e^{-i k rhat.O} (rel)",
          np.abs(F_s - F_multi).max() / np.abs(F_multi).max(), 3e-3)

    # ----------------------------------------------------------------- taper
    if args.taper:
        print("\n[taper] repository tapered sum vs Ewald, per pair block")
        for kRc in ((10.0, 14.0, 20.0), (14.0, 20.0, 28.0),
                    (20.0, 28.0, 40.0)):
            Wtap = sc.block_lattice_sums(k, A1, A2, rho_cb, mb, R0, quad,
                                         kRc=kRc, projector=proj)
            info(f"kRc = {kRc}: max |W_taper - W_Ewald| / max|W|",
                 np.abs(Wtap - Wcb).max() / np.abs(Wcb).max())
            info(f"kRc = {kRc}: sum_t W_st vs its own one-atom sum",
                 np.abs(Wtap.sum(axis=1)
                        - sc.block_lattice_sums(k, (p, 0), (0, p), [(0, 0)],
                                                mb, R0, quad, kRc=kRc,
                                                projector=proj)[0, 0][None]
                        ).max() / np.abs(Wcb).max())
        print("         the sub-lattice identity is exact for the tapered sum")
        print("         too -- the taper error is common to all blocks and")
        print("         cancels in the sum, which is why the one-atom lattice")
        print("         is fine and the individual pair blocks are not.")

    print("\n" + ("ALL CHECKS PASSED" if not _fails
                  else f"{len(_fails)} FAILURE(S): " + ", ".join(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
