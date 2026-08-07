"""Stage-3 reconstruction for an arbitrary tmat.h5 + square lattice pitch.

Same pipeline as run_demo.py, but the data file, pitch, projection radius and
output directory are command-line arguments, so a new CST extraction can be
pushed through T-matrix -> S-parameters without editing code.

Per frequency:
  1. C = Richardson-extrapolated lattice sum of translation operators
     (square lattice, normal incidence, Gaussian taper).
  2. Periodic Foldy-Lax solve: a = (I - C T)^{-1} a_inc,  f = T a.
  3. S11/S21 from the 0th-order plane-wave projection of the per-cell far
     field (valid while lambda > pitch: no propagating diffraction orders).

Also written out: the uncoupled sheet (C = 0), single-scatterer cross
sections, and -- unless --quick -- taper/r0 convergence checks, a reciprocity
spot check (S21 vs S12) and finite N x N Foldy-Lax arrays.

Multipole truncation (--lmax).  The translation operators grow like
h_l^(1)(k*pitch) ~ (2l-1)!! / (k*pitch)^(l+1), so on a deep-subwavelength
lattice the high-l block of C is enormous and (I - C T) becomes ill
conditioned: whatever extraction noise sits in the high-l rows of T is
amplified by cond(I - C T).  The cure is to keep only the multipoles the
lattice can actually resolve.  --lmax auto picks, per frequency, the largest
truncation whose cond(I - C T) stays under --cond-max; --lmax N forces one
value; the default keeps every mode in the file.

Example (the 8 um-pitch spoke-and-wheel cell of test/2x2):

    python run_case.py ../test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5 \
        --pitch 8.0 --r0 3.0 --lmax auto --out results_2x2
"""
import argparse
import os
import time

import numpy as np

from tmat_io import TMatrixData
from vswf import ModeBasis, plane_wave_coeffs, RegularProjector, far_field_amplitude
from translate import lattice_sum_C, make_quad, square_lattice_shells
from aggregate import solve_periodic, build_finite_system, solve_finite
from sparams import sparams_normal, energy_balance, cross_sections

HERE = os.path.dirname(os.path.abspath(__file__))
KRC = (10.0, 14.0, 20.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5", help="tmat.h5 file with the unit-cell T-matrix")
    p.add_argument("--pitch", type=float, required=True,
                   help="square-lattice period, um")
    p.add_argument("--r0", type=float, default=None,
                   help="projection sphere radius, um (default: pitch/2.7)")
    p.add_argument("--out", default="results_case", help="output directory")
    p.add_argument("--quad", type=int, nargs=2, default=(20, 40),
                   metavar=("N_THETA", "N_PHI"))
    p.add_argument("--lmax", default=None,
                   help="multipole truncation: an integer, or 'auto' to pick "
                        "the largest lmax with cond(I - C T) <= --cond-max "
                        "at each frequency (default: keep all file modes)")
    p.add_argument("--cond-max", type=float, default=100.0,
                   help="conditioning budget used by --lmax auto")
    p.add_argument("--finite", type=int, nargs="*", default=(5, 9, 13),
                   help="finite array sizes N (N x N); empty to skip")
    p.add_argument("--quick", action="store_true",
                   help="periodic sweep only: no convergence/finite-array checks")
    return p.parse_args()


def mode_subsets(modes):
    """lmax -> (index array, truncated ModeBasis).

    Truncation commutes with everything downstream: each entry of C, of a_inc
    and each term of the far-field sum is built per mode independently, so the
    truncated quantities are exactly the corresponding sub-blocks of the full
    ones -- no need to recompute the lattice sum per truncation.
    """
    out = {}
    for L in range(1, modes.lmax + 1):
        sel = np.nonzero(modes.l <= L)[0]
        out[L] = (sel, ModeBasis(modes.l[sel], modes.m[sel], modes.pol[sel]))
    return out


def pick_lmax(C, T, subsets, lmax_file, cond_max):
    """Largest truncation whose Foldy-Lax system stays inside the budget."""
    for L in range(lmax_file, 0, -1):
        sel, mb = subsets[L]
        ix = np.ix_(sel, sel)
        cnd = np.linalg.cond(np.eye(mb.n) - C[ix] @ T[ix])
        if cnd <= cond_max or L == 1:
            return L, cnd
    return 1, np.inf


def tmatrix_diagnostics(data):
    """Convention/quality checks on the input file itself."""
    modes = data.modes
    sv = np.array([np.linalg.svd(np.eye(modes.n) + 2 * data.T[i],
                                 compute_uv=False).max()
                   for i in range(len(data.freq))])
    perm = np.array([modes.index(modes.l[i], -modes.m[i], modes.pol[i])
                     for i in range(modes.n)])
    sign = (-1.0) ** (modes.m[:, None] + modes.m[None, :])
    rec = np.array([np.linalg.norm(T - sign * T[np.ix_(perm, perm)].T)
                    / np.linalg.norm(T) for T in data.T])
    print("=== input T-matrix diagnostics ===")
    print(f"  passivity   max SV(I + 2T) = {sv.max():.6f}  (must be <= 1 + eps)")
    print(f"  reciprocity |T - T_rec|/|T| = {rec.min():.4f} .. {rec.max():.4f}")
    return sv, rec


def convergence_check(data, pitch, r0, quad_hi, idx):
    """r0 / quadrature / taper-length sensitivity at a few frequencies."""
    modes = data.modes
    a_inc = plane_wave_coeffs([0, 0, 1], [1, 0, 0], modes)
    print("=== numerical convergence (|dS11| + |dS21| vs reference run) ===")
    print("  lam[um]   r0*0.8      quad hi     taper kRc/1.4")
    for i in idx:
        k = data.k_at(i)
        A_cell = pitch ** 2

        def run(r0_, quad_, kRc_):
            q = make_quad(*quad_)
            C = lattice_sum_C(k, pitch, modes, r0_, q, kRc=kRc_,
                              projector=RegularProjector(modes, q))
            _, f = solve_periodic(data.T[i], C, a_inc)
            S = sparams_normal(k, A_cell, modes, f)
            return S["S11_co"], S["S21_co"]

        ref = run(r0, quad_hi[0], KRC)
        var = [run(0.8 * r0, quad_hi[0], KRC),
               run(r0, quad_hi[1], KRC),
               run(r0, quad_hi[0], tuple(x / 1.4 for x in KRC))]
        d = [abs(v[0] - ref[0]) + abs(v[1] - ref[1]) for v in var]
        print(f"  {data.wavelength_um[i]:7.2f}   " +
              "   ".join(f"{x:.2e}" for x in d))


def main():
    args = parse_args()
    pitch = args.pitch
    r0 = args.r0 if args.r0 is not None else pitch / 2.7
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)

    data = TMatrixData(os.path.abspath(args.h5))
    modes = data.modes
    nf = len(data.freq)
    lam = data.wavelength_um
    A_cell = pitch ** 2

    print(f"Loaded {os.path.basename(args.h5)}: {nf} freqs, "
          f"lambda {lam.min():.2f}-{lam.max():.2f} um, {modes.n} modes "
          f"(lmax={modes.lmax})")
    print(f"  square lattice pitch {pitch} um, A_cell {A_cell} um^2, "
          f"projection radius r0 {r0:.3f} um")
    if lam.min() <= pitch:
        print(f"  !! lambda_min = {lam.min():.3f} um <= pitch: the 0th-order "
              f"S-parameter model is invalid there (diffraction opens)")
    else:
        print(f"  lambda_min / pitch = {lam.min()/pitch:.3f} "
              f"(> 1: 0th order only, Rayleigh onset at lambda = pitch)")
    sv, rec = tmatrix_diagnostics(data)

    quad = make_quad(*args.quad)
    proj = RegularProjector(modes, quad)
    e_x = np.array([1.0, 0, 0])
    a_inc = plane_wave_coeffs([0, 0, 1], e_x, modes)

    if not args.quick:
        convergence_check(data, pitch, r0, (args.quad, (args.quad[0] + 8,
                                                        args.quad[1] + 16)),
                          (0, nf // 2, nf - 1))

    subsets = mode_subsets(modes)
    fixed_lmax = None
    if args.lmax is not None and args.lmax != "auto":
        fixed_lmax = min(int(args.lmax), modes.lmax)
    if args.lmax == "auto":
        print(f"=== periodic sweep (adaptive lmax, cond budget "
              f"{args.cond_max:g}) ===")
    else:
        print(f"=== periodic sweep (lmax = "
              f"{fixed_lmax if fixed_lmax else modes.lmax}) ===")

    rows = []
    C_stack = np.empty((nf, modes.n, modes.n), dtype=complex)
    t0 = time.time()
    for i in range(nf):
        k = data.k_at(i)
        T_full = data.T[i]

        C = lattice_sum_C(k, pitch, modes, r0, quad, kRc=KRC, projector=proj)
        C_stack[i] = C

        if args.lmax == "auto":
            L, cnd = pick_lmax(C, T_full, subsets, modes.lmax, args.cond_max)
        else:
            L = fixed_lmax if fixed_lmax else modes.lmax
            sel_, mb_ = subsets[L]
            ix_ = np.ix_(sel_, sel_)
            cnd = np.linalg.cond(np.eye(mb_.n) - C[ix_] @ T_full[ix_])
        sel, mb = subsets[L]
        ix = np.ix_(sel, sel)
        T, Ci, a_i = T_full[ix], C[ix], a_inc[sel]

        a, f = solve_periodic(T, Ci, a_i)
        S = sparams_normal(k, A_cell, mb, f)
        R, Tr, Ab = energy_balance(S)

        f0 = T @ a_i                            # uncoupled sheet (C = 0)
        S0 = sparams_normal(k, A_cell, mb, f0)
        R0_, T0_, A0_ = energy_balance(S0)

        sext, ssca, sabs = cross_sections(k, mb, f0, [0, 0, 1], e_x)
        rows.append(dict(
            lam=lam[i], f_THz=data.freq[i] / 1e12, lmax=L, cond=cnd,
            S11=S["S11_co"], S21=S["S21_co"],
            S11x=S["S11_cross"], S21x=S["S21_cross"],
            R=R, T=Tr, A=Ab,
            S11_nc=S0["S11_co"], S21_nc=S0["S21_co"],
            R_nc=R0_, T_nc=T0_, A_nc=A0_,
            sig_ext=sext, sig_sca=ssca, sig_abs=sabs))
        print(f"  [{i+1:2d}/{nf}] lam={lam[i]:6.2f} um f={data.freq[i]/1e12:5.1f} THz  "
              f"lmax={L} cond={cnd:8.1f}  "
              f"|S11|={abs(S['S11_co']):.3f} |S21|={abs(S['S21_co']):.3f} "
              f"R={R:.3f} T={Tr:.3f} A={Ab:.3f}  "
              f"[{time.time()-t0:5.1f} s]", flush=True)

    keys = [key for key in rows[0] if key not in ("lam", "f_THz")]
    np.savez(os.path.join(out, "periodic_results.npz"),
             lam=lam, freq=data.freq, pitch=pitch, r0=r0, C=C_stack,
             passivity_sv=sv, reciprocity=rec,
             **{key: np.array([r[key] for r in rows]) for key in keys})

    with open(os.path.join(out, "periodic_results.csv"), "w") as fh:
        cols = ["lam_um", "f_THz", "lmax", "cond_I_minus_CT",
                "Re_S11", "Im_S11", "Re_S21", "Im_S21",
                "absS11", "absS21", "R", "T", "A",
                "absS11_uncoupled", "absS21_uncoupled", "A_uncoupled",
                "sigma_ext_um2", "sigma_sca_um2", "sigma_abs_um2"]
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{v:.10g}" for v in [
                r["lam"], r["f_THz"], r["lmax"], r["cond"],
                r["S11"].real, r["S11"].imag,
                r["S21"].real, r["S21"].imag, abs(r["S11"]), abs(r["S21"]),
                r["R"], r["T"], r["A"], abs(r["S11_nc"]), abs(r["S21_nc"]),
                r["A_nc"], r["sig_ext"], r["sig_sca"], r["sig_abs"]]) + "\n")

    A_arr = np.array([r["A"] for r in rows])
    xpol = max(max(abs(r["S11x"]), abs(r["S21x"])) for r in rows)
    print("=== energy balance ===")
    print(f"  A = 1 - R - T in [{A_arr.min():.4f}, {A_arr.max():.4f}]  "
          f"(must be >= 0 for a passive cell)")
    print(f"  max cross-polarized |S| = {xpol:.2e} "
          f"(C4-symmetric cell at normal incidence: expect ~0)")

    if not args.quick:
        print("=== reciprocity spot check (S21 vs S12) ===")
        a_back = plane_wave_coeffs([0, 0, -1], e_x, modes)
        for i in (0, nf // 2, nf - 1):
            k = data.k_at(i)
            sel, mb = subsets[rows[i]["lmax"]]
            ix = np.ix_(sel, sel)
            _, f_b = solve_periodic(data.T[i][ix], C_stack[i][ix], a_back[sel])
            F_b = far_field_amplitude(k, f_b, mb,
                                      np.array([[0, 0, -1.0]]))[0]
            S12 = 1.0 + 2j * np.pi / (k * A_cell) * np.vdot(e_x, F_b)
            S21 = rows[i]["S21"]
            print(f"  lam={lam[i]:6.2f}: S21={S21:.4f}  S12={S12:.4f}  "
                  f"|diff|={abs(S21 - S12):.2e}")

    if not args.quick and len(args.finite):
        print("=== finite Foldy-Lax arrays vs periodic ===")
        idx = range(0, nf, max(1, nf // 5))
        fin = {N: [] for N in args.finite}
        for i in idx:
            k = data.k_at(i)
            sel, mb = subsets[rows[i]["lmax"]]
            Ti = data.T[i][np.ix_(sel, sel)]
            qmb = make_quad(*args.quad)
            for N in fin:
                g = (np.arange(N) - (N - 1) / 2) * pitch
                X, Y = np.meshgrid(g, g)
                pos = np.stack([X.ravel(), Y.ravel(), np.zeros(N * N)], axis=1)
                M, T_list = build_finite_system(k, Ti, pos, mb, r0, qmb)
                a_blocks = np.tile(a_inc[sel], (N * N, 1))  # normal incidence
                _, f_sites = solve_finite(M, T_list, a_blocks)
                f_tot = f_sites.sum(axis=0)             # in-plane: no phase
                S = sparams_normal(k, N * N * A_cell, mb, f_tot)
                fin[N].append((lam[i], S["S11_co"], S["S21_co"]))
            print(f"  lam={lam[i]:6.2f}: |S21| periodic={abs(rows[i]['S21']):.3f}  "
                  + "  ".join(f"{N}x{N}={abs(fin[N][-1][2]):.3f}" for N in fin),
                  flush=True)
        np.savez(os.path.join(out, "finite_results.npz"),
                 **{f"N{N}": np.array([[l, s11, s21] for l, s11, s21 in v],
                                      dtype=complex) for N, v in fin.items()})

    print(f"done in {time.time()-t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
