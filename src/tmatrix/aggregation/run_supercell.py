"""Periodic heterogeneous-supercell sweep: several tmat.h5 -> Floquet S-matrix.

One driver for both halves of the experiment, so the single-atom reference and
the supercell go through identical code:

  # one atom per cell (reproduces run_case.py's problem with Ewald coupling)
  python -m tmatrix.aggregation.run_supercell --cell 8 \
      --site test/2x2/saw_gold_wl17p30um_10to34THz.tmat.h5 0 0 \
      --lmax 3 --out results_B_ewald

  # the a,b;b,a checkerboard, supercell period 16 um
  python -m tmatrix.aggregation.run_supercell --cell 16 \
      --site test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5 -4 -4 \
      --site test/2x2/saw_gold_wl17p30um_10to34THz.tmat.h5  4 -4 \
      --site test/2x2/saw_gold_wl17p30um_10to34THz.tmat.h5 -4  4 \
      --site test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5  4  4 \
      --lmax 3 --cluster-lmax 14 --out results_2x2_super

Per frequency it
  1. builds the pair-resolved Bloch coupling W_st (manual Eq. 48) -- Ewald by
     default, `--coupling taper` for the repository's Gaussian-taper sum;
  2. solves (I - W T0) a = a_inc, f = T0 a (Eq. 50);
  3. projects onto EVERY propagating Floquet order with the coherent basis
     phase of Eq. (64), power-normalized by sqrt(k_z,G / k_z,in);
  4. optionally forms the single-origin T-matrix of the same atoms taken as an
     ISOLATED finite cluster (Eq. 40) -- a different object from the periodic
     solve, reported alongside it because it is the literal "T-matrix of the
     whole array".

Diffraction is not optional here.  A supercell of side P has Rayleigh onsets at
lambda = P/sqrt(n1^2+n2^2); for P = 16 um and a 10-34 THz band the (+-1,0)/(0,+-1)
orders open at 18.74 THz and the (+-1,+-1) orders at 26.50 THz.  Scalar S11/S21
describe the sheet completely only below the first onset that carries power --
which, for the a,b;b,a checkerboard, is 26.50 THz, because the odd orders are
extinguished by its symmetry.  Everything above that is reported per order.
"""
import argparse
import json
import os
import time

import numpy as np

from tmatrix.aggregation.tmat_io import TMatrixData
from tmatrix.aggregation.vswf import ModeBasis, RegularProjector
from tmatrix.aggregation.translate import make_quad
from tmatrix.aggregation.sparams import cross_sections
from tmatrix.aggregation import supercell as sc

from tmatrix.paths import AGG_DATA
from tmatrix.units import C_UM_THZ as C0



def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", nargs=3, action="append", required=True,
                   metavar=("TMAT_H5", "X", "Y"),
                   help="one basis atom: its tmat.h5 and in-plane position, um")
    p.add_argument("--cell", type=float, default=None,
                   help="square supercell period, um")
    p.add_argument("--a1", type=float, nargs=2, default=None)
    p.add_argument("--a2", type=float, nargs=2, default=None)
    p.add_argument("--lmax", default=None,
                   help="integer truncation, or 'auto' for the largest lmax "
                        "with cond(I - W T0) <= --cond-max")
    p.add_argument("--cond-max", type=float, default=100.0)
    p.add_argument("--coupling", choices=("ewald", "taper"), default="ewald")
    p.add_argument("--r0", type=float, default=3.0,
                   help="projection radius for the tapered coupling, um")
    p.add_argument("--quad", type=int, nargs=2, default=(20, 40))
    p.add_argument("--cluster-lmax", type=int, default=None,
                   help="also build the isolated finite-cluster T^O at this "
                        "multipole order (manual 6.4)")
    p.add_argument("--pol", type=float, nargs=2, default=(1.0, 0.0),
                   metavar=("EX", "EY"), help="incident polarization")
    p.add_argument("--refine", type=int, default=1,
                   help="subdivide each stored frequency interval this many "
                        "times, interpolating T linearly, to resolve lattice "
                        "resonances the stored grid aliases")
    p.add_argument("--denoise", default=None,
                   help="comma-separated physical-constraint repairs applied "
                        "to every site T before the solve: 'reciprocity' "
                        "(project onto the reciprocal subspace) and/or "
                        "'passivity' (clip singular values of I + 2T at 1), "
                        "in the given order -- see aggregation/denoise.py")
    p.add_argument("--out", default="results_supercell")
    return p.parse_args()


def load_sites(sites):
    """-> (unique files, per-site data index, positions (M,2))."""
    files, index, pos = [], [], []
    for h5, x, y in sites:
        path = os.path.abspath(h5)
        if path not in files:
            files.append(path)
        index.append(files.index(path))
        pos.append((float(x), float(y)))
    data = [TMatrixData(f) for f in files]
    ref = data[0]
    for d in data[1:]:
        if not np.allclose(d.freq, ref.freq):
            raise ValueError("all tmat.h5 files must share the frequency grid")
        if (not np.array_equal(d.modes.l, ref.modes.l)
                or not np.array_equal(d.modes.m, ref.modes.m)
                or not np.array_equal(d.modes.pol, ref.modes.pol)):
            raise ValueError("all tmat.h5 files must share the mode order")
        if not (np.isclose(d.eps_emb, ref.eps_emb)
                and np.isclose(d.mu_emb, ref.mu_emb)):
            raise ValueError("all tmat.h5 files must share the embedding")
    return files, data, np.array(index), np.array(pos, dtype=float)


def sub_basis(modes, L):
    sel = np.nonzero(modes.l <= L)[0]
    return sel, ModeBasis(modes.l[sel], modes.m[sel], modes.pol[sel])


def refine_grid(data, refine):
    """Frequency grid subdivided `refine`x, with T linearly interpolated.

    Returns (freq, T) where T[j] is the (nf, n, n) stack of file j on the new
    grid.  refine = 1 returns the stored grid untouched.
    """
    f0 = data[0].freq
    if refine <= 1:
        return f0, [d.T for d in data]
    freq = np.unique(np.concatenate(
        [np.linspace(f0[i], f0[i + 1], refine + 1) for i in range(len(f0) - 1)]))
    out = []
    for d in data:
        j = np.clip(np.searchsorted(f0, freq) - 1, 0, len(f0) - 2)
        w = ((freq - f0[j]) / (f0[j + 1] - f0[j]))[:, None, None]
        out.append((1 - w) * d.T[j] + w * d.T[j + 1])
    return freq, out


def main():
    args = parse_args()
    if args.cell is not None:
        a1, a2 = (args.cell, 0.0), (0.0, args.cell)
    elif args.a1 and args.a2:
        a1, a2 = tuple(args.a1), tuple(args.a2)
    else:
        raise SystemExit("give --cell or both --a1 and --a2")
    out = args.out if os.path.isabs(args.out) else os.path.join(AGG_DATA, args.out)
    os.makedirs(out, exist_ok=True)

    files, data, index, rho = load_sites(args.site)
    ref = data[0]
    modes_file = ref.modes
    # --refine subdivides each stored frequency interval and interpolates T
    # linearly between the two bracketing samples.  A resonance of the LATTICE
    # (a Rayleigh anomaly) is much narrower than the 1 THz sampling of these
    # files, so the stored grid aliases it; T itself is smooth over 1 THz
    # EXCEPT across a band seam of a merged extraction, where the interpolation
    # is not meaningful and the refined curve should not be read.
    freq, Tsite = refine_grid(data, args.refine)
    denoise_steps = ([s.strip() for s in args.denoise.split(",") if s.strip()]
                     if args.denoise else [])
    if denoise_steps:
        from tmatrix.aggregation.denoise import (apply_denoise,
                                                 reciprocity_residual)
        for j in range(len(Tsite)):
            before = float(np.mean(reciprocity_residual(Tsite[j], modes_file)))
            Tsite[j] = apply_denoise(Tsite[j], modes_file, denoise_steps)
            after = float(np.mean(reciprocity_residual(Tsite[j], modes_file)))
            print(f"  denoise {denoise_steps}: file {j} mean reciprocity "
                  f"residual {before:.3f} -> {after:.3f}")
    nf = len(freq)
    lam = 299792458.0 / freq * 1e6
    A_uc = abs(a1[0] * a2[1] - a1[1] * a2[0])
    M = len(rho)
    e_inc = np.array([args.pol[0], args.pol[1], 0.0], dtype=complex)
    e_inc = e_inc / np.linalg.norm(e_inc)

    print(f"{M} basis atom(s) from {len(files)} file(s):")
    for f in files:
        print(f"    {os.path.basename(f)}")
    for s in range(M):
        print(f"    site {s}: ({rho[s,0]:+6.2f}, {rho[s,1]:+6.2f}) um  <- "
              f"{os.path.basename(files[index[s]])}")
    print(f"  lattice a1 = {a1}, a2 = {a2} um, A_uc = {A_uc:g} um^2")
    print(f"  {nf} frequencies, lambda {lam.min():.2f}-{lam.max():.2f} um, "
          f"file lmax {modes_file.lmax} ({modes_file.n} modes)")

    # nearest-neighbour distance across the lattice: the translation-validity
    # and (for the tapered coupling) the projection-radius constraint
    d_all = []
    for s in range(M):
        for t in range(M):
            d, _ = sc.lattice_points(a1, a2, 3 * max(np.linalg.norm(a1),
                                                     np.linalg.norm(a2)),
                                     rho[t] - rho[s], drop_zero_offset=(s == t))
            if len(d):
                d_all.append(np.linalg.norm(d, axis=1).min())
    d_min = min(d_all)
    print(f"  nearest inter-atom distance {d_min:.3f} um "
          f"(manual Eq. 57 needs it > a_i + a_j)")

    b1, b2 = sc.reciprocal_vectors(a1, a2)
    onset = [(n1, n2, 2 * np.pi / np.linalg.norm(n1 * b1 + n2 * b2))
             for n1 in range(-2, 3) for n2 in range(-2, 3)
             if (n1, n2) != (0, 0)]
    lam_first = max(o[2] for o in onset)
    print(f"  first Rayleigh onset at lambda = {lam_first:.3f} um "
          f"= {C0/lam_first:.2f} THz "
          f"({'inside' if lam_first > lam.min() else 'above'} the band)")

    quad = make_quad(*args.quad)
    fixed = None if (args.lmax in (None, "auto")) else min(int(args.lmax),
                                                           modes_file.lmax)
    rows, order_rows = [], []
    T_cluster = {}
    t0 = time.time()

    n_emb = np.sqrt(ref.eps_emb * ref.mu_emb).real
    for i in range(nf):
        k = 2 * np.pi / lam[i] * n_emb
        cand = ([fixed] if fixed else
                list(range(modes_file.lmax, 0, -1)) if args.lmax == "auto"
                else [modes_file.lmax])
        chosen = None
        for L in cand:
            sel, mb = sub_basis(modes_file, L)
            if args.coupling == "ewald":
                from tmatrix.aggregation import ewald_supercell as ew
                W, einfo = ew.converged_W(k, a1, a2, rho, mb, return_info=True)
                if W is None:
                    raise RuntimeError(
                        f"Ewald split did not converge at lambda={lam[i]:.3f}: "
                        + "; ".join(einfo["reasons"]))
                eta_spread = max(einfo["eta_deviations"].values())
            else:
                proj = RegularProjector(mb, quad)
                if args.r0 >= d_min:
                    raise SystemExit(
                        f"--r0 {args.r0} must be below the nearest inter-atom "
                        f"distance {d_min:.3f} um")
                W = sc.block_lattice_sums(k, a1, a2, rho, mb, args.r0, quad,
                                          projector=proj)
                eta_spread = np.nan
            T_list = [Tsite[index[s]][i][np.ix_(sel, sel)] for s in range(M)]
            cond = np.linalg.cond(sc.build_block_system(W, T_list))
            chosen = (L, sel, mb, W, T_list, cond, eta_spread)
            if fixed or args.lmax != "auto" or cond <= args.cond_max or L == 1:
                break
        L, sel, mb, W, T_list, cond, eta_spread = chosen

        a_inc = sc.incident_blocks([0, 0, 1], e_inc, mb, rho, k)
        a, f = sc.solve_supercell(W, T_list, a_inc)
        resid = np.linalg.norm(sc.build_block_system(W, T_list)
                               @ a.reshape(-1) - a_inc.reshape(-1)) \
            / np.linalg.norm(a_inc)
        res = sc.floquet_smatrix(k, a1, a2, rho, mb, f, e_inc=e_inc)
        S = sc.zeroth_order(res, e_co=e_inc,
                            e_cross=np.cross([0, 0, 1.0], e_inc))

        # uncoupled sheet: same atoms, W = 0 (isolated response, no lattice)
        f_nc = np.stack([T_list[s] @ a_inc[s] for s in range(M)])
        res_nc = sc.floquet_smatrix(k, a1, a2, rho, mb, f_nc, e_inc=e_inc)
        S_nc = sc.zeroth_order(res_nc, e_co=e_inc)
        sig = [cross_sections(k, mb, f_nc[s], [0, 0, 1], e_inc)
               for s in range(M)]

        higher_t = sum(r["power"] for r in res["trans"] if r["n"] != (0, 0))
        higher_r = sum(r["power"] for r in res["refl"] if r["n"] != (0, 0))
        rows.append(dict(
            lam=lam[i], f_THz=freq[i] / 1e12, lmax=L, cond=cond,
            resid=resid, eta=eta_spread,
            S11=S["S11_co"], S21=S["S21_co"],
            S11x=S["S11_cross"], S21x=S["S21_cross"],
            R=res["R"], T=res["T"], A=res["A"],
            n_open=len(res["orders"]), R_hi=higher_r, T_hi=higher_t,
            S11_nc=S_nc["S11_co"], S21_nc=S_nc["S21_co"],
            sig_ext=sum(s_[0] for s_ in sig),
            sig_sca=sum(s_[1] for s_ in sig),
            sig_abs=sum(s_[2] for s_ in sig)))
        for side in ("trans", "refl"):
            for r in res[side]:
                order_rows.append((lam[i], freq[i] / 1e12, side,
                                   r["n"][0], r["n"][1], r["kz"],
                                   r["S_te"], r["S_tm"], r["power"]))

        if args.cluster_lmax:
            pos3 = np.column_stack([rho, np.zeros(M)])
            TO, mbc = sc.cluster_T(k, pos3, T_list, mb, args.cluster_lmax,
                                   args.r0, quad)
            T_cluster[i] = TO

        print(f"  [{i+1:2d}/{nf}] lam={lam[i]:6.2f} um f={freq[i]/1e12:6.2f} THz "
              f"lmax={L} cond={cond:8.1f} res={resid:.1e}  "
              f"|S11|={abs(S['S11_co']):.3f} |S21|={abs(S['S21_co']):.3f} "
              f"R={res['R']:.3f} T={res['T']:.3f} A={res['A']:+.3f} "
              f"[{len(res['orders'])} ord, hi {higher_r+higher_t:.3f}] "
              f"[{time.time()-t0:5.1f} s]", flush=True)

    keys = [key for key in rows[0] if key not in ("lam", "f_THz")]
    payload = dict(lam=lam, freq=freq, a1=np.array(a1), a2=np.array(a2),
                   rho=rho, A_uc=A_uc, files=np.array(files),
                   site_index=index, e_inc=e_inc,
                   **{key: np.array([r[key] for r in rows]) for key in keys})
    np.savez(os.path.join(out, "periodic_results.npz"), **payload)
    if T_cluster:
        # kept out of periodic_results.npz on purpose: (nf, D_C, D_C) complex is
        # tens of MB, does not compress, and regenerates from this same command
        # in a couple of minutes, so it is a build artifact rather than a record
        np.savez(os.path.join(out, "cluster_T.npz"),
                 T_cluster=np.stack([T_cluster[i] for i in range(nf)]),
                 cluster_lmax=args.cluster_lmax, lam=lam, freq=freq,
                 rho=rho, site_index=index, files=np.array(files))

    with open(os.path.join(out, "periodic_results.csv"), "w") as fh:
        fh.write("lam_um,f_THz,lmax,cond,solve_residual,eta_spread,"
                 "Re_S11,Im_S11,Re_S21,Im_S21,absS11,absS21,"
                 "absS11_cross,absS21_cross,R,T,A,n_open_orders,"
                 "R_higher_orders,T_higher_orders,"
                 "absS11_uncoupled,absS21_uncoupled,"
                 "sigma_ext_um2,sigma_sca_um2,sigma_abs_um2\n")
        for r in rows:
            fh.write(",".join(f"{v:.10g}" for v in [
                r["lam"], r["f_THz"], r["lmax"], r["cond"], r["resid"],
                r["eta"], r["S11"].real, r["S11"].imag, r["S21"].real,
                r["S21"].imag, abs(r["S11"]), abs(r["S21"]), abs(r["S11x"]),
                abs(r["S21x"]), r["R"], r["T"], r["A"], r["n_open"],
                r["R_hi"], r["T_hi"], abs(r["S11_nc"]), abs(r["S21_nc"]),
                r["sig_ext"], r["sig_sca"], r["sig_abs"]]) + "\n")

    with open(os.path.join(out, "floquet_orders.csv"), "w") as fh:
        fh.write("lam_um,f_THz,side,n1,n2,kz_rad_per_um,"
                 "Re_S_TE,Im_S_TE,Re_S_TM,Im_S_TM,power\n")
        for (lm, ft, side, n1, n2, kz, ste, stm, pw) in order_rows:
            fh.write(f"{lm:.10g},{ft:.10g},{side},{n1},{n2},{kz:.10g},"
                     f"{ste.real:.10g},{ste.imag:.10g},"
                     f"{stm.real:.10g},{stm.imag:.10g},{pw:.10g}\n")

    with open(os.path.join(out, "run.json"), "w") as fh:
        json.dump(dict(sites=[[files[index[s]], *rho[s].tolist()]
                              for s in range(M)],
                       a1=list(a1), a2=list(a2), A_uc=A_uc,
                       coupling=args.coupling, lmax=args.lmax,
                       cond_max=args.cond_max, r0=args.r0, quad=list(args.quad),
                       cluster_lmax=args.cluster_lmax,
                       polarization=[args.pol[0], args.pol[1]],
                       refine=args.refine, denoise=args.denoise,
                       first_rayleigh_onset_um=lam_first,
                       nearest_distance_um=d_min), fh, indent=2)

    A_arr = np.array([r["A"] for r in rows])
    xp = max(max(abs(r["S11x"]), abs(r["S21x"])) for r in rows)
    print("=== summary ===")
    print(f"  A = 1 - R - T in [{A_arr.min():+.4f}, {A_arr.max():+.4f}]")
    print(f"  max cross-polarized |S| = {xp:.2e}")
    print(f"  max linear-solve residual = "
          f"{max(r['resid'] for r in rows):.2e}")
    hi = max(r["R_hi"] + r["T_hi"] for r in rows)
    print(f"  max power into higher diffraction orders = {hi:.4f}")
    print(f"done in {time.time()-t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
