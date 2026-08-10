"""Independent cross-check of run_case.py using treams (Ewald lattice sums).

Same physics, completely different numerics: treams builds the lattice-coupled
T-matrix with Ewald-summed translation operators and projects onto a plane-wave
basis, where this repo uses Gaussian-tapered real-space shell sums plus a
numerical VSWF projection.  Agreement between the two is a genuine check of the
Stage-3 aggregation; disagreement points at the lattice sum.

Conventions match treams_reference.py exactly (see its docstring): normal
incidence from the +z side (treams "down"), E_inc = x_hat exp(-i k z), phase
reference z = 0, S21 including the direct unity term.  Because this repo's
run_case.py illuminates along +z, its S-parameters are compared against these
after the trivial z -> -z relabelling (both are C4-symmetric planar cells, so
S11/S21 are unchanged).

    python -m tmatrix.aggregation.treams_case \
        test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5 \
        --pitch 8.0 --out results_2x2/treams_reference.npz --lmax auto
"""
import argparse
import os

import numpy as np
import treams
import treams.io

from tmatrix import treams_compat

from tmatrix.paths import AGG_DATA


treams_compat.apply()


def truncate(tm, lmax):
    """Restrict a treams TMatrix to l <= lmax (same basis ordering)."""
    keep = np.asarray(tm.basis.l) <= lmax
    if keep.all():
        return tm
    idx = np.nonzero(keep)[0]
    basis = treams.SphericalWaveBasis(
        [(int(tm.basis.l[j]), int(tm.basis.m[j]), int(tm.basis.pol[j]))
         for j in idx])
    return treams.TMatrix(np.asarray(tm)[np.ix_(idx, idx)], k0=tm.k0,
                          material=tm.material, basis=basis,
                          poltype=tm.poltype)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5")
    ap.add_argument("--pitch", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lmax", default=None,
                    help="integer, or 'auto' to read the per-frequency lmax "
                         "chosen by run_case.py from its periodic_results.csv "
                         "next to --out")
    args = ap.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(AGG_DATA, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    lmax_per_freq = None
    if args.lmax == "auto":
        csv = os.path.join(os.path.dirname(out), "periodic_results.csv")
        rows = np.genfromtxt(csv, delimiter=",", names=True)
        lmax_per_freq = np.atleast_1d(rows["lmax"]).astype(int)
        print(f"  per-frequency lmax from {os.path.basename(csv)}: "
              f"{lmax_per_freq.min()}..{lmax_per_freq.max()}")
    elif args.lmax is not None:
        lmax_fixed = int(args.lmax)

    tms = treams.io.load_hdf5(os.path.abspath(args.h5), lunit="um")
    nf = len(tms)
    lattice = treams.Lattice.square(args.pitch)
    kpar = [0.0, 0.0]

    lam = np.array([2 * np.pi / tm.k0 for tm in tms])
    S11 = np.empty(nf, complex)
    S21 = np.empty(nf, complex)
    S11x = np.empty(nf, complex)
    S21x = np.empty(nf, complex)
    used = np.empty(nf, int)

    for i, tm in enumerate(tms):
        assert tm.poltype == "parity"
        if lmax_per_freq is not None:
            tm = truncate(tm, int(lmax_per_freq[i]))
        elif args.lmax is not None:
            tm = truncate(tm, lmax_fixed)
        used[i] = int(np.max(tm.basis.l))
        k0 = tm.k0

        tm_lat = tm.latticeinteraction.solve(lattice, kpar)
        basis = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, 1e-9)
        smat = treams.SMatrices.from_array(tm_lat, basis)

        M_up = np.asarray(treams.efield(
            [0, 0, 0.0], basis=basis, k0=k0, material=tm.material,
            poltype="parity", modetype="up"))
        M_down = np.asarray(treams.efield(
            [0, 0, 0.0], basis=basis, k0=k0, material=tm.material,
            poltype="parity", modetype="down"))

        c_inc, *_ = np.linalg.lstsq(M_down, np.array([1.0, 0.0, 0.0]),
                                    rcond=None)
        resid = np.linalg.norm(M_down @ c_inc - np.array([1.0, 0.0, 0.0]))
        assert resid < 1e-12, f"x-pol decomposition failed at {i}: {resid}"

        E_t = M_down @ (np.asarray(smat[1, 1]) @ c_inc)
        E_r = M_up @ (np.asarray(smat[0, 1]) @ c_inc)
        S21[i], S11[i] = E_t[0], E_r[0]
        S21x[i], S11x[i] = E_t[1], E_r[1]

    np.savez(out, lam_um=lam, S11=S11, S21=S21,
             S11_cross=S11x, S21_cross=S21x, lmax=used, pitch=args.pitch,
             notes=(f"treams Ewald reference, square lattice pitch "
                    f"{args.pitch} um, vacuum embedding, normal incidence, "
                    f"phase reference z=0, S21 includes the direct term."))

    print(f"\n{'lam_um':>8} {'lmax':>5} {'|S11|':>9} {'|S21|':>9} "
          f"{'1-R-T':>9}")
    for i in range(nf):
        print(f"{lam[i]:8.3f} {used[i]:5d} {abs(S11[i]):9.5f} "
              f"{abs(S21[i]):9.5f} "
              f"{1 - abs(S11[i])**2 - abs(S21[i])**2:9.5f}")
    print(f"\nmax |cross-pol|: S11 {abs(S11x).max():.2e}, "
          f"S21 {abs(S21x).max():.2e}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
