"""Independent treams cross-check of the heterogeneous-supercell aggregation.

This is a second implementation of the *whole* chain, not just of the lattice
sums: treams builds its own block-diagonal cluster T-matrix, does its own Ewald
lattice interaction, and does its own plane-wave projection with its own
basis-position phases and power normalization.  Agreement therefore tests
`supercell.py` end to end (coupling, block solve, Eq. (64) output map, Floquet
normalization), which is manual section 8 row 9.

    python -m tmatrix.aggregation.treams_supercell --cell 16 \
        --site test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5 -4 -4 \
        --site test/2x2/saw_gold_wl17p30um_10to34THz.tmat.h5  4 -4 \
        --site test/2x2/saw_gold_wl17p30um_10to34THz.tmat.h5 -4  4 \
        --site test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5  4  4 \
        --lmax 3 --out results_2x2_super_l3/treams_reference.npz

Conventions match tmatrix.aggregation.treams_reference, which was gated against the
repository for `test/single`: normal incidence from the +z side (treams
modetype "down"), incident E = x_hat exp(-i k z) with the phase reference at
z = 0, S21 the total co-polarized 0th-order amplitude below the array
(including the direct term) and S11 the co-polarized reflected amplitude above
it.  The array is z-symmetric here (planar atoms at z = 0), so illuminating
from +z and from -z give the same S11/S21.

Windows/numpy2 gufunc casting is patched by importing `ewald_supercell`.
"""
import argparse
import os

import numpy as np
import treams
import treams.io

from tmatrix import treams_compat

from tmatrix.paths import AGG_DATA

HERE = AGG_DATA
treams_compat.apply()


def truncate(tm, lmax):
    b = tm.basis
    sel = np.nonzero(np.asarray(b.l) <= lmax)[0]
    bn = treams.SphericalWaveBasis(
        list(zip(np.zeros(len(sel), int), np.asarray(b.l)[sel],
                 np.asarray(b.m)[sel], np.asarray(b.pol)[sel])),
        positions=b.positions)
    return treams.TMatrix(np.asarray(tm)[np.ix_(sel, sel)], k0=tm.k0,
                          basis=bn, material=tm.material, poltype=tm.poltype)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", nargs=3, action="append", required=True,
                    metavar=("TMAT_H5", "X", "Y"))
    ap.add_argument("--cell", type=float, required=True)
    ap.add_argument("--lmax", type=int, default=None)
    ap.add_argument("--bmax", type=float, default=None,
                    help="largest |G| kept in the plane-wave basis, rad/um")
    ap.add_argument("--incidence", choices=("up", "down"), default="up",
                    help="'up' = propagating +z (illuminated from below), the "
                         "convention of run_supercell.py and of the CST "
                         "Floquet run, which excites Zmin; 'down' = the "
                         "convention of treams_reference.py.  They differ by "
                         "the input T-matrix's own reciprocity violation, not "
                         "by anything in the aggregation.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = []
    idx, pos = [], []
    for h5, x, y in args.site:
        p = os.path.abspath(h5)
        if p not in files:
            files.append(p)
        idx.append(files.index(p))
        pos.append([float(x), float(y), 0.0])
    loaded = [treams.io.load_hdf5(f, lunit="um") for f in files]
    nf = len(loaded[0])
    lattice = treams.Lattice.square(args.cell)
    kpar = [0.0, 0.0]
    bmax = args.bmax if args.bmax is not None else 2.5 * 2 * np.pi / args.cell

    lam = np.empty(nf)
    S11 = np.empty(nf, complex)
    S21 = np.empty(nf, complex)
    S11x = np.empty(nf, complex)
    S21x = np.empty(nf, complex)
    R = np.empty(nf)
    T = np.empty(nf)

    for i in range(nf):
        tms = [loaded[j][i] for j in range(len(files))]
        if args.lmax:
            tms = [truncate(t, args.lmax) for t in tms]
        k0 = tms[0].k0
        lam[i] = 2 * np.pi / k0
        cluster = treams.TMatrix.cluster([tms[j] for j in idx], pos)
        tm_lat = cluster.latticeinteraction.solve(lattice, kpar)
        basis = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, bmax)
        smat = treams.SMatrices.from_array(tm_lat, basis)

        kx = np.asarray(basis.kx, float)
        ky = np.asarray(basis.ky, float)
        zero = np.nonzero((np.abs(kx) < 1e-12) & (np.abs(ky) < 1e-12))[0]
        kz2 = k0 ** 2 - kx ** 2 - ky ** 2
        prop = kz2 > 1e-12 * k0 ** 2

        M_up = np.asarray(treams.efield([0, 0, 0.0], basis=basis, k0=k0,
                                        material=tms[0].material,
                                        poltype="parity", modetype="up"))
        M_dn = np.asarray(treams.efield([0, 0, 0.0], basis=basis, k0=k0,
                                        material=tms[0].material,
                                        poltype="parity", modetype="down"))
        # x-polarized 0th-order illumination.  SMatrices convention:
        #   field_up   = S[0,0] illu_up + S[0,1] illu_down
        #   field_down = S[1,0] illu_up + S[1,1] illu_down
        M_in, M_t, M_r = ((M_up, M_up, M_dn) if args.incidence == "up"
                          else (M_dn, M_dn, M_up))
        blk_t, blk_r = ((smat[0, 0], smat[1, 0]) if args.incidence == "up"
                        else (smat[1, 1], smat[0, 1]))
        c = np.zeros(len(basis), complex)
        sub = np.linalg.lstsq(M_in[:, zero], np.array([1.0, 0.0, 0.0]),
                              rcond=None)[0]
        c[zero] = sub
        res = np.linalg.norm(M_in[:, zero] @ sub - np.array([1.0, 0, 0]))
        if res > 1e-10:
            raise RuntimeError(f"x-pol decomposition failed at index {i}: {res}")

        b_t = np.asarray(blk_t) @ c
        b_r = np.asarray(blk_r) @ c
        E_t = M_t[:, zero] @ b_t[zero]
        E_r = M_r[:, zero] @ b_r[zero]
        S21[i], S21x[i] = E_t[0], E_t[1]
        S11[i], S11x[i] = E_r[0], E_r[1]
        # treams' plane-wave S-matrix carries FIELD amplitudes, not
        # power-normalized ones: the flux of an order through a plane parallel
        # to the sheet is |E|^2 k_z, so each propagating order enters the power
        # sums with the weight k_z,G / k_z,0.  (Checked: without it the sum of
        # the (+-1,+-1) orders comes out 1/0.469 = k_0/k_z too large at
        # lambda = 9.99 um, and R + T exceeds 1 for a passive sheet.)
        wt = np.where(prop, np.sqrt(np.maximum(kz2, 0.0)) / k0, 0.0)
        T[i] = float(np.sum(wt * np.abs(b_t) ** 2))
        R[i] = float(np.sum(wt * np.abs(b_r) ** 2))
        print(f"  [{i+1:2d}/{nf}] lam={lam[i]:6.2f} um  |S11|={abs(S11[i]):.4f} "
              f"|S21|={abs(S21[i]):.4f}  R={R[i]:.4f} T={T[i]:.4f} "
              f"A={1-R[i]-T[i]:+.4f}  ({int(prop.sum())//2} open orders)",
              flush=True)

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, lam_um=lam, S11=S11, S21=S21, S11_cross=S11x,
             S21_cross=S21x, R=R, T=T, A=1 - R - T,
             sites=np.array([[files[j], *pos[n][:2]]
                             for n, j in enumerate(idx)], dtype=object),
             cell=args.cell, lmax=args.lmax if args.lmax else -1,
             notes=("treams independent supercell reference: "
                    "TMatrix.cluster -> latticeinteraction.solve(square cell) "
                    "-> PlaneWaveBasisByComp.diffr_orders -> "
                    "SMatrices.from_array; x-polarized normal incidence from "
                    "+z, phase reference z = 0; S21 includes the direct term; "
                    "R and T sum |S|^2 over all PROPAGATING orders."))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
