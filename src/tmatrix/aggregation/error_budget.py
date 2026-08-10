"""Where the disagreement with direct CST comes from, frequency by frequency.

    python error_budget.py results_A_ewald_l3
    python error_budget.py results_2x2_super_l3 --draws 32

The aggregation itself is exact -- it reproduces an independent treams
implementation to 1e-12 (`test_supercell.py`).  Everything that remains between
a reconstruction and its direct full-wave reference is therefore inherited from
the input T-matrices, and this script separates the two things that decide how
much of it survives into S:

  how badly T is known        the h5's own stored `residual`, and the isolated
                              scattering cross section that sets its SNR --
                              a weak scatterer is a poorly conditioned fit
  how hard the lattice        |W| resolved by multipole order, and the solve
  amplifies that              amplification ||(I - W T0)^-1||

and then propagates the declared uncertainty end to end: perturb every site's T
by a random matrix of exactly the norm the file declares, re-run the whole
aggregation, and report the RMS |dS11| + |dS21| over `--draws` draws.

Reading the last column.  A random perturbation is the *most benign* error of a
given size: it has no preferred direction, so it cannot move a resonance.  Where
the observed disagreement is only a small multiple of the prediction, the error
really is noise of the declared size, amplified by the lattice.  Where it is
tens of times larger, no perturbation of that norm explains it and the residual
must be systematic -- in practice a pole slightly off frequency, which shows up
hardest where the response varies fastest.
"""
import argparse
import json
import os

import h5py
import numpy as np

from tmatrix.aggregation.tmat_io import TMatrixData
from tmatrix.aggregation.vswf import ModeBasis
from tmatrix.results_io import interp_c, load_cst_reference as load_cst
from tmatrix.aggregation import supercell as sc
from tmatrix.aggregation import ewald_supercell as ew

from tmatrix.paths import AGG_DATA
from tmatrix.units import C_UM_THZ as C0



def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(
        AGG_DATA, args.out_dir)

    cfg = json.load(open(os.path.join(out, "run.json")))
    a1, a2 = tuple(cfg["a1"]), tuple(cfg["a2"])
    files = [s[0] for s in cfg["sites"]]
    rho = np.array([[s[1], s[2]] for s in cfg["sites"]])
    uniq = sorted(set(files))
    data = {f: TMatrixData(f) for f in uniq}
    resid = {}
    for f in uniq:
        with h5py.File(f, "r") as fh:
            resid[f] = fh["computation/analysis/residual"][:]
    ref = data[files[0]]

    m = np.load(os.path.join(out, "periodic_results.npz"), allow_pickle=True)
    lam = m["lam"]
    cst = load_cst(out)
    if cst is None:
        raise SystemExit(f"no direct CST reference CSV in {out}")
    r21 = interp_c(lam, cst["lam"], cst["S21"])
    r11 = interp_c(lam, cst["lam"], cst["S11"])
    obs = np.abs(m["S21"] - r21) + np.abs(m["S11"] - r11)

    lmax = int(m["lmax"].max())
    sel = np.nonzero(ref.modes.l <= lmax)[0]
    mb = ModeBasis(ref.modes.l[sel], ref.modes.m[sel], ref.modes.pol[sel])
    ix = np.ix_(sel, sel)
    rng = np.random.default_rng(args.seed)
    e_inc = np.asarray(m["e_inc"])
    M = len(rho)

    print(f"{os.path.basename(out)}: {M} atom(s), cell {a1} x {a2} um, "
          f"lmax {lmax}, {args.draws} draws")
    print()
    print("  f_THz    lam    |dS| vs CST   h5 resid   sigma_sca   "
          "|W| l=1 / l=2 / l=3      ||(I-WT0)^-1||   predicted |dS|   ratio")
    rows = []
    for i in range(len(lam)):
        k = 2 * np.pi / lam[i] * np.sqrt(ref.eps_emb * ref.mu_emb).real
        W = ew.block_lattice_sums_ewald(k, a1, a2, rho, mb)
        T_list = [data[f].T[i][ix] for f in files]
        a_inc = sc.incident_blocks([0, 0, 1], e_inc, mb, rho, k)
        big = sc.build_block_system(W, T_list)
        amp = np.linalg.norm(np.linalg.inv(big), 2)

        def s_of(Ts):
            _, fu = sc.solve_supercell(W, Ts, a_inc)
            return sc.zeroth_order(
                sc.floquet_smatrix(k, a1, a2, rho, mb, fu, e_inc=e_inc),
                e_co=e_inc)

        S0 = s_of(T_list)
        ds = []
        for _ in range(args.draws):
            pert = []
            for f, T in zip(files, T_list):
                E = rng.normal(size=T.shape) + 1j * rng.normal(size=T.shape)
                E *= resid[f][i] * np.linalg.norm(T) / np.linalg.norm(E)
                pert.append(T + E)
            S1 = s_of(pert)
            ds.append(abs(S1["S11_co"] - S0["S11_co"])
                      + abs(S1["S21_co"] - S0["S21_co"]))
        pred = float(np.sqrt(np.mean(np.square(ds))))

        wl = [max(np.abs(W[s, t][np.ix_(mb.l == L, mb.l == L)]).max()
                  for s in range(M) for t in range(M)
                  if not (s == t and M == 1) or True)
              for L in (1, 2, 3) if L <= lmax]
        rmax = max(resid[f][i] for f in set(files))
        sig = m["sig_sca"][i]
        rows.append((C0 / lam[i], lam[i], obs[i], rmax, sig, wl, amp, pred))
        print(f"  {C0/lam[i]:5.1f} {lam[i]:7.2f}      {obs[i]:.4f}     "
              f"{rmax:.4f}   {sig:9.3f}   "
              + " / ".join(f"{x:8.2f}" for x in wl)
              + f"     {amp:9.2f}       {pred:.4f}      {obs[i]/pred:6.2f}")

    lo = slice(0, max(1, len(lam) // 5))
    hi = slice(-max(1, len(lam) // 5), None)
    print()
    print("  long-wavelength fifth : observed mean "
          f"{obs[lo].mean():.4f}, predicted {np.mean([r[7] for r in rows[lo]]):.4f}, "
          f"ratio {obs[lo].mean()/np.mean([r[7] for r in rows[lo]]):.2f}")
    print("  short-wavelength fifth: observed mean "
          f"{obs[hi].mean():.4f}, predicted {np.mean([r[7] for r in rows[hi]]):.4f}, "
          f"ratio {obs[hi].mean()/np.mean([r[7] for r in rows[hi]]):.2f}")
    print("  whole band            : observed mean "
          f"{obs.mean():.4f}, predicted {np.mean([r[7] for r in rows]):.4f}")


if __name__ == "__main__":
    main()
