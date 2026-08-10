"""Figures and agreement metrics for a run_supercell.py output directory.

Overlays whatever it finds:
  periodic_results.npz            this repo's block-Bloch reconstruction
  floquet_orders.csv              its per-order Floquet amplitudes
  treams_reference.npz            independent treams/Ewald cross-check
  cst_direct_reference.csv        direct CST run, one atom per cell
  cst_direct_supercell.csv        direct CST run, heterogeneous supercell

    python -m tmatrix.aggregation.plot_supercell results_2x2_super_l3 \
        --title "a,b;b,a, 16 um cell"
"""
import argparse
import os

import numpy as np

from tmatrix.paths import AGG_DATA
from tmatrix.plotting import parabola_min, plt, thz_axis
from tmatrix.results_io import interp_c, load_cst_reference as load_cst


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--title", default="")
    ap.add_argument("--fine", default=None,
                    help="a second run_supercell.py directory on a refined "
                         "frequency grid (--refine N), drawn as a thin line so "
                         "the lattice resonances the stored grid aliases are "
                         "visible")
    args = ap.parse_args()
    out = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(
        AGG_DATA, args.out_dir)
    fine = None
    if args.fine:
        fp = args.fine if os.path.isabs(args.fine) else os.path.join(AGG_DATA,
                                                                     args.fine)
        fine = np.load(os.path.join(fp, "periodic_results.npz"),
                       allow_pickle=True)

    d = np.load(os.path.join(out, "periodic_results.npz"), allow_pickle=True)
    lam, S11, S21 = d["lam"], d["S11"], d["S21"]
    R, T, A = d["R"], d["T"], d["A"]
    a1, a2 = d["a1"], d["a2"]
    period = float(np.linalg.norm(a1))

    tr = None
    p = os.path.join(out, "treams_reference.npz")
    if os.path.exists(p):
        tr = np.load(p, allow_pickle=True)
    cst = load_cst(out)

    # ---- fig 1: complex S ---------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    for col, (key, mine) in enumerate((("S21", S21), ("S11", S11))):
        for row, part in enumerate(("abs", "phase")):
            ax = axes[row, col]
            def y(v):
                return np.abs(v) if part == "abs" else np.unwrap(np.angle(v))
            if cst is not None:
                ax.plot(cst["lam"], y(cst[key]), "-", lw=1.8, color="0.6",
                        label=f"direct CST ({cst['name'].split('_')[-1][:-4]})")
            if fine is not None:
                ax.plot(fine["lam"], y(fine[key]), "-", lw=1.0, color="C0",
                        alpha=0.55,
                        label="this repo, refined frequency grid")
            ax.plot(lam, y(mine), "o-", ms=5, lw=1.6, color="C0",
                    label="T-matrix -> block-Bloch (this repo)")
            if tr is not None:
                ax.plot(tr["lam_um"], y(tr[key]), "s--", ms=4, lw=1.1,
                        color="C1", alpha=0.85, label="treams (independent)")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel(f"|{key}|" if part == "abs" else f"arg {key} (rad)")
            ax.set_xlim(lam.min(), lam.max())
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(frameon=False, fontsize=8.5)
                thz_axis(ax)
    fig.suptitle(args.title or os.path.basename(out), fontsize=11)
    f1 = os.path.join(out, "fig1_sparams.png")
    fig.savefig(f1, dpi=170)
    plt.close(fig)

    # ---- fig 2: power balance and diffraction -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    ax = axes[0]
    if cst is not None:
        for yv, c in ((cst["R"], "C3"), (cst["T"], "C2"), (cst["A"], "C4")):
            ax.plot(cst["lam"], yv, "-", lw=1.4, color=c, alpha=0.35)
    ax.plot(lam, R, "o-", ms=4, color="C3", label="R (all open orders)")
    ax.plot(lam, T, "o-", ms=4, color="C2", label="T (all open orders)")
    ax.plot(lam, A, "o-", ms=4, color="C4", label="A = 1 - R - T")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("power fraction")
    ax.set_xlim(lam.min(), lam.max())
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("thin lines: direct CST", fontsize=9)
    thz_axis(ax)

    ax = axes[1]
    hi = d["R_hi"] + d["T_hi"]
    ax.plot(lam, hi, "o-", ms=4, color="C5",
            label="power into higher orders (this repo)")
    if cst is not None and "R_higher" in cst:
        ax.plot(cst["lam"], cst["R_higher"] + cst["T_higher"], "-", lw=1.4,
                color="0.5", label="direct CST")
    for n in ((1, 0), (1, 1), (2, 0)):
        lr = period / np.hypot(*n)
        if lam.min() <= lr <= lam.max():
            ax.axvline(lr, color="k", ls=":", lw=0.9)
            ax.text(lr, ax.get_ylim()[1] * 0.92, f"  {n} opens",
                    fontsize=8, rotation=90, va="top")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("diffracted power fraction")
    ax.set_xlim(lam.min(), lam.max())
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Rayleigh onsets of the {period:g} um cell", fontsize=9)
    thz_axis(ax)
    f2 = os.path.join(out, "fig2_power.png")
    fig.savefig(f2, dpi=170)
    plt.close(fig)

    # ---- metrics ------------------------------------------------------------
    print(f"=== {os.path.basename(out)} ===")
    print(f"  cell {a1} x {a2} um, {len(lam)} frequencies, "
          f"lmax {d['lmax'].min()}-{d['lmax'].max()}")
    print(f"  transmission minimum (parabolic fit in frequency):")
    print(f"    this repo  : {parabola_min(lam, np.abs(S21)):.3f} um")
    if tr is not None:
        print(f"    treams     : {parabola_min(tr['lam_um'], np.abs(tr['S21'])):.3f} um")
    if cst is not None:
        m = (cst["lam"] >= lam.min()) & (cst["lam"] <= lam.max())
        print(f"    direct CST : {parabola_min(cst['lam'][m], np.abs(cst['S21'][m])):.3f} um")
    if tr is not None:
        for key, mine, ref in (("S21", S21, tr["S21"]), ("S11", S11, tr["S11"])):
            dd = np.abs(mine - ref)
            print(f"  vs treams  complex {key}: max {dd.max():.3e}  "
                  f"mean {dd.mean():.3e}")
    if cst is not None:
        for key, mine in (("S21", S21), ("S11", S11)):
            ref = interp_c(lam, cst["lam"], cst[key])
            dd = np.abs(mine - ref)
            print(f"  vs CST     complex {key}: max {dd.max():.4f}  "
                  f"mean {dd.mean():.4f}")
        o = np.argsort(cst["lam"])
        for key, mine in (("R", R), ("T", T), ("A", A)):
            ref = np.interp(lam, cst["lam"][o], cst[key][o])
            print(f"  vs CST     {key}: max |d| {np.abs(mine-ref).max():.4f}  "
                  f"mean {np.abs(mine-ref).mean():.4f}")
    print(f"  saved {f1}\n        {f2}")


if __name__ == "__main__":
    main()
