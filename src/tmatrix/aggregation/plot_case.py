"""Figures + agreement metrics for a run_case.py output directory.

Overlays whatever is present in the directory:
  periodic_results.npz        this repo's Foldy-Lax reconstruction (required)
  treams_reference.npz        independent Ewald cross-check (treams_case.py)
  cst_direct_reference.csv    direct CST periodic run (cst_packed_reference.py)

    python -m tmatrix.aggregation.plot_case results_2x2 \
        --title "spoke-and-wheel, pitch 8 um"
"""
import argparse
import os

import numpy as np

from tmatrix.paths import AGG_DATA
from tmatrix.plotting import parabola_min, plt, thz_axis
from tmatrix.results_io import load_cst_reference, load_treams


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    out = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(
        AGG_DATA, args.out_dir)

    d = np.load(os.path.join(out, "periodic_results.npz"))
    lam, S11, S21 = d["lam"], d["S11"], d["S21"]
    R, T, A = d["R"], d["T"], d["A"]

    tr = load_treams(out)
    # run_case.py cells are one atom per cell, so only the reference export is
    # meaningful here -- the supercell schema would be a different experiment.
    cst = load_cst_reference(out, names=("cst_direct_reference.csv",))

    # ---- fig 1: |S11|, |S21| ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, key, mine in ((axes[0], "S21", S21), (axes[1], "S11", S11)):
        if cst is not None:
            ax.plot(cst["lam"], np.abs(cst[key]), "-", lw=1.8, color="0.6",
                    label=f"|{key}| CST direct (packed .cst in folder)")
        ax.plot(lam, np.abs(mine), "o-", ms=5, lw=1.6, color="C0",
                label=f"|{key}| T-matrix -> Foldy-Lax (this repo)")
        if tr is not None:
            ax.plot(tr["lam_um"], np.abs(tr[key]), "s--", ms=4, lw=1.2,
                    color="C1", alpha=0.85, label=f"|{key}| treams (Ewald)")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel(f"|{key}|")
        ax.set_xlim(lam.min(), lam.max())
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
        thz_axis(ax)
    fig.suptitle(args.title or os.path.basename(out), fontsize=11)
    f1 = os.path.join(out, "fig1_sparams.png")
    fig.savefig(f1, dpi=180)

    # ---- fig 2: power balance + truncation ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axes[0]
    if cst is not None:
        for y, c in ((cst["R"], "C3"), (cst["T"], "C2"), (cst["A"], "C4")):
            ax.plot(cst["lam"], y, "-", lw=1.4, color=c, alpha=0.35)
    ax.plot(lam, R, "o-", ms=4, color="C3", label="R (reflectance)")
    ax.plot(lam, T, "o-", ms=4, color="C2", label="T (transmittance)")
    ax.plot(lam, A, "o-", ms=4, color="C4", label="A = 1 - R - T")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("power fraction")
    ax.set_xlim(lam.min(), lam.max())
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("thin lines: CST direct run in the same folder", fontsize=9)
    thz_axis(ax)

    ax = axes[1]
    ax.plot(lam, d["lmax"], "o-", color="C0", label="lmax used")
    ax.set_ylabel("multipole truncation lmax", color="C0")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylim(0, max(d["lmax"]) + 1)
    ax2 = ax.twinx()
    ax2.semilogy(lam, d["cond"], "s--", ms=4, color="C1")
    ax2.set_ylabel("cond(I - C T)", color="C1")
    ax.grid(alpha=0.3)
    ax.set_title("adaptive truncation and Foldy-Lax conditioning", fontsize=9)
    f2 = os.path.join(out, "fig2_balance_truncation.png")
    fig.savefig(f2, dpi=180)

    # ---- metrics ------------------------------------------------------------
    print(f"transmission minimum (parabolic fit):")
    print(f"  this repo   : {parabola_min(lam, np.abs(S21)):.3f} um")
    if tr is not None:
        print(f"  treams      : {parabola_min(tr['lam_um'], np.abs(tr['S21'])):.3f} um")
    if cst is not None:
        m = (cst["lam"] >= lam.min()) & (cst["lam"] <= lam.max())
        print(f"  CST direct  : {parabola_min(cst['lam'][m], np.abs(cst['S21'][m])):.3f} um")
    if tr is not None:
        d21 = np.abs(np.abs(S21) - np.abs(tr["S21"]))
        d11 = np.abs(np.abs(S11) - np.abs(tr["S11"]))
        print(f"vs treams: max ||S21|-|S21|| = {d21.max():.4f} "
              f"(mean {d21.mean():.4f}); max ||S11|-|S11|| = {d11.max():.4f} "
              f"(mean {d11.mean():.4f})")
    if cst is not None:
        s21c = np.interp(lam, cst["lam"][::-1], np.abs(cst["S21"])[::-1])
        s11c = np.interp(lam, cst["lam"][::-1], np.abs(cst["S11"])[::-1])
        print(f"vs CST direct: max ||S21|-|S21|| = "
              f"{np.abs(s21c - np.abs(S21)).max():.4f}; max ||S11|-|S11|| = "
              f"{np.abs(s11c - np.abs(S11)).max():.4f}")
    print(f"saved {f1}\n      {f2}")


if __name__ == "__main__":
    main()
