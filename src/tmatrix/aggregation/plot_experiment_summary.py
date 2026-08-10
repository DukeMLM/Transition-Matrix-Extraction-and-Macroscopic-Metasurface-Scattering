"""One figure for the whole experiment: three atoms, three mixed cells.

    python -m tmatrix.aggregation.plot_experiment_summary

Left   : the three pure 8 um lattices, each over its own direct CST run.
Middle : the three x,y;y,x checkerboards built from them, same treatment.
Right  : the diffracted power of the mixed cells against the Rayleigh onsets of
         their common 16 um period.

The figure exists to make one point visible: a mixed cell is not an
interpolation between its two constituents. Its dips sit at different
frequencies, and it carries features — the dark 16 um lattice resonance, the
11.31 um diffraction edge — that neither pure lattice possesses.

Directories that do not exist yet are skipped, so this runs before the CST
benchmarks finish.
"""
import os

import numpy as np

from tmatrix.plotting import plt, thz_axis
from tmatrix.results_io import load_cst_reference as load_cst

from tmatrix.paths import AGG_DATA

PURE = [("results_A_ewald_l3", "A alone (scale 4.00)", "C0"),
        ("results_B_ewald_l3", "B alone (scale 5.00)", "C1"),
        ("results_C_ewald_l3", "C alone (scale 3.25)", "C2")]
MIXED = [("results_2x2_super_l3", "a,b;b,a", "C3"),
         ("results_2x2_AC_l3", "a,c;c,a", "C4"),
         ("results_2x2_BC_l3", "b,c;c,b", "C5")]
PERIOD = 16.0


def load(d):
    p = os.path.join(AGG_DATA, d)
    f = os.path.join(p, "periodic_results.npz")
    if not os.path.exists(f):
        return None
    return np.load(f, allow_pickle=True), load_cst(p)


def panel(ax, cases, title):
    for d, label, c in cases:
        got = load(d)
        if got is None:
            continue
        r, cst = got
        ax.plot(r["lam"], np.abs(r["S21"]), "o-", ms=3.5, lw=1.6, color=c,
                label=label)
        if cst is not None:
            ax.plot(cst["lam"], np.abs(cst["S21"]), "-", lw=1.4, color=c,
                    alpha=0.38)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("|S21|  (0th order)")
    ax.set_xlim(8.8, 30)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title(title, fontsize=9)
    thz_axis(ax)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    panel(axes[0], PURE,
          "one atom per 8 um cell — markers: aggregation, pale: direct CST")
    panel(axes[1], MIXED,
          "two atoms per 16 um cell — same treatment")

    ax = axes[2]
    for d, label, c in MIXED:
        got = load(d)
        if got is None:
            continue
        r, cst = got
        ax.plot(r["lam"], r["R_hi"] + r["T_hi"], "o-", ms=3.5, lw=1.6, color=c,
                label=label)
        if cst is not None and "R_higher" in cst:
            ax.plot(cst["lam"], cst["R_higher"] + cst["T_higher"], "-", lw=1.4,
                    color=c, alpha=0.38)
    for n, txt in (((1, 1), "(±1,±1) opens"),
                   ((1, 0), "(±1,0) opens\n(dark: extinguished)")):
        lr = PERIOD / np.hypot(*n)
        ax.axvline(lr, color="k", ls=":", lw=1.0)
        ax.text(lr + 0.15, 0.47, f"{lr:.2f} um\n{txt}", fontsize=7.5,
                rotation=90, va="top")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("power into higher orders")
    ax.set_xlim(8.8, 30)
    ax.set_ylim(0, 0.5)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("what the 16 um period adds", fontsize=9)
    thz_axis(ax)

    out = os.path.join(AGG_DATA, "results_2x2_super_l3", "fig3_experiment.png")
    fig.savefig(out, dpi=170)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
