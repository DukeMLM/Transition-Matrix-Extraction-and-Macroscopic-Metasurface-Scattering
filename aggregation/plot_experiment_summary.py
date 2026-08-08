"""One figure for the whole a,b;b,a experiment: A alone, B alone, and the mix.

    python plot_experiment_summary.py

Left  : |S21| of the two pure 8 um lattices and of the checkerboard, each with
        its own direct CST run underneath.
Right : the same for |S11|, plus the diffracted power fraction of the mixed
        cell and the Rayleigh onsets of its 16 um period.

The point of the figure is that the mixed cell is not interpolation between the
two pure ones: its dips sit at different frequencies, and it has features
(the 16 um dark lattice resonance, the 11.31 um diffraction edge) that neither
pure lattice possesses.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_supercell import load_cst, thz_axis

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [("results_A_ewald_l3", "A alone (scale 4), 8 um lattice", "C0"),
         ("results_B_ewald_l3", "B alone (scale 5), 8 um lattice", "C1"),
         ("results_2x2_super_l3", "a,b;b,a checkerboard, 16 um cell", "C2")]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    mix = None
    for d, label, c in CASES:
        p = os.path.join(HERE, d)
        r = np.load(os.path.join(p, "periodic_results.npz"), allow_pickle=True)
        cst = load_cst(p)
        if d.startswith("results_2x2"):
            mix = (r, cst)
        for ax, key in ((axes[0], "S21"), (axes[1], "S11")):
            ax.plot(r["lam"], np.abs(r[key]), "o-", ms=4, lw=1.7, color=c,
                    label=label)
            if cst is not None:
                ax.plot(cst["lam"], np.abs(cst[key]), "-", lw=1.4, color=c,
                        alpha=0.4)

    for ax, key in ((axes[0], "S21"), (axes[1], "S11")):
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel(f"|{key}|  (0th order)")
        ax.set_xlim(8.8, 30)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
        thz_axis(ax)
    axes[0].set_title("solid+markers: T-matrix aggregation;  "
                      "pale line: direct CST", fontsize=9)

    ax = axes[2]
    if mix is not None:
        r, cst = mix
        ax.plot(r["lam"], r["R_hi"] + r["T_hi"], "o-", ms=4, color="C2",
                label="diffracted power, aggregation")
        if cst is not None and "R_higher" in cst:
            ax.plot(cst["lam"], cst["R_higher"] + cst["T_higher"], "-", lw=1.4,
                    color="0.4", label="diffracted power, direct CST")
        ax.plot(r["lam"], r["A"], "s--", ms=3, lw=1.1, color="C4",
                label="absorption A = 1 - R - T")
    for n, txt in (((1, 0), "(±1,0) opens\n(dark: extinguished)"),
                   ((1, 1), "(±1,±1) opens")):
        lr = 16.0 / np.hypot(*n)
        ax.axvline(lr, color="k", ls=":", lw=1.0)
        ax.text(lr + 0.15, 0.42, f"{lr:.2f} um\n{txt}", fontsize=7.5,
                rotation=90, va="top")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("power fraction")
    ax.set_xlim(8.8, 30)
    ax.set_ylim(0, 0.5)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("mixed cell only: what the 16 um period adds", fontsize=9)
    thz_axis(ax)

    out = os.path.join(HERE, "results_2x2_super_l3", "fig3_experiment.png")
    fig.savefig(out, dpi=170)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
