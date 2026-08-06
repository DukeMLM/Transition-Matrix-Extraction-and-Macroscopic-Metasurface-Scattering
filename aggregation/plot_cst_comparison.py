"""Overlay: direct CST periodic simulation vs T-matrix aggregation vs treams."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_results import thz_axis

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
C0 = 299792458.0


def main(run_dir="cst_direct/run_v2"):
    d = np.load(os.path.join(OUT, "periodic_results.npz"))
    lam = d["lam"]
    tr = np.load(os.path.join(OUT, "treams_reference.npz"), allow_pickle=True)

    csv = os.path.join(HERE, run_dir, "s_params_complex.csv")
    raw = np.loadtxt(csv, delimiter=",", skiprows=1)
    f_thz = raw[:, 0]
    lam_cst = C0 / (f_thz * 1e12) * 1e6
    s11_cst = raw[:, 1] + 1j * raw[:, 2]
    s21_cst = raw[:, 3] + 1j * raw[:, 4]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for ax, key, mine, cst in [(axes[0], "S21", d["S21"], s21_cst),
                               (axes[1], "S11", d["S11"], s11_cst)]:
        ax.plot(lam_cst, np.abs(cst), "-", lw=2.2, color="C2", alpha=0.85,
                label=f"|{key}| CST direct periodic sim")
        ax.plot(lam, np.abs(mine), "o", ms=4, color="C0",
                label=f"|{key}| T-matrix aggregation (this work)")
        ax.plot(tr["lam_um"], np.abs(tr[key]), "s", ms=3, color="C1",
                alpha=0.7, label=f"|{key}| treams")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel(f"|{key}|")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        thz_axis(ax)
    fig.suptitle("Free-standing spoke-and-wheel array, pitch 2 um, "
                 "normal incidence", fontsize=11)
    fig.savefig(os.path.join(OUT, "fig7_cst_direct_comparison.png"), dpi=200)

    # quantify agreement on the common grid
    s21_i = np.interp(lam, lam_cst[::-1], np.abs(s21_cst)[::-1])
    s11_i = np.interp(lam, lam_cst[::-1], np.abs(s11_cst)[::-1])
    print(f"max | |S21|_cst - |S21|_agg | = {np.abs(s21_i - np.abs(d['S21'])).max():.4f}")
    print(f"max | |S11|_cst - |S11|_agg | = {np.abs(s11_i - np.abs(d['S11'])).max():.4f}")
    print("saved fig7_cst_direct_comparison.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
