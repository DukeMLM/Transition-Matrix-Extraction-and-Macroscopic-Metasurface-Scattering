"""Figures for the demo: S-parameters and cross sections vs wavelength/THz."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
C0 = 299792458.0


def thz_axis(ax):
    sec = ax.secondary_xaxis(
        "top", functions=(lambda lam: C0 / np.maximum(lam, 1e-9) * 1e-6,
                          lambda f: C0 / np.maximum(f, 1e-9) * 1e-6))
    sec.set_xlabel("Frequency (THz)")
    return sec


def main():
    d = np.load(os.path.join(OUT, "periodic_results.npz"))
    lam = d["lam"]
    fin = np.load(os.path.join(OUT, "finite_results.npz"))

    # ---- Fig 1: single-scatterer cross sections
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(lam, d["sig_ext"], "-o", ms=3, lw=1.4, label=r"$\sigma_{ext}$")
    ax.plot(lam, d["sig_sca"], "-s", ms=3, lw=1.4, label=r"$\sigma_{sca}$")
    ax.plot(lam, d["sig_abs"], "-^", ms=3, lw=1.4, label=r"$\sigma_{abs}$")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel(r"Cross section (um$^2$)")
    ax.set_title("Isolated spoke-and-wheel resonator (from CST T-matrix)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    thz_axis(ax)
    fig.savefig(os.path.join(OUT, "fig1_cross_sections.png"), dpi=200)

    # ---- Fig 2: periodic-array S-parameters
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(lam, np.abs(d["S21"]), "-o", ms=3, lw=1.5, color="C0",
            label=r"$|S_{21}|$ coupled (Foldy-Lax)")
    ax.plot(lam, np.abs(d["S21_nc"]), "--", lw=1.2, color="C0", alpha=0.55,
            label=r"$|S_{21}|$ uncoupled sheet")
    ax.plot(lam, np.abs(d["S11"]), "-s", ms=3, lw=1.5, color="C3",
            label=r"$|S_{11}|$ coupled")
    ax.plot(lam, np.abs(d["S11_nc"]), "--", lw=1.2, color="C3", alpha=0.55,
            label=r"$|S_{11}|$ uncoupled sheet")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("|S|")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Infinite array, pitch 2 um, normal incidence, x-pol",
                 fontsize=11)
    thz_axis(ax)

    ax = axes[1]
    ax.plot(lam, d["R"], "-s", ms=3, lw=1.5, color="C3", label="R")
    ax.plot(lam, d["T"], "-o", ms=3, lw=1.5, color="C0", label="T")
    ax.plot(lam, d["A"], "-^", ms=3, lw=1.5, color="C2", label="A = 1-R-T")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Power")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Power balance (energy check: A must stay in [0, 1])",
                 fontsize=11)
    thz_axis(ax)
    fig.savefig(os.path.join(OUT, "fig2_sparams_periodic.png"), dpi=200)

    # ---- Fig 3: finite arrays vs periodic
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(lam, np.abs(d["S21"]), "-", lw=2.0, color="k",
            label="infinite (lattice sum)")
    for i, key in enumerate(["N5", "N9", "N13"]):
        arr = fin[key]
        ax.plot(arr[:, 0].real, np.abs(arr[:, 2]), "o--", ms=4, lw=1.0,
                color=f"C{i}", label=f"{key[1:]}x{key[1:]} Foldy-Lax")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel(r"$|S_{21}|$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Finite-array aggregation converges to the periodic result",
                 fontsize=11)
    thz_axis(ax)
    fig.savefig(os.path.join(OUT, "fig3_finite_vs_periodic.png"), dpi=200)

    # ---- treams comparison if available
    tr_path = os.path.join(OUT, "treams_reference.npz")
    if os.path.exists(tr_path):
        tr = np.load(tr_path, allow_pickle=True)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                 constrained_layout=True)
        for ax, key, mine in [(axes[0], "S21", d["S21"]),
                              (axes[1], "S11", d["S11"])]:
            ax.plot(lam, np.abs(mine), "-o", ms=3, lw=1.5,
                    label=f"|{key}| this work")
            ax.plot(tr["lam_um"], np.abs(tr[key]), "--s", ms=3, lw=1.2,
                    label=f"|{key}| treams (independent)")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel(f"|{key}|")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False)
            thz_axis(ax)
        fig.suptitle("Cross-validation against treams (Ewald lattice sums)",
                     fontsize=11)
        fig.savefig(os.path.join(OUT, "fig4_treams_crosscheck.png"), dpi=200)

    # ---- CSV export
    import csv
    with open(os.path.join(OUT, "sparams_periodic.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lambda_um", "freq_THz",
                    "Re_S11", "Im_S11", "Re_S21", "Im_S21",
                    "abs_S11", "abs_S21", "R", "T", "A"])
        fTHz = C0 / (lam * 1e-6) / 1e12
        for i in range(len(lam)):
            w.writerow([f"{lam[i]:.4f}", f"{fTHz[i]:.4f}",
                        f"{d['S11'][i].real:.6f}", f"{d['S11'][i].imag:.6f}",
                        f"{d['S21'][i].real:.6f}", f"{d['S21'][i].imag:.6f}",
                        f"{abs(d['S11'][i]):.6f}", f"{abs(d['S21'][i]):.6f}",
                        f"{d['R'][i]:.6f}", f"{d['T'][i]:.6f}",
                        f"{d['A'][i]:.6f}"])
    print("figures + CSV written to", OUT)


if __name__ == "__main__":
    main()
