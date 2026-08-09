"""The whole study in one figure: 4 atoms, 5 mixed cells, 9 CST benchmarks.

    python plot_comparison.py [--out results_2x2_super_l3/fig4_comparison.png]

Top row     one cell family per panel -- single atoms, two-species checkerboards,
            four-distinct-atom cells.  Markers are the T-matrix prediction, pale
            lines the direct CST run for the same structure.
Bottom left what the 16 um period adds: power into higher diffraction orders,
            against the two Rayleigh onsets.  The two-species cells are dark
            between them by symmetry; the four-atom cells are not.
Bottom mid  the four-atom cells are birefringent -- |t_xx| against |t_yy| -- and
            how much depends on the arrangement, not the composition.
Bottom right the accuracy summary: mean |dS21| against the manual's Eq. (57)
            margin, min over the 8 um neighbour pairs of (8 um - a_i - a_j).

Cases whose directory or CST reference is missing are skipped, so this runs
while benchmarks are still solving.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_supercell import load_cst, interp_c, thz_axis

HERE = os.path.dirname(os.path.abspath(__file__))
PERIOD, PITCH = 16.0, 8.0
R = {"A": 2.87712, "B": 3.59639, "C": 2.33766, "D": 3.95603}

SINGLE = [("results_C_ewald_l3", "C  (scale 3.25)", "C2", "C"),
          ("results_A_ewald_l3", "A  (scale 4.00)", "C0", "A"),
          ("results_B_ewald_l3", "B  (scale 5.00)", "C1", "B"),
          ("results_D_ewald_l3", "D  (scale 5.50)", "C3", "D")]
PAIRS = [("results_2x2_super_l3", "a,b;b,a", "C0", "AB"),
         ("results_2x2_AC_l3", "a,c;c,a", "C1", "AC"),
         ("results_2x2_BC_l3", "b,c;c,b", "C2", "BC")]
QUADS = [("results_2x2_ABCD_l3", "a,b;c,d", "C3", "ABCD"),
         ("results_2x2_ADBC_l3", "a,d;b,c", "C4", "ADBC")]


def margin(spec):
    """Manual Eq. (57) headroom: min over the 8 um neighbour pairs."""
    if len(spec) == 1:                       # one atom per cell
        return PITCH - 2 * R[spec]
    if len(spec) == 2:                       # x,y;y,x -- every neighbour unlike
        return PITCH - R[spec[0]] - R[spec[1]]
    w, x, y, z = spec                        # w,x;y,z -- rows and columns
    return PITCH - max(R[w] + R[x], R[y] + R[z], R[w] + R[y], R[x] + R[z])


def load(d):
    p = os.path.join(HERE, d)
    f = os.path.join(p, "periodic_results.npz")
    if not os.path.exists(f):
        return None
    return np.load(f, allow_pickle=True), load_cst(p)


def spanel(ax, cases, title):
    for d, label, c, _ in cases:
        got = load(d)
        if got is None:
            continue
        r, cst = got
        ax.plot(r["lam"], np.abs(r["S21"]), "o-", ms=3.2, lw=1.5, color=c,
                label=label)
        if cst is not None:
            ax.plot(cst["lam"], np.abs(cst["S21"]), "-", lw=1.4, color=c,
                    alpha=0.35)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("|S21|  (0th order)")
    ax.set_xlim(8.8, 30)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(title, fontsize=9)
    thz_axis(ax)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results_2x2_super_l3/fig4_comparison.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.2), constrained_layout=True)
    spanel(ax[0, 0], SINGLE, "one atom per 8 um cell")
    spanel(ax[0, 1], PAIRS, "two species per 16 um cell (x,y;y,x)")
    spanel(ax[0, 2], QUADS, "four species per 16 um cell")
    ax[0, 0].set_title("one atom per 8 um cell — markers: T-matrix, pale: CST",
                       fontsize=9)

    # ---- diffracted power -------------------------------------------------
    a = ax[1, 0]
    for d, label, c, _ in PAIRS + QUADS:
        got = load(d)
        if got is None:
            continue
        r, cst = got
        a.plot(r["lam"], r["R_hi"] + r["T_hi"], "o-", ms=3.2, lw=1.5, color=c,
               label=label)
        if cst is not None and "R_higher" in cst:
            a.plot(cst["lam"], cst["R_higher"] + cst["T_higher"], "-", lw=1.4,
                   color=c, alpha=0.35)
    for n, txt in (((1, 1), "(±1,±1)"), ((1, 0), "(±1,0), (0,±1)")):
        lr = PERIOD / np.hypot(*n)
        a.axvline(lr, color="k", ls=":", lw=1.0)
        a.text(lr + 0.15, 0.60, f"{lr:.2f} um\n{txt} opens", fontsize=7.5,
               rotation=90, va="top")
    a.set_xlabel("Wavelength (um)")
    a.set_ylabel("power into higher orders")
    a.set_xlim(8.8, 30)
    a.set_ylim(0, 0.65)
    a.grid(alpha=0.3)
    a.legend(frameon=False, fontsize=8, loc="upper right")
    a.set_title("two-species cells are dark between the onsets; four-species "
                "are not", fontsize=9)
    thz_axis(a)

    # ---- birefringence ----------------------------------------------------
    a = ax[1, 1]
    jp = os.path.join(HERE, "results_2x2_ABCD_l3", "jones_xy.npz")
    if os.path.exists(jp):
        j = np.load(jp)
        for key, label, c in (("abcd", "a,b;c,d", "C3"),
                              ("adbc", "a,d;b,c", "C4")):
            a.plot(j["lam"], j[key][:, 0], "o-", ms=3.2, lw=1.5, color=c,
                   label=f"{label}  |t_xx|")
            a.plot(j["lam"], j[key][:, 1], "s--", ms=3.2, lw=1.3, color=c,
                   alpha=0.65, label=f"{label}  |t_yy|")
        a.set_ylabel("|t| of the 0th transmitted order")
    a.set_xlabel("Wavelength (um)")
    a.set_xlim(8.8, 30)
    a.set_ylim(0, 1.02)
    a.grid(alpha=0.3)
    a.legend(frameon=False, fontsize=8, loc="lower right")
    a.set_title("four species -> birefringent; how much is set by the "
                "arrangement", fontsize=9)
    thz_axis(a)

    # ---- accuracy vs the translation-validity margin ----------------------
    a = ax[1, 2]
    for cases, mk in ((SINGLE, "o"), (PAIRS, "s"), (QUADS, "D")):
        for d, label, c, spec in cases:
            got = load(d)
            if got is None or got[1] is None:
                continue
            r, cst = got
            err = np.abs(r["S21"] - interp_c(r["lam"], cst["lam"],
                                             cst["S21"])).mean()
            a.plot(margin(spec), err, mk, ms=9, color=c, mec="k", mew=0.5)
            a.annotate(label.split()[0], (margin(spec), err), fontsize=8,
                       xytext=(6, -3), textcoords="offset points")
    a.axvspan(-0.2, 1.0, color="0.85", zorder=0)
    a.text(0.4, 0.32, "spheres\nnearly touch", fontsize=8, ha="center")
    a.set_yscale("log")
    a.set_xlabel("Eq. (57) margin  min(8 um − a_i − a_j)  over the 8 um pairs")
    a.set_ylabel("mean |ΔS21| vs direct CST")
    a.set_xlim(-0.2, 3.6)
    a.grid(alpha=0.3, which="both")
    a.set_title("accuracy tracks the translation-validity headroom\n"
                "(circles: 1 atom, squares: 2 species, diamonds: 4)",
                fontsize=9)

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    fig.savefig(out, dpi=165)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
