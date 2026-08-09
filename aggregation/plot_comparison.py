"""The whole study in one figure: 4 atoms, 5 mixed cells, 9 CST benchmarks.

    python plot_comparison.py [--out results_2x2_super_l3/fig4_comparison.png]

Top row     single atoms, the two-species checkerboards, then ONE PANEL PER
            four-distinct-atom cell -- those two carry the most structure, and
            overlaying them hides which pale CST curve belongs to which.
            Markers are the T-matrix prediction, pale lines the direct CST run
            of the same structure.
Bottom row  power into higher diffraction orders against the two Rayleigh
            onsets (the two-species cells are dark between them by symmetry,
            the four-atom cells are not); the birefringence of each four-atom
            cell, again one panel each; and the accuracy summary -- mean |dS21|
            against the addition theorem's convergence ratio
            rho = (a_i + a_j)/d, maximised over the 8 um neighbour pairs.

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


def rho(spec):
    """Convergence ratio of the addition theorem, worst 8 um neighbour pair.

    The truncation error of the outgoing->regular translation falls like
    rho^lmax with rho = (a_i + a_j) / d.  Manual Eq. (57) only asks for rho < 1;
    what decides the accuracy is how far below 1 it sits.
    """
    if len(spec) == 1:                       # one atom per cell
        return 2 * R[spec] / PITCH
    if len(spec) == 2:                       # x,y;y,x -- every neighbour unlike
        return (R[spec[0]] + R[spec[1]]) / PITCH
    w, x, y, z = spec                        # w,x;y,z -- rows and columns
    return max(R[w] + R[x], R[y] + R[z], R[w] + R[y], R[x] + R[z]) / PITCH


def load(d):
    p = os.path.join(HERE, d)
    f = os.path.join(p, "periodic_results.npz")
    if not os.path.exists(f):
        return None
    return np.load(f, allow_pickle=True), load_cst(p)


def spanel(ax, cases, title, legend_loc="lower right"):
    for d, label, c, _ in cases:
        got = load(d)
        if got is None:
            continue
        r, cst = got
        ax.plot(r["lam"], np.abs(r["S21"]), "o-", ms=3.4, lw=1.6, color=c,
                label=f"{label} — T-matrix" if len(cases) == 1 else label)
        if cst is not None:
            ax.plot(cst["lam"], np.abs(cst["S21"]), "-", lw=1.8,
                    color=c if len(cases) > 1 else "0.45",
                    alpha=0.35 if len(cases) > 1 else 0.9,
                    label="direct CST" if len(cases) == 1 else None)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("|S21|  (0th order)")
    ax.set_xlim(8.8, 30)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc=legend_loc)
    ax.set_title(title, fontsize=9)
    thz_axis(ax)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results_2x2_super_l3/fig4_comparison.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(2, 4, figsize=(21.5, 9.2), constrained_layout=True)
    spanel(ax[0, 0], SINGLE,
           "one atom per 8 um cell — markers: T-matrix, pale: CST")
    spanel(ax[0, 1], PAIRS, "two species per 16 um cell (x,y;y,x)")
    for col, case in zip((2, 3), QUADS):
        got = load(case[0])
        err = ""
        if got is not None and got[1] is not None:
            r, cst = got
            e = np.abs(r["S21"] - interp_c(r["lam"], cst["lam"],
                                           cst["S21"])).mean()
            err = f" — mean |ΔS21| {e:.3f}, rho {rho(case[3]):.3f}"
        spanel(ax[0, col], [case], f"four species: {case[1]}{err}")

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
    a.set_title("two-species cells are dark between the onsets; "
                "four-species are not", fontsize=9)
    thz_axis(a)

    # ---- birefringence, one panel per four-atom cell ----------------------
    jp = os.path.join(HERE, "results_2x2_ABCD_l3", "jones_xy.npz")
    j = np.load(jp) if os.path.exists(jp) else None
    for col, (key, label, c) in zip((1, 2), (("abcd", "a,b;c,d", "C3"),
                                             ("adbc", "a,d;b,c", "C4"))):
        a = ax[1, col]
        if j is not None:
            sep = np.abs(j[key][:, 0] - j[key][:, 1])
            a.fill_between(j["lam"], j[key][:, 0], j[key][:, 1], color=c,
                           alpha=0.18, lw=0)
            a.plot(j["lam"], j[key][:, 0], "o-", ms=3.2, lw=1.6, color=c,
                   label="|t_xx|")
            a.plot(j["lam"], j[key][:, 1], "s--", ms=3.2, lw=1.4, color="0.35",
                   label="|t_yy|")
            a.set_title(f"{label}: birefringence, max ||t_xx|−|t_yy|| = "
                        f"{sep.max():.3f}", fontsize=9)
        a.set_xlabel("Wavelength (um)")
        a.set_ylabel("|t| of the 0th transmitted order")
        a.set_xlim(8.8, 30)
        a.set_ylim(0, 1.02)
        a.grid(alpha=0.3)
        a.legend(frameon=False, fontsize=8, loc="lower right")
        thz_axis(a)

    # ---- accuracy vs the translation convergence ratio --------------------
    a = ax[1, 3]
    for cases, mk in ((SINGLE, "o"), (PAIRS, "s"), (QUADS, "D")):
        for d, label, c, spec in cases:
            got = load(d)
            if got is None or got[1] is None:
                continue
            r, cst = got
            err = np.abs(r["S21"] - interp_c(r["lam"], cst["lam"],
                                             cst["S21"])).mean()
            x = rho(spec)
            a.plot(x, err, mk, ms=9, color=c, mec="k", mew=0.5)
            right = x > 0.92           # keep labels clear of the right edge
            a.annotate(label.split()[0], (x, err), fontsize=8,
                       ha="right" if right else "left",
                       xytext=(-9, 3) if right else (6, -3),
                       textcoords="offset points")
    a.axvspan(0.93, 1.005, color="0.85", zorder=0)
    a.text(0.967, 0.021, "series barely\ncontracts", fontsize=8, ha="center")
    a.set_yscale("log")
    a.set_xlabel(r"$\rho = (a_i + a_j)\,/\,d$  over the 8 um neighbour pairs")
    a.set_ylabel("mean |ΔS21| vs direct CST")
    a.set_xlim(0.55, 1.005)
    a.grid(alpha=0.3, which="both")
    a.set_title("accuracy tracks the addition theorem's convergence rate\n"
                "(circles: 1 atom, squares: 2 species, diamonds: 4)",
                fontsize=9)

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
