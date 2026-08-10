"""The whole study in one figure: 4 atoms, 6 mixed cells, 10 CST benchmarks.

    python plot_comparison.py [--out results_2x2_super_l3/fig4_comparison.png]

Top row     single atoms, the two-species checkerboards, then ONE PANEL PER
            four-distinct-atom cell -- all three distinct arrangements of the
            same four atoms, which carry the most structure, and overlaying
            them hides which pale CST curve belongs to which.
            Markers are the T-matrix prediction, pale lines the direct CST run
            of the same structure.
Bottom row  power into higher diffraction orders against the two Rayleigh
            onsets (the two-species cells are dark between them by symmetry,
            the four-atom cells are not); the birefringence of each four-atom
            cell, again one panel each; and the accuracy summary -- the MSE of
            complex S21 against the addition theorem's convergence ratio
            rho = (a_i + a_j)/d, maximised over the 8 um neighbour pairs.

MSE is mean(|S21_predicted - S21_CST|^2) over the 25 stored frequencies, on the
complex amplitude rather than its magnitude, so a phase error counts.

Cases whose directory or CST reference is missing are skipped, so this runs
while benchmarks are still solving.
"""
import argparse
import os

import numpy as np

from tmatrix.plotting import plt, thz_axis
from tmatrix.results_io import interp_c, load_cst_reference as load_cst

from tmatrix.paths import AGG_DATA

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
         ("results_2x2_ADBC_l3", "a,d;b,c", "C4", "ADBC"),
         ("results_2x2_ACDB_l3", "a,c;d,b", "C5", "ACDB")]


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
    p = os.path.join(AGG_DATA, d)
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

    fig, ax = plt.subplots(2, 5, figsize=(26.5, 9.2), constrained_layout=True)
    spanel(ax[0, 0], SINGLE,
           "one atom per 8 um cell — markers: T-matrix, pale: CST")
    spanel(ax[0, 1], PAIRS, "two species per 16 um cell (x,y;y,x)")
    for col, case in zip((2, 3, 4), QUADS):
        got = load(case[0])
        err = ""
        if got is not None and got[1] is not None:
            r, cst = got
            dz = r["S21"] - interp_c(r["lam"], cst["lam"], cst["S21"])
            err = (f" — MSE {np.mean(np.abs(dz) ** 2):.4f}, "
                   f"rho {rho(case[3]):.3f}")
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
    jp = os.path.join(AGG_DATA, "results_2x2_ABCD_l3", "jones_xy.npz")
    j = np.load(jp) if os.path.exists(jp) else None
    for col, (key, label, c) in zip((1, 2, 3), (("abcd", "a,b;c,d", "C3"),
                                                ("adbc", "a,d;b,c", "C4"),
                                                ("acdb", "a,c;d,b", "C5"))):
        a = ax[1, col]
        if j is not None and key in j.files:         # jones_xy.py may predate a cell
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
    a = ax[1, 4]
    seen = {}                          # rho -> how many points already at it
    for cases, mk in ((SINGLE, "o"), (PAIRS, "s"), (QUADS, "D")):
        for d, label, c, spec in cases:
            got = load(d)
            if got is None or got[1] is None:
                continue
            r, cst = got
            dz = r["S21"] - interp_c(r["lam"], cst["lam"], cst["S21"])
            err = float(np.mean(np.abs(dz) ** 2))
            x = rho(spec)
            a.plot(x, err, mk, ms=9, color=c, mec="k", mew=0.5)
            # a,b;c,d and a,c;d,b sit at exactly the same rho -- that is the
            # point of the third cell, so their labels must not collide.
            n = seen.get(round(x, 6), 0)
            seen[round(x, 6)] = n + 1
            right = x > 0.92           # keep labels clear of the right edge
            dy = 3 if n % 2 == 0 else -11
            a.annotate(label.split()[0], (x, err), fontsize=8,
                       ha="right" if right else "left",
                       xytext=(-9, dy) if right else (6, -3),
                       textcoords="offset points")
    a.axvspan(0.93, 1.005, color="0.85", zorder=0)
    a.text(0.967, 0.021, "series barely\ncontracts", fontsize=8, ha="center")
    a.set_yscale("log")
    a.set_xlabel(r"$\rho = (a_i + a_j)\,/\,d$  over the 8 um neighbour pairs")
    a.set_ylabel(r"MSE of complex $S_{21}$ vs direct CST")
    a.set_xlim(0.55, 1.005)
    a.grid(alpha=0.3, which="both")
    a.set_title("accuracy tracks the addition theorem's convergence rate\n"
                "(circles: 1 atom, squares: 2 species, diamonds: 4)",
                fontsize=9)

    out = args.out if os.path.isabs(args.out) else os.path.join(AGG_DATA, args.out)
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
