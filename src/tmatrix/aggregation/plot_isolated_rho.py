"""The isolated-atom series: every meta-atom alone on its own 8 um lattice.

    python -m tmatrix.aggregation.plot_isolated_rho
    python -m tmatrix.aggregation.plot_isolated_rho --out-dir paper/figs

Two panels, written as separate files so a paper or deck can place them apart:

  s21_singles.png   |S21| of each atom's own 8 um lattice -- markers are the
                    T-matrix prediction, pale lines the direct CST run of the
                    same lattice (a row of the packed parametric sweep, so no
                    simulation was run for any of these).
  mse_vs_rho.png    MSE of the complex 0th-order S21 against the addition
                    theorem's convergence ratio rho = 2a/p, with the
                    log-linear fit and its correlation.

Why the isolated series is worth its own figure: it is the one place in the
study where rho varies with *everything else held fixed* -- same family, same
lattice, same band, one scatterer per cell, so no arrangement and no symmetry.
It is monotone over two decades of MSE.  The four-species cells are not
(`results_2x2_EBCA_l3/REPORT.md`), which is exactly why the two cases should
not be plotted on one axis and read as one trend.

Atoms E, F and G were extracted after the trend was first drawn on C, A, B, D,
so they are a genuine out-of-sample test of it, and they cost no full-wave time
-- runs 1, 8 and 7 of the packed sweep were already there.
"""
import argparse
import os

import numpy as np

from tmatrix.plotting import plt, thz_axis
from tmatrix.paths import AGG_DATA
from tmatrix.results_io import interp_c, load_cst_reference as load_cst

PITCH = 8.0

# atom -> (scale, circumscribing radius um, results dir, colour).  Ordered by
# size, which is also the order of rho.  A, B, C and D keep the colours they
# have everywhere else in the study so the figures stay comparable.
ATOMS = [
    ("E", 3.00, 2.15784, "results_E_ewald_l3", "#7f7f7f"),
    ("C", 3.25, 2.33766, "results_C_ewald_l3", "#2ca02c"),
    ("F", 3.50, 2.51748, "results_F_ewald_l3", "#17becf"),
    ("A", 4.00, 2.87712, "results_A_ewald_l3", "#1f77b4"),
    ("G", 4.50, 3.23676, "results_G_ewald_l3", "#9467bd"),
    ("B", 5.00, 3.59639, "results_B_ewald_l3", "#ff7f0e"),
    ("D", 5.50, 3.95603, "results_D_ewald_l3", "#d62728"),
]


def load(d):
    p = os.path.join(AGG_DATA, d)
    f = os.path.join(p, "periodic_results.npz")
    if not os.path.exists(f):
        return None
    return np.load(f, allow_pickle=True), load_cst(p)


def collect():
    """-> [(atom, scale, a, rho, colour, prediction, cst, mse)], skipping any
    atom that has not been run."""
    out = []
    for name, scale, a, d, colour in ATOMS:
        got = load(d)
        if got is None or got[1] is None:
            print(f"  skipping {name}: {d} not run, or no CST reference")
            continue
        m, cst = got
        dz = m["S21"] - interp_c(m["lam"], cst["lam"], cst["S21"])
        out.append(dict(name=name, scale=scale, a=a, rho=2 * a / PITCH,
                        colour=colour, m=m, cst=cst,
                        mse=float(np.mean(np.abs(dz) ** 2))))
    return out


def panel_spectra(rows, path):
    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    for r in rows:
        ax.plot(r["cst"]["lam"], np.abs(r["cst"]["S21"]), "-", lw=2.6,
                color=r["colour"], alpha=0.28, zorder=1)
        ax.plot(r["m"]["lam"], np.abs(r["m"]["S21"]), "o-", ms=4.0, lw=1.5,
                color=r["colour"], zorder=2,
                label=f"{r['name']}  (scale {r['scale']:.2f})")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("|S21|  (0th order)")
    ax.set_xlim(8.8, 30)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, framealpha=0.9, fontsize=9, loc="lower right")
    thz_axis(ax)
    fig.savefig(path, dpi=200)
    print(f"saved {path}")
    plt.close(fig)


def panel_mse(rows, path):
    rho = np.array([r["rho"] for r in rows])
    mse = np.array([r["mse"] for r in rows])
    y = np.log10(mse)
    corr = float(np.corrcoef(rho, y)[0, 1])
    slope, intercept = np.polyfit(rho, y, 1)

    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    xs = np.linspace(rho.min() - 0.04, rho.max() + 0.04, 100)
    ax.plot(xs, 10 ** (slope * xs + intercept), "-", color="0.45", lw=1.6,
            label=f"fit: r = {corr:+.3f}")
    for r in rows:
        ax.plot(r["rho"], r["mse"], "o", ms=13, color=r["colour"], mec="k",
                mew=0.7, zorder=3)
        ax.annotate(f"{r['name']}", (r["rho"], r["mse"]), fontsize=11,
                    fontweight="bold", color=r["colour"],
                    xytext=(0, 13), textcoords="offset points", ha="center")
        ax.annotate(f"a={r['a']:.2f} um", (r["rho"], r["mse"]), fontsize=8.5,
                    color=r["colour"],
                    xytext=(11, -4), textcoords="offset points", ha="left")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\rho = 2a/p$")
    ax.set_ylabel(r"MSE of complex $S_{21}$ vs direct CST")
    ax.set_xlim(rho.min() - 0.06, rho.max() + 0.09)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=True, fontsize=9, loc="lower right")
    ax.set_title(r"(b)  $\rho \rightarrow 1$  $\Rightarrow$  larger error"
                 f"   ({len(rows)} isolated atoms)", fontsize=11)
    fig.savefig(path, dpi=200)
    print(f"saved {path}")
    plt.close(fig)
    return corr, slope


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results_2x2_super_l3",
                    help="relative to aggregation/, or an absolute path")
    args = ap.parse_args(argv)
    out = (args.out_dir if os.path.isabs(args.out_dir)
           else os.path.join(AGG_DATA, args.out_dir))
    os.makedirs(out, exist_ok=True)

    rows = collect()
    if not rows:
        raise SystemExit("no isolated-atom cases have both a run and a CST "
                         "reference")
    panel_spectra(rows, os.path.join(out, "s21_singles.png"))
    corr, slope = panel_mse(rows, os.path.join(out, "mse_vs_rho.png"))

    print(f"\n{len(rows)} isolated atoms, rho {rows[0]['rho']:.3f} to "
          f"{rows[-1]['rho']:.3f}")
    print(f"{'atom':<5} {'scale':>6} {'a um':>7} {'rho':>6} {'MSE S21':>9}")
    for r in rows:
        print(f"{r['name']:<5} {r['scale']:6.2f} {r['a']:7.4f} {r['rho']:6.3f} "
              f"{r['mse']:9.5f}")
    print(f"\nlog10(MSE) against rho: r = {corr:+.3f}, slope {slope:+.2f} "
          f"decades per unit rho ({10 ** (slope * 0.1):.1f}x per 0.1 in rho)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
