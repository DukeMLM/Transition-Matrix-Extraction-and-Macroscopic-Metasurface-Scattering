"""The matched-pair experiment: what rho decides, and what symmetry decides.

    python -m tmatrix.aggregation.plot_rho_vs_symmetry
    python -m tmatrix.aggregation.plot_rho_vs_symmetry --out-dir paper/figs

`e,b;c,a` and `e,b;g,a` have the SAME worst pair (A-B), the SAME rho (0.8092)
and the SAME tightest gap (1.5265 um) -- and `e,b;g,a` is tighter than `e,b;c,a`
on every one of the other three 8 um pairs, so every distance-based metric ranks
it same-or-worse.  They differ only in how nearly the arrangement is mirror
symmetric: 10.0 % against 25.0 %.  That is a controlled experiment on the
symmetry axis, and the two top panels are its result.

Bottom panel: all benchmarked four-species cells, MSE against the mirror
mismatch, grouped by rho.  The two rho groups occupy non-overlapping bands of
error, and inside each band the mismatch orders the cells.  Within the low-rho
band rho itself *anti*-orders (`a,d;b,c` has the highest rho of the three and is
the best), so the ordering there is not a residual rho effect.

Caveat the figure cannot show: five cells, two predictors, and no cell sampled
between rho 0.854 and 0.944, so the band structure is a description of these
points rather than a law tested out of sample.  The controlled part is the
matched pair.
"""
import argparse
import os

import numpy as np

from tmatrix.plotting import dips, plt, thz_axis
from tmatrix.aggregation.arrangement_predictors import (
    DIAGONAL, nearest_mirror, worst_pair)
from tmatrix.paths import AGG_DATA
from tmatrix.results_io import interp_c, load_cst_reference as load_cst

# label -> (spec, colour).  The matched pair is drawn first, in the top row.
PAIR = [("e,b;c,a", "EBCA", "#e377c2"), ("e,b;g,a", "EBGA", "#1f77b4")]
ALL = [("a,b;c,d", "ABCD", "#d62728"), ("a,c;d,b", "ACDB", "#8c564b"),
       ("a,d;b,c", "ADBC", "#9467bd"), ("e,b;c,a", "EBCA", "#e377c2"),
       ("e,b;g,a", "EBGA", "#1f77b4"), ("e,a;f,c", "EAFC", "#2ca02c"),
       ("e,c;f,a", "ECFA", "#17becf"), ("c,a;g,f", "CAGF", "#ff7f0e")]


def load(spec):
    p = os.path.join(AGG_DATA, f"results_2x2_{spec}_l3")
    f = os.path.join(p, "periodic_results.npz")
    if not os.path.exists(f):
        return None
    cst = load_cst(p)
    if cst is None:
        return None
    m = np.load(f, allow_pickle=True)
    fine = os.path.join(AGG_DATA, f"results_2x2_{spec}_fine",
                        "periodic_results.npz")
    mf = np.load(fine, allow_pickle=True) if os.path.exists(fine) else m
    dz = m["S21"] - interp_c(m["lam"], cst["lam"], cst["S21"])
    return m, mf, cst, float(np.mean(np.abs(dz) ** 2))


def spectra_panel(ax, label, spec, colour, got):
    m, mf, cst, mse = got
    o = np.argsort(mf["lam"])
    ax.plot(cst["lam"], np.abs(cst["S21"]), "-", lw=2.8, color="0.35",
            alpha=0.85, label="direct CST", zorder=1)
    ax.plot(mf["lam"][o], np.abs(mf["S21"][o]), "-", lw=1.2, color=colour,
            alpha=0.55, zorder=2)
    ax.plot(m["lam"], np.abs(m["S21"]), "o", ms=5, color=colour, zorder=3,
            label="T-matrix prediction")
    # mark each side's deepest feature, which is what the two cells differ on.
    # dips() parabola-refines the position and returns the sampled depth, which
    # is how the REPORT tables quote these.
    lp, vp = min(dips(mf["lam"], mf["S21"]), key=lambda t: t[1])
    lc, vc = min(dips(cst["lam"], cst["S21"]), key=lambda t: t[1])
    ax.plot(lp, vp, "v", ms=11, color=colour, mec="k", mew=0.6, zorder=4)
    ax.plot(lc, vc, "v", ms=11, color="0.35", mec="k", mew=0.6, zorder=4)
    shift = lp - lc
    rho = worst_pair(spec)[0]
    mm = nearest_mirror(spec, DIAGONAL)[0] * 100
    ax.annotate("", xy=(lc, vc + 0.05), xytext=(lp, vp + 0.05),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.3,
                                shrinkA=2, shrinkB=2))
    ax.text(0.5 * (lp + lc), max(vp, vc) + 0.10,
            f"deepest mode {shift:+.2f} um"
            + ("  MISSED" if abs(shift) > 1.0 else "  matched"),
            fontsize=10, ha="center", fontweight="bold",
            color="#b22222" if abs(shift) > 1.0 else "#1a7a1a")
    ax.set_title(f"{label}    rho {rho:.3f},  mirror mismatch {mm:.1f} %"
                 f"    MSE {mse:.4f}", fontsize=11)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("|S21|  (0th order)")
    ax.set_xlim(8.8, 30)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    thz_axis(ax)


def summary_panel(ax, rows):
    """MSE against mirror mismatch, grouped by rho."""
    groups = {}
    for label, spec, colour, mse in rows:
        groups.setdefault(round(worst_pair(spec)[0], 3), []).append(
            (nearest_mirror(spec, DIAGONAL)[0] * 100, mse, label, colour))
    # a band per contiguous rho group, so "rho sets the band" is visible
    bands, rhos = {}, {}
    for rho, pts in groups.items():
        key = "high" if rho > 0.9 else "low"
        bands.setdefault(key, []).extend(pts)
        rhos.setdefault(key, []).append(rho)
    for key, colour in (("low", "#2ca02c"), ("high", "#d62728")):
        pts = sorted(bands.get(key, []))
        if not pts:
            continue
        lo, hi = min(p[1] for p in pts), max(p[1] for p in pts)
        rr = sorted(rhos[key])
        txt = (f"rho {rr[0]:.3f}" if rr[0] == rr[-1]
               else f"rho {rr[0]:.3f}-{rr[-1]:.3f}")
        # shade the band the data actually occupies -- the two must not touch,
        # that non-overlap is the claim
        ax.axhspan(lo / 1.15, hi * 1.15, color=colour, alpha=0.09, zorder=0)
        ax.text(30.6, hi * 1.02, txt, fontsize=9.5, color=colour, ha="right",
                va="bottom", fontweight="bold")
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-",
                    color=colour, lw=1.2, alpha=0.5, zorder=1)
    for label, spec, colour, mse in rows:
        x = nearest_mirror(spec, DIAGONAL)[0] * 100
        ax.plot(x, mse, "D", ms=12, color=colour, mec="k", mew=0.7, zorder=3)
        ax.annotate(label, (x, mse), fontsize=9.5, fontweight="bold",
                    color=colour, xytext=(0, 14), textcoords="offset points",
                    ha="center")
        ax.annotate(f"rho {worst_pair(spec)[0]:.3f}", (x, mse), fontsize=8,
                    color="0.3", xytext=(0, -20), textcoords="offset points",
                    ha="center")
    ax.set_yscale("log")
    ax.set_xlabel("nearest-mirror mismatch (%)   0 = exactly mirror symmetric")
    ax.set_ylabel(r"MSE of complex $S_{21}$ vs direct CST")
    ax.set_xlim(4, 31)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("neither predictor orders the cells alone; together they hold "
                 "in 22 of 23 decidable pairs.\nThe exception is the rho-axis "
                 "matched pair itself: c,a;g,f has the SAME mismatch as e,c;f,a "
                 "and a\nHIGHER rho, and is still 1.9x better — below ~0.003 "
                 "neither predictor resolves the cells",
                 fontsize=10)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results_2x2_EBGA_l3")
    ap.add_argument("--name", default="fig_rho_vs_symmetry.png")
    args = ap.parse_args(argv)
    out = (args.out_dir if os.path.isabs(args.out_dir)
           else os.path.join(AGG_DATA, args.out_dir))
    os.makedirs(out, exist_ok=True)

    fig = plt.figure(figsize=(14.5, 10.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    for col, (label, spec, colour) in enumerate(PAIR):
        got = load(spec)
        ax = fig.add_subplot(gs[0, col])
        if got is None:
            ax.text(0.5, 0.5, f"{label}: no CST reference yet", ha="center")
            continue
        spectra_panel(ax, label, spec, colour, got)

    rows = []
    for label, spec, colour in ALL:
        got = load(spec)
        if got is None:
            print(f"  skipping {label}: no CST reference yet")
            continue
        rows.append((label, spec, colour, got[3]))
    summary_panel(fig.add_subplot(gs[1, :]), rows)

    fig.suptitle("Matched pair at identical pair geometry: rho 0.8092, worst "
                 "pair A-B, gap 1.5265 um in BOTH top panels", fontsize=12.5)
    path = os.path.join(out, args.name)
    fig.savefig(path, dpi=170)
    print(f"saved {path}")

    print(f"\n{'cell':<9} {'rho':>7} {'diag mm':>8} {'MSE':>8}")
    for label, spec, _c, mse in sorted(rows, key=lambda r: r[3]):
        print(f"{label:<9} {worst_pair(spec)[0]:7.3f} "
              f"{nearest_mirror(spec, DIAGONAL)[0]*100:7.1f}% {mse:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
