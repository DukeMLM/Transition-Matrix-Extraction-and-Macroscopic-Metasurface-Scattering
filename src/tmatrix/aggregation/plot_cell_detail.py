"""One cell in four panels: layout, its atoms alone, and Im/Re of S21.

    python -m tmatrix.aggregation.plot_cell_detail EAFC
    python -m tmatrix.aggregation.plot_cell_detail EAFC ECFA --out-dir paper/figs

  a  the 2x2 layout, to scale.  Ring and spokes are drawn edgeless in one
     colour so each atom reads as the single united solid it is in the CST
     model (`Solid.Add` unites them there too), not as a disc with four bricks
     laid on top.  Geometry from
     `cst_supercell.build_2x2_supercell: ATOMS`, so it is to scale.

  b  each of the four atoms ALONE on its own 8 um lattice, same colours --
     the reference the cell has to be judged against, because a supercell
     prediction cannot be better than the T-matrices going into it.

  c  Im S21   of the 0th diffraction order
  d  |S21|    of the 0th diffraction order

Panels c and d carry three curves: the direct CST run, the prediction at the
25 stored frequencies, and the prediction on the --refine 4 grid.  The refined
grid is a re-SOLVE at every new frequency -- the Ewald sums, the block solve
and the Floquet projection are all recomputed; only the input T is
interpolated between its two bracketing stored samples, hence "complex
interpolation" (see run_supercell.refine_grid).  `PART` also has a "Re" mode:
Im with Re is the pair the reported MSE actually scores, since that MSE is on
the complex amplitude, while |S21| can let a magnitude and a phase error
cancel.

Not drawn, but worth knowing when reading the refined line: each tmat.h5 is
stitched from two CST band runs, and interpolating T across the join mixes two
independent extractions.  `SEAM` holds the interval for each atom -- 18-19 THz
for A (15.78-16.66 um), 20-21 THz for every other atom (14.28-14.99 um).
"""
import argparse
import os

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Wedge

from tmatrix.plotting import plt, thz_axis
from tmatrix.aggregation.arrangement_predictors import (  # noqa: F401
    DIAGONAL, DIP, MIRRORS, R, SCALE, nearest_mirror, sites, worst_pair)
from tmatrix.aggregation.cst_supercell.build_2x2_supercell import ATOMS
from tmatrix.paths import AGG_DATA
from tmatrix.results_io import interp_c, load_cst_reference as load_cst

PITCH, CELL = 8.0, 16.0
COLOR = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728",
         "E": "#7f7f7f", "F": "#17becf", "G": "#9467bd"}
SEAM = {a: (20.0, 21.0) for a in COLOR}
SEAM["A"] = (18.0, 19.0)                 # read off computation.bands in the h5
C0 = 299.792458
GRID = "0.85"                            # solid, not black-at-alpha-0.3


def pale(colour, frac):
    """`colour` blended `frac` of the way from white, returned opaque.

    Postscript has no alpha channel: matplotlib writes a semi-transparent line
    into EPS at full opacity, so a pale reference curve comes out solid and
    hides the markers it is meant to sit behind.  Pre-blending against the
    white page gives the same pixels in PNG and survives the EPS round trip.
    """
    r, g, b = mcolors.to_rgb(colour)
    return (1 - frac + frac * r, 1 - frac + frac * g, 1 - frac + frac * b)


def draw_atom(ax, cx, cy, key):
    """Ring plus four inward spokes as ONE united solid: no internal edges."""
    p, c = ATOMS[key], COLOR[key]
    ax.add_patch(Wedge((cx, cy), p["r"], 0, 360, width=p["w_ring"],
                       facecolor=c, edgecolor="none", zorder=3))
    tip, root, hw = p["gap"] / 2, p["r"] - p["w_ring"] / 2, p["w"] / 2
    L = root - tip
    for x0, y0, dx, dy in ((cx + tip, cy - hw, L, p["w"]),
                           (cx - root, cy - hw, L, p["w"]),
                           (cx - hw, cy + tip, p["w"], L),
                           (cx - hw, cy - root, p["w"], L)):
        ax.add_patch(Rectangle((x0, y0), dx, dy, facecolor=c,
                               edgecolor="none", zorder=3))


def local_labels(spec):
    """Relabel this cell's four atoms a, b, c, d by ASCENDING resonance
    frequency -- i.e. descending dip wavelength, so `a` is the largest atom.

    Local to one figure.  Everywhere else in the study an atom's letter is its
    row of the parametric sweep (A = scale 4.00 and so on) and is the same in
    every cell; here it is a within-cell rank, so the two do not agree and the
    mapping is printed on the figure.
    """
    order = sorted(set(spec), key=lambda a: -DIP[a])
    return {atom: "abcd"[i] for i, atom in enumerate(order)}


def local_name(spec, loc):
    """'ECFA' -> 'd,c;b,a' under the local relabelling."""
    w, x, y, z = spec
    return f"{loc[w]},{loc[x]};{loc[y]},{loc[z]}"


def panel_layout(ax, spec, loc):
    """(a) the pure 2x2 layout -- no axes, no annotation."""
    pos = sites(spec)
    ax.add_patch(Rectangle((-CELL / 2, -CELL / 2), CELL, CELL, fill=False,
                           edgecolor="0.55", lw=1.3, ls="--", zorder=1))
    for atom, (cx, cy) in pos.items():
        draw_atom(ax, cx, cy, atom)
        ax.text(cx, cy, loc[atom], fontsize=17, fontweight="bold",
                ha="center", va="center", color="k", zorder=5)
    ax.set_title("(a)  2x2 layout", fontsize=11)
    ax.set_xlim(-CELL / 2 - 0.4, CELL / 2 + 0.4)
    ax.set_ylim(-CELL / 2 - 0.4, CELL / 2 + 0.4)
    ax.set_aspect("equal")
    ax.set_axis_off()


def panel_atoms(ax, spec, loc):
    """(b) each atom of this cell alone on its own 8 um lattice."""
    for atom in sorted(set(spec), key=lambda a: -DIP[a]):
        d = os.path.join(AGG_DATA, f"results_{atom}_ewald_l3")
        f = os.path.join(d, "periodic_results.npz")
        if not os.path.exists(f):
            continue
        m = np.load(f, allow_pickle=True)
        cst = load_cst(d)
        if cst is not None:
            ax.plot(cst["lam"], np.abs(cst["S21"]), "-", lw=2.2,
                    color=pale(COLOR[atom], 0.30), zorder=1)
        ax.plot(m["lam"], np.abs(m["S21"]), "o-", ms=3.6, lw=1.4,
                color=COLOR[atom], zorder=2,
                label=f"{loc[atom]}   dip {DIP[atom]:.2f} um,  "
                      f"r = {R[atom]:.2f} um,  rho = {2 * R[atom] / PITCH:.3f}")
    ax.set_title("(b)  isolated atoms", fontsize=11)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("|S21|  (0th order)")
    ax.set_xlim(8.8, 30)
    ax.set_ylim(0, 1.05)
    ax.grid(color=GRID, lw=0.6)
    key = ",  ".join(f"{loc[a]} = {a}" for a in
                     sorted(set(spec), key=lambda a: -DIP[a]))
    lg = ax.legend(frameon=True, framealpha=1.0, fontsize=8.5,
                   loc="lower right",
                   title=f"labelled by rising resonance frequency\n"
                         f"study labels:  {key}")
    lg.get_title().set_fontsize(8)
    thz_axis(ax)


PART = {"arg": (np.angle, r"arg $S_{21}$ (rad)", "arg S21 (rad)"),
        "abs": (np.abs, r"$|S_{21}|$", "|S21|"),
        "Im": (np.imag, r"Im $S_{21}$", "Im S21"),
        "Re": (np.real, r"Re $S_{21}$", "Re S21")}


def panel_part(ax, spec, colour, part, tag):
    """(c)/(d) one component of the complex 0th-order S21."""
    take, ylabel, name = PART[part]
    p = os.path.join(AGG_DATA, f"results_2x2_{spec}_l3")
    m = np.load(os.path.join(p, "periodic_results.npz"), allow_pickle=True)
    fine = os.path.join(AGG_DATA, f"results_2x2_{spec}_fine",
                        "periodic_results.npz")
    mf = np.load(fine, allow_pickle=True) if os.path.exists(fine) else None
    cst = load_cst(p)

    if cst is not None:
        ax.plot(cst["lam"], take(cst["S21"]), "-", lw=2.6, color="0.4",
                zorder=2, label="direct CST")
    if mf is not None:
        o = np.argsort(mf["lam"])
        ax.plot(mf["lam"][o], take(mf["S21"][o]), "-", lw=1.2, color=colour,
                zorder=3, label="Complex Interpolation")
    ax.plot(m["lam"], take(m["S21"]), "o", ms=6, color=colour, mec="k",
            mew=0.5, zorder=4, label="Prediction")

    if part not in ("abs",):
        ax.axhline(0.0, color="0.7", lw=0.8, zorder=1)
    ax.set_title(f"({tag})  {name}", fontsize=11)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(8.8, 30)
    if part == "abs":
        ax.set_ylim(0, 1.05)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=True, framealpha=1.0, fontsize=9, loc="lower right")
    thz_axis(ax)


CELLS = {"ABCD": ("a,b;c,d", "#d62728"), "ACDB": ("a,c;d,b", "#8c564b"),
         "ADBC": ("a,d;b,c", "#9467bd"), "EBCA": ("e,b;c,a", "#e377c2"),
         "EBGA": ("e,b;g,a", "#1f77b4"), "EAFC": ("e,a;f,c", "#2ca02c"),
         "ECFA": ("e,c;f,a", "#17becf"), "CAGF": ("c,a;g,f", "#ff7f0e")}


def figure(spec):
    colour = CELLS[spec][1]
    loc = local_labels(spec)
    fig, ax = plt.subplots(2, 2, figsize=(14.0, 10.4),
                           constrained_layout=True)
    panel_layout(ax[0, 0], spec, loc)
    panel_atoms(ax[0, 1], spec, loc)
    panel_part(ax[1, 0], spec, colour, "arg", "c")
    panel_part(ax[1, 1], spec, colour, "abs", "d")
    fig.suptitle(local_name(spec, loc), fontsize=14)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="*", help="e.g. EAFC ECFA")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", default="results_2x2_super_l3")
    ap.add_argument("--format", default="png",
                    choices=("png", "eps", "pdf", "svg"),
                    help="eps/pdf/svg are vector and stay editable; the figure "
                         "carries no alpha so eps round-trips faithfully")
    args = ap.parse_args(argv)
    # Type 42 embeds the TrueType outlines and keeps the text as TEXT, so
    # Illustrator can retype a label.  The PS/PDF default is Type 3, which most
    # editors import as uneditable glyph paths.
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    specs = list(CELLS) if args.all else [s.upper() for s in args.specs]
    if not specs:
        ap.error("give one or more cell specs, or --all")
    out = (args.out_dir if os.path.isabs(args.out_dir)
           else os.path.join(AGG_DATA, args.out_dir))
    os.makedirs(out, exist_ok=True)

    for spec in specs:
        if not os.path.exists(os.path.join(AGG_DATA, f"results_2x2_{spec}_l3",
                                           "periodic_results.npz")):
            print(f"  skipping {spec}: not run")
            continue
        path = os.path.join(out, f"cell_{spec.lower()}.{args.format}")
        figure(spec).savefig(path, dpi=170)
        plt.close("all")
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
