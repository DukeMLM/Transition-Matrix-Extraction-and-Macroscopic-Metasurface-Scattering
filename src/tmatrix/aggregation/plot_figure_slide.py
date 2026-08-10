"""Presentation figure: single atoms, all three four-species cells, and their layouts.

    python -m tmatrix.aggregation.plot_figure_slide \
        [--out results_2x2_super_l3/fig5_slide.png]

Eight panels, top row spectra and bottom row the geometry that explains them:

  a  the four meta-atoms, each alone on its own 8 um lattice
  b  a,b;c,d  — T-matrix prediction against its own direct CST run
  c  a,d;b,c  — the same four atoms rearranged
  d  a,c;d,b  — the third and last distinct arrangement

Panels b-d are scored by the MSE of the complex 0th-order S21,
mean(|S21_predicted - S21_CST|^2) over the 25 stored frequencies -- on the
complex amplitude rather than its magnitude, so a phase error counts.
  e  the four atoms drawn to scale, so the size series in (a) is visible
  f  the a,b;c,d cell, to scale, with the tightest neighbour gap marked
  g  the a,d;b,c cell, likewise
  h  the a,c;d,b cell, likewise

The 24 assignments of four distinct atoms to the four sites collapse under the
D4 symmetry of the site square to exactly the three cells in b-d, so this is the
complete set.  Panels f and h are the reason the third one was run: a,c;d,b
reproduces a,b;c,d's tightest gap and rho exactly while placing a different pair
on the diagonal, which separates the geometry explanation of the b-versus-c
difference from the symmetry one (`OPEN_QUESTIONS.md` section 1).

The layouts are drawn from the same ring/spoke parameters the CST models are
built from (`tmatrix.aggregation.cst_supercell.build_2x2_supercell: ATOMS`),
not traced from screenshots, so they are to scale and stay correct if the
design changes.  The gap annotation in (f)-(h) is the quantity under test: the
addition theorem's truncation error falls like rho^lmax with rho = (a_i + a_j)/d,
so the tightest pair sets the accuracy -- if that is the whole story.
"""
import argparse
import os

import numpy as np
from matplotlib.patches import Rectangle, Wedge

from tmatrix.plotting import plt, thz_axis
from tmatrix.results_io import interp_c, load_cst_reference as load_cst

from tmatrix.paths import AGG_DATA


# ring/spoke geometry, um -- the same numbers the CST models are built from
ATOMS = {
    "A": dict(scale=4.00, r=2.87712, w_ring=0.644552, gap=2.800, w=0.80),
    "B": dict(scale=5.00, r=3.59639, w_ring=0.805690, gap=3.500, w=1.00),
    "C": dict(scale=3.25, r=2.33766, w_ring=0.523698, gap=2.275, w=0.65),
    "D": dict(scale=5.50, r=3.95603, w_ring=0.886259, gap=3.850, w=1.10),
}
COLOR = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}
PITCH, CELL = 8.0, 16.0

SINGLE = [("results_C_ewald_l3", "C", 3.25), ("results_A_ewald_l3", "A", 4.00),
          ("results_B_ewald_l3", "B", 5.00), ("results_D_ewald_l3", "D", 5.50)]
# site order: top-left, top-right, bottom-left, bottom-right
QUADS = [("results_2x2_ABCD_l3", "a,b;c,d", "ABCD", "#d62728"),
         ("results_2x2_ADBC_l3", "a,d;b,c", "ADBC", "#9467bd"),
         ("results_2x2_ACDB_l3", "a,c;d,b", "ACDB", "#8c564b")]


def draw_atom(ax, cx, cy, key, alpha=1.0, edge=True):
    """Ring plus four inward spokes, to scale, centred at (cx, cy)."""
    p = ATOMS[key]
    c = COLOR[key]
    ax.add_patch(Wedge((cx, cy), p["r"], 0, 360, width=p["w_ring"],
                       facecolor=c, edgecolor="k" if edge else "none",
                       lw=0.4, alpha=alpha, zorder=3))
    tip, root, hw = p["gap"] / 2, p["r"] - p["w_ring"] / 2, p["w"] / 2
    L = root - tip
    for x0, y0, dx, dy in ((cx + tip, cy - hw, L, p["w"]),
                           (cx - root, cy - hw, L, p["w"]),
                           (cx - hw, cy + tip, p["w"], L),
                           (cx - hw, cy - root, p["w"], L)):
        ax.add_patch(Rectangle((x0, y0), dx, dy, facecolor=c,
                               edgecolor="k" if edge else "none", lw=0.4,
                               alpha=alpha, zorder=3))


def label(ax, tag, text=""):
    ax.text(-0.10, 1.06, tag, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="left")
    if text:
        ax.set_title(text, fontsize=10.5, pad=8)


def spectra_single(ax):
    for d, key, scale in SINGLE:
        f = os.path.join(AGG_DATA, d, "periodic_results.npz")
        if not os.path.exists(f):
            continue
        r = np.load(f, allow_pickle=True)
        cst = load_cst(os.path.join(AGG_DATA, d))
        ax.plot(r["lam"], np.abs(r["S21"]), "o-", ms=3.4, lw=1.7,
                color=COLOR[key], label=f"{key}  (scale {scale:.2f})")
        if cst is not None:
            ax.plot(cst["lam"], np.abs(cst["S21"]), "-", lw=1.6,
                    color=COLOR[key], alpha=0.35)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    label(ax, "a", "one atom per 8 µm cell\nmarkers: T-matrix   pale: direct CST")


def spectra_quad(ax, case, tag):
    d, name, spec, col = case
    f = os.path.join(AGG_DATA, d, "periodic_results.npz")
    r = np.load(f, allow_pickle=True)
    cst = load_cst(os.path.join(AGG_DATA, d))
    ax.plot(r["lam"], np.abs(r["S21"]), "o-", ms=3.6, lw=1.8, color=col,
            label="T-matrix prediction", zorder=3)
    sub = ""
    if cst is not None:
        ax.plot(cst["lam"], np.abs(cst["S21"]), "-", lw=1.9, color="0.35",
                label="direct CST", zorder=2)
        dz = r["S21"] - interp_c(r["lam"], cst["lam"], cst["S21"])
        mse = float(np.mean(np.abs(dz) ** 2))
        sub = f"\nMSE = {mse:.4f}    (RMSE {np.sqrt(mse):.3f})"
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    label(ax, tag, f"2×2 — {name}{sub}")


def layout(ax, spec, tag, name):
    """The four sites to scale, with the tightest neighbour gap marked."""
    h = PITCH / 2
    sites = {"TL": (-h, +h), "TR": (+h, +h), "BL": (-h, -h), "BR": (+h, -h)}
    order = dict(zip(("TL", "TR", "BL", "BR"), spec))
    ax.add_patch(Rectangle((-CELL / 2, -CELL / 2), CELL, CELL, fill=False,
                           edgecolor="0.4", lw=1.2, ls="--", zorder=1))
    for site, key in order.items():
        cx, cy = sites[site]
        draw_atom(ax, cx, cy, key)
        ax.text(cx, cy, key.lower(), fontsize=13, fontweight="bold",
                ha="center", va="center", color="k", zorder=4)

    # the tightest of the four 8 um neighbour pairs sets the accuracy.  The
    # arrow spans ring edge to ring edge, in data coordinates, so it shows the
    # actual gap rather than a centre-to-centre line trimmed by eye.
    edges = [("TL", "TR"), ("BL", "BR"), ("TL", "BL"), ("TR", "BR")]
    g, u, v = min((PITCH - ATOMS[order[a]]["r"] - ATOMS[order[b]]["r"], a, b)
                  for a, b in edges)
    (x1, y1), (x2, y2) = sites[u], sites[v]
    r1, r2 = ATOMS[order[u]]["r"], ATOMS[order[v]]["r"]
    horizontal = y1 == y2
    # the gap can be a few hundred nm, far too narrow to letter inside, so the
    # label sits 1.6 um to the side of the arrow with a leader back to it
    if horizontal:
        (xa, xb) = (x1 + r1, x2 - r2) if x1 < x2 else (x1 - r1, x2 + r2)
        p0, p1 = (xa, y1), (xb, y2)
        mid = ((xa + xb) / 2, y1)
        tpos = (mid[0], y1 + 1.6)
    else:
        (ya, yb) = (y1 + r1, y2 - r2) if y1 < y2 else (y1 - r1, y2 + r2)
        p0, p1 = (x1, ya), (x2, yb)
        mid = (x1, (ya + yb) / 2)
        tpos = (x1 + 1.6, mid[1])
    ax.annotate("", xy=p1, xytext=p0, zorder=5,
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.5,
                                shrinkA=0, shrinkB=0))
    ax.annotate(f"{g:.2f} µm", xy=mid, xytext=tpos, fontsize=10.5,
                fontweight="bold", ha="center", va="center", zorder=6,
                bbox=dict(fc="w", ec="0.5", lw=0.6, alpha=0.95,
                          boxstyle="round,pad=0.25"),
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8))
    rho = (PITCH - g) / PITCH
    ax.set_xlim(-CELL / 2 - 0.6, CELL / 2 + 0.6)
    ax.set_ylim(-CELL / 2 - 0.6, CELL / 2 + 0.6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    label(ax, tag, f"layout of {name}   (16 µm cell)\n"
                   f"tightest gap {g:.2f} µm  →  ρ = {rho:.3f}")


def atom_row(ax):
    """The four atoms to scale, in size order."""
    keys = ["C", "A", "B", "D"]
    x = 0.0
    for k in keys:
        r = ATOMS[k]["r"]
        x += r + 0.5
        draw_atom(ax, x, 0, k)
        ax.text(x, -ATOMS["D"]["r"] - 1.5,
                f"{k}\nscale {ATOMS[k]['scale']:.2f}\nr = {r:.2f} µm",
                fontsize=9, ha="center", va="top")
        x += r + 0.5
    ax.set_xlim(-0.5, x)
    ax.set_ylim(-ATOMS["D"]["r"] - 5.2, ATOMS["D"]["r"] + 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    label(ax, "e", "the four meta-atoms, to scale")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results_2x2_super_l3/fig5_slide.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(2, 4, figsize=(21.5, 9.6),
                           gridspec_kw=dict(height_ratios=[1.15, 1.0]),
                           constrained_layout=True)

    spectra_single(ax[0, 0])
    for col, (case, tag) in enumerate(zip(QUADS, "bcd"), start=1):
        spectra_quad(ax[0, col], case, tag)
    for a in ax[0]:
        a.set_xlabel("Wavelength (µm)")
        a.set_ylabel("|S21|  (0th order)")
        a.set_xlim(8.8, 30)
        a.set_ylim(0, 1.02)
        a.grid(alpha=0.3)
        thz_axis(a)

    atom_row(ax[1, 0])
    for col, (case, tag) in enumerate(zip(QUADS, "fgh"), start=1):
        layout(ax[1, col], case[2], tag, case[1])

    out = args.out if os.path.isabs(args.out) else os.path.join(AGG_DATA, args.out)
    fig.savefig(out, dpi=200, facecolor="w")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
