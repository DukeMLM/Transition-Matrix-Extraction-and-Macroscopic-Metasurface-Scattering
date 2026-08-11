"""The two competing predictors of aggregation accuracy, for every distinct
arrangement of four atoms on the 2x2 site square.

    python -m tmatrix.aggregation.arrangement_predictors
    python -m tmatrix.aggregation.arrangement_predictors --search

`OPEN_QUESTIONS.md` section 1 poses geometry against symmetry as explanations
of why `a,d;b,c` reconstructs far better than `a,b;c,d`.  Both were quoted there
as bare numbers; this script is where they come from, so the discriminating
experiments can be scored against something reproducible rather than against a
table someone typed.

**Geometry.**  The outgoing->regular translation of the addition theorem
truncates with an error falling like rho^lmax, rho = (a_i + a_j) / d.  Manual
Eq. (57) only asks for rho < 1; what sets the accuracy is how far below 1 the
*worst* pair sits.  Reported as `rho` over all pairs (the 8 um edge neighbours
always win -- the 11.31 um diagonals are far looser).

**Symmetry.**  If the arrangement is close to a mirror symmetry, equivalent
pairs recur and their coupling errors can partially cancel, which would make
symmetric cells *look* accurate without the solver being more correct.  Scored
as the mismatch of the nearest mirror: for each mirror of the site square, the
worst relative size difference over the atom pairs that mirror exchanges,
    mismatch = max_pairs |s_i - s_j| / max(s_i, s_j),
minimised over the mirrors.  0 means the arrangement is exactly mirror
symmetric.  Reported for the two diagonal mirrors (which exchange one pair and
fix the other two atoms) and, separately, over all four mirrors, because the
choice is a judgement call and the two orderings differ.

First experiment -- fix rho, change the symmetry class
-----------------------------------------------------
`a,c;d,b` has *identical* pair geometry to `a,b;c,d` -- same worst pair, same
rho, same gap -- but is in a different symmetry class.  So

    a,c;d,b reconstructs like a,b;c,d          -> geometry is the driver
    a,c;d,b reconstructs measurably better     -> rho is not the whole story

It landed with `a,b;c,d` (MSE 0.1207 against 0.1654, both ~10x worse than
`a,d;b,c`'s 0.0118), which settled section 1: at rho = 0.944 the symmetry class
does not rescue the answer.

Second experiment -- fix nothing, and take rho down while symmetry gets worse
----------------------------------------------------------------------------
That left the converse untested.  The one accurate cell, `a,d;b,c`, is *also*
the most nearly mirror-symmetric of the three (9 % against 27 % and 20 %), so in
the cell that works, "low rho" and "nearly symmetric" are still confounded.
Three more atoms extracted from the same sweep (E, F, G) open up 105 distinct
cells instead of 3, and `--search` shows that exactly one pair of them --
`e,b;c,a` and its twin `e,b;f,a` -- is simultaneously

  * lower in rho than `a,d;b,c` (0.809 against 0.854), and
  * further from a mirror than *either* failing cell under *both* metrics
    (25 % diagonal against 27 % / 20 %, and 20 % all-mirror against the 18.8 %
    the two failing cells share).

That 18.8 % matters: the all-mirror ordering is the variant section 1 could not
falsify, because under it the two rho = 0.944 cells tie.  `e,b;c,a` is the most
asymmetric cell in the study under exactly that metric, and it reuses three of
the four original atoms, so only one new T-matrix enters.  The two hypotheses
therefore made predictions a factor of ten apart:

    rho drives it       -> e,b;c,a scores at or below a,d;b,c (MSE <= 0.012)
    symmetry drives it  -> e,b;c,a scores with the failing cells (MSE ~ 0.12-0.17)

MEASURED: 0.0612 -- neither.  The cell with the LOWEST rho of the four is 5.2x
worse than the cell at rho = 0.854, so rho is not sufficient; and it is better
than `a,c;d,b`, which is *more* nearly symmetric than it is, so the mirror
mismatch does not order the four either.  Each predictor gets exactly one
inversion out of four cells.  What survives is the weaker statement that rho
bounds the error at fixed composition -- A, B, C, D still go
0.944 -> 0.1654 / 0.1206 and 0.854 -> 0.0118 -- while ordering cells against
each other needs something neither column here measures.  The full argument,
including what the prediction actually gets wrong (it puts the depth on the
wrong one of two collective modes), is in
aggregation/results_2x2_EBCA_l3/REPORT.md.
"""
import argparse
import itertools

import numpy as np

# scale and circumscribing radius, from
# tmatrix.aggregation.cst_supercell.build_2x2_supercell -- itself the
# `Result/db.parmap` sweep table of test/2x2/SAW_gold_noSub_packed.cst.  `dip`
# is the atom's own transmission minimum on its own 8 um lattice as measured by
# that run, which is what names its tmat.h5 file (saw_gold_wl<dip>um...).
SCALE = {"E": 3.00, "C": 3.25, "F": 3.50, "A": 4.00,
         "G": 4.50, "B": 5.00, "D": 5.50}
R = {"E": 2.15784, "C": 2.33766, "F": 2.51748, "A": 2.87712,
     "G": 3.23676, "B": 3.59639, "D": 3.95603}
DIP = {"E": 10.29, "C": 10.93, "F": 11.59, "A": 13.13,
       "G": 14.90, "B": 17.34, "D": 23.51}
PITCH = 8.0                       # atom-to-atom spacing, um
H = PITCH / 2

# The 24 assignments of four distinct atoms to the four sites collapse under
# the D4 symmetry of the site square to exactly three cells.  These are the
# ones that have been benchmarked against a direct CST supercell run.
CELLS = {"a,b;c,d": "ABCD", "a,d;b,c": "ADBC", "a,c;d,b": "ACDB",
         "e,b;c,a": "EBCA", "e,b;g,a": "EBGA", "e,a;f,c": "EAFC",
         "e,c;f,a": "ECFA"}
# MSE of the complex 0th-order S21 against the direct CST run, from
# tmatrix.aggregation.compare_cases.  None until the run exists.
MSE = {"ABCD": 0.1654, "ADBC": 0.0118, "ACDB": 0.1206, "EBCA": 0.0612,
       "EBGA": 0.0182, "EAFC": 0.0011, "ECFA": 0.0028}

# Mirrors of the site square, as maps on (x, y).  The two diagonal mirrors
# exchange one pair of sites and fix the other two; the two axis mirrors
# exchange two pairs.
MIRRORS = {
    "diag y=x": lambda x, y: (y, x),
    "diag y=-x": lambda x, y: (-y, -x),
    "axis x": lambda x, y: (x, -y),
    "axis y": lambda x, y: (-x, y),
}
DIAGONAL = ("diag y=x", "diag y=-x")


def sites(spec):
    """'wxyz' -> {atom: (x, y)}, read as the matrix the name suggests:

           w  x        w at (-4, +4)   x at (+4, +4)
           y  z        y at (-4, -4)   z at (+4, -4)
    """
    w, x, y, z = spec
    return {w: (-H, +H), x: (+H, +H), y: (-H, -H), z: (+H, -H)}


def worst_pair(spec):
    """(rho, 'i-j', surface gap um) for the pair with the largest rho.

    In a 2x2 supercell every 8 um neighbour relation is one of the four in-cell
    edges -- each atom's other two 8 um neighbours are the periodic images of
    the same two partners -- so ranging over the in-cell pairs is exhaustive.
    """
    pos = sites(spec)
    out = []
    for i, j in itertools.combinations(sorted(pos), 2):
        d = float(np.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]))
        out.append(((R[i] + R[j]) / d, f"{i}-{j}", d - R[i] - R[j]))
    return max(out)


def mirror_mismatch(spec, mirror):
    """Worst relative size difference over the pairs this mirror exchanges.

    Returns (mismatch, [exchanged pairs]).  A mirror that exchanges nothing --
    impossible here, the four atoms are distinct -- would score 0.
    """
    pos = sites(spec)
    at = {(round(p[0], 9), round(p[1], 9)): a for a, p in pos.items()}
    swaps, worst = [], 0.0
    for a, (x, y) in pos.items():
        b = at[tuple(round(v, 9) for v in mirror(x, y))]
        if a >= b:
            continue
        swaps.append(f"{a}-{b}")
        worst = max(worst, abs(SCALE[a] - SCALE[b]) / max(SCALE[a], SCALE[b]))
    return worst, swaps


def nearest_mirror(spec, names):
    """(mismatch, name, exchanged pairs) of the best-matching mirror in `names`."""
    cands = [(*mirror_mismatch(spec, MIRRORS[n]), n) for n in names]
    best = min(cands, key=lambda c: c[0])
    return best[0], best[2], best[1]


def diagonal_pairs(spec):
    pos = sites(spec)
    return [f"{i}-{j}" for i, j in itertools.combinations(sorted(pos), 2)
            if abs(pos[i][0] - pos[j][0]) > 1e-9
            and abs(pos[i][1] - pos[j][1]) > 1e-9]


def distinct_cells(quad):
    """One spec per distinct cell of `quad` under the D4 site symmetry.

    The 24 assignments collapse to three, indexed by which atom shares a
    diagonal with the first.
    """
    a, b, c, d = quad
    return [a + b + c + d, a + d + b + c, a + c + d + b]


def table(specs, labels=None):
    print(f"{'cell':<9} {'worst pair':>10} {'rho':>7} {'gap um':>8}   "
          f"{'diagonals':<14} {'diag mm':>8} {'all mm':>7}  {'MSE S21':>8}")
    for spec in specs:
        rho, pair, gap = worst_pair(spec)
        md = nearest_mirror(spec, DIAGONAL)[0]
        ma = nearest_mirror(spec, MIRRORS)[0]
        mse = MSE.get(spec)
        label = (labels or {}).get(spec, spec)
        print(f"{label:<9} {pair:>10} {rho:7.3f} {gap:8.3f}   "
              f"{','.join(diagonal_pairs(spec)):<14} {md * 100:7.1f}% "
              f"{ma * 100:6.1f}%  "
              f"{'-' if mse is None else format(mse, '8.4f'):>8}")


def search():
    """Every distinct cell of every four-atom subset, on the two predictors."""
    rows = []
    for quad in itertools.combinations(sorted(SCALE, key=SCALE.get), 4):
        for spec in distinct_cells(quad):
            rho = worst_pair(spec)[0]
            rows.append((spec, rho,
                         nearest_mirror(spec, DIAGONAL)[0],
                         nearest_mirror(spec, MIRRORS)[0]))
    print(f"\n{len(rows)} distinct cells from {len(SCALE)} atoms\n")

    ref_rho, ref_md, ref_ma = 0.854, 0.090, 0.090      # a,d;b,c, the cell to beat
    fail_ma = 0.188                                    # both rho = 0.944 cells
    good = [r for r in rows if r[1] < ref_rho - 1e-9
            and r[2] > ref_md + 1e-9 and r[3] >= fail_ma - 1e-9]
    good.sort(key=lambda r: (-r[3], -r[2], r[1]))
    print("cells that beat a,d;b,c on rho AND are further from a mirror than\n"
          "either failing cell under both metrics (rho < 0.854, diagonal\n"
          "mismatch > 9 %, all-mirror mismatch >= 18.8 %):")
    table([r[0] for r in good])
    if not good:
        print("  (none)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", action="store_true",
                    help="enumerate every distinct cell of the whole atom pool")
    args = ap.parse_args(argv)

    print("atom pool (radius sets rho; scale sets the symmetry mismatch; "
          "dip is the 8 um-lattice resonance that names the tmat.h5)\n")
    print(f"{'atom':<5} {'scale':>6} {'r um':>8} {'2r/8':>7} {'dip um':>8}")
    for a in sorted(SCALE, key=SCALE.get):
        print(f"{a:<5} {SCALE[a]:6.2f} {R[a]:8.4f} {2 * R[a] / PITCH:7.3f} "
              f"{DIP[a]:8.2f}")

    print("\nThe benchmarked cells\n")
    table(list(CELLS.values()), {v: k for k, v in CELLS.items()})

    print("\nFirst experiment -- a,b;c,d against a,c;d,b at fixed geometry\n")
    ref, test = CELLS["a,b;c,d"], CELLS["a,c;d,b"]
    r_ref, p_ref, g_ref = worst_pair(ref)
    r_test, p_test, g_test = worst_pair(test)
    same = (p_ref == p_test and abs(r_ref - r_test) < 1e-12
            and abs(g_ref - g_test) < 1e-12)
    print(f"  a,b;c,d and a,c;d,b share their pair geometry: {same}"
          f"  (both {p_ref}, rho {r_ref:.3f}, gap {g_ref:.3f} um)")
    print(f"  ... and differ in symmetry class: "
          f"{nearest_mirror(ref, DIAGONAL)[0] * 100:.0f} % vs "
          f"{nearest_mirror(test, DIAGONAL)[0] * 100:.0f} % (diagonal), "
          f"{nearest_mirror(ref, MIRRORS)[0] * 100:.0f} % vs "
          f"{nearest_mirror(test, MIRRORS)[0] * 100:.0f} % (all)")
    print("  MEASURED: 0.1207 against 0.1654 -- it landed with a,b;c,d, so at "
          "rho = 0.944\n            the symmetry class is not what decides.")

    print("\nSecond experiment -- e,b;c,a takes rho down while symmetry gets "
          "worse\n")
    new = CELLS["e,b;c,a"]
    good = CELLS["a,d;b,c"]
    print(f"  rho          {worst_pair(new)[0]:.3f}  against "
          f"{worst_pair(good)[0]:.3f} for a,d;b,c   (lower -> predict better)")
    fail = [CELLS["a,b;c,d"], CELLS["a,c;d,b"]]
    print(f"  diagonal mm  {nearest_mirror(new, DIAGONAL)[0] * 100:.1f} %  "
          f"against {nearest_mirror(good, DIAGONAL)[0] * 100:.1f} % for "
          f"a,d;b,c and "
          f"{' / '.join(f'{nearest_mirror(c, DIAGONAL)[0] * 100:.1f}' for c in fail)}"
          f" % for the failing cells")
    print(f"  all-mirror   {nearest_mirror(new, MIRRORS)[0] * 100:.1f} %  "
          f"against {nearest_mirror(good, MIRRORS)[0] * 100:.1f} % for "
          f"a,d;b,c and "
          f"{' / '.join(f'{nearest_mirror(c, MIRRORS)[0] * 100:.1f}' for c in fail)}"
          f" % for BOTH failing cells")
    print("\n  The two hypotheses made incompatible predictions:\n"
          "    geometry  -> e,b;c,a scores at or below a,d;b,c (MSE <= 0.012)\n"
          "    symmetry  -> e,b;c,a scores with the failing cells (MSE ~ 0.12-0.17)")
    print(f"  MEASURED: {MSE[new]:.4f} -- neither.  The lowest-rho cell of the "
          f"four is {MSE[new] / MSE[good]:.1f}x worse than\n"
          f"            a,d;b,c, so rho is not sufficient; and it beats "
          f"a,c;d,b, which is more nearly\n"
          f"            symmetric, so the mirror mismatch does not order them "
          f"either.")

    if args.search:
        search()


if __name__ == "__main__":
    main()
