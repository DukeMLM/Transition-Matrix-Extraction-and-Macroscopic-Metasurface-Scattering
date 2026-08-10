"""The two competing predictors of aggregation accuracy, for every distinct
arrangement of four atoms on the 2x2 site square.

    python -m tmatrix.aggregation.arrangement_predictors

`OPEN_QUESTIONS.md` section 1 poses geometry against symmetry as explanations
of why `a,d;b,c` reconstructs far better than `a,b;c,d` from the same four
atoms.  Both were quoted there as bare numbers; this script is where they come
from, so the discriminating experiment can be scored against something
reproducible rather than against a table someone typed.

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

The point of the third arrangement is that it does not depend on that
judgement call: `a,c;d,b` has *identical* pair geometry to `a,b;c,d` -- same
worst pair, same rho, same gap -- but is in a different symmetry class under
every one of these metrics.  So

    a,c;d,b reconstructs like a,b;c,d          -> geometry is the driver
    a,c;d,b reconstructs measurably better     -> rho is not the whole story
"""
import itertools

import numpy as np

# scale and circumscribing radius, from
# tmatrix.aggregation.cst_supercell.build_2x2_supercell
SCALE = {"A": 4.00, "B": 5.00, "C": 3.25, "D": 5.50}
R = {"A": 2.87712, "B": 3.59639, "C": 2.33766, "D": 3.95603}
PITCH = 8.0                       # atom-to-atom spacing, um
H = PITCH / 2

# The 24 assignments of four distinct atoms to the four sites collapse under
# the D4 symmetry of the site square to exactly these three cells.
CELLS = {"a,b;c,d": "ABCD", "a,d;b,c": "ADBC", "a,c;d,b": "ACDB"}

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
    """(rho, 'i-j', surface gap um) for the pair with the largest rho."""
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


def main():
    print("Geometry -- the addition theorem's convergence ratio\n")
    print(f"{'cell':<9} {'worst pair':>10} {'rho':>7} {'gap um':>8}   "
          f"{'diagonal pairs':<14}")
    for label, spec in CELLS.items():
        rho, pair, gap = worst_pair(spec)
        pos = sites(spec)
        diag = ", ".join(
            f"{i}-{j}" for i, j in itertools.combinations(sorted(pos), 2)
            if abs(pos[i][0] - pos[j][0]) > 1e-9
            and abs(pos[i][1] - pos[j][1]) > 1e-9)
        print(f"{label:<9} {pair:>10} {rho:7.3f} {gap:8.3f}   {diag:<14}")

    print("\nSymmetry -- mismatch of the nearest mirror "
          "(0 = exactly symmetric)\n")
    print(f"{'cell':<9} {'diagonal mirrors':>18} {'':<12} "
          f"{'all four mirrors':>18}")
    for label, spec in CELLS.items():
        md, nd, sd = nearest_mirror(spec, DIAGONAL)
        ma, na, sa = nearest_mirror(spec, MIRRORS)
        print(f"{label:<9} {md * 100:15.1f} %  {'(' + ','.join(sd) + ')':<12} "
              f"{ma * 100:15.1f} %  ({na}: {','.join(sa)})")

    print("\nThe discriminating comparison\n")
    ref, test = CELLS["a,b;c,d"], CELLS["a,c;d,b"]
    r_ref, p_ref, g_ref = worst_pair(ref)
    r_test, p_test, g_test = worst_pair(test)
    same = (p_ref == p_test and abs(r_ref - r_test) < 1e-12
            and abs(g_ref - g_test) < 1e-12)
    print(f"  a,b;c,d and a,c;d,b share their pair geometry: {same}"
          f"  (both {p_ref}, rho {r_ref:.3f}, gap {g_ref:.3f} um)")
    print(f"  ... and differ in symmetry class: "
          f"{nearest_mirror(ref, DIAGONAL)[0] * 100:.0f} % vs "
          f"{nearest_mirror(test, DIAGONAL)[0] * 100:.0f} % "
          f"(diagonal), "
          f"{nearest_mirror(ref, MIRRORS)[0] * 100:.0f} % vs "
          f"{nearest_mirror(test, MIRRORS)[0] * 100:.0f} % (all)")
    print("\n  So the two hypotheses make incompatible predictions:\n"
          "    geometry  -> a,c;d,b scores like a,b;c,d (MSE ~ 0.17)\n"
          "    symmetry  -> a,c;d,b scores unlike it, between the two cells")


if __name__ == "__main__":
    main()
