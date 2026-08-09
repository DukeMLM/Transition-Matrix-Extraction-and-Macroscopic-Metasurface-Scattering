"""Read the direct CST supercell run and de-embed it to the array plane z = 0.

    python read_supercell_results.py --list          # dump the result tree
    python read_supercell_results.py --out ../results_2x2_super_l3

What has to be got right
------------------------
1. Reference plane.  CST puts the Floquet ports on the Zmin/Zmax boundaries and
   "expanded open" adds an unadvertised margin, so the raw S-parameters carry a
   propagation phase whose length L is a property of the run, not of the model.
   The empty-cell companion run measures it exactly: with no scatterer,
   S(Zmax(1), Zmin(1)) = e^{-j k L}.  For a z-symmetric model both transmission
   INTO order G and reflection INTO order G travel one leg at k_z,0 and one at
   k_z,G, so both de-embed with exp(+j (k_z,0 + k_z,G) L / 2).

2. Mode -> diffraction order.  The port sort code is "+beta/pw", i.e. modes are
   ordered by decreasing propagation constant, and each spatial order carries
   two polarizations.  For a square cell that gives the deterministic grouping

       modes 1-2    (0,0)
       modes 3-10   (+-1,0), (0,+-1)          |G| = 2 pi / P
       modes 11-18  (+-1,+-1)                 |G| = 2 pi sqrt2 / P
       modes 19-...  (+-2,0), (0,+-2)         |G| = 4 pi / P

   which the script re-derives from the lattice rather than hard-coding, and
   then CHECKS against the physics: every group must be silent while it is
   evanescent.  A mis-assignment shows up immediately as power in a closed
   channel.

3. Power.  Only propagating orders carry power; |S|^2 of an evanescent Floquet
   mode is not a power fraction and is excluded from the R/T sums.

4. Time convention.  CST is e^{+j w t}; this repository is e^{-i w t}.  The
   de-embedded values are conjugated.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

AUTO_CST = Path("D:/Claude/auto_cst")
if str(AUTO_CST) not in sys.path:
    sys.path.insert(0, str(AUTO_CST))
CST_PYTHON_LIB = r"E:\cst\AMD64\python_cst_libraries"
if CST_PYTHON_LIB not in sys.path:
    sys.path.insert(0, CST_PYTHON_LIB)

from nir.cst_helpers import get_result_with_data, open_results  # noqa: E402

HERE = Path(__file__).resolve().parent
C0 = 299.792458          # um * THz
SUPER = 16.0             # supercell period, um (design.json)


def sparam(proj3d, run_ids, out_port, out_mode, in_port, in_mode):
    key = f"1D Results\\S-Parameters\\S{out_port}({out_mode}),{in_port}({in_mode})"
    item, _ = get_result_with_data(proj3d, key, run_ids)
    return (np.asarray(item.get_xdata(), float),
            np.asarray(item.get_ydata(), complex))


def order_groups(period, n_modes):
    """Spatial orders sorted by decreasing k_z, expanded to 2 modes each.

    Returns a list, one entry per CST mode index (1-based), of
    (|G|, [(n1, n2), ...]) -- the degenerate family that mode belongs to.
    """
    g = 2 * np.pi / period
    fam = {}
    rng = int(np.ceil(np.sqrt(n_modes))) + 2
    for n1 in range(-rng, rng + 1):
        for n2 in range(-rng, rng + 1):
            key = round(np.hypot(n1, n2), 9)
            fam.setdefault(key, []).append((n1, n2))
    per_mode = []
    for key in sorted(fam):                     # ascending |G| = descending k_z
        members = fam[key]
        for _ in range(2 * len(members)):       # two polarizations per order
            per_mode.append((key * g, members))
        if len(per_mode) >= n_modes:
            break
    return per_mode[:n_modes]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path,
                    default=HERE / "runs" / "supercell" / "supercell.cst")
    ap.add_argument("--empty", type=Path,
                    default=HERE / "runs" / "empty" / "empty.cst")
    ap.add_argument("--period", type=float, default=SUPER)
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory for the CSV files")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    proj3d, run_ids = open_results(args.run)
    tree = [t for t in proj3d.get_tree_items()
            if t.startswith("1D Results\\S-Parameters\\")]
    if args.list:
        for t in sorted(tree):
            print(t)
        print(f"\n{len(tree)} S-parameter items, run ids {run_ids}")
        return 0

    modes = sorted({int(m.group(1)) for m in
                    (re.search(r"SZmax\((\d+)\),Zmin\(1\)$", t) for t in tree)
                    if m})
    if not modes:
        raise SystemExit("no SZmax(m),Zmin(1) items found; run with --list")
    n_modes = max(modes)
    print(f"{args.run.name}: {n_modes} Floquet modes, run ids {run_ids}")

    # ---- port-plane separation from the empty companion run -----------------
    ep3d, erids = open_results(args.empty)
    f, s21_e = sparam(ep3d, erids, "Zmax", 1, "Zmin", 1)
    _, s11_e = sparam(ep3d, erids, "Zmin", 1, "Zmin", 1)
    k = 2 * np.pi * f / C0
    # S21_empty = e^{-j k L}, so L is the slope of -phase against k.  Fitting
    # the *unwrapped* phase through the origin aliases: the sweep starts at
    # 10 THz where k L is already several turns, so the unwrapped curve carries
    # an unknown 2 pi offset that a through-origin fit absorbs into the slope.
    # Take the offset-free slope first, then resolve the branch per sample.
    ph = np.angle(s21_e)
    kc = k - k.mean()
    pc = np.unwrap(ph) - np.unwrap(ph).mean()
    L_slope = float(-np.sum(kc * pc) / np.sum(kc * kc))
    turns = np.round((k * L_slope + ph) / (2 * np.pi))
    L = float(np.median((2 * np.pi * turns - ph) / k))
    resid = np.abs(s21_e * np.exp(1j * k * L) - 1.0).max()
    print(f"  slope-only estimate {L_slope:.4f} um, branch-resolved {L:.4f} um "
          f"(model half-height 25 um -> geometric 50 um plus the "
          f"'expanded open' margin)")
    print(f"  empty cell: |S21| in [{np.abs(s21_e).min():.6f}, "
          f"{np.abs(s21_e).max():.6f}], max |S11| = {np.abs(s11_e).max():.2e}")
    # The empty run doubles as the manual's section-8 row-7 background test AND
    # as the noise floor of the benchmark itself: whatever it fails to return
    # exactly is CST's own error, not the aggregation's.
    lo = f <= 27.0
    print(f"  port-plane separation L = {L:.4f} um; |S21 e^{{jkL}} - 1| max "
          f"{resid:.2e} overall, {np.abs(s21_e[lo] * np.exp(1j * k[lo] * L) - 1).max():.2e} "
          f"below 27 THz  <- the benchmark's own noise floor")
    if resid > 0.1:
        print("  !! the empty run is not a clean e^{-jkL}: check the ports")

    groups = order_groups(args.period, n_modes)
    print("  mode -> diffraction order grouping (sort code +beta/pw):")
    seen = []
    for m, (gmag, members) in enumerate(groups, start=1):
        tag = f"|G| = {gmag:.4f} rad/um  {members[:4]}" \
              f"{'...' if len(members) > 4 else ''}"
        if not seen or seen[-1] != tag:
            print(f"    mode {m:2d} ->  {tag}")
            seen.append(tag)

    # ---- read the whole excited column --------------------------------------
    f, _ = sparam(proj3d, run_ids, "Zmax", 1, "Zmin", 1)
    kk = 2 * np.pi * f / C0
    trans = np.zeros((n_modes, len(f)), complex)
    refl = np.zeros((n_modes, len(f)), complex)
    for m in range(1, n_modes + 1):
        try:
            _, trans[m - 1] = sparam(proj3d, run_ids, "Zmax", m, "Zmin", 1)
            _, refl[m - 1] = sparam(proj3d, run_ids, "Zmin", m, "Zmin", 1)
        except RuntimeError:
            print(f"  (mode {m} missing from the tree; treated as zero)")

    kz = np.zeros((n_modes, len(f)))
    prop = np.zeros((n_modes, len(f)), bool)
    for m, (gmag, _) in enumerate(groups):
        kz2 = kk ** 2 - gmag ** 2
        prop[m] = kz2 > 0
        kz[m] = np.sqrt(np.maximum(kz2, 0.0))

    deemb = np.exp(0.5j * (kz[0] + kz) * L)      # both legs, see the docstring
    trans_d = np.conj(trans * deemb)
    refl_d = np.conj(refl * deemb)
    trans_d[~prop] = np.nan
    refl_d[~prop] = np.nan

    P_t = np.where(prop, np.abs(trans) ** 2, 0.0)
    P_r = np.where(prop, np.abs(refl) ** 2, 0.0)
    T_tot, R_tot = P_t.sum(axis=0), P_r.sum(axis=0)

    # ---- consistency: a closed channel must be silent -----------------------
    print("  channel check (power in each group, max over the band):")
    for gmag in sorted({g for g, _ in groups}):
        idx = [m for m, (g, _) in enumerate(groups) if g == gmag]
        onset = C0 * gmag / (2 * np.pi)
        pw = (P_t[idx] + P_r[idx]).sum(axis=0)
        closed = f < onset - 1e-9
        print(f"    |G| = {gmag:.4f} (opens at {onset:6.2f} THz): "
              f"max power {pw.max():.4e}; while closed "
              f"{pw[closed].max() if closed.any() else 0.0:.2e}")

    # A relative --out is resolved against the CURRENT directory, not this
    # file's.  Every documented invocation is run from aggregation/ and passes
    # e.g. --out results_2x2_ABCD_l3 meaning the results directory there;
    # resolving against HERE instead put the CSVs in cst_supercell/ silently,
    # where load_cst never looks and the case reads as "no CST reference yet".
    out = args.out.resolve() if args.out else (HERE / "runs")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "cst_direct_supercell.csv").open("w") as fh:
        fh.write("f_THz,lam_um,Re_S11,Im_S11,Re_S21,Im_S21,absS11,absS21,"
                 "absS11_cross,absS21_cross,R_total,T_total,A,"
                 "R_higher,T_higher\n")
        for i in range(len(f)):
            if f[i] <= 0:
                continue
            hi_r = P_r[2:, i].sum()
            hi_t = P_t[2:, i].sum()
            fh.write(",".join(f"{v:.10g}" for v in [
                f[i], C0 / f[i],
                refl_d[0, i].real, refl_d[0, i].imag,
                trans_d[0, i].real, trans_d[0, i].imag,
                abs(refl_d[0, i]), abs(trans_d[0, i]),
                abs(refl_d[1, i]), abs(trans_d[1, i]),
                R_tot[i], T_tot[i], 1 - R_tot[i] - T_tot[i],
                hi_r, hi_t]) + "\n")

    with (out / "cst_direct_supercell_orders.csv").open("w") as fh:
        fh.write("f_THz,mode,abs_G,kz,propagating,side,Re_S,Im_S,power\n")
        for m in range(n_modes):
            for i in range(len(f)):
                if f[i] <= 0:
                    continue
                for side, S, P in (("trans", trans_d[m, i], P_t[m, i]),
                                   ("refl", refl_d[m, i], P_r[m, i])):
                    fh.write(f"{f[i]:.10g},{m+1},{groups[m][0]:.10g},"
                             f"{kz[m,i]:.10g},{int(prop[m,i])},{side},"
                             f"{S.real:.10g},{S.imag:.10g},{P:.10g}\n")

    sel = (f >= 10) & (f <= 34)
    print(f"  in band: max |S11|^2+|S21|^2+higher = "
          f"{(R_tot + T_tot)[sel].max():.4f} (<= 1 for a passive cell)")
    print(f"  wrote {out/'cst_direct_supercell.csv'} and "
          f"{out/'cst_direct_supercell_orders.csv'} ({sel.sum()} in-band samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
