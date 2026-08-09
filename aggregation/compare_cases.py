"""One table across several run_supercell.py output directories.

    python compare_cases.py results_A_ewald_l3 results_B_ewald_l3 ...
    python compare_cases.py --all

For each case it reports the agreement with the independent treams
implementation (which measures the aggregation) and with the direct CST run
(which measures the input T-matrices), so the two are never confused, plus the
observables that distinguish one cell from another: where the transmission dips
sit, how much power leaves in higher orders, and whether the answer is passive.

The CST columns are the MSE of the complex S-parameter, mean(|S_pred - S_CST|^2)
over the 25 stored frequencies -- on the complex amplitude, not its magnitude,
so a phase error counts.

Directories without a CST reference are reported with the CST columns blank.
"""
import argparse
import os

import numpy as np

from plot_supercell import load_cst, interp_c, parabola_min

HERE = os.path.dirname(os.path.abspath(__file__))
C0 = 299.792458
ALL = ["results_A_ewald_l3", "results_B_ewald_l3", "results_C_ewald_l3",
       "results_D_ewald_l3", "results_2x2_super_l3", "results_2x2_AC_l3",
       "results_2x2_BC_l3", "results_2x2_ABCD_l3", "results_2x2_ADBC_l3"]
LABEL = {"results_A_ewald_l3": "A alone", "results_B_ewald_l3": "B alone",
         "results_C_ewald_l3": "C alone", "results_D_ewald_l3": "D alone",
         "results_2x2_super_l3": "a,b;b,a", "results_2x2_AC_l3": "a,c;c,a",
         "results_2x2_BC_l3": "b,c;c,b", "results_2x2_ABCD_l3": "a,b;c,d",
         "results_2x2_ADBC_l3": "a,d;b,c"}


def dips(lam, s21, n=3, sep=1.5):
    """Up to n local minima of |S21|, deepest first, at least `sep` um apart."""
    y = np.abs(s21)
    loc = [i for i in range(1, len(y) - 1) if y[i] <= y[i - 1] and y[i] <= y[i + 1]]
    loc.sort(key=lambda i: y[i])
    out = []
    for i in loc:
        if all(abs(lam[i] - l) > sep for l, _ in out):
            out.append((parabola_min(lam[i - 1:i + 2], y[i - 1:i + 2]), y[i]))
        if len(out) == n:
            break
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    dirs = ALL if (args.all or not args.dirs) else args.dirs

    print(f"{'case':<10} {'vs treams':>10} {'vs CST S21':>18} "
          f"{'vs CST S11':>18} {'A range':>16} {'diffr':>7}  dips |S21| (um)")
    print(f"{'':<10} {'max |dS|':>10} {'MSE / max |d|':>18} "
          f"{'MSE / max |d|':>18}")
    for d in dirs:
        p = d if os.path.isabs(d) else os.path.join(HERE, d)
        f = os.path.join(p, "periodic_results.npz")
        if not os.path.exists(f):
            print(f"{LABEL.get(d, os.path.basename(d)):<10}  (not run yet)")
            continue
        m = np.load(f, allow_pickle=True)
        lam = m["lam"]
        tp = os.path.join(p, "treams_reference.npz")
        tre = "-"
        if os.path.exists(tp):
            t = np.load(tp, allow_pickle=True)
            tre = f"{max(np.abs(m['S21'] - t['S21']).max(), np.abs(m['S11'] - t['S11']).max()):.1e}"
        cst = load_cst(p)
        if cst is None:
            c21 = c11 = "        -         "
        else:
            d21 = np.abs(m["S21"] - interp_c(lam, cst["lam"], cst["S21"]))
            d11 = np.abs(m["S11"] - interp_c(lam, cst["lam"], cst["S11"]))
            c21 = f"{np.mean(d21 ** 2):.5f} / {d21.max():.3f}".rjust(18)
            c11 = f"{np.mean(d11 ** 2):.5f} / {d11.max():.3f}".rjust(18)
        arange = f"[{m['A'].min():+.3f}, {m['A'].max():+.3f}]".rjust(16)
        diffr = f"{(m['R_hi'] + m['T_hi']).max():.3f}".rjust(7)
        dd = ", ".join(f"{l:.2f} ({v:.3f})" for l, v in dips(lam, m["S21"]))
        print(f"{LABEL.get(d, os.path.basename(d)):<10} {tre:>10} {c21} {c11} "
              f"{arange} {diffr}  {dd}")


if __name__ == "__main__":
    main()
