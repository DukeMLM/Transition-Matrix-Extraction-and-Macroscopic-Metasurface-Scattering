"""Normal-incidence Jones diagonal of each four-distinct-atom cell.

    python jones_xy.py [--out results_2x2_ABCD_l3/jones_xy.npz]

`run_supercell.py` solves one incident polarization per run, so the cell's
0th-order Jones matrix needs two sweeps:

    --pol 1 0  ->  S21   = t_xx,   S21x = t_yx
    --pol 0 1  ->  S21   = t_yy,   S21x = t_xy

This drives both for every cell in CELLS and stores |t_xx|, |t_yy|, |t_xy| on
the shared wavelength grid, which is what the birefringence panels of
`plot_comparison.py` and `plot_figure_slide.py` read.  The x-polarized sweep is
reused from the case's own results directory when it is already there, so only
the y-polarized half is actually recomputed.

Cross-polarization is ~1e-12 for every one of these cells even though four
distinct atoms leave the cell no point-group symmetry to forbid it -- see
`OPEN_QUESTIONS.md` section 2, which this file is the data for.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TMAT = {"A": "saw_gold_wl13p10um_10to34THz.tmat.h5",
        "B": "saw_gold_wl17p30um_10to34THz.tmat.h5",
        "C": "saw_gold_wl10p90um_10to34THz.tmat.h5",
        "D": "saw_gold_wl23p50um_10to34THz.tmat.h5"}
BANK = os.path.join(HERE, "..", "test", "2x2")
H = 4.0

# key in the npz -> (spec read as w,x;y,z, results dir of the x-polarized run)
CELLS = {"abcd": ("ABCD", "results_2x2_ABCD_l3"),
         "adbc": ("ADBC", "results_2x2_ADBC_l3"),
         "acdb": ("ACDB", "results_2x2_ACDB_l3")}
LMAX = "3"


def site_args(spec):
    w, x, y, z = spec
    pos = [(w, -H, +H), (x, +H, +H), (y, -H, -H), (z, +H, -H)]
    out = []
    for atom, px, py in pos:
        out += ["--site", os.path.join(BANK, TMAT[atom]), str(px), str(py)]
    return out


def read(path, keys=("lam", "S21", "S21x", "e_inc")):
    """Load just the fields we need and CLOSE the file.

    np.load holds the .npz open until the NpzFile is closed, and on Windows an
    open handle blocks the temporary directory from being removed -- which
    otherwise kills the run at cleanup, after all the solving is done.
    """
    with np.load(path, allow_pickle=True) as f:
        return {k: np.array(f[k]) for k in keys}


def sweep(spec, pol, out_dir):
    """Run run_supercell.py for one polarization; -> its results as arrays."""
    cmd = ([sys.executable, os.path.join(HERE, "run_supercell.py"),
            "--cell", "16", "--lmax", LMAX,
            "--pol", str(pol[0]), str(pol[1]), "--out", out_dir]
           + site_args(spec))
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"run_supercell failed for {spec} pol {pol}:\n"
                           f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return read(os.path.join(out_dir, "periodic_results.npz"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results_2x2_ABCD_l3/jones_xy.npz")
    args = ap.parse_args()

    data, lam = {}, None
    with tempfile.TemporaryDirectory(prefix="jones_",
                                     ignore_cleanup_errors=True) as tmp:
        for key, (spec, xdir) in CELLS.items():
            xp = os.path.join(HERE, xdir, "periodic_results.npz")
            if os.path.exists(xp):
                mx = read(xp)
                if not np.allclose(mx["e_inc"], [1, 0, 0]):
                    raise ValueError(f"{xdir} is not the x-polarized run")
                print(f"{key}: reusing {xdir} for x")
            else:
                print(f"{key}: solving x ...")
                mx = sweep(spec, (1, 0), os.path.join(tmp, key + "_x"))
            print(f"{key}: solving y ...")
            my = sweep(spec, (0, 1), os.path.join(tmp, key + "_y"))
            if not np.allclose(mx["lam"], my["lam"]):
                raise ValueError("the two polarizations disagree on the grid")
            if lam is None:
                lam = mx["lam"]
            elif not np.allclose(lam, mx["lam"]):
                raise ValueError("cells disagree on the wavelength grid")
            data[key] = np.column_stack([np.abs(mx["S21"]),      # t_xx
                                         np.abs(my["S21"]),      # t_yy
                                         np.abs(my["S21x"])])    # t_xy
            sep = np.abs(data[key][:, 0] - data[key][:, 1])
            print(f"{key}: max ||t_xx|-|t_yy|| = {sep.max():.3f} "
                  f"(mean {sep.mean():.3f}), max |t_xy| = "
                  f"{data[key][:, 2].max():.1e}")

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    np.savez(out, lam=lam,
             note="columns: |t_xx|, |t_yy|, |t_xy| of the 0th transmitted "
                  "order at normal incidence", **data)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
