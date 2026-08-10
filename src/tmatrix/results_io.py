"""Readers for the files a run directory contains.

A run_case.py / run_supercell.py output directory holds up to four things:

  periodic_results.npz       this repo's reconstruction  (always)
  treams_reference.npz       independent Ewald cross-check
  cst_direct_reference.csv   direct CST run, one atom per cell
  cst_direct_supercell.csv   direct CST run, heterogeneous supercell

The two CSVs share a prefix but not a schema -- the supercell export adds the
cross-polarized magnitudes and the higher-order power, and renames R,T to
R_total,T_total.  They used to be parsed in three places by three different
rules, one of them by *column index*.  That reader was correct for the layout
it looked for, but it was one filename away from silently reading R,T,A out of
the supercell export's cross-polarization columns.  Everything goes through
load_cst_reference() now, which reads by column name and normalizes the two
schemas onto one dict.
"""
from pathlib import Path

import numpy as np

__all__ = ["load_periodic", "load_treams", "load_cst_reference",
           "interp_c"]

#: filenames tried by load_cst_reference, in order
CST_CSV_NAMES = ("cst_direct_supercell.csv", "cst_direct_reference.csv")


def load_periodic(run_dir):
    """The repo's own reconstruction.  Raises if the directory has no run."""
    return np.load(_path(run_dir, "periodic_results.npz"), allow_pickle=True)


def load_treams(run_dir):
    """The treams/Ewald cross-check, or None if it was never run."""
    p = _path(run_dir, "treams_reference.npz")
    return np.load(p, allow_pickle=True) if p.exists() else None


def load_cst_reference(run_dir, names=CST_CSV_NAMES):
    """The direct CST run in `run_dir`, or None if there is none.

    Returns a dict with keys lam, S11, S21 (complex), R, T, A, plus name, plus
    R_higher/T_higher and the cross-polarized magnitudes when the file is a
    supercell export.  Rows with f_THz <= 0 are dropped: CST writes a DC row
    that is not a measurement.
    """
    for name in names:
        p = _path(run_dir, name)
        if not p.exists():
            continue
        raw = np.genfromtxt(p, delimiter=",", names=True)
        cols = raw.dtype.names
        m = raw["f_THz"] > 0
        out = {
            "name": name,
            "lam": raw["lam_um"][m],
            "S11": raw["Re_S11"][m] + 1j * raw["Im_S11"][m],
            "S21": raw["Re_S21"][m] + 1j * raw["Im_S21"][m],
            # the supercell export renames R,T -> R_total,T_total
            "R": raw["R_total"][m] if "R_total" in cols else raw["R"][m],
            "T": raw["T_total"][m] if "T_total" in cols else raw["T"][m],
            "A": raw["A"][m],
        }
        for key in ("R_higher", "T_higher", "absS11_cross", "absS21_cross"):
            if key in cols:
                out[key] = raw[key][m]
        return out
    return None


def interp_c(lam, lam_src, y_src):
    """Interpolate a complex series onto `lam`, real and imaginary separately.

    Sorts the source first: the CST sweeps are stored in ascending frequency,
    i.e. descending wavelength, which np.interp rejects.
    """
    lam_src = np.asarray(lam_src, dtype=float)
    y_src = np.asarray(y_src)
    o = np.argsort(lam_src)
    return (np.interp(lam, lam_src[o], y_src[o].real)
            + 1j * np.interp(lam, lam_src[o], y_src[o].imag))


def _path(run_dir, name):
    return Path(run_dir) / name
