r"""Merge per-band .tmat.h5 extractions of ONE scatterer into a single file.

Thin command-line wrapper around cst_tmatrix.storage.merge_tmatrix_files,
which verifies identical lmax (same mode basis) and non-overlapping
frequency ranges, concatenates T and the diagnostics along the frequency
axis, records per-band provenance, and round-trip-verifies the band
boundaries.  The config-driven runner (cst_tmatrix.runner) and the batch
script call the same function; this wrapper exists for manual merges.

Usage:
    python merge_band_tmatrices.py out.tmat.h5 in_band1.tmat.h5 in_band2.tmat.h5 [...]
Defaults (no args): merges the 13.1 um spoke-wheel bands into
    library/saw_gold_wl13p10um_10to34THz.tmat.h5
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst_tmatrix.storage import merge_tmatrix_files  # noqa: E402

LIB = Path(__file__).resolve().parent.parent / "library"
DEFAULT_OUT = LIB / "saw_gold_wl13p10um_10to34THz.tmat.h5"
DEFAULT_IN = [LIB / "saw_gold_wl13p10um__10to18THz.tmat.h5",
              LIB / "saw_gold_wl13p10um__18to34THz.tmat.h5"]


def main(argv):
    if len(argv) >= 3:
        out, inputs = Path(argv[1]), [Path(p) for p in argv[2:]]
    else:
        out, inputs = DEFAULT_OUT, DEFAULT_IN
    try:
        info = merge_tmatrix_files(out, inputs)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"merged {info['n_bands']} bands -> {out}")
    print(f"  {info['n_freq']} frequencies "
          f"{info['f_min_hz'] / 1e12:g}-{info['f_max_hz'] / 1e12:g} THz")
    print("round-trip verification: OK (band boundaries intact)")


if __name__ == "__main__":
    main(sys.argv)
