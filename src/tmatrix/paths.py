"""Where the data lives.

Only the *code* moved under src/.  The data directories stayed exactly where
they were, because `retrieval/cst_runs/campaign_manifest.json` records the
absolute path of every project CST solved, and that record is the campaign's
provenance -- relocating the tree would invalidate it.  So the repository root
holds:

    src/tmatrix/...        all Python code
    tests/                 all test suites
    aggregation/           aggregation run outputs (results_*, cst_* runs)
    retrieval/             retrieval run outputs (results, cst_runs, assets)
    test/                  the tmat.h5 / .cst benchmark inputs

This module is the only place that knows those names.  Import the constants
from here rather than deriving anything from `__file__`, which after the move
points into src/ and no longer sits next to the data.
"""
import os
from pathlib import Path

_MARKER = "pyproject.toml"


def _find_root():
    env = os.environ.get("TMATRIX_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if not (root / _MARKER).is_file():
            raise RuntimeError(
                f"TMATRIX_ROOT={root} does not contain {_MARKER}")
        return root
    # src/tmatrix/paths.py -> src/tmatrix -> src -> repo root
    for cand in Path(__file__).resolve().parents[2:4]:
        if (cand / _MARKER).is_file():
            return cand
    raise RuntimeError(
        "cannot locate the repository root from "
        f"{Path(__file__).resolve()}.  This happens when tmatrix is installed "
        "as a copy rather than in editable mode; set TMATRIX_ROOT to the "
        "checkout that holds the data directories, or reinstall with "
        "`pip install -e .`")


REPO_ROOT = _find_root()

#: the package source tree, used for provenance hashing (opt_marginalized)
SRC = Path(__file__).resolve().parent

# ---- benchmark inputs (tracked; the extraction project's outputs) ----------
BENCHMARKS = REPO_ROOT / "test"
BENCHMARK_SINGLE = BENCHMARKS / "single"
BENCHMARK_2X2 = BENCHMARKS / "2x2"

#: the one-atom T-matrix the demos and most gates run on
DEMO_TMAT = BENCHMARK_SINGLE / "saw_gold_wl15p0025um.tmat.h5"

# ---- aggregation outputs ---------------------------------------------------
AGG_DATA = REPO_ROOT / "aggregation"
AGG_RESULTS = AGG_DATA / "results"
CST_DIRECT_DATA = AGG_DATA / "cst_direct"
CST_SUPERCELL_DATA = AGG_DATA / "cst_supercell"

# ---- retrieval outputs -----------------------------------------------------
RETRIEVAL_DATA = REPO_ROOT / "retrieval"
RETRIEVAL_RESULTS = RETRIEVAL_DATA / "results"
RETRIEVAL_ASSETS = RETRIEVAL_DATA / "assets"
CST_RUNS = RETRIEVAL_DATA / "cst_runs"

#: the fastfull subpackage's own published artifacts (m1_study.json, the
#: candidate registry, the gate-A study); the modules there sit one directory
#: deeper than the rest of retrieval, so they cannot spell this relatively
FASTFULL_RESULTS = RETRIEVAL_RESULTS / "fastfull"


def resolve(arg, base):
    """A command-line path argument, resolved against `base` when relative.

    Every runner accepts `--out results_2x2` and means "next to the other
    aggregation results", not "next to my current working directory".
    """
    p = Path(arg).expanduser()
    return p if p.is_absolute() else Path(base) / p


def agg_case(arg):
    """Resolve a run_case.py / run_supercell.py output directory argument."""
    return resolve(arg, AGG_DATA)


def retrieval_case(arg):
    """Resolve a retrieval output directory argument."""
    return resolve(arg, RETRIEVAL_DATA)
