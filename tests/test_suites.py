"""pytest front end for the standalone validation suites.

Each file under tests/aggregation is a self-contained
suite: it prints its own PASS/FAIL table and exits non-zero if anything
failed.  That contract predates pytest here and is what the reports in
README.md and experiment.md quote, so pytest runs each suite as a subprocess
and asserts its exit status rather than collecting its functions.

(Collecting them directly would not work anyway: many `test_*` functions take
the loaded T-matrix, the quadrature or the symmetry basis as ordinary
arguments, which pytest would try, and fail, to resolve as fixtures.)

    pytest                    every suite that has its inputs
    pytest -m "not slow"      skip the search-heavy ones (~10 min saved)
    pytest -k vswf            one suite
    pytest -s                 show each suite's own output

A suite whose input data is not in the checkout is skipped, not failed.
"""
import subprocess
import sys

import pytest

from tmatrix.paths import BENCHMARK_2X2, DEMO_TMAT, REPO_ROOT

TESTS = REPO_ROOT / "tests"

#: (path, marks, required input paths, extra argv)
SUITES = [
    ("aggregation/test_vswf.py", (), (), ()),
    ("aggregation/test_translate.py", ("slow",), (), ()),
    ("aggregation/test_mirror.py", (), (DEMO_TMAT,), ()),
    ("aggregation/test_supercell.py", ("slow",), (BENCHMARK_2X2,), ()),
    ("aggregation/test_feature_fidelity.py", ("treams",), (DEMO_TMAT,), ()),
]


def _param(entry):
    rel, marks = entry[0], entry[1]
    return pytest.param(entry, id=rel,
                        marks=[getattr(pytest.mark, m) for m in marks])


@pytest.mark.parametrize("suite", [_param(s) for s in SUITES])
def test_suite(suite):
    rel, _, needs, argv = suite
    for path in needs:
        if not path.exists():
            pytest.skip(f"input not in this checkout: {path}")
    proc = subprocess.run([sys.executable, str(TESTS / rel), *argv],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    if proc.returncode != 0:
        # the suite's own table is the useful diagnosis, so surface it whole
        pytest.fail(f"{rel} exited {proc.returncode}\n\n"
                    f"--- stdout ---\n{proc.stdout[-8000:]}\n"
                    f"--- stderr ---\n{proc.stderr[-4000:]}",
                    pytrace=False)
    print(proc.stdout)
