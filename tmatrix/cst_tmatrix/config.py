"""Configuration: directories, machine settings, extraction settings.

MACHINE SETTINGS
================
Settings that belong to the machine rather than to a physics problem are
read from an optional JSON file ``settings.json`` at the repository root
(next to the ``cst_tmatrix`` package).  Recognized keys, all optional:

    {
      "run_root": "D:/cst_runs",       working area for CST projects and
                                       solver output; must be a LOCAL
                                       disk (cloud-synchronized folders
                                       lock files during solves)
      "library": "",                   destination of the final .tmat.h5
                                       files; default: library/ in this
                                       repository
      "max_cpus": 6,                   solver threads on this machine
      "hardware_acceleration": false   GPU use on this machine
    }

The CPU count and acceleration setting differ between machines, which is
why they live here and not in the per-extraction configuration; a value
given in an extraction configuration overrides the machine setting for
that run.  The environment variables CST_TMATRIX_RUN_ROOT and
CST_TMATRIX_LIBRARY take precedence over settings.json where both are
given.

The package runs under whatever Python interpreter executes it; the
interpreter must provide numpy, scipy, h5py, and the CST Studio Suite
Python interface (see README).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SETTINGS_FILE = _REPO_ROOT / "settings.json"


def machine_settings() -> dict:
    """The contents of settings.json (empty dict if absent or invalid;
    an invalid file is reported once rather than raising, so that a
    malformed settings file does not make the package unimportable)."""
    if not _SETTINGS_FILE.exists():
        return {}
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as e:
        print(f"warning: {_SETTINGS_FILE.name} could not be read ({e}); "
              f"using defaults")
        return {}


_machine = machine_settings()

RUN_ROOT = Path(os.environ.get("CST_TMATRIX_RUN_ROOT",
                               _machine.get("run_root")
                               or (Path.home() / "cst_tmatrix_runs")))

LIBRARY_ROOT = Path(os.environ.get("CST_TMATRIX_LIBRARY",
                                   _machine.get("library")
                                   or (_REPO_ROOT / "library")))

# Backwards-compatible alias used by earlier scripts.
LOCAL_RUN_ROOT = RUN_ROOT


# ---------------------------------------------------------------------------
# Extraction settings
# ---------------------------------------------------------------------------

@dataclass
class ExtractionConfig:
    """Settings of the vector-spherical-wave projection and the linear
    solve for T.

    lmax : multipole truncation order.
    illumination_factor : the number of illumination directions is
        ceil(illumination_factor * N / 2), where N = 2 lmax (lmax + 2) is
        the number of unknown columns; two orthogonal polarizations are
        simulated per direction.  A factor of 1.0 gives a determined
        system; 1.5 (default) gives 50% overdetermination, which averages
        solver noise in the least-squares solution.
    illumination_scheme : distribution of the incidence directions.
        "fibonacci" (default) — quasi-uniform spiral points, available for
        any direction count and free of the coordinate poles.
        "lebedev"   — Lebedev quadrature nodes; the smallest tabulated
        rule with at least the required number of directions is used, so
        the actual number of solver runs may exceed the requested count
        (Lebedev rules exist only for specific point counts).
    quadrature_oversample : multiplier on the surface-projection
        quadrature density (1.0 is exact for the design bandwidth).
    symmetry_n_fold : if set, the order of the scatterer's principal
        rotation axis (about z).  Together with symmetry_mirror_phi0_deg
        this activates SYMMETRY-REDUCED illumination: a scatterer with
        point group G satisfies T D(g) = D(g) T, so each solved
        illumination also supplies the columns (D(g) a, D(g) f) for every
        g in G without another solve.  For C4v (n_fold=4 plus four mirror
        planes, |G| = 8) this cuts the solve count ~8x -- measured 7.8x at
        lmax=9 (297 -> 38 solves).  Leave None to solve every illumination
        explicitly.  ONLY set this when the symmetry is real: verify with
        postprocess.symmetry.detect_axial_point_group() on an STL export,
        and confirm on an existing extraction that the C_n/mirror
        violations sit at the reciprocity noise level.
    symmetry_mirror_phi0_deg : azimuths (degrees) of vertical mirror
        planes, e.g. [0, 45, 90, 135] for C4v.  None = rotations only (C_n).
    """
    lmax: int
    illumination_factor: float = 1.5
    illumination_scheme: str = "fibonacci"
    quadrature_oversample: float = 1.2
    symmetry_n_fold: int | None = None
    symmetry_mirror_phi0_deg: tuple | None = None

    # ---- automatic checks -------------------------------------------------
    # Most conditions that invalidate an extraction can be tested in
    # seconds to minutes; these thresholds control the checks that do so.
    # Both motivating cases were avoidable: an out-of-memory failure that
    # was predictable from the cell count at meshing time, and a band
    # extraction whose first solve measured an incident deviation of
    # 0.11 yet ran 2.5 h to completion.
    preflight: bool = True
    # refuse to start when the monitor-placement criteria fail (both the
    # lmax-conditioning and the scatterer-near-field clearance); override
    # only for deliberate experiments
    allow_monitor_violation: bool = False
    # refuse to start when the meshed cell count exceeds this (None = off)
    max_cells: int | None = 8_000_000
    # refuse to start when the estimated solver memory exceeds this
    # fraction of currently available physical RAM
    memory_headroom: float = 0.9
    # per-solve check on the measured incident field: log a warning above
    # the first threshold, stop the run above the second (previously
    # solved columns stay cached, so a stopped run loses only the current
    # solve)
    incident_deviation_warn: float = 1e-2
    incident_deviation_abort: float = 3e-2
    # after the first solve: fraction of the scattered column's per-order
    # content permitted at l = lmax before a warning that the truncation
    # may be cutting into the physical content
    lmax_content_warn: float = 0.1
    # stop after the first solve if the projected total run time exceeds
    # this many hours (None = no limit)
    max_projected_hours: float | None = None

    @property
    def n_modes(self) -> int:
        return 2 * self.lmax * (self.lmax + 2)

    @property
    def n_illuminations(self) -> int:
        import math
        return math.ceil(self.illumination_factor * self.n_modes / 2)


@dataclass
class FrequencyPlan:
    """Frequencies at which T is extracted.  Units: the project frequency
    unit declared in the ScattererPlan."""
    fmin: float
    fmax: float
    n_freq: int = 1
    unit: str = "THz"

    @property
    def frequencies(self):
        import numpy as np
        if self.n_freq == 1:
            return np.array([0.5 * (self.fmin + self.fmax)])
        return np.linspace(self.fmin, self.fmax, self.n_freq)


# ---------------------------------------------------------------------------
# Per-scatterer paths
# ---------------------------------------------------------------------------

def grid_tag(freqs_hz, lmax: int) -> str:
    """Short filesystem-safe tag identifying an extraction's frequency grid
    and truncation order (e.g. "n49_df250GHz_l3").

    Two extractions of the SAME design at different resolution or lmax are
    a different amount of physics, not a re-run of the same one — they must
    not share a run directory or a library file path (see RunPaths.name vs.
    run_tag, and extract_tmatrix's overwrite guard in pipeline.py)."""
    import numpy as np
    freqs_hz = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
    n = freqs_hz.size
    step_hz = float(np.median(np.abs(np.diff(np.sort(freqs_hz))))) if n > 1 else 0.0
    step_str = f"{step_hz / 1e9:.4g}".replace(".", "p")
    return f"n{n}_df{step_str}GHz_l{lmax}"


@dataclass
class RunPaths:
    """Directory layout of one scatterer's CST runs and exports.

    name : identifies the physical design (geometry + material) — stable
        across extractions run at different fidelity.
    run_tag : identifies the extraction settings (frequency grid, lmax).
        Two runs of the same `name` with different `run_tag` get separate
        project/export directories, so switching between a coarse
        timing run and the final fine-grid extraction never overwrites the
        other's CST project or cached illuminations.  Empty by default for
        callers that construct RunPaths directly (e.g. tests); extract_tmatrix
        always supplies one (see grid_tag()).
    """
    name: str
    run_root: Path = RUN_ROOT
    run_tag: str = ""

    @property
    def _dir_name(self) -> str:
        return f"{self.name}__{self.run_tag}" if self.run_tag else self.name

    @property
    def project_dir(self) -> Path:
        return self.run_root / self._dir_name

    @property
    def cst_file(self) -> Path:
        return self.project_dir / f"{self.name}.cst"

    @property
    def export_dir(self) -> Path:
        return self.project_dir / "exports"

    def ensure(self):
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        return self
