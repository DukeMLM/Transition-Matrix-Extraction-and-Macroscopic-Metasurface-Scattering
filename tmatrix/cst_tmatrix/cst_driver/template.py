"""Extraction from a manually prepared CST project (template mode).

For geometries that are easier to build interactively than to script, the
user supplies a finished CST project containing ONLY the scatterer geometry
and its materials, centered at the origin.  This module

1. copies the project into the run directory (the original file is never
   modified),
2. verifies, as far as the scripting interface permits, that the model is
   consistent with the requirements of a free-space T-matrix extraction,
   and
3. imposes the settings the extraction owns — frequency range, vacuum
   background, open boundaries, mesh and adaptation settings, field
   monitors, and the frequency-domain solver configuration — as history
   blocks appended to (or replacing blocks of) the copied project.

Requirements on the template (see the user manual for the rationale):
  - geometry centered at the origin (the multipole origin is (0,0,0));
  - no ports, no periodic/unit-cell boundaries, no symmetry planes;
  - no substrate or other structure extending to the domain boundary —
    the surface on which fields are exported must lie in the homogeneous
    background;
  - project length/frequency units as declared in the ScattererPlan
    (units are NOT changed by this module, because reinterpreting units
    on an existing model would rescale the geometry).

Verification is best-effort: every check reports PASS, FAIL, or
UNVERIFIED (the scripting interface does not expose the quantity).  With
strict=True a FAIL raises; UNVERIFIED never raises but is printed so the
user can confirm manually.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .session import CSTSession, ProjectHandle, CSTError


class CheckReport:
    """Collected verification results: list of (name, status, detail)."""

    def __init__(self):
        self.entries = []

    def add(self, name: str, status: str, detail: str = ""):
        self.entries.append((name, status, detail))

    @property
    def failed(self):
        return [e for e in self.entries if e[1] == "FAIL"]

    def __str__(self):
        lines = ["Template verification:"]
        for name, status, detail in self.entries:
            lines.append(f"  [{status:>10s}] {name}"
                         + (f" — {detail}" if detail else ""))
        return "\n".join(lines)


def copy_template(template: str | Path, dest_cst: Path) -> Path:
    """Copy the template project file into the run directory.  Any project
    already present at the destination is overwritten (the run directory
    is disposable by definition)."""
    template = Path(template)
    if not template.is_file():
        raise FileNotFoundError(f"Template project not found: {template}")
    dest_cst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, dest_cst)
    return dest_cst


def open_template(session: CSTSession, template: str | Path,
                  dest_cst: Path) -> ProjectHandle:
    """Copy the template and open the copy."""
    # close a stale open project at the destination before overwriting
    try:
        old = session.de.get_open_project(str(dest_cst))
        if old is not None:
            old.close()
    except Exception:                                     # noqa: BLE001
        pass
    copy_template(template, dest_cst)
    return session.open_project(dest_cst)


def _try(fn, *args):
    """Call a scripting getter, returning (ok, value_or_exception)."""
    try:
        return True, fn(*args)
    except Exception as e:                                # noqa: BLE001
        return False, e


def verify_template(h: ProjectHandle, r_circ_proj: float,
                    tolerance: float = 0.05) -> CheckReport:
    """Best-effort consistency checks of a template project.

    r_circ_proj : circumscribing radius in PROJECT length units, from the
        ScattererPlan (r_circ_m / length-unit scale).  The union bounding
        box of all shapes is compared against it: the box must be centered
        near the origin and its half-diagonal must not exceed the declared
        radius by more than `tolerance` (relative).  This catches the two
        most damaging template mistakes: geometry not centered at the
        multipole origin, and a wrong declared unit or radius.
    """
    rep = CheckReport()

    # -- solids exist -------------------------------------------------------
    ok, nshapes = _try(h.m3d.Solid.GetNumberOfShapes)
    if not ok:
        rep.add("solid count", "UNVERIFIED", str(nshapes))
        nshapes = None
    elif int(nshapes) < 1:
        rep.add("solid count", "FAIL", "template contains no solids")
    else:
        rep.add("solid count", "PASS", f"{int(nshapes)} solid(s)")

    # -- bounding box vs declared circumscribing radius ---------------------
    box = None
    if nshapes:
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        got_any = False
        for i in range(int(nshapes)):
            ok, sname = _try(h.m3d.Solid.GetNameOfShapeFromIndex, i)
            if not ok:
                continue
            ok, bb = _try(h.m3d.Solid.GetLooseBoundingBoxOfShape, sname)
            if ok and isinstance(bb, (list, tuple)) and len(bb) >= 7 and bb[0]:
                vals = np.asarray(bb[1:7], dtype=float)
                lo = np.minimum(lo, vals[[0, 2, 4]])
                hi = np.maximum(hi, vals[[1, 3, 5]])
                got_any = True
        if got_any:
            box = (lo, hi)
    if box is None:
        rep.add("bounding box", "UNVERIFIED",
                "shape bounding boxes not readable through the proxy; "
                "confirm manually that the geometry is centered at the "
                "origin and fits the declared circumscribing radius")
    else:
        lo, hi = box
        center = 0.5 * (lo + hi)
        half_diag = float(np.linalg.norm(0.5 * (hi - lo)))
        if np.linalg.norm(center) > tolerance * max(r_circ_proj, 1e-300):
            rep.add("geometry centered", "FAIL",
                    f"bounding-box center {center} is displaced from the "
                    f"origin; the multipole origin is (0,0,0)")
        else:
            rep.add("geometry centered", "PASS",
                    f"bounding-box center offset {np.linalg.norm(center):.3g}")
        if half_diag > (1.0 + tolerance) * r_circ_proj:
            rep.add("circumscribing radius", "FAIL",
                    f"bounding-box half-diagonal {half_diag:.4g} exceeds "
                    f"declared r_circ {r_circ_proj:.4g} (project units) — "
                    f"check the declared radius and length unit")
        else:
            rep.add("circumscribing radius", "PASS",
                    f"half-diagonal {half_diag:.4g} <= declared "
                    f"{r_circ_proj:.4g}")

    # -- active solver ------------------------------------------------------
    ok, stype = _try(h.m3d.GetSolverType)
    if ok:
        if "Frequency" in str(stype):
            rep.add("solver type", "PASS", str(stype))
        else:
            rep.add("solver type", "PASS",
                    f"currently '{stype}'; will be switched to the "
                    f"frequency-domain solver")
    else:
        rep.add("solver type", "UNVERIFIED",
                "will be set explicitly before the run")

    return rep


def prepare_template_project(session: CSTSession, template: str | Path,
                             dest_cst: Path, r_circ_proj: float,
                             strict: bool = True, verbose: bool = True):
    """Copy, open, and verify a template project.

    Returns the ProjectHandle.  With strict=True (default) any FAIL in the
    verification raises CSTError; UNVERIFIED entries are reported only.
    The caller (pipeline.build_project) subsequently imposes frequency
    range, background, boundaries, mesh, monitors, and solver settings.
    """
    h = open_template(session, template, dest_cst)
    rep = verify_template(h, r_circ_proj)
    if verbose:
        print(rep)
    if strict and rep.failed:
        raise CSTError(
            "Template verification failed:\n"
            + "\n".join(f"  {n}: {d}" for n, _, d in rep.failed)
            + "\nFix the template or call with strict=False to proceed "
              "anyway.")
    return h
