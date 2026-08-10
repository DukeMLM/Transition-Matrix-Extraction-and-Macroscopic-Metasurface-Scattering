"""Field data extraction from CST into numpy.

Near fields: ASCIIExport (VBA-only object) driven through Mode 3 with a
custom evaluation-point file — the only documented route for complex vector
E/H at arbitrary 3D points.  The parser verifies that the coordinates in the
exported file match the requested points (catching unit or format surprises
immediately instead of corrupting the projection silently).

Farfield: FarfieldCalculator list evaluation (Mode 2 getters) — complex
E_theta/E_phi directly in memory, no file round-trip.
"""

from __future__ import annotations

import re
import numpy as np
from pathlib import Path

from .session import ProjectHandle, CSTError

Z0 = 376.730313668          # free-space impedance, ohms


# ---------------------------------------------------------------------------
# Point files and ASCII export
# ---------------------------------------------------------------------------

def write_point_file(path: str | Path, points_project_units: np.ndarray):
    """Evaluation-point file for ASCIIExport.SetPointFile: one 'x y z' line
    per point, project length units."""
    pts = np.asarray(points_project_units, dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as fh:
        for p in pts:
            fh.write(f"{p[0]:.9e} {p[1]:.9e} {p[2]:.9e}\n")
    return path


def export_monitor_ascii(h: ProjectHandle, tree_path: str,
                         out_file: str | Path, point_file: str | Path):
    """Run the ASCIIExport of a selected 3D field monitor result at the
    points in point_file.  Mode 3 (no history pollution)."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()                       # never trust stale exports
    body = f"""SelectTreeItem ("{tree_path}")
With ASCIIExport
  .Reset
  .FileName "{str(out_file)}"
  .SetPointFile "{str(Path(point_file))}"
  .SetFileType "ascii"
  .ExportCoordinatesInMeter "False"
  .Execute
End With"""
    h.execute_vba(body)
    if not out_file.exists():
        raise CSTError(
            f"ASCIIExport produced no file for '{tree_path}'.\n"
            f"Likely causes: tree item does not exist (check monitor name / "
            f"excitation label), or the point-file format was rejected.\n"
            f"CST messages: {h._messages()}")
    return out_file


_FIELD_COLS = {
    "e": ["ExRe", "EyRe", "EzRe", "ExIm", "EyIm", "EzIm"],
    "h": ["HxRe", "HyRe", "HzRe", "HxIm", "HyIm", "HzIm"],
}


def parse_field_export(path: str | Path, expected_points: np.ndarray | None = None,
                       rtol: float = 1e-3):
    """Parse a CST ASCII field export -> (points (N,3), F (N,3) complex).

    Column mapping is header-driven when a header exists (tokens like
    'ExRe [V/m]'); otherwise assumes x y z Re-triplet Im-triplet.  If
    expected_points is given, verifies the exported coordinates match
    (allowing a single global unit scale factor) and reorders rows to the
    requested order if CST permuted them.
    """
    path = Path(path)
    header_cols = None
    rows = []
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            toks = line.replace(",", " ").split()
            if not toks:
                continue
            try:
                rows.append([float(t) for t in toks])
            except ValueError:
                joined = " ".join(toks).lower()
                if re.search(r"[eh]x.?re", joined):
                    header_cols = re.findall(r"([EHeh][xyz](?:Re|Im|re|im))",
                                             line)
    data = np.array([r for r in rows if len(r) == len(rows[-1])], dtype=float)
    if data.ndim != 2 or data.shape[1] < 9:
        raise CSTError(f"Unexpected export format in {path}: shape {data.shape}")

    xyz = data[:, :3]
    vals = data[:, 3:9]
    if header_cols and len(header_cols) >= 6:
        order = [c.lower()[1:] for c in header_cols[:6]]   # e.g. 'xre','yre'..
        want = ["xre", "yre", "zre", "xim", "yim", "zim"]
        try:
            perm = [order.index(wc) for wc in want]
            vals = vals[:, perm]
        except ValueError:
            pass                                            # fall back to default
    F = vals[:, :3] + 1j * vals[:, 3:6]

    if expected_points is not None:
        exp = np.asarray(expected_points, dtype=float)
        if exp.shape[0] != xyz.shape[0]:
            raise CSTError(
                f"Export row count {xyz.shape[0]} != requested {exp.shape[0]} "
                f"({path})")
        # allow one global scale (unit) factor between file and request
        scale = (np.linalg.norm(xyz) / max(np.linalg.norm(exp), 1e-300))
        xr = xyz / max(scale, 1e-300)
        err = np.max(np.linalg.norm(xr - exp, axis=1))
        ref = max(np.max(np.linalg.norm(exp, axis=1)), 1e-300)
        if err > rtol * ref:
            # rows may be permuted: match by nearest neighbour
            from scipy.spatial import cKDTree
            tree = cKDTree(xr)
            d, idx = tree.query(exp)
            if np.max(d) > rtol * ref or len(set(idx)) != len(idx):
                raise CSTError(
                    f"Exported coordinates do not match requested points "
                    f"(max mismatch {np.max(d):.3g}, scale {scale:.3g}) — "
                    f"check point-file units ({path})")
            F = F[idx]
            xyz = xyz[idx]
    return xyz, F


def get_EH_on_points(h: ProjectHandle, e_tree: str, h_tree: str,
                     export_dir: Path, tag: str, point_file: Path,
                     points_project_units: np.ndarray):
    """Export + parse E and Z0*H at the evaluation points.
    Returns (E, Z0H) complex (N,3) in V/m."""
    e_file = export_monitor_ascii(h, e_tree, export_dir / f"{tag}_E.txt",
                                  point_file)
    h_file = export_monitor_ascii(h, h_tree, export_dir / f"{tag}_H.txt",
                                  point_file)
    _, E = parse_field_export(e_file, points_project_units)
    _, Hf = parse_field_export(h_file, points_project_units)
    return E, Z0 * Hf


# ---------------------------------------------------------------------------
# Farfield list evaluation (Mode 2)
# ---------------------------------------------------------------------------

def get_farfield_pattern(h: ProjectHandle, tree_path: str,
                         theta_rad: np.ndarray, phi_rad: np.ndarray):
    """Complex farfield E_theta, E_phi at the given angles via
    FarfieldCalculator (in-memory, no files).

    Returns (E_th, E_ph) arrays as exported by CST ('efield' plot mode,
    reference distance 1 m).  Conversion to the dimensionless pattern F of
    vswf.project_farfield: F = k * (1 m) * exp(+j k 1m) * E_cst — but any
    global per-frequency factor cancels in T when the incident columns come
    from the same normalization, so the raw values are usable directly for
    cross-checks.
    """
    fc = h.m3d.FarfieldCalculator
    th_deg = np.degrees(np.asarray(theta_rad)).tolist()
    ph_deg = np.degrees(np.asarray(phi_rad)).tolist()
    n = len(th_deg)
    try:
        fc.AddListEvaluationPoints(th_deg, ph_deg, [0.0] * n,
                                   ["spherical"] * n, [""] * n, [0.0] * n)
    except Exception:                                     # noqa: BLE001
        for t, p in zip(th_deg, ph_deg):
            fc.AddListEvaluationPoint(t, p, 0.0, "spherical", "", 0.0)
    fc.CalculateList(tree_path, "farfield")
    comp = {}
    for pol in ("theta", "phi"):
        for part in ("re", "im"):
            comp[(pol, part)] = np.array(
                fc.GetList("efield", f"spherical linear {pol} {part}"),
                dtype=float)
    E_th = comp[("theta", "re")] + 1j * comp[("theta", "im")]
    E_ph = comp[("phi", "re")] + 1j * comp[("phi", "im")]
    return E_th, E_ph
