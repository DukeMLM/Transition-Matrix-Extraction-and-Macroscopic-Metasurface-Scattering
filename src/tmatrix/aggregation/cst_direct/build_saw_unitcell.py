"""Direct CST reference: free-standing spoke-and-wheel unit cell, periodic
(Floquet) boundaries, normal incidence — the configuration predicted by the
T-matrix aggregation pipeline (tmatrix.aggregation.run_demo).

Geometry/material from the tmat.h5 attributes (matching the extraction
project): pitch 2 um, r_out = 0.7193, w_ring = 0.1611, gap = 0.5627,
w = 0.3624, thickness = 0.1 (um), gold sigma = 4.561e7 S/m, vacuum
everywhere (no substrate, no ground plane — this is T_iso's world).

Band: 8-20 um -> 14.99-37.47 THz.

Adapted from D:/Claude/auto_cst/build_tao2010_dualband.py (proven VBA on
this machine's CST installation).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from tmatrix.cst_env import ensure_on_path
from tmatrix.paths import CST_DIRECT_DATA

ensure_on_path()

import cst.interface as cstint  # noqa: E402

from nir.cst_helpers import (  # noqa: E402
    HistoryBuilder,
    find_reflection_sparams,
    get_result_with_data,
    open_results,
    save_project_at,
)

# ---- design (um) -----------------------------------------------------------
P = dict(pitch=2.0, r_out=0.7193, w_ring=0.1611, gap=0.5627, w=0.3624,
         t_mm=0.1)
AU_SIGMA = 4.561e7          # S/m, from tmat.h5 scatterer/material
F_MIN_THZ, F_MAX_THZ = 14.99, 37.47

VBA_UNITS = """\
With Units
    .SetUnit "Length", "um"
    .SetUnit "Frequency", "THz"
    .SetUnit "Time", "ps"
End With
"""

VBA_FREQ_RANGE = f'Solver.FrequencyRange "{F_MIN_THZ}", "{F_MAX_THZ}"\n'

VBA_BOUNDARY = f"""\
With Boundary
    .Xmin "unit cell"
    .Xmax "unit cell"
    .Ymin "unit cell"
    .Ymax "unit cell"
    .Zmin "expanded open"
    .Zmax "expanded open"
    .SetPeriodicBoundaryAngles "0", "0"
End With
' CRITICAL: without an explicit unit-cell size CST fits the cell to the
' geometry bounding box (1.4386 um = the ring diameter), fusing neighboring
' rings into a connected reflective mesh.  Verified by control tests
' (control_tests.py): a 1 um patch behaved identically to a continuous sheet.
With Boundary
    .UnitCellFitToBoundingBox "False"
    .UnitCellDs1 "{P['pitch']}"
    .UnitCellDs2 "{P['pitch']}"
    .UnitCellAngle "90.0"
End With
"""

VBA_BACKGROUND = """\
With Background
    .Type "Normal"
    .Epsilon "1.0"
    .Mu "1.0"
End With
"""

VBA_AU = f"""\
With Material
    .Reset
    .Name "Au"
    .Folder ""
    .FrqType "all"
    .Type "Lossy metal"
    .Sigma "{AU_SIGMA}"
    .Colour "1.0", "0.84", "0.0"
    .Create
End With
"""


def geometry_vba(p: dict) -> str:
    """Ring (annulus) + 4 inward spokes, united into component1:saw.

    Spokes run from |u| = gap/2 to the middle of the ring annulus
    (r_out - w_ring/2) so the boolean unite always overlaps the ring.
    """
    r_in = p["r_out"] - p["w_ring"]
    z0, z1 = -p["t_mm"] / 2, p["t_mm"] / 2
    tip, root = p["gap"] / 2, p["r_out"] - p["w_ring"] / 2
    hw = p["w"] / 2
    vba = [f"""\
With Cylinder
    .Reset
    .Name "ring"
    .Component "component1"
    .Material "Au"
    .OuterRadius "{p['r_out']}"
    .InnerRadius "{r_in}"
    .Axis "z"
    .Zrange "{z0}", "{z1}"
    .Xcenter "0"
    .Ycenter "0"
    .Create
End With
"""]
    spokes = [("spoke_n", -hw, hw, tip, root),
              ("spoke_s", -hw, hw, -root, -tip),
              ("spoke_e", tip, root, -hw, hw),
              ("spoke_w", -root, -tip, -hw, hw)]
    for name, x0, x1, y0, y1 in spokes:
        vba.append(f"""\
With Brick
    .Reset
    .Name "{name}"
    .Component "component1"
    .Material "Au"
    .Xrange "{x0}", "{x1}"
    .Yrange "{y0}", "{y1}"
    .Zrange "{z0}", "{z1}"
    .Create
End With
""")
    for name in ("spoke_n", "spoke_s", "spoke_e", "spoke_w"):
        vba.append(f'Solid.Add "component1:ring", "component1:{name}"\n')
    vba.append('Solid.Rename "component1:ring", "saw"\n')
    return "".join(vba)


VBA_PORTS = """\
With FloquetPort
    .Reset
    .SetDialogTheta "0"
    .SetDialogPhi "0"
    .SetPolarizationIndependentOfScanAnglePhi "0", "False"
    .SetSortCode "+beta/pw"
    .SetCustomizedListFlag "False"
    .Port "Zmax"
    .SetNumberOfModesConsidered "2"
    .SetDistanceToReferencePlane "0"
    .SetUseCircularPolarization "False"
    .Port "Zmin"
    .SetNumberOfModesConsidered "2"
End With
"""

VBA_FREQ_SAMPLES = f"""\
With FDSolver
    .Reset
    .Stimulation "List", "List"
    .AddSampleInterval "{F_MIN_THZ}", "{F_MAX_THZ}", "", "Automatic", "True"
    .AccuracyTet "1e-4"
    .OrderTet "Second"
End With
"""

VBA_MESH_ADAPT = """\
With MeshAdaption3D
    .SetType "HighFrequencyTet"
    .SetAdaptionStrategy "Energy"
    .MinPasses 3
    .MaxPasses 8
    .ErrorLimit 0.0001
End With
"""

# Local mesh constraint on the metal: the v1 run false-converged on a
# 1372-3141-cell mesh that could not resolve the 161 nm ring / 100 nm
# thickness (log: "82% of all mesh edges seem to be longer than expected"),
# smearing the array into a continuous reflective sheet.  Force the surface
# mesh on the resonator down to 60 nm edges.
VBA_LOCAL_MESH = """\
With MeshSettings
    With .ItemMeshSettings ("solid$component1:saw")
        .SetMeshType "Tet"
        .Set "Size", "0.06"
    End With
End With
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=CST_DIRECT_DATA / "run_v1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "build_log.txt"
    log_path.write_text("")

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"saw unit cell build -> {run_dir}")
    log(f"design (um): {json.dumps(P)}")

    target_cst = run_dir / "saw_unitcell.cst"
    if target_cst.exists():
        target_cst.unlink()
    folder = run_dir / "saw_unitcell"
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)

    log("opening CST DesignEnvironment...")
    env = cstint.DesignEnvironment()
    project = env.new_mws()
    m3d = project.model3d
    save_project_at(project, target_cst)

    builder = HistoryBuilder(project, verify=True)
    try:
        builder.add("saw: Step 1 Units", VBA_UNITS)
        builder.add("saw: Step 2 Freq range", VBA_FREQ_RANGE)
        builder.add("saw: Step 3 Boundary", VBA_BOUNDARY)
        builder.add("saw: Step 4 Background", VBA_BACKGROUND)
        builder.add("saw: Step 5 Au material", VBA_AU,
                    expects_materials=["Au"])
        builder.add("saw: Step 6 Geometry", geometry_vba(P),
                    expects_solids=["saw"])
        builder.add("saw: Step 7 Floquet ports", VBA_PORTS)
        builder.add("saw: Step 8 Freq samples", VBA_FREQ_SAMPLES)
        builder.add("saw: Step 9 Mesh adaptation", VBA_MESH_ADAPT)
        builder.add("saw: Step 10 Local mesh on metal", VBA_LOCAL_MESH)
    except RuntimeError as e:
        log(f"[FATAL] VBA failure:\n{e}\n{builder.summary()}")
        try:
            project.save(); project.close(); env.close()
        except Exception:
            pass
        return 3
    log("history applied + verified:")
    for line in builder.summary().splitlines():
        log("  " + line)
    project.save()

    if args.dry_run:
        log("[DRY RUN] stopping before solver; project saved for inspection")
        project.close(); env.close()
        return 0

    log("starting FD solver...")
    t0 = time.time()
    try:
        m3d.FDSolver.Start()
        log(f"FDSolver finished after {time.time()-t0:.0f}s")
    except Exception as e:
        log(f"[ERROR] FDSolver: {e}")
        project.save(); project.close(); env.close()
        return 4
    project.save()
    project.close()
    env.close()

    log("reading S-parameters...")
    proj3d, run_ids = open_results(target_cst)
    tree = proj3d.get_tree_items()
    s_items = [t for t in tree if t.startswith("1D Results\\S-Parameters\\")]
    for t in s_items:
        log(f"  {t}")
    refl = find_reflection_sparams(tree)
    import re as _re
    best = None
    for c11 in refl:
        m = _re.search(r"Zmax\((\d+)\),Zmax\(\1\)", c11)
        if not m:
            continue
        idx = m.group(1)
        c21 = next((t for t in s_items if f"Zmin({idx}),Zmax({idx})" in t),
                   None)
        if c21 is None:
            continue
        try:
            item, _ = get_result_with_data(proj3d, c11, run_ids)
        except Exception:
            continue
        y = np.array(item.get_ydata(), dtype=complex)
        std = float(np.std(np.abs(y)))
        log(f"  mode {idx}: std|S11|={std:.5f}")
        if best is None or std > best[2]:
            best = (c11, c21, std)
    if best is None:
        log("[ERROR] no S-param pair found")
        return 5
    s11_item, _ = get_result_with_data(proj3d, best[0], run_ids)
    s21_item, _ = get_result_with_data(proj3d, best[1], run_ids)
    f = np.array(s11_item.get_xdata(), dtype=float)
    s11 = np.array(s11_item.get_ydata(), dtype=complex)
    f21 = np.array(s21_item.get_xdata(), dtype=float)
    s21 = np.array(s21_item.get_ydata(), dtype=complex)
    if not np.allclose(f, f21):
        s21 = (np.interp(f, f21, s21.real) + 1j * np.interp(f, f21, s21.imag))
    out = run_dir / "s_params_complex.csv"
    with out.open("w") as fh:
        fh.write("freq_THz,Re_S11,Im_S11,Re_S21,Im_S21\n")
        for fr, a, b in zip(f, s11, s21):
            fh.write(f"{fr:.6f},{a.real:.8e},{a.imag:.8e},"
                     f"{b.real:.8e},{b.imag:.8e}\n")
    log(f"wrote {out} ({len(f)} points); |S11| range "
        f"[{np.abs(s11).min():.4f}, {np.abs(s11).max():.4f}], |S21| range "
        f"[{np.abs(s21).min():.4f}, {np.abs(s21).max():.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
