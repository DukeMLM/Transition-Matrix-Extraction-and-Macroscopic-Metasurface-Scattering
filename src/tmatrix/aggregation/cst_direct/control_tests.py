"""Three controls with the IDENTICAL boundary/port/solver setup:
  vacuum : brick with eps=1.000001 spanning the cell -> |S21| ~ 1, |S11| ~ 0
  sheet  : continuous 100 nm gold sheet              -> |S11| ~ 0.999
  patch  : 1.0 um square gold patch, 100 nm thick    -> transmissive
           (patch resonance ~2-3 um, far below the 8-20 um band)
Whichever controls fail indict the setup; if all pass, the spoke-wheel
result is real physics.
"""
import time
from pathlib import Path

import numpy as np

from tmatrix.cst_env import ensure_on_path

ensure_on_path()
import cst.interface as cstint                              # noqa: E402

from nir.cst_helpers import (HistoryBuilder, get_result_with_data,   # noqa: E402
                             open_results, save_project_at)
from tmatrix.aggregation.cst_direct.build_saw_unitcell import (  # noqa: E402
    VBA_AU, VBA_BACKGROUND, VBA_BOUNDARY, VBA_FREQ_RANGE,
    VBA_FREQ_SAMPLES, VBA_PORTS, VBA_UNITS)
from tmatrix.paths import CST_DIRECT_DATA                       # noqa: E402

VBA_VAC = """\
With Material
    .Reset
    .Name "AlmostVac"
    .Type "Normal"
    .Epsilon "1.000001"
    .Mu "1"
    .Create
End With
"""

GEOMS = {
    "vacuum": """\
With Brick
    .Reset
    .Name "slab"
    .Component "component1"
    .Material "AlmostVac"
    .Xrange "-1", "1"
    .Yrange "-1", "1"
    .Zrange "-0.05", "0.05"
    .Create
End With
""",
    "sheet": """\
With Brick
    .Reset
    .Name "sheet"
    .Component "component1"
    .Material "Au"
    .Xrange "-1", "1"
    .Yrange "-1", "1"
    .Zrange "-0.05", "0.05"
    .Create
End With
""",
    "patch": """\
With Brick
    .Reset
    .Name "patch"
    .Component "component1"
    .Material "Au"
    .Xrange "-0.5", "0.5"
    .Yrange "-0.5", "0.5"
    .Zrange "-0.05", "0.05"
    .Create
End With
""",
}


def run_case(name, geom_vba, solid):
    run_dir = CST_DIRECT_DATA / f"run_ctrl_{name}"
    run_dir.mkdir(exist_ok=True)
    target = run_dir / f"{name}.cst"
    if target.exists():
        target.unlink()
    env = cstint.DesignEnvironment()
    prj = env.new_mws()
    m3d = prj.model3d
    save_project_at(prj, target)
    b = HistoryBuilder(prj, verify=True)
    b.add("Units", VBA_UNITS)
    b.add("Freq", VBA_FREQ_RANGE)
    b.add("Boundary", VBA_BOUNDARY)
    b.add("Background", VBA_BACKGROUND)
    b.add("Vac material", VBA_VAC)
    b.add("Au material", VBA_AU, expects_materials=["Au"])
    b.add("Geometry", geom_vba, expects_solids=[solid])
    b.add("Ports", VBA_PORTS)
    b.add("Samples", VBA_FREQ_SAMPLES)
    prj.save()
    t0 = time.time()
    m3d.FDSolver.Start()
    dt = time.time() - t0
    prj.save(); prj.close(); env.close()

    proj3d, run_ids = open_results(target)
    out = {}
    for label, path in [("S11", "SZmax(1),Zmax(1)"),
                        ("S21", "SZmin(1),Zmax(1)")]:
        item, _ = get_result_with_data(
            proj3d, f"1D Results\\S-Parameters\\{path}", run_ids)
        y = np.array(item.get_ydata(), dtype=complex)
        out[label] = (np.abs(y).min(), np.abs(y).max())
    print(f"[{name}] solve {dt:.0f}s  |S11| in [{out['S11'][0]:.4f}, "
          f"{out['S11'][1]:.4f}]  |S21| in [{out['S21'][0]:.4f}, "
          f"{out['S21'][1]:.4f}]", flush=True)


if __name__ == "__main__":
    for nm, (geom, solid) in {
        "vacuum": (GEOMS["vacuum"], "slab"),
        "sheet": (GEOMS["sheet"], "sheet"),
        "patch": (GEOMS["patch"], "patch"),
    }.items():
        run_case(nm, geom, solid)
