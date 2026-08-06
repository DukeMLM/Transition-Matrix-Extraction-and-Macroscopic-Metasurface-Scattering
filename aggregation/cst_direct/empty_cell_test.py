"""Control test: EMPTY unit cell with the identical boundary/port/solver
setup must give |S21| = 1, |S11| = 0. Any deviation indicts the setup,
not the scatterer."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\Claude\auto_cst")
sys.path.insert(0, r"E:\cst\AMD64\python_cst_libraries")
import cst.interface as cstint

from nir.cst_helpers import (HistoryBuilder, get_result_with_data,
                             open_results, save_project_at)
from build_saw_unitcell import (VBA_UNITS, VBA_FREQ_RANGE, VBA_BOUNDARY,
                                VBA_BACKGROUND, VBA_PORTS, VBA_FREQ_SAMPLES)

run_dir = Path(__file__).parent / "run_empty"
run_dir.mkdir(exist_ok=True)
target = run_dir / "empty_cell.cst"
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
b.add("Ports", VBA_PORTS)
b.add("Samples", VBA_FREQ_SAMPLES)
prj.save()
print("solving empty cell...", flush=True)
t0 = time.time()
m3d.FDSolver.Start()
print(f"done in {time.time()-t0:.0f}s")
prj.save(); prj.close(); env.close()

proj3d, run_ids = open_results(target)
tree = proj3d.get_tree_items()
for name in ("SZmax(1),Zmax(1)", "SZmin(1),Zmax(1)",
             "SZmax(1),Zmin(1)", "SZmin(1),Zmin(1)"):
    path = f"1D Results\\S-Parameters\\{name}"
    if path not in tree:
        print(f"{name}: NOT IN TREE")
        continue
    item, _ = get_result_with_data(proj3d, path, run_ids)
    y = np.array(item.get_ydata(), dtype=complex)
    print(f"{name}: |S| range [{np.abs(y).min():.6f}, {np.abs(y).max():.6f}]")
