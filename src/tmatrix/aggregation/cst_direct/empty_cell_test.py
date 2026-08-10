"""Control test: EMPTY unit cell with the identical boundary/port/solver
setup must give |S21| = 1, |S11| = 0. Any deviation indicts the setup,
not the scatterer."""
import time

import numpy as np

from tmatrix.cst_env import ensure_on_path
from tmatrix.paths import CST_DIRECT_DATA

ensure_on_path()
import cst.interface as cstint                              # noqa: E402

from nir.cst_helpers import (HistoryBuilder, get_result_with_data,  # noqa: E402
                             open_results, save_project_at)
from tmatrix.aggregation.cst_direct.build_saw_unitcell import (  # noqa: E402
    VBA_BACKGROUND, VBA_BOUNDARY, VBA_FREQ_RANGE, VBA_FREQ_SAMPLES,
    VBA_PORTS, VBA_UNITS)


def main():
    run_dir = CST_DIRECT_DATA / "run_empty"
    run_dir.mkdir(parents=True, exist_ok=True)
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
    prj.save()
    prj.close()
    env.close()

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
        print(f"{name}: |S| range "
              f"[{np.abs(y).min():.6f}, {np.abs(y).max():.6f}]")


if __name__ == "__main__":
    main()
