r"""Mesh-density check for the GOLD spoke-and-wheel geometry (13.1 um design).

Purpose: the deterministic non-adaptive mesh strategy (mesh_adaption=False +
plan.mesh_steps_per_wave) was validated on DIELECTRIC scatterers -- the eps=4
sphere vs exact Mie and the TiO2 cylinder vs the JCMsuite reference
(2026-08-05/06).  The spoke-and-wheel is a lossy-metal (SIBC gold) resonator
with sharp sub-wavelength features; its density requirement must be measured,
not assumed.

Method (no analytic reference exists for this shape):
  - ONE normal-incidence illumination, TWO frequencies per solve:
      22.885 THz (the 13.1 um design's center) and 34.0 THz (band top --
      the finest per-wavelength mesh requirement of the eventual full run)
  - solve on 15/6, 30/12, 45/18 steps-per-wavelength fixed meshes and once
    with ADAPTIVE refinement (relaxed 2-pass/0.02 settings, the reference
    configuration that matched exact Mie on the sphere)
  - compare the SCATTERED coefficient column f_meas across meshes: this is
    exactly the quantity that enters T = F A^+, so its convergence is the
    relevant metric.  Reported both over all orders (lmax=9) and restricted
    to l <= 7 (the Wiscombe-physical orders at 22.885 THz; all 9 orders are
    physical at 34 THz).

The monitor factor is auto-computed for THESE two frequencies (~8.0), not
the full-band value (~18.2): steps-per-wavelength adequacy transfers to the
full-band run (the mesh is sized per wavelength at fmax, identical here and
there), while the domain -- and so the check's cost -- stays ~2.3x smaller
in radius.

Also reported per case: cell count and run time, the inputs for
projecting the cost of the full extraction.

Run on the CST machine:
    python density_check_spoke_wheel.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract_spoke_wheel_scaled as sw                          # noqa: E402
from cst_tmatrix import vswf                                     # noqa: E402
from cst_tmatrix.config import ExtractionConfig, RunPaths, LOCAL_RUN_ROOT  # noqa: E402
from cst_tmatrix.pipeline import build_project, C0               # noqa: E402
from cst_tmatrix.quadrature import quadrature_for_monitor, unit_vectors  # noqa: E402
from cst_tmatrix.cst_driver import (CSTSession, solvers, excitation,  # noqa: E402
                                    monitors, export)

PICK_WL = 13.1
LMAX = 9
FREQS_HZ = np.array([C0 / 13.1e-6, 34.0e12])      # 22.885, 34.0 THz
L_PHYS_LOW = 7          # Wiscombe at 22.885 THz for this r_circ; 9 at 34 THz

# Case-set note (revised 2026-08-06, live): the original ladder 15/6 ->
# 30/12 -> 45/18 scaled near AND far together; at this domain size the
# far-zone VACUUM dominates the meshed volume, so 30/12 ballooned (~8x
# cells) and was killed as impractical.  The 15/6 result showed
# incident_deviation 2e-2 at 34 THz -- a propagation-accuracy symptom
# (the incident wave crosses several wavelengths of far zone to reach the
# monitor), pointing at far=6, not the near zone, as the marginal setting.
# Revised ladder: raise far 6->8 and near 15->20, and let the ADAPTIVE
# reference arbitrate.  Completed cases are cached (density_case.npz) and
# reused on restart.
CASES = [
    ("15_6", (15, 6), False),
    ("20_8", (20, 8), False),
    ("adaptive", (15, 6), True),      # base density + adaptive refinement
]


def main():
    designs = sw.load_designs(sw.DESIGN_FILE)
    design = min(designs, key=lambda d: abs(d["wl"] - PICK_WL))
    print(f"design: wl={design['wl']:.4f} um  r={design['r']:.6g} "
          f"gap={design['gap']:.6g} w_ring={design['w_ring']:.6g} "
          f"w={design['w']:.6g} t={sw.T_MM_UM} um", flush=True)

    # one shared measurement geometry for every case
    ref_plan = sw.make_plan(design, FREQS_HZ, LMAX, monitor_factor=None)
    r_mon_m = ref_plan.monitor_factor * ref_plan.r_circ_m
    k_list = 2 * np.pi * FREQS_HZ / C0
    th_q, ph_q, w_q, qshape = quadrature_for_monitor(
        LMAX, float(np.max(k_list)), r_mon_m)
    pts_m = r_mon_m * unit_vectors(th_q, ph_q)
    pts_proj = pts_m / 1e-6
    a_nom = vswf.plane_wave_coefficients(0.0, 0.0, "theta", LMAX)
    _, ns, _ = vswf.mode_list(LMAX)
    print(f"monitor radius {r_mon_m*1e6:.3f} um (factor "
          f"{ref_plan.monitor_factor:.2f}), quadrature "
          f"{qshape[0]} x {qshape[1]} = {len(th_q)} points\n", flush=True)

    session = CSTSession()
    results = {}
    for tag, steps, adaptive in CASES:
        cache = (RunPaths(f"sw13p1_density_{tag}", LOCAL_RUN_ROOT).export_dir
                 / "density_case.npz")
        if cache.exists():
            z = np.load(cache)
            if np.allclose(z["freqs_hz"], FREQS_HZ):
                results[tag] = {"cells": int(z["cells"]),
                                "t_solve": float(z["t_solve"]),
                                "f": {0: z["f0"], 1: z["f1"]},
                                "dev": {0: float(z["dev"][0]),
                                        1: float(z["dev"][1])}}
                print(f"[{tag:>8}] reusing cached result "
                      f"(cells={int(z['cells']):,}, "
                      f"solve={float(z['t_solve']):.1f}s)", flush=True)
                continue
        plan = sw.make_plan(design, FREQS_HZ, LMAX,
                            monitor_factor=ref_plan.monitor_factor)
        plan.name = f"sw13p1_density_{tag}"
        plan.mesh_steps_per_wave = steps
        paths = RunPaths(plan.name, LOCAL_RUN_ROOT).ensure()

        t0 = time.time()
        h, freqs_proj, _ = build_project(
            session, plan, paths, ExtractionConfig(lmax=LMAX),
            solver_kwargs={"max_cpus": sw.MAX_CPUS,
                           "hardware_acceleration": sw.HARDWARE_ACCELERATION,
                           "mesh_adaption": adaptive})
        if adaptive:
            # relaxed reference settings (validated vs Mie on the sphere);
            # same caption replaces the tighter default block
            solvers.configure_adaptation(h, min_passes=2, max_passes=8,
                                         max_delta_s=0.02)
        t_build = time.time() - t0

        point_file = export.write_point_file(
            paths.export_dir / "points.txt", pts_proj)
        excitation.set_plane_wave(h, 0.0, 0.0, "theta")
        h.save()
        t0 = time.time()
        h.run_solver()
        t_solve = time.time() - t0
        cells = h.m3d.Mesh.GetNumberOfMeshCells()

        rec = {"cells": int(cells), "t_solve": t_solve, "t_build": t_build,
               "f": {}, "dev": {}}
        for jj, f_proj in enumerate(freqs_proj):
            E, ZH = export.get_EH_on_points(
                h, monitors.efield_tree_path(f_proj),
                monitors.hfield_tree_path(f_proj),
                paths.export_dir, f"d{tag}_f{jj}", point_file, pts_proj)
            k = 2 * np.pi * FREQS_HZ[jj] / C0
            a_meas, f_meas = vswf.separate_surface_field(
                E, ZH, LMAX, k, r_mon_m, th_q, ph_q, w_q)
            scale = (a_meas @ np.conj(a_nom)) / max(
                np.vdot(a_nom, a_nom).real, 1e-300)
            rec["dev"][jj] = float(np.linalg.norm(a_meas - scale * a_nom)
                                   / max(np.linalg.norm(a_meas), 1e-300))
            rec["f"][jj] = f_meas / scale       # source-calibrated column
        results[tag] = rec
        np.savez(paths.export_dir / "density_case.npz",
                 cells=cells, t_solve=t_solve,
                 f0=rec["f"][0], f1=rec["f"][1],
                 dev=[rec["dev"][0], rec["dev"][1]], freqs_hz=FREQS_HZ)
        print(f"[{tag:>8}] cells={cells:>9,}  solve={t_solve:7.1f}s  "
              f"inc_dev={rec['dev'][0]:.2e}/{rec['dev'][1]:.2e}  "
              f"|f|={np.linalg.norm(rec['f'][0]):.4e}/"
              f"{np.linalg.norm(rec['f'][1]):.4e}", flush=True)

    # ------------- convergence table -------------------------------------
    def rel(fa, fb, lcut=None):
        if lcut is not None:
            m = ns <= lcut
            fa, fb = fa[m], fb[m]
        return np.linalg.norm(fa - fb) / max(np.linalg.norm(fb), 1e-300)

    print("\n" + "=" * 74)
    print("scattered-column convergence (rel diff of f_meas), per frequency")
    print("=" * 74)
    pairs = [("15_6", "30_12"), ("30_12", "45_18"),
             ("15_6", "adaptive"), ("30_12", "adaptive"),
             ("45_18", "adaptive")]
    print(f"{'pair':>20} {'f[THz]':>8} {'all l<=9':>12} "
          f"{'l<=7 (phys@22.9)':>17}")
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        for jj, f_hz in enumerate(FREQS_HZ):
            print(f"{a + ' vs ' + b:>20} {f_hz/1e12:>8.2f} "
                  f"{rel(results[a]['f'][jj], results[b]['f'][jj]):>12.3e} "
                  f"{rel(results[a]['f'][jj], results[b]['f'][jj], L_PHYS_LOW):>17.3e}")
    print("\nreading: pick the cheapest fixed density whose diff vs the")
    print("next-denser mesh AND vs adaptive sits at the extraction noise")
    print("floor (~1e-2 relative on a strong column, per the cylinder run);")
    print("per-case cells/solve-time above give the full-run cost input.")


if __name__ == "__main__":
    main()
