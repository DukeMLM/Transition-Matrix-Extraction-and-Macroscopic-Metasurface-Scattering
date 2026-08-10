r"""Fast calibration test before committing to a full extraction: times TWO
illuminations of the real scaled spoke-wheel geometry PER BAND, at the real
lmax/monitor_factor/mesh settings from extract_spoke_wheel_scaled.py,
sampling that band's full frequency list per solve (matching the actual
per-illumination cost of a real extraction).

MESH MODEL (2026-08-06): CST never carries an adapted mesh across solver
runs, so there is no "one-time adaptation + cheap steady state".  The
pipeline uses a DETERMINISTIC non-adaptive mesh (sw.MESH_STEPS_PER_WAVE,
density-checked on this geometry in examples/density_check_spoke_wheel.py),
under which every illumination costs the same.  Two illuminations per band
VERIFY that uniformity (a large gap now flags a problem, not a warm-up).

BAND SPLIT + lmax (2026-08-06): the single-band 10-34 THz configuration
coupled the 10 THz monitor radius (52.5 um at lmax=9) to the 34 THz mesh
density and exhausted 32 GB on the remote machine ("Could not compute
preconditioner").  sw.BAND_EDGES_THZ splits the band so each segment gets
its own (smaller) monitor and its own mesh; sw.LMAX=5 is the evidence-based
truncation (measured content ends at l~4; the old lmax=9 tail was
conditioning noise), which further shrinks the monitor.

Prints per band: mesh cells, per-illumination times, incident_deviation_max,
and wall-clock projections for the unreduced AND C4v-reduced solve counts;
then the grand totals.

Run on the CST machine:
    python calibrate_spoke_wheel_scaled.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract_spoke_wheel_scaled as sw  # noqa: E402
from cst_tmatrix import vswf  # noqa: E402
from cst_tmatrix.config import RunPaths, LOCAL_RUN_ROOT, ExtractionConfig  # noqa: E402
from cst_tmatrix.pipeline import build_project, C0  # noqa: E402
from cst_tmatrix.quadrature import (quadrature_for_monitor, unit_vectors,  # noqa: E402
                                    illumination_directions)
from cst_tmatrix.postprocess.symmetry import (  # noqa: E402
    point_group_operations, reduced_illumination_directions)
from cst_tmatrix.cst_driver import CSTSession  # noqa: E402
from cst_tmatrix.cst_driver import monitors, excitation, export  # noqa: E402

N_ILLUM_TO_TIME = 2


def calibrate_band(session, design, band_tag, freqs_hz, lmax, ops):
    label = band_tag.lstrip("_") or "full band"
    N = vswf.n_modes(lmax)
    n_dirs = int(np.ceil(1.5 * N / 2))
    n_illum_total = 2 * n_dirs
    plan = sw.make_plan(design, freqs_hz, lmax, sw.MONITOR_FACTOR)
    plan.name = f"{plan.name}_calib{band_tag}"
    print(f"\n--- {label}: {len(freqs_hz)} freqs "
          f"{freqs_hz.min()/1e12:g}-{freqs_hz.max()/1e12:g} THz, "
          f"monitor_factor={plan.monitor_factor:.2f} "
          f"(r_mon={plan.monitor_factor*plan.r_circ_m*1e6:.1f} um), "
          f"mesh {plan.mesh_steps_per_wave}, adaption {sw.MESH_ADAPTION}",
          flush=True)

    paths = RunPaths(plan.name, LOCAL_RUN_ROOT).ensure()
    t0 = time.time()
    h, freqs_proj, r_mon_proj = build_project(
        session, plan, paths, ExtractionConfig(lmax=lmax),
        solver_kwargs={"max_cpus": sw.MAX_CPUS,
                       "hardware_acceleration": sw.HARDWARE_ACCELERATION,
                       "mesh_adaption": sw.MESH_ADAPTION})
    print(f"project build: {time.time()-t0:.1f}s", flush=True)

    k_max = 2 * np.pi * np.max(freqs_hz) / C0
    r_mon_m = plan.monitor_factor * plan.r_circ_m
    th_q, ph_q, w_q, qshape = quadrature_for_monitor(lmax, k_max, r_mon_m)
    pts_proj = (r_mon_m * unit_vectors(th_q, ph_q)) / 1e-6
    point_file = export.write_point_file(paths.export_dir / "points.txt",
                                         pts_proj)

    th_i, ph_i = illumination_directions("fibonacci", n_dirs)
    idx_red, n_exp = reduced_illumination_directions(th_i, ph_i, ops,
                                                     n_illum_total)
    n_red = 2 * len(idx_red)
    # time the directions the reduced run would actually solve
    timed = [(float(th_i[i]), float(ph_i[i]), pol)
             for i in idx_red[:1] for pol in ("theta", "phi")]
    if len(timed) < N_ILLUM_TO_TIME and len(idx_red) > 1:
        timed += [(float(th_i[idx_red[1]]), float(ph_i[idx_red[1]]), "theta")]
    timed = timed[:N_ILLUM_TO_TIME]

    times, worst_dev = [], 0.0
    for kk, (ti, pi_, pol) in enumerate(timed):
        t0 = time.time()
        excitation.set_plane_wave(h, ti, pi_, pol)
        h.save()
        h.run_solver()
        dt_solve = time.time() - t0
        if kk == 0:
            print(f"mesh cells: {h.m3d.Mesh.GetNumberOfMeshCells():,}",
                  flush=True)
        devs = []
        for jj, f_proj in enumerate(freqs_proj):
            E, ZH = export.get_EH_on_points(
                h, monitors.efield_tree_path(f_proj),
                monitors.hfield_tree_path(f_proj),
                paths.export_dir, f"calib_i{kk}_f{jj}", point_file, pts_proj)
            k = 2 * np.pi * freqs_hz[jj] / C0
            a_meas, _f_meas = vswf.separate_surface_field(
                E, ZH, lmax, k, r_mon_m, th_q, ph_q, w_q)
            a_nom = vswf.plane_wave_coefficients(ti, pi_, pol, lmax)
            scale = (a_meas @ np.conj(a_nom)) / max(
                np.vdot(a_nom, a_nom).real, 1e-300)
            devs.append(np.linalg.norm(a_meas - scale * a_nom)
                        / max(np.linalg.norm(a_meas), 1e-300))
        dt_total = time.time() - t0
        times.append(dt_total)
        worst_dev = max(worst_dev, max(devs))
        print(f"illum {kk} (th={ti:.3f} ph={pi_:.3f} pol={pol}): "
              f"solve={dt_solve:.1f}s, total(incl. export)={dt_total:.1f}s, "
              f"worst incident_deviation = {max(devs):.3e}", flush=True)

    mean_t = float(np.mean(times))
    spread = (abs(times[1] - times[0]) / max(mean_t, 1e-9)
              if len(times) > 1 else 0.0)
    print(f"per-illumination mean {mean_t/60:.1f} min (spread {spread:.0%} "
          f"-- expect small; no mesh warm-up exists)")
    print(f"worst incident_deviation: {worst_dev:.3e}  "
          f"({'FAIL (>1e-2)' if worst_dev > 1e-2 else 'OK'})")
    print(f"projection, unreduced ({n_illum_total} solves): "
          f"{n_illum_total*mean_t/3600:.1f} h")
    print(f"projection, C4v-reduced ({n_red} solves -> {n_exp} columns): "
          f"{n_red*mean_t/3600:.2f} h")
    return {"label": label, "mean_t": mean_t, "n_full": n_illum_total,
            "n_red": n_red, "worst_dev": worst_dev}


def main():
    designs = sw.load_designs(sw.DESIGN_FILE)
    design = min(designs, key=lambda d: abs(d["wl"] - sw.PICK_WL))
    _wl, freqs_all = sw.frequency_plan(freq_min_thz=sw.FREQ_MIN_THZ,
                                       freq_max_thz=sw.FREQ_MAX_THZ,
                                       freq_step_thz=sw.FREQ_STEP_THZ)
    bands = sw.split_bands(freqs_all)
    lmax = sw.LMAX
    print(f"design: wl={design['wl']:.4f} um | lmax={lmax} "
          f"(N={vswf.n_modes(lmax)} modes) | "
          f"{len(bands)} band(s) | C4v={'on' if sw.USE_C4V else 'off'} | "
          f"solver: {sw.MAX_CPUS} CPUs, hw accel {sw.HARDWARE_ACCELERATION}")

    ops = point_group_operations(lmax, n_fold=4,
                                 mirror_phi0_deg=[0., 45., 90., 135.])
    session = CSTSession()
    recs = [calibrate_band(session, design, tag, fb, lmax, ops)
            for tag, fb in bands]

    print("\n" + "=" * 66)
    print("grand totals for one design, all bands:")
    tot_full = sum(r["n_full"] * r["mean_t"] for r in recs)
    tot_red = sum(r["n_red"] * r["mean_t"] for r in recs)
    print(f"  unreduced:   {tot_full/3600:.1f} h "
          f"({sum(r['n_full'] for r in recs)} solves)")
    print(f"  C4v-reduced: {tot_red/3600:.2f} h "
          f"({sum(r['n_red'] for r in recs)} solves)")
    print("=" * 66)


if __name__ == "__main__":
    main()
