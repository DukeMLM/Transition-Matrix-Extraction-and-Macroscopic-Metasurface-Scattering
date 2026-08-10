r"""Unattended sequential T-matrix extraction of several spoke-wheel designs.

Runs the designs in DESIGN_WLS one after another (each starts the moment
the previous finishes), each split into the bands of
extract_spoke_wheel_scaled.BAND_EDGES_THZ, with all of that script's
settings (lmax, mesh, C4v reduction, auto monitor factor, per-run
log.txt).  After both bands of a design succeed, they are merged into a
single {name}_10to34THz.tmat.h5 and the continuity of T across the band
boundary is checked
automatically.

The automatic checks of the pipeline are active: the monitor-placement
conditions are verified before each band starts (kr >= lmax+2 for the
conditioning of the incident/scattered decomposition, and kr >=
Wiscombe(k r_circ)+1 so the monitor lies outside the reactive near
field); the mesh cell count and estimated memory are compared with the
available memory after meshing and before any solve; and the incident
deviation of every completed solve is tested at once, so unusable field
data stops a band within one solve instead of after hours, without
caching the affected column.  A stopped or failed band is logged and
the batch continues with the remaining bands and designs; completed
illuminations are cached, so a corrected rerun of a stopped band
continues from its last valid solve.  The summary at the end lists the
outcome per band.

All designs run at the same lmax (extract_spoke_wheel_scaled.LMAX), so
every stored T-matrix has the same mode dimension.  For lmax=5 across
this design list: the measured per-order content of the 17.3 um
extraction (size parameter 2.56 at the band top, compared with 2.82
for the largest design) has 2.1% of the matrix norm at l=4 and 1.6% at
l=5, the latter at the level of the extraction error; the per-band
first-solve report re-measures this for each new design.

Run on the CST machine:
    python batch_extract_spoke_wheel.py            # full batch
    python batch_extract_spoke_wheel.py --dry-run  # plan + preflight math only
"""
import datetime
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract_spoke_wheel_scaled as sw  # noqa: E402
import merge_band_tmatrices  # noqa: E402
from cst_tmatrix.pipeline import (recommended_monitor_factor,  # noqa: E402
                                  check_monitor_lmax_margin, ExtractionCheckError, C0)
from cst_tmatrix.storage import load_tmatrix  # noqa: E402

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
DESIGN_WLS = [10.3,9.7]      # center wavelengths (um), run in this order
MERGED_SUFFIX = "_10to34THz"   # merged output: {name}{MERGED_SUFFIX}.tmat.h5
# Automatic retry (2026-08-08; a band stopped by the incident-deviation
# check during an unattended run had left the machine idle for hours):
# such a band is rerun once with the monitor radius one Wiscombe order
# larger (nearfield_margin=2 instead of 1).  Nothing is cached when a
# solve is stopped, so the rerun starts from unaffected data; a second
# stop is left for the operator.
AUTO_RETRY_LARGER_MONITOR = True


def _design_name(design):
    """Exactly make_plan's naming formula (kept in sync by an assertion in
    the dry run)."""
    wl_tag = f"{design['wl']:.2f}".replace(".", "p")
    return f"saw_gold_wl{wl_tag}um"


def paper_preflight(design, bands, lmax):
    """No-CST verification of the monitor criteria for every band -- the
    same conditions the pipeline verifies at extraction start, printed
    here before any run begins."""
    import math
    r_circ = math.hypot(design["r"], sw.T_MM_UM / 2) * 1e-6
    print(f"  r_circ = {r_circ * 1e6:.3f} um")
    ok = True
    for tag, fb in bands:
        mf = recommended_monitor_factor(r_circ, fb, lmax)
        k_min = 2 * np.pi * float(np.min(fb)) / C0
        msg = check_monitor_lmax_margin(k_min, mf * r_circ, lmax,
                                        r_circ_m=r_circ)
        status = "OK" if msg is None else f"FAIL: {msg}"
        print(f"  band {tag.lstrip('_'):>10}: mf={mf:6.3f}  "
              f"r_mon={mf * r_circ * 1e6:6.2f} um  "
              f"kr(fmin)={k_min * mf * r_circ:5.2f}  {status}")
        ok = ok and msg is None
    return ok


def boundary_check(path):
    d = load_tmatrix(path)
    T, f = d["tmatrix"], d["frequencies"]
    steps = []
    b_idx = None
    for j in range(len(f) - 1):
        dT = np.linalg.norm(T[j + 1] - T[j])
        m = 0.5 * (np.linalg.norm(T[j]) + np.linalg.norm(T[j + 1]))
        steps.append(dT / max(m, 1e-300))
        if abs(f[j] / 1e12 - 20.0) < 0.01:
            b_idx = j
    if b_idx is None or b_idx in (0, len(steps) - 1):
        print("  band-boundary check: boundary not interior to the grid -- skipped")
        return True
    step, lo, hi = steps[b_idx], steps[b_idx - 1], steps[b_idx + 1]
    neighborhood = max(lo, hi)
    ok = step < 2.0 * neighborhood
    print(f"  band-boundary check (20->21 THz): step {step:.4f} vs neighbors "
          f"{lo:.4f}/{hi:.4f} -> {'OK' if ok else 'step exceeds twice the '
          'neighboring steps; different mesh and monitor per side -- inspect'}")
    return ok


def main():
    dry = "--dry-run" in sys.argv
    designs = sw.load_designs(sw.DESIGN_FILE)
    _wl, freqs_all = sw.frequency_plan(freq_min_thz=sw.FREQ_MIN_THZ,
                                       freq_max_thz=sw.FREQ_MAX_THZ,
                                       freq_step_thz=sw.FREQ_STEP_THZ)
    bands = sw.split_bands(freqs_all)
    args = types.SimpleNamespace(lmax=sw.LMAX, monitor_factor=sw.MONITOR_FACTOR,
                                 max_cpus=sw.MAX_CPUS,
                                 hw_accel=sw.HARDWARE_ACCELERATION)
    print(f"batch: {len(DESIGN_WLS)} designs {DESIGN_WLS} um, "
          f"{len(bands)} bands each, lmax={sw.LMAX}, "
          f"C4v={'on' if sw.USE_C4V else 'off'}, "
          f"mesh {sw.MESH_STEPS_PER_WAVE} adaption={sw.MESH_ADAPTION}")

    selected = []
    for wl in DESIGN_WLS:
        d = min(designs, key=lambda x: abs(x["wl"] - wl))
        selected.append(d)
        print(f"\ndesign wl={d['wl']:.4f} um "
              f"(requested {wl}):")
        # keep the local naming helper in lockstep with make_plan
        assert sw.make_plan(d, freqs_all, sw.LMAX, 1.0).name == \
            _design_name(d), "naming drift between batch and make_plan"
        if not paper_preflight(d, bands, sw.LMAX):
            raise SystemExit(f"paper preflight FAILED for wl={d['wl']} -- "
                             f"fix the configuration before launching.")
    if dry:
        print("\n--dry-run: all paper preflights pass; no CST touched.")
        return

    from cst_tmatrix.cst_driver import CSTSession
    results = []
    t_batch0 = time.time()
    for d in selected:
        band_files, design_ok = [], True
        for band_tag, freqs_b in bands:
            wl_b = C0 / freqs_b / 1e-6
            label = f"wl={d['wl']:.4f} {band_tag.lstrip('_')}"
            t0 = time.time()
            try:
                # fresh connection per band: survives a CST frontend crash
                # in an earlier band (connect_to_any_or_new respawns)
                session = CSTSession()
                sw.extract_one(session, d, args, wl_b, freqs_b, band_tag)
                dt = time.time() - t0
                results.append((label, "ok", dt))
                band_files.append(sw.LIBRARY_DIR /
                                  f"{_design_name(d)}{band_tag}.tmat.h5")
                print(f"[batch] {label}: ok ({dt / 3600:.2f} h)", flush=True)
            except ExtractionCheckError as e:
                retried = False
                if (AUTO_RETRY_LARGER_MONITOR
                        and "incident-deviation" in str(e)):
                    import math
                    r_circ = math.hypot(d["r"], sw.T_MM_UM / 2) * 1e-6
                    mf2 = recommended_monitor_factor(
                        r_circ, freqs_b, sw.LMAX, nearfield_margin=2.0)
                    print(f"[batch] {label}: stopped by check -- retrying "
                          f"ONCE with enlarged monitor "
                          f"(factor {mf2:.3f}, +1 Wiscombe order)",
                          flush=True)
                    args2 = types.SimpleNamespace(
                        lmax=sw.LMAX, monitor_factor=mf2,
                        max_cpus=sw.MAX_CPUS,
                        hw_accel=sw.HARDWARE_ACCELERATION)
                    t0 = time.time()
                    try:
                        session = CSTSession()
                        sw.extract_one(session, d, args2, wl_b, freqs_b,
                                       band_tag)
                        dt = time.time() - t0
                        results.append((label, "ok (auto-retry, enlarged "
                                        "monitor)", dt))
                        band_files.append(
                            sw.LIBRARY_DIR /
                            f"{_design_name(d)}{band_tag}.tmat.h5")
                        print(f"[batch] {label}: ok after auto-retry "
                              f"({dt / 3600:.2f} h)", flush=True)
                        retried = True
                    except Exception as e2:               # noqa: BLE001
                        design_ok = False
                        results.append((label, f"FAILED after auto-retry: "
                                        f"{e2}", time.time() - t0))
                        print(f"[batch] {label}: FAILED after auto-retry "
                              f"-- {e2}", flush=True)
                        retried = True
                if not retried:
                    design_ok = False
                    results.append((label, f"stopped by check: {e}", time.time() - t0))
                    print(f"[batch] {label}: stopped by check -- {e}",
                          flush=True)
            except KeyboardInterrupt:
                raise
            except Exception as e:                        # noqa: BLE001
                design_ok = False
                results.append((label, f"FAILED: {e}", time.time() - t0))
                print(f"[batch] {label}: FAILED -- {e}", flush=True)

        if design_ok and len(band_files) == len(bands):
            merged = (sw.LIBRARY_DIR /
                      f"{_design_name(d)}{MERGED_SUFFIX}.tmat.h5")
            try:
                merge_band_tmatrices.main(
                    ["batch", str(merged)] + [str(p) for p in band_files])
                boundary_check(merged)
                results.append((f"wl={d['wl']:.4f} merge", "ok", 0.0))
            except SystemExit as e:
                results.append((f"wl={d['wl']:.4f} merge", f"FAILED: {e}",
                                0.0))
            except Exception as e:                        # noqa: BLE001
                results.append((f"wl={d['wl']:.4f} merge", f"FAILED: {e}",
                                0.0))

    print(f"\n{'=' * 70}\nbatch summary "
          f"({(time.time() - t_batch0) / 3600:.1f} h total, "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}):")
    for label, status, dt in results:
        line = f"  {label:>28}  {str(status).splitlines()[0][:80]}"
        if dt:
            line += f"  ({dt / 3600:.2f} h)"
        print(line)
    log = sw.LIBRARY_DIR / "batch_log.txt"
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n==== batch {datetime.datetime.now():%Y-%m-%d %H:%M} "
                 f"====\n")
        for label, status, dt in results:
            fh.write(f"{label}\t{str(status).splitlines()[0]}\t"
                     f"{dt / 3600:.2f} h\n")
    print(f"summary appended to {log}")


if __name__ == "__main__":
    main()
