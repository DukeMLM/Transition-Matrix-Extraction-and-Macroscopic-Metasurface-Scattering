r"""Extraction driven by a JSON configuration file.

Intended for users who are not developing the package: build the
geometry interactively in CST (geometry and materials only, centered at
the origin; see the template requirements in the manual), write a short
JSON configuration file, and run

    python -m cst_tmatrix.runner  my_extraction.json
    python -m cst_tmatrix.runner  my_extraction.json --dry-run

The runner applies the package defaults throughout: the monitor radius
is computed from the placement conditions, the mesh is the fixed
deterministic density, a wide frequency range is split into sub-ranges
each with its own monitor, the automatic checks are active, symmetry
reduction is applied if configured, the sub-ranges are merged into a
single file with a continuity test at their junction, and a sub-range
stopped by the incident-deviation check is rerun once with a larger
monitor.

Example configuration (all keys; see examples/example_extraction.json;
keys beginning with an underscore are ignored and may hold comments):

    {
      "name": "my_resonator",
      "template": "C:/work/my_resonator_geometry.cst",
      "r_circ_um": 3.6,
      "frequencies": {
        "min_thz": 10.0, "max_thz": 34.0, "step_thz": 1.0,
        "range_edges_thz": [10.0, 20.0, 34.0]
      },
      "extraction": {
        "lmax": 5,
        "symmetry_n_fold": 0,
        "symmetry_mirror_planes_deg": [0.0, 45.0, 90.0, 135.0]
      },
      "mesh": {"steps_per_wave": [20, 8]},
      "solver": {"max_cpus": 6, "hardware_acceleration": false}
    }

Machine-dependent settings (working directory, library directory, CPU
count, hardware acceleration) can also be placed once per machine in
settings.json at the repository root (see config.py); values given in
the extraction configuration take precedence.

One lmax should be used for a whole family of related designs: the
physics requires lmax to cover each design's measured multipole content
plus one order, and a shared value gives every stored T-matrix the same
dimension, which merging within a design and any joint use of the
family both require.  Choose the largest value needed in the family.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .config import ExtractionConfig, LIBRARY_ROOT, LOCAL_RUN_ROOT
from .pipeline import (C0, ExtractionCheckError, ScattererPlan,
                       check_monitor_lmax_margin, extract_tmatrix,
                       recommended_monitor_factor)
from .storage import load_tmatrix, merge_tmatrix_files


def _strip_comment_keys(obj):
    """Remove keys beginning with an underscore, recursively.  JSON has no
    comment syntax; such keys hold documentation in the example file."""
    if isinstance(obj, dict):
        return {k: _strip_comment_keys(v) for k, v in obj.items()
                if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_comment_keys(v) for v in obj]
    return obj


def load_config(path):
    """Parse and validate the JSON configuration into a plain dict with
    defaults applied.  A missing or inconsistent entry raises SystemExit
    with a message naming the entry, so that a configuration mistake is
    reported as a sentence rather than a traceback."""
    import json
    from .config import machine_settings
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"configuration file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = _strip_comment_keys(json.load(fh))
    except json.JSONDecodeError as e:
        raise SystemExit(f"configuration error: {path.name} is not valid "
                         f"JSON ({e})")

    def need(key, section=None):
        src = raw.get(section, {}) if section else raw
        if key not in src:
            where = f"'{section}'.{key}" if section else key
            raise SystemExit(f"configuration error: missing {where} "
                             f"in {path.name}")
        return src[key]

    machine = machine_settings()
    solver = raw.get("solver", {})
    freq = raw.get("frequencies", {})
    cfg = {
        "name": str(need("name")),
        "template": Path(str(need("template"))),
        "r_circ_m": float(need("r_circ_um")) * 1e-6,
        "length_unit": str(raw.get("length_unit", "um")),
        "freq_unit": str(raw.get("freq_unit", "THz")),
        "f_min_hz": float(need("min_thz", "frequencies")) * 1e12,
        "f_max_hz": float(need("max_thz", "frequencies")) * 1e12,
        "f_step_hz": float(need("step_thz", "frequencies")) * 1e12,
        "band_edges_hz": [float(x) * 1e12 for x in
                          freq.get("range_edges_thz",
                                   freq.get("band_edges_thz", []))],
        "lmax": int(need("lmax", "extraction")),
        "n_fold": int(raw.get("extraction", {}).get("symmetry_n_fold", 0)),
        "mirrors_deg": [float(x) for x in raw.get("extraction", {}).get(
            "symmetry_mirror_planes_deg", [])],
        "mesh_steps": tuple(raw.get("mesh", {}).get("steps_per_wave",
                                                    (20, 8))),
        "mesh_adaptive": bool(raw.get("mesh", {}).get("adaptive", False)),
        # solver resources are properties of the machine, not of the
        # physics problem: default from the per-machine settings.json,
        # overridable per configuration
        "max_cpus": solver.get("max_cpus", machine.get("max_cpus")),
        "hw_accel": solver.get("hardware_acceleration",
                               machine.get("hardware_acceleration", False)),
        "run_root": Path(raw["paths"]["run_root"]) if
        raw.get("paths", {}).get("run_root") else LOCAL_RUN_ROOT,
        "library": Path(raw["paths"]["library"]) if
        raw.get("paths", {}).get("library") else LIBRARY_ROOT,
    }
    if not cfg["template"].exists():
        raise SystemExit(f"configuration error: template project not "
                         f"found: {cfg['template']}")
    if cfg["r_circ_m"] <= 0:
        raise SystemExit("configuration error: r_circ_um must be positive "
                         "(radius of the smallest origin-centered sphere "
                         "enclosing the geometry).")
    if not cfg["f_min_hz"] < cfg["f_max_hz"]:
        raise SystemExit("configuration error: 'frequencies'.min_thz must "
                         "be below max_thz.")
    if cfg["n_fold"] < 0 or cfg["n_fold"] == 1:
        raise SystemExit("configuration error: symmetry_n_fold must be 0 "
                         "(off) or >= 2.")
    return cfg


def frequency_bands(cfg):
    """The frequency grid, split by band edges.  Returns
    [(tag, freqs_hz_array), ...]; a single unlabeled band without edges."""
    n = int(round((cfg["f_max_hz"] - cfg["f_min_hz"]) / cfg["f_step_hz"])) + 1
    freqs = cfg["f_min_hz"] + cfg["f_step_hz"] * np.arange(n)
    edges = cfg["band_edges_hz"]
    if len(edges) < 2:
        return [("", freqs)]
    bands = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        m = ((freqs > lo) if k else (freqs >= lo)) & (freqs <= hi)
        if np.any(m):
            bands.append((f"__{lo / 1e12:g}to{hi / 1e12:g}THz", freqs[m]))
    if not bands:
        raise SystemExit("configuration error: band_edges_thz selects no "
                         "frequencies from the grid.")
    return bands


def extraction_config(cfg):
    if cfg["n_fold"]:
        return ExtractionConfig(
            lmax=cfg["lmax"], symmetry_n_fold=cfg["n_fold"],
            symmetry_mirror_phi0_deg=tuple(cfg["mirrors_deg"]) or None)
    return ExtractionConfig(lmax=cfg["lmax"])


def paper_preflight(cfg, bands, nearfield_margin=1.0):
    """Verify the monitor-placement criteria for every band before CST is
    touched; returns [(tag, monitor_factor), ...] or raises SystemExit."""
    out = []
    print(f"paper preflight for '{cfg['name']}' "
          f"(r_circ = {cfg['r_circ_m'] * 1e6:.3f} um, lmax = {cfg['lmax']}):")
    for tag, fb in bands:
        mf = recommended_monitor_factor(cfg["r_circ_m"], fb, cfg["lmax"],
                                        nearfield_margin=nearfield_margin)
        k_min = 2 * np.pi * float(np.min(fb)) / C0
        msg = check_monitor_lmax_margin(k_min, mf * cfg["r_circ_m"],
                                        cfg["lmax"],
                                        r_circ_m=cfg["r_circ_m"])
        label = tag.lstrip("_") or "full band"
        print(f"  {label:>14}: {len(fb)} freqs, monitor_factor {mf:6.3f} "
              f"(r_mon {mf * cfg['r_circ_m'] * 1e6:6.2f} um, "
              f"kr(fmin) {k_min * mf * cfg['r_circ_m']:5.2f})  "
              f"{'OK' if msg is None else 'FAIL'}")
        if msg is not None:
            raise SystemExit(f"paper preflight FAILED for band {label}: "
                             f"{msg}")
        out.append((tag, mf))
    return out


def run_band(cfg, tag, freqs_b, monitor_factor):
    from .cst_driver.session import CSTSession
    plan = ScattererPlan(
        name=cfg["name"],
        template=str(cfg["template"]),
        r_circ_m=cfg["r_circ_m"],
        freqs_hz=freqs_b,
        lmax=cfg["lmax"],
        monitor_factor=monitor_factor,
        length_unit=cfg["length_unit"],
        freq_unit=cfg["freq_unit"],
        mesh_steps_per_wave=tuple(cfg["mesh_steps"]),
        metadata={"configured_via": "cst_tmatrix.runner"})
    session = CSTSession()          # fresh connection per band
    return extract_tmatrix(
        session, plan, extraction_config(cfg),
        run_root=cfg["run_root"],
        solver_kwargs={"max_cpus": cfg["max_cpus"],
                       "hardware_acceleration": cfg["hw_accel"],
                       "mesh_adaption": cfg["mesh_adaptive"]},
        library_path=cfg["library"] / f"{cfg['name']}{tag}.tmat.h5")


def band_boundary_report(path, edges_hz):
    d = load_tmatrix(path)
    T, f = d["tmatrix"], d["frequencies"]
    steps = [np.linalg.norm(T[j + 1] - T[j])
             / max(0.5 * (np.linalg.norm(T[j]) + np.linalg.norm(T[j + 1])),
                   1e-300)
             for j in range(len(f) - 1)]
    for e in edges_hz[1:-1]:
        idx = [j for j in range(len(f) - 1)
               if abs(f[j] - e) < 0.51 * (f[1] - f[0])]
        for j in idx:
            if 0 < j < len(steps) - 1:
                ok = steps[j] < 2.0 * max(steps[j - 1], steps[j + 1])
                print(f"  band boundary at {f[j] / 1e12:g} THz: step {steps[j]:.4f} "
                      f"vs neighbors {steps[j - 1]:.4f}/{steps[j + 1]:.4f} "
                      f"-> {'OK' if ok else 'step exceeds twice the neighboring steps -- inspect'}")


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    args = [a for a in argv[1:] if not a.startswith("-")]
    if not args:
        raise SystemExit("usage: python -m cst_tmatrix.runner "
                         "<config.json> [--dry-run]")
    dry = "--dry-run" in argv
    cfg = load_config(args[0])
    bands = frequency_bands(cfg)
    mfs = paper_preflight(cfg, bands)
    if dry:
        print("--dry-run: configuration valid, all preflights pass; "
              "no CST touched.")
        return

    band_files, results = [], []
    t0_all = time.time()
    for (tag, freqs_b), (_tag2, mf) in zip(bands, mfs):
        label = tag.lstrip("_") or "full band"
        t0 = time.time()
        try:
            _T, _diags, out = run_band(cfg, tag, freqs_b, mf)
            band_files.append(Path(out))
            results.append((label, "ok", time.time() - t0))
        except ExtractionCheckError as e:
            if "incident-deviation" in str(e):
                # one automatic retry with the monitor pushed one further
                # Wiscombe order out -- the recovery that fixed the only
                # such failure observed in production
                mf2 = recommended_monitor_factor(
                    cfg["r_circ_m"], freqs_b, cfg["lmax"],
                    nearfield_margin=2.0)
                print(f"[runner] {label}: stopped by check -- retrying once "
                      f"with enlarged monitor (factor {mf2:.3f})",
                      flush=True)
                try:
                    _T, _diags, out = run_band(cfg, tag, freqs_b, mf2)
                    band_files.append(Path(out))
                    results.append((label, "ok (auto-retry)",
                                    time.time() - t0))
                except Exception as e2:                   # noqa: BLE001
                    results.append((label, f"FAILED after retry: {e2}",
                                    time.time() - t0))
            else:
                results.append((label, f"stopped by check: {e}", time.time() - t0))
        except KeyboardInterrupt:
            raise
        except Exception as e:                            # noqa: BLE001
            results.append((label, f"FAILED: {e}", time.time() - t0))

    merged = None
    if len(bands) > 1 and len(band_files) == len(bands):
        lo = cfg["band_edges_hz"][0] / 1e12
        hi = cfg["band_edges_hz"][-1] / 1e12
        merged = cfg["library"] / f"{cfg['name']}_{lo:g}to{hi:g}THz.tmat.h5"
        info = merge_tmatrix_files(merged, band_files, name=cfg["name"])
        print(f"merged {info['n_bands']} bands -> {merged} "
              f"({info['n_freq']} frequencies)")
        band_boundary_report(merged, cfg["band_edges_hz"])

    print(f"\nsummary ({(time.time() - t0_all) / 3600:.2f} h):")
    for label, status, dt in results:
        print(f"  {label:>14}: {str(status).splitlines()[0][:90]} "
              f"({dt / 3600:.2f} h)")
    if merged:
        print(f"library entry: {merged}")


if __name__ == "__main__":
    main()
