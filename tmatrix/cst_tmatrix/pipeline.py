"""End-to-end T-matrix extraction pipeline.

    CST (per illumination: plane wave -> FD solve -> E/H on monitor sphere)
    + Python (separate incident/scattered VSWFs -> least-squares T -> HDF5).

Robustness features:
- every illumination's projected coefficients are cached to .npz in the
  local run directory; a crash or restart resumes where it left off;
- incident coefficients are MEASURED from the same exported fields
  (self-calibrating: CST amplitude/phase conventions cancel in T);
- the measured incident coefficients are cross-checked against the analytic
  plane-wave expansion — large deviation flags a setup problem;
- exported coordinates are verified against the requested quadrature points;
- per-frequency diagnostics (residual, cond, passivity) are stored with T.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import vswf
from .config import ExtractionConfig, RunPaths, grid_tag
from .mie import wiscombe_lmax
from .quadrature import (illumination_directions, quadrature_for_monitor,
                         unit_vectors)
from .tmatrix_solve import solve_tmatrix, passivity_check, reciprocity_check
from .storage import save_tmatrix, extraction_matches

C0 = 299792458.0


@dataclass
class ScattererPlan:
    """Everything the pipeline needs to know about one library entry.

    Exactly one of `build` and `template` must be given:

    build : callable(ProjectHandle) that constructs materials and geometry
        through the builder module.  It must not define units, frequency
        range, boundaries, or monitors — the pipeline owns those.
    template : path to a manually prepared CST project containing only the
        geometry and materials, centered at the origin, in the declared
        length/frequency units.  The file is copied into the run directory
        and verified; the original is never modified (see
        cst_driver.template).

    r_circ_m : radius (meters) of the minimum sphere circumscribing the
        scatterer, centered at the origin.
    """
    name: str
    build: "callable | None" = None
    r_circ_m: float = 0.0
    freqs_hz: np.ndarray = None
    template: "str | None" = None
    lmax: int | None = None                # default: Wiscombe at fmax
    monitor_factor: float = 1.5            # r_monitor = factor * r_circ
    length_unit: str = "um"
    freq_unit: str = "THz"
    template_strict: bool = True
    # (near, far) tetrahedral mesh density in steps per wavelength.
    # Effective only because set_tet_mesh pins CellsPerWavelengthPolicy;
    # raise it (e.g. (30, 12)) when running with mesh_adaption=False, in
    # which case the fixed mesh alone determines the accuracy.
    mesh_steps_per_wave: tuple = (15, 6)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if (self.build is None) == (self.template is None):
            raise ValueError(
                "ScattererPlan requires exactly one of `build` (callable) "
                "or `template` (path to a .cst file).")
        if self.r_circ_m <= 0:
            raise ValueError("r_circ_m (circumscribing radius in meters) "
                             "must be positive.")


_FREQ_SCALE = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
_LEN_SCALE = {"nm": 1e-9, "um": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0}


def plan_lmax(plan: ScattererPlan) -> int:
    if plan.lmax is not None:
        return plan.lmax
    k_max = 2 * np.pi * np.max(plan.freqs_hz) / C0
    return wiscombe_lmax(k_max * plan.r_circ_m)


def physical_lmax_per_frequency(freqs_hz, r_circ_m: float) -> np.ndarray:
    """Wiscombe truncation order at EACH frequency (not just fmax).

    plan_lmax sizes the extraction at fmax, which is correct for the
    projection basis but means that at the low end of a wide frequency
    range the top
    orders are physically unpopulated: an extraction over 10-34 THz at
    lmax = 9 carries orders at 10 THz that the physics supports only
    above ~20 THz.  Such orders contain the absolute extraction error
    and no signal.  They do not corrupt the physical orders (each (n, m)
    pair is determined by an independent 2x2 system), but they inflate
    norm-based diagnostics and enter downstream sums over all modes.
    """
    freqs_hz = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
    k = 2 * np.pi * freqs_hz / C0
    return np.array([wiscombe_lmax(x) for x in k * r_circ_m], dtype=int)


def noise_order_content(T: np.ndarray, lmax: int, freqs_hz,
                        r_circ_m: float) -> np.ndarray:
    """Per frequency: fraction of ||T(f)||_F carried by matrix entries whose
    row OR column order exceeds the Wiscombe order at that frequency.

    Near zero at the top of the frequency range by construction
    (every order is physical there).  An O(1) value at the bottom of a
    wide range is expected for undetermined orders and is not by
    itself an error; a large value where l_phys(f) >= lmax does
    indicate a problem.
    Report alongside residual/reciprocity; it also indicates how far T(f)
    is determined order-by-order for downstream use.
    """
    T = np.asarray(T)
    if T.ndim == 2:
        T = T[None, :, :]
    freqs_hz = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
    _, ns, _ = vswf.mode_list(lmax)
    l_phys = physical_lmax_per_frequency(freqs_hz, r_circ_m)
    out = np.zeros(len(freqs_hz))
    for j, lp in enumerate(l_phys):
        noisy = (ns[:, None] > lp) | (ns[None, :] > lp)
        tot = np.linalg.norm(T[j])
        out[j] = np.linalg.norm(T[j][noisy]) / max(tot, 1e-300)
    return out


def mask_noise_orders(T: np.ndarray, lmax: int, freqs_hz, r_circ_m: float,
                      margin: int = 1) -> np.ndarray:
    """Zero the rows/columns of T(f) above the per-frequency Wiscombe order
    (+ `margin`).  Returns a copy; the stored library entry keeps the raw T
    -- masking is a downstream choice, applied where a sum over all modes
    (passivity, forward-scattering S-parameters, Foldy-Lax) would otherwise
    integrate noise from orders the physics cannot populate at f.
    """
    T = np.array(T, copy=True)
    squeeze = T.ndim == 2
    if squeeze:
        T = T[None, :, :]
    freqs_hz = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
    _, ns, _ = vswf.mode_list(lmax)
    l_phys = physical_lmax_per_frequency(freqs_hz, r_circ_m) + margin
    for j, lp in enumerate(l_phys):
        noisy = (ns[:, None] > lp) | (ns[None, :] > lp)
        T[j][noisy] = 0.0
    return T[0] if squeeze else T


def recommended_monitor_factor(r_circ_m: float, freqs_hz, lmax: int,
                               margin: float = 2.0,
                               nearfield_margin: float = 1.0) -> float:
    """Smallest monitor_factor satisfying BOTH monitor-placement criteria at
    every frequency in `freqs_hz`.

    1. Truncation conditioning: k * r_mon >= lmax + margin, so the
       regular/outgoing separation is well conditioned at l = lmax
       (see check_monitor_lmax_margin).  Binding at the lowest frequency.

    2. Scatterer near-field clearance: k * r_mon >=
       wiscombe_lmax(k * r_circ) + nearfield_margin AT EACH frequency.
       The scatterer radiates multipoles up to its full Wiscombe order
       REGARDLESS of the truncation lmax; orders with l > k*r_mon are
       reactive at the monitor radius, with steeply growing amplitude that
       enlarges the total field there -- the solver's relative error on
       that enlarged field becomes absolute error in every projected
       coefficient.  In five recorded extractions, those with
       k*r_mon - n(x) >= +0.67 (n the continuous Wiscombe quantity)
       satisfied the diagnostics, while those at +0.44 and -0.25 gave
       incident deviations of 0.11-0.15, sv(S) up to 1.37, and an
       l=lmax block an order of magnitude too large, largest at the
       low-frequency end where the reactive content is strongest;
       requiring w(x)+1 places every new extraction above all
       satisfactory observations.

    Each design has a different r_circ_m, so this must be recomputed per
    design AND per band -- never reuse a monitor_factor across designs.
    """
    freqs = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
    k = 2 * np.pi * freqs / C0
    req_cond = (lmax + margin) / np.min(k)
    req_near = float(np.max(
        [(wiscombe_lmax(kk * r_circ_m) + nearfield_margin) / kk for kk in k]))
    # tiny relative safety pad: the round trip through check_monitor_lmax_margin
    # (mf -> r_mon -> kr_mon) is not perfectly floating-point-invertible, so
    # without this the result can sit fractionally below its own threshold
    return (1.0 + 1e-9) * max(req_cond, req_near) / r_circ_m


def check_monitor_lmax_margin(k_min: float, r_mon_m: float, lmax: int,
                              margin: float = 2.0,
                              r_circ_m: float | None = None,
                              nearfield_margin: float = 1.0) -> str | None:
    """None if k_min*r_mon_m safely clears lmax; otherwise a warning string.

    The regular/outgoing field separation inverts a 2x2 system per (n,m)
    built from j_n(kr_mon)/h_n(kr_mon). j_n(x) is intrinsically tiny
    whenever x < n (the standard centrifugal-barrier falloff of spherical
    Bessel functions) -- e.g. j_3(0.9) ~ 0.007 vs h_3(0.9) ~ 24 -- so if
    kr_mon at the LOWEST frequency in the band doesn't comfortably clear
    lmax, the l=lmax separation becomes numerically ill-conditioned and
    amplifies ordinary solver/measurement noise into wildly wrong "incident"
    estimates specifically at l=lmax. That then contaminates residual,
    passivity, reciprocity, and incident_deviation_max together -- a sharp,
    single-mode failure, not a gradual broad-spectrum mesh-quality issue.
    Found by direct field-level tracing of a real extraction where
    kr_mon(fmin) = 0.90 against lmax = 3 (incident deviation up to 0.99).
    """
    kr_mon_min = k_min * r_mon_m
    if kr_mon_min < lmax + margin:
        return (f"k*r_monitor at the lowest frequency = {kr_mon_min:.2f}, "
                f"only {kr_mon_min / lmax:.2f}x lmax ({lmax}) -- the "
                f"regular/outgoing field separation becomes ill-conditioned "
                f"at l=lmax when kr_mon is this close to (or below) lmax. "
                f"Increase monitor_factor (currently gives r_mon = "
                f"{r_mon_m * 1e6:.3f} um) so kr_mon(fmin) exceeds "
                f"lmax+{margin:g} ({lmax + margin:g}), or raise the lowest "
                f"frequency in the band, before trusting this extraction's "
                f"low-frequency end.")
    if r_circ_m is not None:
        # scatterer near-field clearance (2026-08-07): the scatterer
        # radiates up to its FULL Wiscombe order regardless of lmax; if the
        # monitor sits inside the reactive zone of those orders, the
        # enlarged total field turns the solver's relative error into a
        # large additive contribution to every projected coefficient.
        # Most restrictive at the lowest frequency (the requirement
        # r_circ + (4x^(1/3)+2)/k decreases with k).  Observed when
        # violated: incident deviation 0.11-0.15, sv(S) up to 1.37, an
        # l=lmax block an order of magnitude too large, largest at the
        # low-frequency end.
        w = wiscombe_lmax(k_min * r_circ_m)
        if kr_mon_min < w + nearfield_margin:
            return (f"k*r_monitor at the lowest frequency = "
                    f"{kr_mon_min:.2f} does not clear the scatterer's own "
                    f"Wiscombe order there ({w}) + {nearfield_margin:g}: "
                    f"physically radiated multipoles above l = "
                    f"{int(kr_mon_min)} are reactive at the monitor radius "
                    f"and inflate the measured field. Increase "
                    f"monitor_factor (currently gives r_mon = "
                    f"{r_mon_m * 1e6:.3f} um) so kr_mon(fmin) >= "
                    f"{w + nearfield_margin:g}.")
    return None


class ExtractionCheckError(RuntimeError):
    """An automatic check found the extraction configuration or its data
    to be invalid, and the run was stopped.

    Raised before or during a run so that hours of computation are not
    spent on a configuration a short check shows to be unusable.  Solved
    illuminations remain cached; after correcting the configuration a
    rerun continues from them."""


def estimate_fd_memory_gb(n_cells: int) -> float:
    """Crude peak-memory estimate (GB) for the iterative second-order tet
    FD solve.  Calibration points (2026-08-06/07, CST 2026): 0.96M cells
    solved within ~19 GB available; 0.4M within a few GB; the ~10-20M-cell
    single-band spoke-wheel configuration exhausted 32 GB ("Could not
    compute preconditioner").  12 kB/cell reproduces that split with
    margin.  Deliberately rough -- used only to refuse obviously-infeasible
    runs, not to admit marginal ones."""
    return float(n_cells) * 12e3 / 1024 ** 3


def available_memory_gb() -> float | None:
    """Currently available physical RAM in GB (None if not measurable)."""
    try:
        import ctypes
        import ctypes.wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.wintypes.DWORD),
                        ("dwMemoryLoad", ctypes.wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_uint64),
                        ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / 1024 ** 3
    except Exception:                                     # noqa: BLE001
        return None


def incident_deviation_action(worst_dev: float, warn: float,
                              abort: float) -> str:
    """'ok' / 'warn' / 'abort' for a solve's worst incident deviation."""
    if worst_dev > abort:
        return "abort"
    if worst_dev > warn:
        return "warn"
    return "ok"


def top_order_content(f_cols: np.ndarray, lmax: int) -> float:
    """Fraction of the scattered columns' per-order content sitting at
    l = lmax: max over frequencies of |f_(l=lmax)| / max_l |f_l|.

    ~0 when the truncation comfortably contains the physics (the top order
    is a guard); O(1) when real content is being cut off at the truncation
    boundary -- measurable from the FIRST solve, which is the point."""
    _, ns, _ = vswf.mode_list(lmax)
    f_cols = np.atleast_2d(np.asarray(f_cols))
    worst = 0.0
    for row in f_cols:
        per_l = np.array([np.linalg.norm(row[ns == l])
                          for l in range(1, lmax + 1)])
        peak = float(np.max(per_l))
        if peak > 0:
            worst = max(worst, float(per_l[-1]) / peak)
    return worst


def build_project(session, plan: ScattererPlan, paths: RunPaths,
                  extraction: ExtractionConfig,
                  solver_kwargs: dict | None = None):
    """Create (or copy from a template) and fully configure the CST project.

    In both modes the pipeline owns the frequency range, background,
    boundaries, mesh, monitors, and solver configuration; in build mode it
    additionally owns the units, while in template mode the units of the
    supplied project are respected and only verified indirectly through
    the bounding-box check."""
    from .cst_driver import builder, monitors, solvers

    paths.ensure()
    lu = _LEN_SCALE[plan.length_unit]
    fu = _FREQ_SCALE[plan.freq_unit]
    freqs_proj = plan.freqs_hz / fu
    fmax_proj = float(np.max(freqs_proj))
    fmin_proj = float(np.min(freqs_proj))
    lam_max_proj = C0 / np.min(plan.freqs_hz) / lu     # longest wavelength, proj units
    r_mon_proj = plan.monitor_factor * plan.r_circ_m / lu

    if plan.template is not None:
        from .cst_driver.template import prepare_template_project
        h = prepare_template_project(session, plan.template, paths.cst_file,
                                     r_circ_proj=plan.r_circ_m / lu,
                                     strict=plan.template_strict)
    else:
        h = session.new_mws_project(paths.cst_file)
        builder.set_units(h, length=plan.length_unit,
                          frequency=plan.freq_unit)

    # domain padding: monitor sphere + >= lambda_max/4 clearance to the
    # boundary; "expanded open" adds its own lambda/8, this keeps margin
    padding = max(0.3 * lam_max_proj, 1.2 * r_mon_proj - plan.r_circ_m / lu)
    builder.set_frequency_range(h, 0.8 * fmin_proj, 1.05 * fmax_proj)
    builder.set_background_vacuum(h, padding=padding)
    builder.set_open_boundaries(h)
    builder.set_tet_mesh(h, steps_per_wave_near=plan.mesh_steps_per_wave[0],
                         steps_per_wave_far=plan.mesh_steps_per_wave[1])

    if plan.build is not None:
        plan.build(h)                                   # user geometry

    monitors.define_field_monitors(h, freqs_proj, fields=("Efield", "Hfield"))
    monitors.define_farfield_monitors(h, freqs_proj)
    solvers.configure_adaptation(h)
    solvers.configure_fd_solver(h, freqs_proj, **(solver_kwargs or {}))
    h.save()
    return h, freqs_proj, r_mon_proj


def extract_tmatrix(session, plan: ScattererPlan,
                    extraction: ExtractionConfig | None = None,
                    run_root: Path | None = None,
                    solver_kwargs: dict | None = None,
                    library_path: Path | None = None):
    """Full extraction.  Returns (T (n_freq,N,N), diagnostics dict, h5 path)."""
    from .cst_driver import export, monitors, solvers
    from .config import LIBRARY_ROOT, LOCAL_RUN_ROOT

    lmax = plan_lmax(plan)
    if extraction is None:
        extraction = ExtractionConfig(lmax=lmax)
    N = vswf.n_modes(lmax)
    tag = grid_tag(plan.freqs_hz, lmax)
    paths = RunPaths(plan.name, run_root or LOCAL_RUN_ROOT, run_tag=tag)
    paths.ensure()

    # --- geometry of the measurement ---------------------------------------
    r_mon_m = plan.monitor_factor * plan.r_circ_m
    n_dirs = int(np.ceil(extraction.illumination_factor * N / 2))
    th_i, ph_i = illumination_directions(extraction.illumination_scheme,
                                         n_dirs)
    n_dirs = len(th_i)                     # Lebedev rules may return more

    # --- optional symmetry reduction ---------------------------------------
    # A scatterer with point group G satisfies T D(g) = D(g) T, so a solved
    # illumination (a, f) also yields (D(g) a, D(g) f) for every g -- valid
    # columns for the SAME T, at no solver cost.  Solve only a reduced set of
    # directions whose orbits together supply enough independent columns.
    sym_ops = None
    if extraction.symmetry_n_fold:
        from .postprocess.symmetry import (point_group_operations,
                                           reduced_illumination_directions)
        mirrors = extraction.symmetry_mirror_phi0_deg
        sym_ops = point_group_operations(
            lmax, n_fold=extraction.symmetry_n_fold,
            mirror_phi0_deg=list(mirrors) if mirrors else None)
        n_cols_needed = 2 * n_dirs
        idx, n_expected = reduced_illumination_directions(
            th_i, ph_i, sym_ops, n_cols_needed)
        if len(idx) == 0 or n_expected < n_cols_needed:
            print(f"WARNING: symmetry reduction could not cover "
                  f"{n_cols_needed} columns from the candidate directions "
                  f"(got {n_expected}); solving all illuminations instead.")
            sym_ops = None
        else:
            th_i, ph_i = np.asarray(th_i)[idx], np.asarray(ph_i)[idx]
            n_dirs = len(th_i)
            print(f"symmetry reduction |G|={len(sym_ops)}: solving "
                  f"{2 * n_dirs} illuminations instead of {n_cols_needed} "
                  f"({n_cols_needed / (2 * n_dirs):.1f}x fewer), expanding "
                  f"to {n_expected} columns")

    illuminations = [(float(th_i[i]), float(ph_i[i]), pol)
                     for i in range(n_dirs) for pol in ("theta", "phi")]

    # quadrature sized by the WORST case bandwidth (highest frequency)
    k_list = 2 * np.pi * plan.freqs_hz / C0
    th_q, ph_q, w_q, qshape = quadrature_for_monitor(lmax, float(np.max(k_list)),
                                                     r_mon_m)
    pts_m = r_mon_m * unit_vectors(th_q, ph_q)

    # --- check 1: monitor placement ----------------------------------------
    warning = check_monitor_lmax_margin(float(np.min(k_list)), r_mon_m, lmax,
                                        r_circ_m=plan.r_circ_m)
    if warning:
        if extraction.allow_monitor_violation:
            print(f"WARNING (override active): {warning}")
        else:
            raise ExtractionCheckError(
                f"monitor placement check: {warning}\n"
                f"Set ExtractionConfig.allow_monitor_violation=True only "
                f"for deliberate experiments.")

    # --- CST project --------------------------------------------------------
    h, freqs_proj, r_mon_proj = build_project(session, plan, paths,
                                              extraction, solver_kwargs)
    lu = _LEN_SCALE[plan.length_unit]
    pts_proj = pts_m / lu
    point_file = export.write_point_file(paths.export_dir / "sphere_points.txt",
                                         pts_proj)

    # --- check 2: mesh size and memory, before any solve --------------------
    if extraction.preflight:
        h.m3d.Mesh.Update()
        cells = int(h.m3d.Mesh.GetNumberOfMeshCells())
        est_gb = estimate_fd_memory_gb(cells)
        avail_gb = available_memory_gb()
        print(f"preflight: mesh {cells:,} cells, est. solver memory "
              f"~{est_gb:.1f} GB, available "
              f"{avail_gb:.1f} GB" if avail_gb is not None else
              f"preflight: mesh {cells:,} cells, est. solver memory "
              f"~{est_gb:.1f} GB, available RAM unknown", flush=True)
        if extraction.max_cells is not None and cells > extraction.max_cells:
            raise ExtractionCheckError(
                f"mesh-size check: {cells:,} mesh cells exceeds "
                f"max_cells={extraction.max_cells:,}. This configuration "
                f"would be prohibitively slow or exhaust memory; split the "
                f"band or reduce lmax/density before spending solver hours.")
        if (avail_gb is not None
                and est_gb > extraction.memory_headroom * avail_gb):
            raise ExtractionCheckError(
                f"memory check: estimated solver memory {est_gb:.1f} GB "
                f"exceeds {extraction.memory_headroom:.0%} of the "
                f"{avail_gb:.1f} GB currently available (the observed "
                f"failure signature is 'Could not compute preconditioner' "
                f"after meshing succeeds). Split the band, reduce the mesh "
                f"density, or free memory.")

    # --- illumination loop (cached, resumable) ------------------------------
    from .cst_driver import excitation as exc
    n_freq = len(plan.freqs_hz)
    A = np.zeros((n_freq, N, len(illuminations)), dtype=complex)
    F = np.zeros_like(A)
    a_nominal_dev = np.zeros((n_freq, len(illuminations)))

    # Mesh strategy note (established live, 2026-08-05): CST re-runs full
    # adaptive refinement on EVERY FDSolver.Start -- an adapted mesh is never
    # carried across solves, by any mechanism (flag toggle, parametric
    # update, ParameterSweep: all tested, all refuted).  Toggling
    # MeshAdaptionTet off mid-loop does NOT reuse the adapted mesh either;
    # it silently reverts to the coarse base mesh, which would put
    # illumination 1 and illuminations 2+ on different discretizations of
    # the same T-matrix.  Therefore: either leave adaptation ON (every
    # illumination pays the full adaptation cost, meshes differ per column)
    # or run with mesh_adaption=False in solver_kwargs plus an adequately
    # dense plan.mesh_steps_per_wave (deterministic, identical mesh for
    # every column -- validated against the adaptive result and exact Mie).

    first_live_done = False
    for kk, (ti, pi_, pol) in enumerate(illuminations):
        cache = paths.export_dir / f"illum_{kk:04d}.npz"
        if cache.exists():
            z = np.load(cache)
            # reuse only if the cache was built on THIS frequency grid
            # (older caches lack freqs_hz and are checked by count only)
            same_grid = (z["a"].shape[0] == n_freq
                         and ("freqs_hz" not in z
                              or np.allclose(z["freqs_hz"], plan.freqs_hz)))
            # ... AND for THIS illumination direction.  The cache is keyed
            # by index, but the illumination LIST is not immutable (e.g.
            # symmetry reduction picks a different subset of directions), so
            # index-only matching could silently attribute a cached column
            # to the wrong direction (latent bug found 2026-08-06).  Caches
            # written before this fix lack the direction record and are
            # conservatively re-solved.
            same_dir = ("th" in z
                        and abs(float(z["th"]) - ti) < 1e-12
                        and abs(float(z["ph"]) - pi_) < 1e-12
                        and str(z["pol"]) == pol)
            # ... AND at THIS monitor radius: the cached a/f columns are
            # projections of fields exported on the monitor sphere, so a
            # changed monitor_factor (e.g. after the 2026-08-07 near-field
            # clearance fix) invalidates them even though grid and
            # direction match.  Legacy caches without the record re-solve.
            same_mon = ("r_mon_m" in z
                        and abs(float(z["r_mon_m"]) - r_mon_m)
                        < 1e-12 * max(r_mon_m, 1.0))
            if same_grid and same_dir and same_mon:
                A[:, :, kk] = z["a"]
                F[:, :, kk] = z["f"]
                a_nominal_dev[:, kk] = z["dev"]
                continue
            print(f"[{kk + 1}/{len(illuminations)}] cache exists but was "
                  f"built on a different frequency grid, illumination "
                  f"direction, or monitor radius — re-solving", flush=True)

        print(f"[{kk + 1}/{len(illuminations)}] th={ti:.3f} ph={pi_:.3f} "
              f"pol={pol}: solving ...", flush=True)
        import time as _time
        t_iter0 = _time.time()
        exc.set_plane_wave(h, ti, pi_, pol)
        h.save()
        h.run_solver()

        a_cols = np.zeros((n_freq, N), dtype=complex)
        f_cols = np.zeros((n_freq, N), dtype=complex)
        devs = np.zeros(n_freq)
        for jj, (f_hz, f_proj) in enumerate(zip(plan.freqs_hz, freqs_proj)):
            k = 2 * np.pi * f_hz / C0
            E, ZH = export.get_EH_on_points(
                h, monitors.efield_tree_path(f_proj),
                monitors.hfield_tree_path(f_proj),
                paths.export_dir, f"i{kk:04d}_f{jj:03d}", point_file, pts_proj)
            a_meas, f_meas = vswf.separate_surface_field(
                E, ZH, lmax, k, r_mon_m, th_q, ph_q, w_q)
            # cross-check measured incident against analytic expansion
            a_nom = vswf.plane_wave_coefficients(ti, pi_, pol, lmax)
            scale = (a_meas @ np.conj(a_nom)) / max(np.vdot(a_nom, a_nom).real,
                                                    1e-300)
            devs[jj] = (np.linalg.norm(a_meas - scale * a_nom)
                        / max(np.linalg.norm(a_meas), 1e-300))
            a_cols[jj], f_cols[jj] = a_meas, f_meas

        # --- check 3: incident deviation of this solve ----------------------
        # A recorded field far from the known incident wave invalidates this
        # column whatever the cause (misplaced monitor, boundary problem,
        # meshing defect, or something not yet understood), and it can be
        # detected now: one earlier run measured a deviation of 0.11 on its
        # first solve and ran 2.5 h to completion before the problem was
        # seen.  The column from a stopped solve is not cached, so a
        # corrected rerun cannot reuse it unnoticed.
        worst_dev = float(np.max(devs))
        action = incident_deviation_action(
            worst_dev, extraction.incident_deviation_warn,
            extraction.incident_deviation_abort)
        if action == "abort":
            raise ExtractionCheckError(
                f"incident-deviation check at illumination {kk + 1}/"
                f"{len(illuminations)}: worst deviation {worst_dev:.2e} "
                f"exceeds abort threshold "
                f"{extraction.incident_deviation_abort:.1e}. The recorded "
                f"field on the monitor sphere is unreliable; see the "
                f"sources-of-error section of the manual before re-running. "
                f"({kk} completed illuminations remain cached.)")
        if action == "warn":
            print(f"WARNING: incident deviation {worst_dev:.2e} exceeds "
                  f"{extraction.incident_deviation_warn:.1e} at "
                  f"illumination {kk + 1}", flush=True)

        np.savez(cache, a=a_cols, f=f_cols, dev=devs,
                 freqs_hz=plan.freqs_hz, th=ti, ph=pi_, pol=pol,
                 r_mon_m=r_mon_m)
        A[:, :, kk] = a_cols
        F[:, :, kk] = f_cols
        a_nominal_dev[:, kk] = devs

        # --- first-live-solve advisories ------------------------------------
        if not first_live_done:
            first_live_done = True
            dt = _time.time() - t_iter0
            toc = top_order_content(f_cols, lmax)
            if toc > extraction.lmax_content_warn:
                print(f"WARNING: {toc:.0%} of the scattered column's "
                      f"per-order peak sits at l=lmax={lmax} -- the "
                      f"truncation may be cutting real content; consider "
                      f"raising lmax (see manual, truncation section)",
                      flush=True)
            n_remaining = len(illuminations) - kk - 1
            projected_h = dt * n_remaining / 3600
            print(f"first live solve: {dt / 60:.1f} min -> projected "
                  f"remaining (upper bound, cached skip): "
                  f"{projected_h:.1f} h", flush=True)
            if (extraction.max_projected_hours is not None
                    and projected_h > extraction.max_projected_hours):
                raise ExtractionCheckError(
                    f"projected-time check: {projected_h:.1f} h remaining "
                    f"exceeds max_projected_hours="
                    f"{extraction.max_projected_hours:g}. (1 illumination "
                    f"cached; re-run continues from it after "
                    f"reconfiguration.)")

    # --- solve per frequency ------------------------------------------------
    T = np.zeros((n_freq, N, N), dtype=complex)
    diags = {"residual": [], "cond": [], "s_max_sv": [], "unitarity": [],
             "reciprocity": [], "incident_deviation_max": []}
    for jj in range(n_freq):
        A_j, F_j = A[jj], F[jj]
        if sym_ops is not None:
            from .postprocess.symmetry import expand_columns_by_symmetry
            A_j, F_j = expand_columns_by_symmetry(A_j, F_j, sym_ops)
        T[jj], d = solve_tmatrix(A_j, F_j)
        smax, uni = passivity_check(T[jj])
        diags["residual"].append(d["residual"])
        diags["cond"].append(d["cond"])
        diags["s_max_sv"].append(smax)
        diags["unitarity"].append(uni)
        diags["reciprocity"].append(reciprocity_check(T[jj], lmax))
        diags["incident_deviation_max"].append(float(np.max(a_nominal_dev[jj])))
    # fraction of ||T(f)|| in orders above the per-frequency Wiscombe limit:
    # expected O(1) at the low end of a wide frequency range (undetermined
    # orders), ~0 at
    # the top -- see noise_order_content docstring before treating it as an
    # error signal
    diags["noise_order_content"] = list(
        noise_order_content(T, lmax, plan.freqs_hz, plan.r_circ_m))

    # --- store (tmat.h5 format; see storage module) -------------------------
    # raw solve data FIRST: if the (metadata-rich, and therefore more
    # failure-prone) tmat.h5 write below raises, the extraction itself must
    # already be on disk (learned live 2026-08-06: a metadata type error in
    # save_tmatrix was the only thing standing after 10 completed solves)
    np.savez(paths.export_dir / "AF_matrices.npz", A=A, F=F,
             freqs_hz=plan.freqs_hz)
    out = (library_path or (LIBRARY_ROOT / f"{plan.name}.tmat.h5"))
    if not extraction_matches(out, plan.freqs_hz, lmax):
        # a different grid/lmax already lives at this path (e.g. a coarse
        # timing run vs. the final fine-grid extraction) — fork the
        # filename instead of destroying the earlier result
        forked = out.with_name(f"{out.stem}__{tag}{out.suffix}")
        print(f"NOTE: {out.name} exists with different extraction settings "
              f"(grid/lmax); writing this run to {forked.name} instead. "
              f"Pass library_path=... explicitly to control the output path.")
        out = forked
    save_tmatrix(
        out, plan.freqs_hz, T, lmax,
        name=plan.name,
        keywords="cst,fem,plane-wave illumination",
        scatterer={"circumscribing_radius_m": plan.r_circ_m,
                   **plan.metadata},
        computation={
            "lmax": lmax, "n_modes": N,
            "n_illuminations": len(illuminations),
            "n_columns_solved": len(illuminations),
            "n_columns_used": (len(illuminations) * len(sym_ops)
                               if sym_ops else len(illuminations)),
            "symmetry_group_order": len(sym_ops) if sym_ops else 1,
            "symmetry_n_fold": extraction.symmetry_n_fold or 0,
            "illumination_scheme": extraction.illumination_scheme,
            "monitor_radius_m": r_mon_m,
            "quadrature_grid": list(qshape),
            "cst_project": str(paths.cst_file),
        },
        diagnostics={k: np.array(v) for k, v in diags.items()})
    return T, diags, out
