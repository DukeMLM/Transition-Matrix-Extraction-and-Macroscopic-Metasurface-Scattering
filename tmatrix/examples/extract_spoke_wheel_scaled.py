r"""End-to-end T-matrix extraction of a gold spoke-and-wheel resonator.

One unit-cell design of the SAW metasurface (the top metamaterial layer
ONLY — no ground plane, no Ge spacer, no substrate, no periodic
boundaries) is built free-standing in vacuum, centered at the origin, and
its T-matrix is extracted per the package manual (docs/Tmatrix_manual.pdf):
plane-wave illuminations -> FD tetrahedral solves -> E/H recorded on the
monitor sphere -> self-calibrating separation -> least-squares T -> tmat.h5.

Geometry (from MM_spoke_wheel.py, all lengths um):
    ring    annulus, outer radius r, inner radius r_in = r - w_ring
    spokes  four bars of width w along +/-x and +/-y, from the central
            gap (+/- gap/2) to the ring, merged with it into one solid
    t_mm    layer thickness (0.1 um), centered on z = 0
Designs are read from SAW_noSub_designs.txt next to this file
(columns: Wavelength (um), r (um), gap (um), w_ring (um), w (um)).

Material: gold as CST-standard lossy metal (surface impedance),
sigma = 4.561e7 S/m.  At 8-20 um the skin depth is ~17-27 nm, well below
t_mm = 100 nm, so the surface-impedance model is applicable.  Swap the
block in `define_gold` for a Drude fit if dispersive gold is preferred.

Cost per design: 2*ceil(1.5*N/2) FD solves with N = 2*lmax*(lmax+2)
(lmax=3 -> 46 solves), each covering ALL wavelength points as exact solver
samples — so total time scales with n_solves x n_wavelengths.  Completed
illuminations are cached (.npz) in the run directory and the extraction
resumes after interruption without recomputation.

Usage (run on the CST machine):

  From an editor (the VS Code run button, Spyder, etc.): edit the USER
  SETTINGS block below and run the file — no command line needed.

  From a terminal, flags override the USER SETTINGS:
    python extract_spoke_wheel_scaled.py --list              # show designs
    python extract_spoke_wheel_scaled.py --dry-run           # plan + VBA
    python extract_spoke_wheel_scaled.py --wl 15             # nearest design
    python extract_spoke_wheel_scaled.py --wl 15 --wl-step 0.25  # finer grid
    python extract_spoke_wheel_scaled.py --all               # all designs
"""

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

# Fail with a diagnosis, not a bare ImportError: report WHICH interpreter
# actually ran this file.  In VS Code, "module not found" with the correct
# interpreter selected means the run used a different one (see message).
_missing = [m for m in ("numpy", "scipy", "h5py")
            if importlib.util.find_spec(m) is None]
if _missing:
    sys.exit(
        "\n".join([
            "Required packages are missing from the interpreter that ran "
            "this file.",
            f"  interpreter used: {sys.executable}",
            f"  missing:          {', '.join(_missing)}",
            "",
            "If this is not the interpreter you selected in VS Code, the",
            "run button belongs to another extension (typically Code",
            "Runner, which uses 'python' from PATH and prints to the",
            "OUTPUT panel).  Use the dropdown next to the run button and",
            "pick 'Run Python File', or run via F5 (Run and Debug) — both",
            "honor the selected interpreter.",
            "",
            "If the interpreter path IS correct, install into it:",
            f'  "{sys.executable}" -m pip install ' + " ".join(_missing),
        ]))

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst_tmatrix import config as cfg
from cst_tmatrix.config import ExtractionConfig, RunPaths, grid_tag
from cst_tmatrix.pipeline import ScattererPlan, extract_tmatrix, C0

# ---------------------------------------------------------------------------
# USER SETTINGS — these apply when the script is run WITHOUT command-line
# flags (e.g. the VS Code run button).  Any flag given on a command line
# overrides the corresponding setting here.
# ---------------------------------------------------------------------------
DRY_RUN = False        # True: print the plan and geometry VBA; CST not touched
LIST_ONLY = False      # True: print the design list and exit
RUN_ALL = False        # True: extract EVERY design in the list, in order
PICK_WL = 13.1         # extract the design nearest this center wavelength
                       # (um); set to None to select by row instead:
PICK_INDEX = 0         # row of the design list (0 = the 20 um design)
WL_MIN_UM = 9.0        # extraction band, specified by WAVELENGTH (um);
WL_MAX_UM = 30.0       # leave FREQ_MIN_THZ/FREQ_MAX_THZ/FREQ_STEP_THZ at
WL_STEP_UM = 1         # None (the default) to use these three instead.
FREQ_MIN_THZ = 10    # extraction band, specified by FREQUENCY (THz)
FREQ_MAX_THZ = 34    # instead of wavelength; set all three (not the
FREQ_STEP_THZ = 1   # WL_* settings above) to use this instead.
LMAX = 5               # multipole truncation (chosen from measurement,
                       # 2026-08-06).  Wiscombe's criterion at 34 THz gives
                       # 9, but that is where the Mie series converges to
                       # machine precision -- far beyond what the data can
                       # determine.  The measured multipole content of this
                       # design (per-order scattered amplitudes at 22.9 and
                       # 34 THz, and the earlier lmax=9 matrix) is resolved
                       # up to l~4; above that the lmax=9 matrix grows with
                       # l and is frequency-independent, i.e. amplified
                       # noise.  lmax=5 is the last resolved order plus one.
                       # A smaller lmax also shrinks the monitor sphere
                       # (r_mon ~ (lmax+2)/k_min) and the mesh with it.
# Split the extraction band into per-segment extractions, each with its own
# monitor radius (set by ITS lowest frequency) and mesh (set by ITS highest):
# the single-band 10-34 THz configuration couples the 10 THz monitor radius
# to the 34 THz mesh density and exhausted 32 GB on the remote machine
# (2026-08-06, "Could not compute preconditioner").  Edges in THz; band k
# spans (edges[k], edges[k+1]], first band inclusive of its lower edge.
# Each band writes its own {name}__{lo}to{hi}THz.tmat.h5.
BAND_EDGES_THZ = (10.0, 20.0, 34.0)
# C4v symmetry-reduced illumination (the spoke-wheel is C4v; detected from
# STL and the reduction validated against the treams cylinder reference,
# 2026-08-06): ~7.6x fewer solves.  Set False for the unreduced base path.
USE_C4V = True
# Deterministic non-adaptive mesh (2026-08-06 density check on THIS
# geometry, examples/density_check_spoke_wheel.py): CST never reuses an
# adapted mesh across solves, so every illumination re-pays adaptation --
# a fixed 20/8 steps-per-wavelength mesh instead makes every solve
# identical and deterministic.  Measured at the design frequency: dominant
# orders (l<=2, ~all of |f|^2) agree 0.4-2% with a 15/6 mesh (mesh-
# converged); incident deviation 6.1e-3 / 1.2e-2 at 22.9/34 THz.
MESH_STEPS_PER_WAVE = (20, 8)
MESH_ADAPTION = False
MONITOR_FACTOR = None  # monitor sphere radius = MONITOR_FACTOR * r_circ.
                       # None (default): auto-computed per design via
                       # pipeline.recommended_monitor_factor, so k*r_monitor
                       # always clears lmax+2 at the lowest extraction
                       # frequency regardless of this design's own r_circ
                       # (see pipeline.check_monitor_lmax_margin -- the old
                       # fixed 1.5 default gave k*r_monitor=0.90 at 30 um,
                       # deep inside lmax=3, and produced incident
                       # deviations up to 0.99; the domain-padding bug that
                       # ALSO contributed to that failure is now fixed in
                       # cst_driver/builder.py). Set a specific number here
                       # only to override the auto value for one run --
                       # doing that across a batch of differently-sized
                       # designs is exactly what breaks silently.
# Solver CPU-thread count and GPU offload are properties of the machine,
# not of the physics problem: set them in settings.json ("max_cpus",
# "hardware_acceleration").  None leaves the CST defaults untouched.
_machine = cfg.machine_settings()
MAX_CPUS = _machine.get("max_cpus")
HARDWARE_ACCELERATION = _machine.get("hardware_acceleration")

DESIGN_FILE = Path(__file__).resolve().parent / "SAW_noSub_designs.txt"

# CST projects + solver output; must be a LOCAL disk on the machine that
# runs CST (never a cloud-synced folder).  Set per machine in settings.json
# ("run_root") or through the CST_TMATRIX_RUN_ROOT environment variable;
# the fallback is ~/cst_tmatrix_runs (see cst_tmatrix/config.py).
RUN_ROOT = cfg.RUN_ROOT
# Final .tmat.h5 files (small; a synced folder would also be safe here).
# settings.json "library" or CST_TMATRIX_LIBRARY; the fallback is the
# library/ directory of this repository.
LIBRARY_DIR = cfg.LIBRARY_ROOT

T_MM_UM = 0.2                 # spoke-and-wheel layer thickness (um)
GOLD_SIGMA = 4.561e7          # S/m, CST library gold


def load_designs(path):
    """Parse the tab-delimited design list into dicts {wl, r, gap, w_ring, w}
    (same conventions as MM_spoke_wheel.load_designs)."""
    lines = [ln.rstrip("\n") for ln in open(path) if ln.strip()]
    header = [h.strip().split("(")[0].split()[0].lower()
              for h in lines[0].split("\t")]
    designs = []
    for ln in lines[1:]:
        row = dict(zip(header, [float(v) for v in ln.split()]))
        designs.append({"wl": row["wavelength"], "r": row["r"],
                        "gap": row["gap"], "w_ring": row["w_ring"],
                        "w": row["w"]})
    return designs


# ---------------------------------------------------------------------------
# Geometry (Mode-1 history blocks through the pipeline's build hook)
# ---------------------------------------------------------------------------

def _f(x) -> str:
    return f'"{x:.12g}"'


def define_gold(h):
    """CST-standard lossy-metal gold (surface-impedance model)."""
    h.add_history("define material: gold", f"""With Material
  .Reset
  .Name "gold"
  .Folder ""
  .FrqType "all"
  .Type "Lossy metal"
  .MaterialUnit "Frequency", "THz"
  .MaterialUnit "Geometry", "um"
  .Mu "1.0"
  .Sigma {_f(GOLD_SIGMA)}
  .Rho "19320.0"
  .Colour "1", "0.84", "0"
  .Create
End With""")


def make_build(design):
    """Return the pipeline build callable for one spoke-and-wheel design.

    The pipeline owns units (um/THz), frequency range, background,
    boundaries, mesh, and monitors; this callable adds only materials and
    solids, centered at the origin as the T-matrix expansion requires."""
    r, gap = design["r"], design["gap"]
    w_ring, w = design["w_ring"], design["w"]
    r_in = r - w_ring
    half_w, half_gap, half_t = w / 2.0, gap / 2.0, T_MM_UM / 2.0
    # spokes reach to mid-annulus: guaranteed overlap for the boolean
    # union, no change to the merged metal shape
    reach = r - w_ring / 2.0

    if not (0 < half_gap < r_in):
        raise ValueError(f"invalid design (gap/2={half_gap} vs r_in={r_in})")
    if not w < gap:
        raise ValueError(f"invalid design (w={w} closes the central gap={gap})")

    def build(h):
        from cst_tmatrix.cst_driver import builder

        define_gold(h)
        builder.new_component(h, "scatterer")
        h.add_history("define cylinder: scatterer:ring", f"""With Cylinder
  .Reset
  .Name "ring"
  .Component "scatterer"
  .Material "gold"
  .Axis "z"
  .Outerradius {_f(r)}
  .Innerradius {_f(r_in)}
  .Zrange {_f(-half_t)}, {_f(half_t)}
  .Segments "0"
  .Create
End With""")
        spokes = {"spoke_px": ((half_gap, reach), (-half_w, half_w)),
                  "spoke_mx": ((-reach, -half_gap), (-half_w, half_w)),
                  "spoke_py": ((-half_w, half_w), (half_gap, reach)),
                  "spoke_my": ((-half_w, half_w), (-reach, -half_gap))}
        for name, (xr, yr) in spokes.items():
            builder.build_brick(h, xr, yr, (-half_t, half_t),
                                material="gold", name=name)
        for name in spokes:
            h.add_history(
                f"boolean add shapes: scatterer:ring, scatterer:{name}",
                f'Solid.Add "scatterer:ring", "scatterer:{name}"')

    return build


class PrintHandle:
    """Stand-in ProjectHandle for --dry-run: prints the VBA blocks."""

    def add_history(self, label, vba, timeout=None):
        print(f"' ===== {label} =====")
        print(vba)
        print()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def frequency_plan(wl_min=None, wl_max=None, wl_step=None,
                   freq_min_thz=None, freq_max_thz=None, freq_step_thz=None):
    """Build the extraction frequency grid from EITHER a wavelength range
    (um) or a frequency range (THz) -- give exactly one complete set (all
    three wl_* or all three freq_*_thz). Returns (wl_um, freqs_hz): wl_um
    descending, freqs_hz ascending, either way."""
    wl_given = wl_min is not None and wl_max is not None and wl_step is not None
    freq_given = (freq_min_thz is not None and freq_max_thz is not None
                 and freq_step_thz is not None)
    if wl_given == freq_given:
        raise SystemExit(
            "frequency_plan: give exactly one complete parameter set -- "
            "either wl_min/wl_max/wl_step (um) or freq_min_thz/"
            "freq_max_thz/freq_step_thz (THz), not both or neither.")
    if freq_given:
        n = (freq_max_thz - freq_min_thz) / freq_step_thz
        if abs(n - round(n)) > 1e-6:
            raise SystemExit(f"freq range {freq_min_thz}-{freq_max_thz} THz "
                             f"is not a whole number of steps of "
                             f"{freq_step_thz} THz")
        freqs_hz = np.linspace(freq_min_thz, freq_max_thz,
                               int(round(n)) + 1) * 1e12
        return C0 / freqs_hz * 1e6, freqs_hz
    n = (wl_max - wl_min) / wl_step
    if abs(n - round(n)) > 1e-6:
        raise SystemExit(f"wl range {wl_min}-{wl_max} um is not a whole "
                         f"number of steps of {wl_step} um")
    wl_um = np.linspace(wl_max, wl_min, int(round(n)) + 1)
    return wl_um, C0 / (wl_um * 1e-6)


def make_plan(design, freqs_hz, lmax, monitor_factor=None):
    """monitor_factor=None (the default): auto-computed per design via
    pipeline.recommended_monitor_factor, so kr_mon(fmin) always clears
    lmax+2 regardless of this design's own r_circ -- do not hand-tune a
    single monitor_factor across a batch of differently-sized designs, it
    silently stops being sufficient the moment r_circ or the band changes
    (this is exactly what produced the incident_deviation~1.0 failure the
    first time -- see conversation/session notes)."""
    wl_tag = f"{design['wl']:.2f}".replace(".", "p")
    r_circ_m = math.hypot(design["r"], T_MM_UM / 2.0) * 1e-6
    if monitor_factor is None:
        from cst_tmatrix.pipeline import recommended_monitor_factor
        monitor_factor = recommended_monitor_factor(r_circ_m, freqs_hz, lmax)
        print(f"  auto monitor_factor = {monitor_factor:.2f} "
             f"(r_circ={r_circ_m*1e6:.4f} um, lmax={lmax})")
    return ScattererPlan(
        name=f"saw_gold_wl{wl_tag}um",
        build=make_build(design),
        r_circ_m=r_circ_m,
        freqs_hz=freqs_hz,
        lmax=lmax,
        monitor_factor=monitor_factor,
        mesh_steps_per_wave=MESH_STEPS_PER_WAVE,
        metadata={
            "geometry": {"shape": "spoke-and-wheel",
                         "r": design["r"], "gap": design["gap"],
                         "w_ring": design["w_ring"], "w": design["w"],
                         "thickness": T_MM_UM},
            "geometry_units": {k: "um" for k in
                               ("r", "gap", "w_ring", "w", "thickness")},
            "design_center_wavelength_um": design["wl"],
            "material": {"name": "gold (lossy metal)",
                         "conductivity_S_per_m": GOLD_SIGMA}})


def describe(plan, design, wl_um, args):
    lmax = args.lmax
    N = 2 * lmax * (lmax + 2)
    n_dirs = math.ceil(1.5 * N / 2)
    print(f"design: center wl {design['wl']:.4f} um | r={design['r']:.6g} "
          f"gap={design['gap']:.6g} w_ring={design['w_ring']:.6g} "
          f"w={design['w']:.6g} t={T_MM_UM} um")
    print(f"  project: {plan.name}")
    print(f"  r_circ = {plan.r_circ_m * 1e6:.4f} um, monitor sphere at "
          f"{plan.monitor_factor * plan.r_circ_m * 1e6:.4f} um")
    print(f"  wavelengths: {wl_um.max():.4g} -> {wl_um.min():.4g} um in "
          f"{len(wl_um)} points ({C0 / wl_um.max() / 1e6:.4g} - "
          f"{C0 / wl_um.min() / 1e6:.4g} THz)")
    print(f"  lmax={lmax}: N={N} modes, {n_dirs} directions x 2 pol = "
          f"{2 * n_dirs} FD solves, each sampling {len(wl_um)} frequencies")
    cpus = "machine default" if args.max_cpus is None else args.max_cpus
    accel = {True: "on", False: "off", None: "machine default"}[args.hw_accel]
    print(f"  solver: {cpus} CPU threads, hardware acceleration {accel}")


def diagnostics_table(wl_um, diags):
    print(f"\n{'wl (um)':>8} {'f (THz)':>8} {'residual':>9} {'cond':>6} "
          f"{'max sv(S)':>10} {'recip':>9} {'inc.dev':>9}")
    for jj, wl in enumerate(wl_um):
        print(f"{wl:8.3f} {C0 / wl / 1e6:8.3f} "
              f"{diags['residual'][jj]:9.2e} {diags['cond'][jj]:6.1f} "
              f"{diags['s_max_sv'][jj]:10.6f} "
              f"{diags['reciprocity'][jj]:9.2e} "
              f"{diags['incident_deviation_max'][jj]:9.1e}")
    worst = float(np.max(diags["incident_deviation_max"]))
    if worst > 1e-2:
        print(f"\nWARNING: incident deviation up to {worst:.1e} (> 1e-2). "
              f"Two known causes, check both: (1) per the manual, the "
              f"single mesh (adapted at the highest frequency) degrading "
              f"at the far band edge — consider splitting the band into "
              f"per-segment extractions; (2) k*r_monitor too close to "
              f"lmax at the lowest extraction frequency, which makes the "
              f"regular/outgoing separation ill-conditioned specifically "
              f"at l=lmax (see the pipeline's own WARNING at extraction "
              f"start, and pipeline.check_monitor_lmax_margin) — fixed by "
              f"raising --monitor-factor, not by splitting the band.")


def extraction_config(lmax):
    if USE_C4V:
        return ExtractionConfig(lmax=lmax, symmetry_n_fold=4,
                                symmetry_mirror_phi0_deg=(0., 45., 90., 135.))
    return ExtractionConfig(lmax=lmax)


def split_bands(freqs_hz):
    """Partition the frequency grid by BAND_EDGES_THZ.  Returns a list of
    (label, freqs_subarray); a single unlabeled band when no split is set."""
    # two edges legitimately define ONE band that also FILTERS the grid
    # (e.g. a single-band targeted re-run); only None/empty/one-edge means
    # "no banding" -- a 2-edge tuple silently returning the FULL grid was a
    # live near-miss on 2026-08-07
    if not BAND_EDGES_THZ or len(BAND_EDGES_THZ) < 2:
        return [("", freqs_hz)]
    e = [x * 1e12 for x in BAND_EDGES_THZ]
    bands = []
    for k in range(len(e) - 1):
        lo, hi = e[k], e[k + 1]
        m = ((freqs_hz > lo) if k else (freqs_hz >= lo)) & (freqs_hz <= hi)
        if np.any(m):
            bands.append((f"__{BAND_EDGES_THZ[k]:g}to"
                          f"{BAND_EDGES_THZ[k + 1]:g}THz", freqs_hz[m]))
    return bands


class _Tee:
    """Mirror writes to several streams (console + per-run log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def extract_one(session, design, args, wl_um, freqs_hz, band_tag=""):
    plan = make_plan(design, freqs_hz, args.lmax, args.monitor_factor)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    # everything printed below is also appended to the run directory's
    # exports/log.txt (same RunPaths the pipeline itself uses), so warnings
    # and diagnostics survive the console -- repeated/resumed runs of the
    # same configuration append to the same file
    paths = RunPaths(plan.name, RUN_ROOT,
                     run_tag=grid_tag(freqs_hz, args.lmax)).ensure()
    log_path = paths.export_dir / "log.txt"
    old_out, old_err = sys.stdout, sys.stderr
    with open(log_path, "a", encoding="utf-8") as lf:
        sys.stdout = _Tee(old_out, lf)
        sys.stderr = _Tee(old_err, lf)
        try:
            import datetime
            print(f"\n{'=' * 72}\n==== run started "
                  f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  "
                  f"band={band_tag.lstrip('_') or 'full'}  "
                  f"lmax={args.lmax} ====")
            describe(plan, design, wl_um, args)
            t0 = time.time()
            T, diags, out = extract_tmatrix(
                session, plan, extraction_config(args.lmax),
                run_root=RUN_ROOT,
                solver_kwargs={"max_cpus": args.max_cpus,
                               "hardware_acceleration": args.hw_accel,
                               "mesh_adaption": MESH_ADAPTION},
                library_path=LIBRARY_DIR / f"{plan.name}{band_tag}.tmat.h5")
            dt = time.time() - t0
            N = 2 * args.lmax * (args.lmax + 2)
            n_solves = 2 * math.ceil(1.5 * N / 2)
            print(f"\nsaved: {out}")
            print(f"wall time {dt / 3600:.2f} h ({dt / n_solves / 60:.1f} "
                  f"min per solve if none were cached; cached illuminations "
                  f"skip; with C4v reduction the actual solve count is far "
                  f"lower)")
            diagnostics_table(wl_um, diags)
            return dt
        except BaseException as e:
            print(f"\nRUN FAILED ({type(e).__name__}): {e}")
            raise
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design-file", type=Path, default=DESIGN_FILE)
    ap.add_argument("--index", type=int, default=None,
                    help="row of the design list (0 = longest wavelength)")
    ap.add_argument("--wl", type=float, default=None,
                    help="pick the design nearest this center wavelength (um)")
    ap.add_argument("--all", action="store_true", default=RUN_ALL,
                    help="extract every design in the list, in order")
    ap.add_argument("--list", action="store_true", default=LIST_ONLY,
                    help="print the design list and exit")
    ap.add_argument("--wl-min", type=float, default=WL_MIN_UM)
    ap.add_argument("--wl-max", type=float, default=WL_MAX_UM)
    ap.add_argument("--wl-step", type=float, default=WL_STEP_UM,
                    help="wavelength spacing of the extraction grid (um); "
                         "0.25 is the goal, start coarse to gauge run time")
    ap.add_argument("--freq-min", type=float, default=FREQ_MIN_THZ,
                    help="extraction band lower edge (THz); give all three "
                         "--freq-* flags to specify the band by frequency "
                         "instead of --wl-min/--wl-max/--wl-step")
    ap.add_argument("--freq-max", type=float, default=FREQ_MAX_THZ,
                    help="extraction band upper edge (THz)")
    ap.add_argument("--freq-step", type=float, default=FREQ_STEP_THZ,
                    help="frequency spacing of the extraction grid (THz)")
    ap.add_argument("--lmax", type=int, default=LMAX,
                    help="multipole truncation (x = ka <= 0.71 here, so "
                         "lmax=3 already carries the octupole)")
    ap.add_argument("--monitor-factor", type=float, default=MONITOR_FACTOR,
                    help="monitor sphere radius = factor * r_circ; must be "
                         "large enough that k*r_monitor clears lmax by a "
                         "few units at the LOWEST extraction frequency")
    ap.add_argument("--dry-run", action="store_true", default=DRY_RUN,
                    help="print the plan and the geometry VBA; no CST")
    ap.add_argument("--max-cpus", type=int, default=MAX_CPUS,
                    help="CPU threads for the FD solver")
    ap.add_argument("--hw-accel", action=argparse.BooleanOptionalAction,
                    default=HARDWARE_ACCELERATION,
                    help="GPU hardware acceleration (--no-hw-accel disables)")
    args = ap.parse_args()

    designs = load_designs(args.design_file)
    if args.list:
        for i, d in enumerate(designs):
            print(f"{i:3d}  wl={d['wl']:8.4f} um  r={d['r']:.6g}  "
                  f"gap={d['gap']:.6g}  w_ring={d['w_ring']:.6g}  "
                  f"w={d['w']:.6g}")
        return

    if args.freq_min is not None or args.freq_max is not None or \
            args.freq_step is not None:
        wl_um, freqs_hz = frequency_plan(freq_min_thz=args.freq_min,
                                         freq_max_thz=args.freq_max,
                                         freq_step_thz=args.freq_step)
    else:
        wl_um, freqs_hz = frequency_plan(args.wl_min, args.wl_max, args.wl_step)

    # design selection: command line first (--wl / --index), then the
    # USER SETTINGS block (PICK_WL, else PICK_INDEX)
    if args.all:
        selected = designs
    elif args.wl is not None:
        selected = [min(designs, key=lambda d: abs(d["wl"] - args.wl))]
    elif args.index is not None:
        selected = [designs[args.index]]
    elif PICK_WL is not None:
        selected = [min(designs, key=lambda d: abs(d["wl"] - PICK_WL))]
    else:
        selected = [designs[PICK_INDEX]]

    if args.dry_run:
        for design in selected:
            plan = make_plan(design, freqs_hz, args.lmax, args.monitor_factor)
            describe(plan, design, wl_um, args)
            print(f"  run dir: {RUN_ROOT / plan.name}")
            print(f"  output:  {LIBRARY_DIR / (plan.name + '.tmat.h5')}\n")
        print("' geometry history blocks of the first selected design:\n")
        make_build(selected[0])(PrintHandle())
        return

    from cst_tmatrix.cst_driver import CSTSession
    session = CSTSession()
    results = []
    bands = split_bands(freqs_hz)
    if len(bands) > 1:
        print(f"band-split extraction: "
              + ", ".join(f"{tag.lstrip('_')} ({len(fb)} freqs)"
                          for tag, fb in bands))
    for design in selected:
        for band_tag, freqs_b in bands:
            wl_b = C0 / freqs_b / 1e-6
            try:
                dt = extract_one(session, design, args, wl_b, freqs_b,
                                 band_tag)
                results.append((design["wl"], band_tag or "full", "ok", dt))
            except KeyboardInterrupt:
                raise
            except Exception as e:                        # noqa: BLE001
                print(f"\nFAILED wl={design['wl']:.4f} um "
                      f"{band_tag or 'full'}: {e}\n")
                results.append((design["wl"], band_tag or "full",
                                "FAILED", float("nan")))
    if len(results) > 1:
        print("\nsummary:")
        for wl, band, status, dt in results:
            print(f"  wl={wl:8.4f} um  {band:>14}  {status}  "
                  f"{dt / 3600:6.2f} h")


if __name__ == "__main__":
    main()
