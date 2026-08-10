"""CST oblique-incidence Floquet campaign generator (checklist item 7).

Implements every "required edit" of INVERSE_TMATRIX_FROM_FLOQUET.md par. 7 on
top of the validated normal-incidence script
aggregation/cst_direct/build_saw_unitcell.py (run_v3, REPORT par. 4b):

1.  Geometry: free-standing gold spoke-and-wheel array, no spacer, no ground
    plane (T_iso's world).  Geometry/material VBA is COPIED from the base
    script (not imported-and-mutated).  Pitch trap handled exactly as
    validated: UnitCellFitToBoundingBox "False" + explicit
    UnitCellDs1/Ds2 = pitch = 2.0 um (REPORT par. 4b).

2.  Pinned domain: an explicit vacuum cellpad brick p x p x [-Z_PAD, +Z_PAD]
    (the cma_infinite/cst/build_run_paper.py vba_cellpad pattern) in BOTH the
    structure and the empty projects, so removing the metal cannot change the
    simulation domain.  Z_PAD = 3.0 um, justified from the validated run's
    measured auto open-space: run_v3's CADSurf log gives a model+background
    bounding-box diagonal of 6.4661101155608458 um => domain z-extent
    L = sqrt(diag^2 - 2*pitch^2) = 5.814687 um = metal thickness 0.1 um +
    2 x 2.857343 um, and 2.857343 um is EXACTLY lambda_center/4 with
    lambda_center = c / ((14.99+37.47)/2 THz) = 11.429373 um (match to 10
    significant digits, so that is CST 2026.0's "expanded open" auto-space
    rule for this setup).  Z_PAD = 3.0 um therefore places the cellpad face
    at >= the validated port clearance (2.857 um above the metal surface at
    z = 0.05 um -> 2.907 um from cell center), and the slowest evanescent
    Floquet order in the band (worst planned case theta = 60 deg,
    lambda = 20 um) decays by exp(-kappa*z_pad) ~ 2e-4 over the pad alone --
    below the 5e-3 closure gate.  We KEEP the validated boundary type
    "expanded open"; CST then adds lambda_center/4 = 2.857343 um of vacuum
    beyond the cellpad on each side, identically in both project kinds
    (the bounding box is the cellpad in both), giving the expected
    port-to-port distance

        L_expected = 2*Z_PAD + 2*(lambda_center/4) = 11.714687 um.

    L is re-measured at de-embed time from the empty-cell phase slope
    (deembed.check_empty_phase); the manifest records both L_pad = 2*Z_PAD
    and L_expected.

    The legacy run_empty project (aggregation/cst_direct/run_empty)
    demonstrates the failure mode this edit fixes in the strongest way: it
    contains NO solids at all, so CST could not even build a mesh
    ("Could not read mesh", Result/Model.log 2026-08-05) -- an un-pinned
    empty domain is not merely phase-shifted, it is unsolvable.

3.  Both Floquet modes excited: the base script's
    .Stimulation "List","List" effectively drives one Zmax mode; here
    .Stimulation "All","All" + .ResetExcitationList (the cma_infinite
    pattern, sanctioned by par. 7) so the 2x2 Jones blocks
    SZmax(a),Zmax(b) and SZmin(a),Zmax(b), a,b in {1,2}, are all driven.

4.  theta/phi scan parametrization: the cma_infinite vba_floquet angle block
    is ported into the cst.interface automation stack of the base script and
    parametrized in BOTH theta and phi (the cma template hard-codes phi=0).

        SetPeriodicBoundaryAnglesDirection "inward"   (NORMATIVE, par. 7)

    NOTE the two in-repo templates DISAGREE on this property:
    cma_infinite/cst/build_run_paper.py uses "inward" while
    cma_infinite/cst/cst_common.py's VBA_FLOQUET_PORTS uses "outward".
    Par. 7 resolves the conflict normatively in favor of "inward", to be
    verified at de-embed time via the empty-cell S21 phase slope vs
    k_z = k cos(theta) (deembed.check_empty_phase), which also pins the
    k_par sign.  Channel dictionary (par. 7, normative): exciting Zmax
    modes 1,2, the incident wave travels -z,
    k_hat = (sin th cos ph, sin th sin ph, -cos th) in CST coordinates.

5.  Campaign table (par. 7): 13 structure runs
    (theta in {0,15,30,45,60} x phi in {0,22.5,45} minus the 2 duplicate
    normal-incidence entries) + 5 empty runs (one per theta; the empty cell
    is phi-independent) + 1 perturbed empty repeat (AccuracyTet 1e-4 ->
    3e-4, for discretization scatter; par. 7 noise-floor calibration) = 19
    FD solves.  Starter subset (--starter): theta in {0,30,60} x
    phi in {0,22.5} minus duplicate = 5 structure + 3 empty runs.
    The 13 structure angles coincide with rows 0-12 of the normative
    17-angle table in retrieval/precompute_C.py.

6.  Frequency sampling: the base script's AddSampleInterval scheme,
    14.99-37.47 THz, "Automatic","True" -- covering the tmat.h5 49-frequency
    grid (lambda = 20.0 : -0.25 : 8.0 um).  The CST band clips the grid
    endpoints by <= 0.004 THz (14.99 > 14.98962, 37.47 < 37.47406); the
    downstream Re/Im np.interp clamps there (recorded in the manifest).

Modes of operation
------------------
--dry-run (DEFAULT)  Write per-run VBA history text + run.json under
                     retrieval/cst_runs/<runid>/ and the campaign manifest
                     retrieval/cst_runs/campaign_manifest.json, with NO CST
                     interaction (works on any machine), then self-verify
                     the generated VBA (cellpad in both kinds, All/All
                     excitation, "inward", explicit Ds1/Ds2, per-run angle
                     values) and print a summary table.
--build              Create the .cst projects via cst.interface
                     (nir.cst_helpers.HistoryBuilder) but DO NOT solve.
--solve              INTENTIONALLY NOT IMPLEMENTED.  Prints the par. 7
                     ordering requirement instead (see SOLVE_ORDER_NOTE).

deembed.py consumes campaign_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from tmatrix.cst_env import AUTO_CST, CST_PYTHON_LIB, ensure_on_path
from tmatrix.paths import CST_RUNS, RETRIEVAL_DATA
from tmatrix.units import C_UM_THZ

HERE = RETRIEVAL_DATA              # where the run/result directories live
RUNS_DIR_DEFAULT = CST_RUNS

# Template provenance (clone) of the base script (par. 7).
CMA_INFINITE = Path("D:/Claude/cma_infinite")

# ---- design (um), copied from build_saw_unitcell.py (do not import) -------
P = dict(pitch=2.0, r_out=0.7193, w_ring=0.1611, gap=0.5627, w=0.3624,
         t_mm=0.1)
AU_SIGMA = 4.561e7          # S/m, from tmat.h5 scatterer/material
F_MIN_THZ, F_MAX_THZ = 14.99, 37.47

# ---- pinned-domain numbers (see module docstring, item 2) ------------------
Z_PAD_UM = 3.0
LAMBDA_CENTER_UM = C_UM_THZ / (0.5 * (F_MIN_THZ + F_MAX_THZ))
AUTOSPACE_PER_SIDE_UM = LAMBDA_CENTER_UM / 4.0     # verified on run_v3
L_PAD_UM = 2.0 * Z_PAD_UM
L_EXPECTED_UM = L_PAD_UM + 2.0 * AUTOSPACE_PER_SIDE_UM

# ---- campaign angle set (par. 7) -------------------------------------------
THETAS_FULL = (0.0, 15.0, 30.0, 45.0, 60.0)
PHIS_FULL = (0.0, 22.5, 45.0)
THETAS_STARTER = (0.0, 30.0, 60.0)
PHIS_STARTER = (0.0, 22.5)
ACCURACY_TET_DEFAULT = "1e-4"
ACCURACY_TET_PERTURBED = "3e-4"

SOLVE_ORDER_NOTE = """\
--solve is INTENTIONALLY NOT IMPLEMENTED in this module (no solver may be
launched from here).  When solving the generated projects, par. 7 REQUIRES
this order:

  1. struct_th00_ph00 + empty_th00 FIRST, and re-establish the complex
     closure gate at normal incidence (deembed.closure_normal vs
     aggregation/results/periodic_results.npz, <= 5e-3): the pinned cellpad
     changes the mesh of the validated run_v3 configuration, so the
     normal-incidence closure must be re-earned before ANY oblique solve.
  2. struct_th30_ph00 + empty_th30 second: the channel-dictionary acceptance
     test -- all 8 de-embedded complex numbers must match the forward model
     evaluated with the reference T under EXACTLY ONE TE/TM label hypothesis
     to <= 1e-2 (deembed.select_hypothesis).
  3. Only then the remaining runs (empty_th00_pert last is fine; it feeds
     the noise-floor calibration, not the gates).
"""

# ---- VBA blocks ------------------------------------------------------------
# Copied from build_saw_unitcell.py where validated, edited per par. 7 where
# required.  Each function returns (label, vba) history steps.

VBA_UNITS = """\
With Units
    .SetUnit "Length", "um"
    .SetUnit "Frequency", "THz"
    .SetUnit "Time", "ps"
End With
"""

VBA_FREQ_RANGE = f'Solver.FrequencyRange "{F_MIN_THZ}", "{F_MAX_THZ}"\n'

# Base-script boundary block (types + REPORT par. 4b pitch trap), WITHOUT the
# scan angles: angles + direction now live in the Floquet step so both theta
# and phi are parametrized per run (par. 7 required edit 4).
VBA_BOUNDARY = f"""\
With Boundary
    .Xmin "unit cell"
    .Xmax "unit cell"
    .Ymin "unit cell"
    .Ymax "unit cell"
    .Zmin "expanded open"
    .Zmax "expanded open"
End With
' CRITICAL (REPORT par. 4b pitch trap): without an explicit unit-cell size
' CST fits the cell to the geometry bounding box (1.4386 um = the ring
' diameter), fusing neighboring rings into a connected reflective mesh.
With Boundary
    .UnitCellFitToBoundingBox "False"
    .UnitCellDs1 "{P['pitch']}"
    .UnitCellDs2 "{P['pitch']}"
    .UnitCellAngle "90.0"
End With
"""

# Zero explicit background padding (cma_infinite build_run_paper.py pattern):
# the domain derives from the cellpad bounding box + the deterministic
# "expanded open" auto-space only, identically in both project kinds.
VBA_BACKGROUND = """\
With Background
    .Type "Normal"
    .Epsilon "1.0"
    .Mu "1.0"
    .XminSpace "0.0"
    .XmaxSpace "0.0"
    .YminSpace "0.0"
    .YmaxSpace "0.0"
    .ZminSpace "0.0"
    .ZmaxSpace "0.0"
    .ApplyInAllDirections "False"
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


def vba_cellpad() -> str:
    """Pinned-domain vacuum brick, p x p x [-Z_PAD, +Z_PAD] (par. 7 edit 2;
    cma_infinite/cst/build_run_paper.py vba_cellpad pattern).  Present in
    BOTH structure and empty projects so removing the metal cannot change
    the domain (and so the empty project has a meshable solid at all --
    cf. the legacy run_empty 'Could not read mesh' failure)."""
    h = P["pitch"] / 2.0
    return f"""\
With Brick
    .Reset
    .Name "cellpad"
    .Component "component1"
    .Material "Vacuum"
    .Xrange "{-h}", "{h}"
    .Yrange "{-h}", "{h}"
    .Zrange "{-Z_PAD_UM}", "{Z_PAD_UM}"
    .Create
End With
"""


def geometry_vba(p: dict) -> str:
    """Ring (annulus) + 4 inward spokes, united into component1:saw.
    Copied verbatim from build_saw_unitcell.py (validated geometry)."""
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


def vba_floquet(theta_deg: float, phi_deg: float) -> str:
    """Floquet ports + scan angles, BOTH theta and phi parametrized.

    Ported from cma_infinite/cst/build_run_paper.py vba_floquet into the
    cst.interface stack of the base script (par. 7 required edit 4; the cma
    template hard-codes phi=0).  SetPeriodicBoundaryAnglesDirection "inward"
    is NORMATIVE (par. 7) -- build_run_paper.py says "inward",
    cst_common.py says "outward"; the conflict is resolved to "inward" and
    verified at de-embed time by the empty-cell S21 phase slope vs
    k_z = k cos(theta) (deembed.check_empty_phase).  Channel dictionary:
    exciting Zmax modes, the incident wave travels -z,
    k_hat = (sin th cos ph, sin th sin ph, -cos th) in CST coordinates.
    """
    th = f"{theta_deg:g}"
    ph = f"{phi_deg:g}"
    return f"""\
With FloquetPort
    .Reset
    .SetDialogTheta "{th}"
    .SetDialogPhi "{ph}"
    .SetPolarizationIndependentOfScanAnglePhi "0", "False"
    .SetSortCode "+beta/pw"
    .SetCustomizedListFlag "False"
    .Port "Zmax"
    .SetNumberOfModesConsidered "2"
    .SetDistanceToReferencePlane "0"
    .SetUseCircularPolarization "False"
    .Port "Zmin"
    .SetNumberOfModesConsidered "2"
    .SetDistanceToReferencePlane "0"
    .SetUseCircularPolarization "False"
End With
With Boundary
    .SetPeriodicBoundaryAngles "{th}", "{ph}"
    .SetPeriodicBoundaryAnglesDirection "inward"
End With
"""


def vba_fd_solver(accuracy_tet: str) -> str:
    """FD solver settings: the base script's AddSampleInterval frequency
    sampling KEPT (par. 7), excitation switched to "All","All" (par. 7
    required edit 3 -- both Floquet modes of the excitation port driven;
    doubles the excitation count).  AccuracyTet parametrized for the
    perturbed empty repeat (1e-4 -> 3e-4, noise-floor calibration)."""
    return f"""\
With FDSolver
    .Reset
    .Stimulation "All", "All"
    .ResetExcitationList
    .AddSampleInterval "{F_MIN_THZ}", "{F_MAX_THZ}", "", "Automatic", "True"
    .AccuracyTet "{accuracy_tet}"
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

# Local mesh refinement on the metal (base script rationale: the v1 run
# false-converged on a mesh that could not resolve the 161 nm ring / 100 nm
# thickness).  Structure projects only -- the target solid does not exist in
# the empty project; all OTHER mesh settings are identical in both kinds.
VBA_LOCAL_MESH = """\
With MeshSettings
    With .ItemMeshSettings ("solid$component1:saw")
        .SetMeshType "Tet"
        .Set "Size", "0.06"
    End With
End With
"""


# ---- campaign table --------------------------------------------------------

def _fmt_angle(x: float) -> str:
    """0.0 -> '00', 22.5 -> '22p5', 60.0 -> '60'."""
    if float(x) == int(x):
        return f"{int(x):02d}"
    return f"{x:g}".replace(".", "p")


def target_grid_THz() -> np.ndarray:
    """The tmat.h5 49-frequency grid: lambda = 20.0 : -0.25 : 8.0 um.

    Reconstructed from the closed form and cross-checked against the file /
    aggregation/results/periodic_results.npz when present (they match:
    freq[0] = 14.9896229 THz, freq[48] = 37.4740572 THz)."""
    lam = 20.0 - 0.25 * np.arange(49)
    f = C_UM_THZ / lam
    npz = HERE.parent / "aggregation" / "results" / "periodic_results.npz"
    if npz.exists():
        with np.load(npz) as z:
            f_ref = z["freq"] / 1e12
        if not np.allclose(f, f_ref, rtol=0, atol=1e-9):
            raise RuntimeError(
                "closed-form 49-point grid does not match "
                "periodic_results.npz -- refusing to write a wrong manifest")
    return f


def structure_history(theta_deg: float, phi_deg: float,
                      accuracy_tet: str = ACCURACY_TET_DEFAULT):
    """(label, vba) steps for a structure project."""
    return [
        ("campaign: Step 1 Units", VBA_UNITS),
        ("campaign: Step 2 Freq range", VBA_FREQ_RANGE),
        ("campaign: Step 3 Boundary", VBA_BOUNDARY),
        ("campaign: Step 4 Background", VBA_BACKGROUND),
        ("campaign: Step 5 Au material", VBA_AU),
        ("campaign: Step 6 Cellpad (pinned domain)", vba_cellpad()),
        ("campaign: Step 7 Geometry", geometry_vba(P)),
        ("campaign: Step 8 Floquet ports + scan angles",
         vba_floquet(theta_deg, phi_deg)),
        ("campaign: Step 9 FD solver + freq samples",
         vba_fd_solver(accuracy_tet)),
        ("campaign: Step 10 Mesh adaptation", VBA_MESH_ADAPT),
        ("campaign: Step 11 Local mesh on metal", VBA_LOCAL_MESH),
    ]


def empty_history(theta_deg: float,
                  accuracy_tet: str = ACCURACY_TET_DEFAULT):
    """(label, vba) steps for an empty (reference) project: identical to the
    structure history minus Au material, geometry and the metal-local mesh
    step (its target solid does not exist); cellpad + all other mesh/solver
    settings identical.  phi is irrelevant for the empty cell (S depends
    only on k_z) and is fixed to 0."""
    return [
        ("campaign: Step 1 Units", VBA_UNITS),
        ("campaign: Step 2 Freq range", VBA_FREQ_RANGE),
        ("campaign: Step 3 Boundary", VBA_BOUNDARY),
        ("campaign: Step 4 Background", VBA_BACKGROUND),
        ("campaign: Step 5 Cellpad (pinned domain)", vba_cellpad()),
        ("campaign: Step 6 Floquet ports + scan angles",
         vba_floquet(theta_deg, 0.0)),
        ("campaign: Step 7 FD solver + freq samples",
         vba_fd_solver(accuracy_tet)),
        ("campaign: Step 8 Mesh adaptation", VBA_MESH_ADAPT),
    ]


def expected_s_tree() -> list[str]:
    """The 8 S-tree entries the de-embedding consumes (par. 7 channel
    dictionary): reflection block SZmax(a),Zmax(b), transmission block
    SZmin(a),Zmax(b), a,b in {1,2}."""
    out = []
    for a in (1, 2):
        for b in (1, 2):
            out.append(f"1D Results\\S-Parameters\\SZmax({a}),Zmax({b})")
    for a in (1, 2):
        for b in (1, 2):
            out.append(f"1D Results\\S-Parameters\\SZmin({a}),Zmax({b})")
    return out


def build_campaign(runs_dir: Path = RUNS_DIR_DEFAULT) -> dict:
    """Assemble the full 19-run campaign table (par. 7) as a manifest dict.

    Structure runs: theta x phi grid minus the theta=0 phi-duplicates.
    Empty runs: one per theta at phi=0 (+ 1 perturbed repeat at theta=0).
    Starter-subset membership is flagged per run (in_starter)."""
    runs = []

    def starter_struct(th, ph):
        return (th in THETAS_STARTER and ph in PHIS_STARTER
                and not (th == 0.0 and ph != 0.0))

    for th in THETAS_FULL:
        phis = (0.0,) if th == 0.0 else PHIS_FULL
        for ph in phis:
            rid = f"struct_th{_fmt_angle(th)}_ph{_fmt_angle(ph)}"
            runs.append(dict(
                runid=rid, kind="structure", theta_deg=th, phi_deg=ph,
                accuracy_tet=ACCURACY_TET_DEFAULT,
                vba_path=str(runs_dir / rid / "build_history.vba"),
                project_path=str(runs_dir / rid / f"{rid}.cst"),
                expected_s_tree=expected_s_tree(),
                in_starter=starter_struct(th, ph)))
    for th in THETAS_FULL:
        rid = f"empty_th{_fmt_angle(th)}"
        runs.append(dict(
            runid=rid, kind="empty", theta_deg=th, phi_deg=0.0,
            accuracy_tet=ACCURACY_TET_DEFAULT,
            vba_path=str(runs_dir / rid / "build_history.vba"),
            project_path=str(runs_dir / rid / f"{rid}.cst"),
            expected_s_tree=expected_s_tree(),
            in_starter=th in THETAS_STARTER))
    rid = "empty_th00_pert"
    runs.append(dict(
        runid=rid, kind="empty_perturbed", theta_deg=0.0, phi_deg=0.0,
        accuracy_tet=ACCURACY_TET_PERTURBED,
        vba_path=str(runs_dir / rid / "build_history.vba"),
        project_path=str(runs_dir / rid / f"{rid}.cst"),
        expected_s_tree=expected_s_tree(),
        in_starter=False))

    return dict(
        campaign_id="floquet_oblique_v1",
        created=datetime.now().isoformat(timespec="seconds"),
        doc="INVERSE_TMATRIX_FROM_FLOQUET.md par. 7",
        pitch_um=P["pitch"], cell_area_um2=P["pitch"] ** 2,
        geometry_um=P, au_sigma_S_per_m=AU_SIGMA,
        band_THz=[F_MIN_THZ, F_MAX_THZ],
        frequency_sampling=dict(
            scheme="AddSampleInterval",
            args=[str(F_MIN_THZ), str(F_MAX_THZ), "", "Automatic", "True"],
            note=("CST band clips the 49-point target grid endpoints by "
                  "<= 0.004 THz; Re/Im np.interp clamps there "
                  "(build_saw_unitcell.py ~line 317 pattern)")),
        target_grid_THz=[float(f) for f in target_grid_THz()],
        z_pad_um=Z_PAD_UM,
        L_pad_um=L_PAD_UM,
        autospace_per_side_um=AUTOSPACE_PER_SIDE_UM,
        autospace_rule=("lambda_center/4 per side beyond the bounding box "
                        "('expanded open', CST 2026.0) -- verified to 10 "
                        "significant digits against run_v3's CADSurf "
                        "bounding-box diagonal 6.4661101155608458 um"),
        L_expected_um=L_EXPECTED_UM,
        angles_direction="inward",
        angles_direction_note=("normative par. 7; build_run_paper.py says "
                               "'inward', cst_common.py says 'outward'; "
                               "verified via deembed.check_empty_phase"),
        channel_dictionary=("excite Zmax modes 1,2; incident wave travels "
                            "-z: k_hat = (sin th cos ph, sin th sin ph, "
                            "-cos th) in CST coordinates"),
        stimulation=["All", "All"],
        floquet_modes_per_port=2,
        mesh=dict(adaption="HighFrequencyTet/Energy 3-8 passes, "
                           "ErrorLimit 1e-4",
                  local_metal_mesh_um=0.06,
                  note="local metal refinement exists only where metal "
                       "exists (structure runs); all other settings "
                       "identical in both kinds"),
        cst_data_convention=("e^{+j omega t}: CONJUGATE on load before any "
                             "physics (deembed.py)"),
        solve_order_note=SOLVE_ORDER_NOTE,
        runs=runs)


def history_for_run(run: dict):
    if run["kind"] == "structure":
        return structure_history(run["theta_deg"], run["phi_deg"],
                                 run["accuracy_tet"])
    return empty_history(run["theta_deg"], run["accuracy_tet"])


def serialize_history(steps) -> str:
    out = []
    for label, vba in steps:
        out.append(f"'==== STEP: {label} ====\n{vba}")
    return "\n".join(out)


# ---- dry-run generation + self-verification --------------------------------

def generate_dry_run(runs_dir: Path = RUNS_DIR_DEFAULT,
                     manifest: dict | None = None) -> dict:
    """Write per-run VBA + run.json + campaign_manifest.json.  No CST."""
    manifest = manifest or build_campaign(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    for run in manifest["runs"]:
        d = runs_dir / run["runid"]
        d.mkdir(parents=True, exist_ok=True)
        Path(run["vba_path"]).write_text(
            serialize_history(history_for_run(run)), encoding="utf-8")
        (d / "run.json").write_text(json.dumps(run, indent=2),
                                    encoding="utf-8")
    (runs_dir / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_campaign(manifest: dict) -> list[str]:
    """Assert the generated VBA implements every par. 7 required edit.
    Returns a list of human-readable check lines; raises AssertionError on
    any failure."""
    runs = manifest["runs"]
    checks = []
    n_struct = sum(r["kind"] == "structure" for r in runs)
    n_empty = sum(r["kind"] == "empty" for r in runs)
    n_pert = sum(r["kind"] == "empty_perturbed" for r in runs)
    assert len(runs) == 19, f"expected 19 runs, got {len(runs)}"
    assert (n_struct, n_empty, n_pert) == (13, 5, 1), \
        f"run-kind counts off: {n_struct}/{n_empty}/{n_pert}"
    checks.append(f"run counts: 19 total = {n_struct} structure + "
                  f"{n_empty} empty + {n_pert} perturbed empty")

    starter_s = [r for r in runs if r["in_starter"]
                 and r["kind"] == "structure"]
    starter_e = [r for r in runs if r["in_starter"] and r["kind"] == "empty"]
    assert len(starter_s) == 5, f"starter structure {len(starter_s)} != 5"
    assert len(starter_e) == 3, f"starter empty {len(starter_e)} != 3"
    checks.append("starter subset: 5 structure + 3 empty runs")

    struct_angles = sorted((r["theta_deg"], r["phi_deg"]) for r in runs
                           if r["kind"] == "structure")
    want = sorted({(th, ph) for th in THETAS_FULL
                   for ph in ((0.0,) if th == 0 else PHIS_FULL)})
    assert struct_angles == want, f"structure angle set wrong: {struct_angles}"
    empty_thetas = sorted(r["theta_deg"] for r in runs if r["kind"] == "empty")
    assert empty_thetas == sorted(THETAS_FULL), \
        f"empty theta set wrong: {empty_thetas}"
    checks.append("angle sets match par. 7 (13 structure pairs, "
                  "5 empty thetas, perturbed repeat at theta=0)")

    pitch = f"{P['pitch']}"
    for run in runs:
        vba = Path(run["vba_path"]).read_text(encoding="utf-8")
        rid = run["runid"]
        th, ph = f"{run['theta_deg']:g}", f"{run['phi_deg']:g}"
        assert '.Name "cellpad"' in vba and \
            f'.Zrange "{-Z_PAD_UM}", "{Z_PAD_UM}"' in vba, \
            f"{rid}: cellpad brick missing/wrong z-range"
        assert '.Stimulation "All", "All"' in vba, \
            f"{rid}: both-mode excitation missing"
        assert '.SetPeriodicBoundaryAnglesDirection "inward"' in vba, \
            f"{rid}: 'inward' missing"
        assert '.UnitCellFitToBoundingBox "False"' in vba \
            and f'.UnitCellDs1 "{pitch}"' in vba \
            and f'.UnitCellDs2 "{pitch}"' in vba, \
            f"{rid}: explicit Ds1/Ds2 pitch trap handling missing"
        assert f'.SetDialogTheta "{th}"' in vba \
            and f'.SetPeriodicBoundaryAngles "{th}", "{ph}"' in vba, \
            f"{rid}: scan angles wrong (want theta={th}, phi={ph})"
        assert vba.count('.SetNumberOfModesConsidered "2"') == 2, \
            f"{rid}: need 2 Floquet modes on both ports"
        assert f'.AccuracyTet "{run["accuracy_tet"]}"' in vba, \
            f"{rid}: AccuracyTet wrong"
        assert f'.AddSampleInterval "{F_MIN_THZ}", "{F_MAX_THZ}"' in vba, \
            f"{rid}: base AddSampleInterval scheme missing"
        if run["kind"] == "structure":
            assert 'Solid.Rename "component1:ring", "saw"' in vba \
                and '.Name "Au"' in vba, f"{rid}: saw geometry missing"
            assert f'.SetDialogPhi "{ph}"' in vba, f"{rid}: phi wrong"
        else:
            assert "saw" not in vba and '"Au"' not in vba, \
                f"{rid}: empty project contains metal"
    checks.append("per-run VBA: cellpad in BOTH kinds, All/All excitation, "
                  "'inward', explicit Ds1/Ds2, per-run theta/phi values, "
                  "2 modes on both ports, AccuracyTet per kind")
    assert len(manifest["target_grid_THz"]) == 49
    checks.append("manifest target grid: 49 frequencies "
                  f"({manifest['target_grid_THz'][0]:.6f} - "
                  f"{manifest['target_grid_THz'][-1]:.6f} THz)")
    return checks


def summary_table(manifest: dict) -> str:
    lines = [f"{'runid':<22} {'kind':<16} {'theta':>6} {'phi':>6} "
             f"{'AccTet':>7} {'starter':>7}"]
    for r in manifest["runs"]:
        lines.append(f"{r['runid']:<22} {r['kind']:<16} "
                     f"{r['theta_deg']:>6g} {r['phi_deg']:>6g} "
                     f"{r['accuracy_tet']:>7} "
                     f"{'yes' if r['in_starter'] else '-':>7}")
    lines.append(f"total runs: {len(manifest['runs'])}   "
                 f"L_pad = {L_PAD_UM} um   "
                 f"L_expected = {L_EXPECTED_UM:.6f} um   "
                 f"direction = inward")
    return "\n".join(lines)


# ---- --build: create projects via cst.interface, DO NOT SOLVE --------------

def build_projects(manifest: dict, starter_only: bool = False) -> int:
    """Create the .cst projects (cst.interface + nir.cst_helpers
    HistoryBuilder, the base script's automation stack).  NO SOLVER CALL
    exists in this module; solving is deferred (see SOLVE_ORDER_NOTE)."""
    ensure_on_path()
    try:
        import cst.interface as cstint
        from nir.cst_helpers import HistoryBuilder, save_project_at
    except ImportError as e:
        print(f"[FATAL] --build needs the CST python libraries at "
              f"{CST_PYTHON_LIB} and nir.cst_helpers at {AUTO_CST}: {e}")
        return 2

    runs = [r for r in manifest["runs"]
            if r["in_starter"] or not starter_only]
    print(f"building {len(runs)} projects (no solves)...")
    env = cstint.DesignEnvironment()
    try:
        for run in runs:
            target = Path(run["project_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                print(f"  [skip] {run['runid']}: {target} exists")
                continue
            print(f"  [build] {run['runid']} (theta={run['theta_deg']}, "
                  f"phi={run['phi_deg']}, {run['kind']})", flush=True)
            project = env.new_mws()
            save_project_at(project, target)
            builder = HistoryBuilder(project, verify=True)
            expects = {"cellpad": ["cellpad"], "saw": ["cellpad", "saw"]}
            for label, vba in history_for_run(run):
                kw = {}
                if "Geometry" in label:
                    kw["expects_solids"] = expects["saw"]
                elif "Cellpad" in label:
                    kw["expects_solids"] = expects["cellpad"]
                elif "Au material" in label:
                    kw["expects_materials"] = ["Au"]
                builder.add(label, vba, **kw)
            project.save()
            project.close()
    finally:
        env.close()
    print("all projects built; NOT solved (see --solve).")
    return 0


# ---- CLI -------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="par. 7 CST campaign generator (no solver launch here)")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR_DEFAULT)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=False,
                      help="write VBA + manifest, verify, print summary "
                           "(DEFAULT mode)")
    mode.add_argument("--build", action="store_true",
                      help="create .cst projects via cst.interface "
                           "(does NOT solve)")
    mode.add_argument("--solve", action="store_true",
                      help="intentionally not implemented")
    ap.add_argument("--starter", action="store_true",
                    help="with --build: only the starter subset")
    args = ap.parse_args(argv)

    if args.solve:
        print(SOLVE_ORDER_NOTE)
        return 1

    manifest = generate_dry_run(args.runs_dir)
    checks = verify_campaign(manifest)
    print(f"campaign manifest -> {args.runs_dir / 'campaign_manifest.json'}")
    for c in checks:
        print(f"  [ok] {c}")
    print()
    print(summary_table(manifest))

    if args.build:
        return build_projects(manifest, starter_only=args.starter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
