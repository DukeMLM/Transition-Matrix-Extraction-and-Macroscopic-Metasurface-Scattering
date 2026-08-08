"""Direct CST reference for an x,y;y,x periodic supercell (manual 6.5.5 test 4).

Builds, meshes and solves a 16 x 16 um free-standing unit cell containing four
spoke-and-wheel gold resonators in a checkerboard -- for --pair AB,

        (-4, +4) B     (+4, +4) A
        (-4, -4) A     (+4, -4) B

The atoms come from the parametric family stored in
`test/2x2/SAW_gold_noSub_packed.cst`: A = `scale` 4.0 (3D Run ID 6),
B = `scale` 5.0 (run 2), C = `scale` 3.25 (run 10), i.e. exactly the meta-atoms
whose isolated T-matrices are saw_gold_wl13p10um / wl17p30um / wl10p90um.
Every solver, mesh, material and boundary setting is copied from that project's
own periodic run (its ModelHistory.json, steps 4/5/163/169/176-179), so the
comparison is like for like: the only difference is the cell contents.

Two projects are produced:

  supercell/   the structure
  empty/       the identical cell with no metal.  Its S(Zmax(m), Zmin(m)) is
               exactly e^{-j k_z,m L}, which measures the Floquet port-plane
               separation L that CST inserts with "expanded open" boundaries --
               so the de-embedding to z = 0 is measured, not fitted.  It is
               also the manual's section-8 row-7 background test.

Reference planes.  CST puts the Floquet ports on the Zmin/Zmax boundaries.  For
a z-symmetric model, transmission from the (0,0) mode into order G travels
L/2 at k_z,0 and L/2 at k_z,G; reflection into G travels the same two legs.
Both therefore de-embed to z = 0 with the SAME factor exp(+j (k_z,0 + k_z,G) L/2),
which `read_supercell_results.py` applies before conjugating CST's e^{+j w t}
to this repository's e^{-i w t}.

    python build_2x2_supercell.py --pair AB --dry-run   # build + save, no solve
    python build_2x2_supercell.py --pair AB            # build + solve both
    python build_2x2_supercell.py --pair AC --only supercell

The empty companion run depends only on the cell, not on its contents, so one
`--only empty` run serves every pair; point read_supercell_results.py --empty at
it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

AUTO_CST = Path("D:/Claude/auto_cst")
if str(AUTO_CST) not in sys.path:
    sys.path.insert(0, str(AUTO_CST))
CST_PYTHON_LIB = r"E:\cst\AMD64\python_cst_libraries"
if CST_PYTHON_LIB not in sys.path:
    sys.path.insert(0, CST_PYTHON_LIB)

import cst.interface as cstint  # noqa: E402

from nir.cst_helpers import HistoryBuilder, save_project_at  # noqa: E402

HERE = Path(__file__).resolve().parent

# ---- design (um), from Result/db.parmap of SAW_gold_noSub_packed.cst --------
# p = 8 in every row of that sweep; `run` is the 3D Run ID whose periodic
# solution is the direct reference for that atom on its own 8 um lattice.
ATOMS = {
    "A": dict(name="A", scale=4.00, r=2.87712, w_ring=0.644552, gap=2.800,
              w=0.80, t=0.2, run=6),
    "B": dict(name="B", scale=5.00, r=3.59639, w_ring=0.805690, gap=3.500,
              w=1.00, t=0.2, run=2),
    "C": dict(name="C", scale=3.25, r=2.33766, w_ring=0.523698, gap=2.275,
              w=0.65, t=0.2, run=10),
}
BASE_PITCH = 8.0                     # atom-to-atom spacing
SUPER = 2 * BASE_PITCH               # supercell period, um
ZH = 25.0                            # half height of the vacuum cell, um
F_MIN, F_MAX = 10.0, 34.0            # THz, the tmat.h5 band
N_FLOQUET_MODES = 20                 # >= 18 propagating at 34 THz (9 orders x 2)
AU_SIGMA = 45610000.0                # S/m, packed project step 5
AU_RHO = 19320.0

def layout(pair):
    """x,y;y,x checkerboard for a two-letter pair such as 'AB' or 'AC'."""
    first, second = ATOMS[pair[0]], ATOMS[pair[1]]
    h = BASE_PITCH / 2
    return [(-h, -h, first), (+h, +h, first), (+h, -h, second), (-h, +h, second)]


VBA_UNITS = """\
With Units
  .SetUnit "Length", "um"
  .SetUnit "Frequency", "THz"
  .SetUnit "Time", "ps"
  .SetUnit "Temperature", "K"
End With
"""

VBA_FREQ = f'Solver.FrequencyRange "{F_MIN}", "{F_MAX}"\n'

VBA_BACKGROUND = """\
With Background
  .ResetBackground
  .Type "normal"
  .Epsilon "1.0"
  .Mu "1.0"
  .XminSpace "0"
  .XmaxSpace "0"
  .YminSpace "0"
  .YmaxSpace "0"
  .ZminSpace "0"
  .ZmaxSpace "0"
  .ApplyInAllDirections "True"
End With
"""

# Packed project step 169, but with the cell size stated explicitly instead of
# inferred from the bounding box.  README warning: with FitToBoundingBox the
# period silently collapses onto the geometry extent.  Here the vacuum brick is
# exactly SUPER x SUPER anyway, so the two agree -- stating it makes that a
# check rather than a hope.
VBA_BOUNDARY = f"""\
With Boundary
  .Xmin "unit cell"
  .Xmax "unit cell"
  .Ymin "unit cell"
  .Ymax "unit cell"
  .Zmin "expanded open"
  .Zmax "expanded open"
  .Xsymmetry "none"
  .Ysymmetry "none"
  .Zsymmetry "none"
  .ApplyInAllDirections "False"
  .OpenAddSpaceFactor "0.5"
  .XPeriodicShift "0.0"
  .YPeriodicShift "0.0"
  .ZPeriodicShift "0.0"
  .PeriodicUseConstantAngles "False"
  .SetPeriodicBoundaryAngles "0.0", "0.0"
  .SetPeriodicBoundaryAnglesDirection "outward"
  .UnitCellFitToBoundingBox "False"
  .UnitCellDs1 "{SUPER}"
  .UnitCellDs2 "{SUPER}"
  .UnitCellAngle "90.0"
End With
"""

VBA_MESH_TYPE = """\
Mesh.MeshType "Tetrahedral"
With MeshSettings
  .SetMeshType "Tet"
  .Set "StepsPerWaveNear", "15"
  .Set "StepsPerWaveFar", "6"
  .Set "CurvatureOrder", "2"
End With
"""

VBA_GOLD = f"""\
With Material
  .Reset
  .Name "gold"
  .Folder ""
  .FrqType "all"
  .Type "Lossy metal"
  .MaterialUnit "Frequency", "THz"
  .MaterialUnit "Geometry", "um"
  .Mu "1.0"
  .Sigma "{AU_SIGMA:.0f}"
  .Rho "{AU_RHO}"
  .Colour "1", "0.84", "0"
  .Create
End With
"""

VBA_COMPONENT = 'Component.New "scatterer"\n'

VBA_MESH_ADAPT = """\
With MeshAdaption3D
  .SetType "HighFrequencyTet"
  .SetAdaptionStrategy "ExpertSystem"
  .MinPasses "3"
  .MaxPasses "8"
  .ClearStopCriteria
  .MaxDeltaS "0.01"
  .NumberOfDeltaSChecks "1"
  .EnableInnerSParameterAdaptation "True"
  .PropagationConstantAccuracy "0.005"
  .NumberOfPropConstChecks "2"
  .EnablePortPropagationConstantAdaptation "True"
  .AddStopCriterion "All S-Parameters", "0.01", "1", "True"
  .AddStopCriterion "Portmode kz/k0", "0.005", "2", "True"
  .MinimumAcceptedCellGrowth "0.5"
  .SetMinimumMeshCellGrowth "5"
  .ErrorEstimatorType "Automatic"
  .RefinementType "Automatic"
  .SnapToGeometry "True"
  .WavelengthBasedRefinement "True"
  .EnableLinearGrowthLimitation "True"
  .SingularEdgeRefinement "2"
End With
"""

VBA_FLOQUET = f"""\
With FloquetPort
  .Reset
  .SetDialogTheta "0"
  .SetDialogPhi "0"
  .SetPolarizationIndependentOfScanAnglePhi "0.0", "False"
  .SetSortCode "+beta/pw"
  .SetCustomizedListFlag "False"
  .Port "Zmin"
  .SetNumberOfModesConsidered "{N_FLOQUET_MODES}"
  .SetDistanceToReferencePlane "0.0"
  .SetUseCircularPolarization "False"
  .Port "Zmax"
  .SetNumberOfModesConsidered "{N_FLOQUET_MODES}"
  .SetDistanceToReferencePlane "0.0"
  .SetUseCircularPolarization "False"
End With
"""

VBA_SOLVER = f"""\
ChangeSolverType "HF Frequency Domain"
Mesh.SetCreator "High Frequency"
With FDSolver
  .Reset
  .SetMethod "Tetrahedral", "General purpose"
  .OrderTet "Second"
  .OrderSrf "First"
  .Stimulation "Zmin", "TE(0,0)"
  .ResetExcitationList
  .AutoNormImpedance "False"
  .NormingImpedance "50"
  .ModesOnly "False"
  .ConsiderPortLossesTet "True"
  .AccuracyTet "0.0001"
  .LimitIterations "False"
  .MaxIterations "0"
  .StoreAllResults "False"
  .StoreResultsInCache "False"
  .UseHelmholtzEquation "True"
  .LowFrequencyStabilization "True"
  .Type "Auto"
  .MeshAdaptionHex "False"
  .MeshAdaptionTet "True"
  .AcceleratedRestart "True"
  .FreqDistAdaptMode "Distributed"
  .NewIterativeSolver "True"
  .AddMonitorSamples "True"
  .CalcPowerLoss "True"
  .SetKeepSolutionCoefficients "MonitorsAndMeshAdaptation"
  .UseDoublePrecision "False"
  .UseDoublePrecision_ML "True"
  .RemoveAllStopCriteria "Tet"
  .AddStopCriterion "All S-Parameters", "0.01", "2", "Tet", "True"
  .SweepMinimumSamples "3"
  .SetResultDataSamplingMode "Automatic"
  .SweepWeightEvanescent "1.0"
  .ResetSampleIntervals "all"
  .AddSampleInterval "{F_MIN}", "{F_MAX}", "", "Automatic", "True"
End With
"""
# (.UseFastFrequencySweep is in the packed project's history but is not a
#  property of FDSolver in this installation -- (10091) no such method.  The
#  broadband model-order-reduction sweep is what .Type "Auto" selects anyway.)


def atom_vba(cx, cy, atom, tag):
    """Ring + 4 inward spokes centred at (cx, cy), united into scatterer:<tag>."""
    r, wr, gap, w, t = (atom["r"], atom["w_ring"], atom["gap"], atom["w"],
                        atom["t"])
    z0, z1 = -t / 2, t / 2
    tip, root = gap / 2, r - wr / 2
    hw = w / 2
    out = [f"""\
With Cylinder
  .Reset
  .Name "{tag}"
  .Component "scatterer"
  .Material "gold"
  .OuterRadius "{r}"
  .InnerRadius "{r - wr}"
  .Axis "z"
  .Zrange "{z0}", "{z1}"
  .Xcenter "{cx}"
  .Ycenter "{cy}"
  .Segments "0"
  .Create
End With
"""]
    spokes = [(f"{tag}_px", cx + tip, cx + root, cy - hw, cy + hw),
              (f"{tag}_mx", cx - root, cx - tip, cy - hw, cy + hw),
              (f"{tag}_py", cx - hw, cx + hw, cy + tip, cy + root),
              (f"{tag}_my", cx - hw, cx + hw, cy - root, cy - tip)]
    for nm, x0, x1, y0, y1 in spokes:
        out.append(f"""\
With Brick
  .Reset
  .Name "{nm}"
  .Component "scatterer"
  .Material "gold"
  .Xrange "{x0}", "{x1}"
  .Yrange "{y0}", "{y1}"
  .Zrange "{z0}", "{z1}"
  .Create
End With
""")
    for nm, *_ in spokes:
        out.append(f'Solid.Add "scatterer:{tag}", "scatterer:{nm}"\n')
    return "".join(out)


def bbox_vba(tags):
    """Vacuum cell brick; the metal is inserted so nothing overlaps."""
    out = [f"""\
With Brick
  .Reset
  .Name "bbox"
  .Component "scatterer"
  .Material "Vacuum"
  .Xrange "{-SUPER/2}", "{SUPER/2}"
  .Yrange "{-SUPER/2}", "{SUPER/2}"
  .Zrange "{-ZH}", "{ZH}"
  .Create
End With
"""]
    for tag in tags:
        out.append(f'Solid.Insert "scatterer:bbox", "scatterer:{tag}"\n')
    return "".join(out)


def build(run_dir: Path, sites, log, dry_run=False):
    """`sites` is a layout() list, or empty for the metal-free companion run."""
    with_metal = bool(sites)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / ("supercell.cst" if with_metal else "empty.cst")
    if target.exists():
        target.unlink()
    folder = target.with_suffix("")
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)

    log(f"opening CST DesignEnvironment for {target.name} ...")
    env = cstint.DesignEnvironment()
    project = env.new_mws()
    save_project_at(project, target)
    builder = HistoryBuilder(project, verify=True)
    tags = []
    try:
        builder.add("sc: 1 units", VBA_UNITS)
        builder.add("sc: 2 frequency range", VBA_FREQ)
        builder.add("sc: 3 background", VBA_BACKGROUND)
        builder.add("sc: 4 boundaries (unit cell 16 um)", VBA_BOUNDARY)
        builder.add("sc: 5 mesh type", VBA_MESH_TYPE)
        builder.add("sc: 6 gold", VBA_GOLD, expects_materials=["gold"])
        builder.add("sc: 7 component", VBA_COMPONENT)
        if with_metal:
            for i, (cx, cy, atom) in enumerate(sites):
                tag = f"{atom['name'].lower()}{i}"
                tags.append(tag)
                builder.add(f"sc: 8.{i} atom {atom['name']} at "
                            f"({cx:+.1f}, {cy:+.1f})",
                            atom_vba(cx, cy, atom, tag),
                            expects_solids=[f"scatterer:{tag}"])
        builder.add("sc: 9 vacuum cell brick", bbox_vba(tags),
                    expects_solids=["scatterer:bbox"])
        builder.add("sc: 10 mesh adaptation", VBA_MESH_ADAPT)
        builder.add("sc: 11 Floquet ports", VBA_FLOQUET)
        builder.add("sc: 12 FD solver", VBA_SOLVER)
    except RuntimeError as exc:
        log(f"[FATAL] VBA failure:\n{exc}\n{builder.summary()}")
        try:
            project.save(); project.close(); env.close()
        except Exception:
            pass
        raise
    log("history applied and verified:")
    for line in builder.summary().splitlines():
        log("  " + line)
    project.save()

    if dry_run:
        log("[DRY RUN] project saved, solver not started")
        project.close(); env.close()
        return target

    log("starting FD solver ...")
    t0 = time.time()
    try:
        project.model3d.FDSolver.Start()
        log(f"FDSolver finished after {time.time() - t0:.0f} s")
    except Exception as exc:
        log(f"[ERROR] FDSolver: {exc}")
        project.save(); project.close(); env.close()
        raise
    project.save()
    project.close()
    env.close()
    return target


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="AB",
                    help="two atom letters, e.g. AB, AC, BC: the first goes on "
                         "one diagonal of the cell and the second on the other")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: runs_<pair>")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=("supercell", "empty"), default=None)
    args = ap.parse_args(argv)
    args.pair = args.pair.upper()
    if len(args.pair) != 2 or any(c not in ATOMS for c in args.pair):
        ap.error(f"--pair must be two letters from {sorted(ATOMS)}")
    if args.out is None:
        args.out = HERE / f"runs_{args.pair}"

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "build_log.txt"

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    sites = layout(args.pair)
    meta = dict(pair=args.pair,
                atoms={c: ATOMS[c] for c in args.pair},
                base_pitch=BASE_PITCH,
                supercell=SUPER, z_half=ZH, f_min_THz=F_MIN, f_max_THz=F_MAX,
                n_floquet_modes=N_FLOQUET_MODES, au_sigma=AU_SIGMA,
                layout=[(cx, cy, a["name"]) for cx, cy, a in sites])
    (args.out / "design.json").write_text(json.dumps(meta, indent=2))
    log(f"design: {json.dumps(meta)}")
    lam_ray = SUPER / np.sqrt(2)
    log(f"Rayleigh onsets: lambda = {SUPER:.2f} um (orders +-1,0 / 0,+-1; "
        f"extinguished by the x,y;y,x symmetry) and "
        f"{lam_ray:.2f} um = {299.792458/lam_ray:.2f} THz (orders +-1,+-1)")

    todo = [args.only] if args.only else ["empty", "supercell"]
    for which in todo:
        log(f"===== building {which} =====")
        build(args.out / which, sites if which == "supercell" else [], log,
              dry_run=args.dry_run)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
