r"""Standalone symmetry checker: geometry (CAD) -> point group -> T-matrix
conformance, in one pass.

Two independent questions, chained:
1. What point group does the SOLID actually have?  Exports the named
   solid to STL via CST's STL object (Model3D.STL.WriteCAD) and runs
   cst_tmatrix.postprocess.symmetry.detect_axial_point_group() -- a
   numerical point-cloud symmetry search about the z-axis, independent of
   whatever the build script's parametrization claims.
2. Does the EXTRACTED T-matrix conform to that point group, within
   extraction noise?  Feeds the detected rotation order and mirror
   azimuths into point_group_report() against a stored tmat.h5.

Two ways to supply the geometry:
- --cst-file PATH [--solid component:name]: open an EXISTING .cst project
  (any project -- does not have to come from this package) and check one
  of its solids.  Use --list-solids first if you don't know the exact
  component:name path.
- --wl (default): build one of this package's own spoke-wheel designs
  in a scratch project, as before.

Usage:
    # an arbitrary existing CST project
    python check_symmetry.py --cst-file "C:\path\to\some_project.cst" --list-solids
    python check_symmetry.py --cst-file "C:\path\to\some_project.cst" --solid patch:metal1

    # this package's own spoke-wheel design generator + a stored T-matrix
    python check_symmetry.py --wl 15 --tmat ..\library\saw_gold_wl15p0025um.tmat.h5

Run on the CST machine (needs a live CST connection for the STL export);
--tmat is optional -- omit it to run the geometry check alone.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst_tmatrix.postprocess.symmetry import (  # noqa: E402
    detect_axial_point_group, point_group_report)


def list_solids(h):
    """component:name for every solid in the model tree (Components\\...)."""
    names = []
    for item in h.m3d.get_tree_items():
        parts = item.split("\\")
        if len(parts) == 3 and parts[0] == "Components":
            names.append(f"{parts[1]}:{parts[2]}")
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cst-file", type=Path, default=None,
                    help="open this EXISTING .cst project instead of "
                         "building a spoke-wheel design")
    ap.add_argument("--list-solids", action="store_true",
                    help="print component:name for every solid in the "
                         "opened project (with --cst-file) and exit")
    ap.add_argument("--design-file", type=Path, default=None,
                    help="(spoke-wheel mode) design list; default: the "
                         "one shipped next to extract_spoke_wheel_scaled.py")
    ap.add_argument("--wl", type=float, default=15.0,
                    help="(spoke-wheel mode) pick the design nearest this "
                         "center wavelength (um)")
    ap.add_argument("--solid", default=None,
                    help="component:solid path to export; default "
                         "'scatterer:ring' in spoke-wheel mode, REQUIRED "
                         "with --cst-file unless --list-solids")
    ap.add_argument("--scratch-dir", type=Path,
                    default=Path.home() / "cst_tmatrix_runs" / "symcheck")
    ap.add_argument("--tmat", type=Path, default=None,
                    help="optional tmat.h5 to check for conformance with "
                         "the detected point group")
    args = ap.parse_args()

    from cst_tmatrix.cst_driver import CSTSession
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    session = CSTSession()

    if args.cst_file is not None:
        h = session.open_project(args.cst_file)
        if args.list_solids:
            for name in list_solids(h):
                print(name)
            return
        if args.solid is None:
            raise SystemExit(
                "--solid component:name is required with --cst-file "
                "(run with --list-solids first to see what's available)")
    else:
        from extract_spoke_wheel_scaled import (DESIGN_FILE, load_designs,
                                                make_build)
        from cst_tmatrix.cst_driver import builder
        design_file = args.design_file or DESIGN_FILE
        designs = load_designs(design_file)
        design = min(designs, key=lambda d: abs(d["wl"] - args.wl))
        print(f"design: wl={design['wl']:.4f} um  r={design['r']:.6g} "
              f"gap={design['gap']:.6g} w_ring={design['w_ring']:.6g} "
              f"w={design['w']:.6g}")
        h = session.new_mws_project(args.scratch_dir / "symcheck.cst")
        builder.set_units(h, length="um", frequency="THz")
        make_build(design)(h)
        h.save()
        if args.solid is None:
            args.solid = "scatterer:ring"

    component, solid = args.solid.split(":", 1)
    stl_path = args.scratch_dir / "scatterer.stl"
    if stl_path.exists():
        stl_path.unlink()
    h.execute_vba(f"""With STL
  .Reset
  .FileName "{stl_path}"
  .Name "{solid}"
  .Component "{component}"
  .ExportFileUnits "um"
  .ScaleToUnit "False"
  .SurfaceTolerance "0.001"
  .NormalTolerance "5"
  .WriteCAD
End With""")

    geo = detect_axial_point_group(stl_path)
    print(f"\ngeometry point group (about z): {geo['label']}"
          f"  (n_points={geo['n_points']}, axis_is_principal="
          f"{geo['axis_is_principal']}, tol={geo['tol']:.2e})")
    if geo["mirror_planes_deg"]:
        print(f"  mirror planes at (deg): {geo['mirror_planes_deg']}")

    if args.tmat is not None:
        from cst_tmatrix.storage import load_tmatrix
        import numpy as np
        d = load_tmatrix(args.tmat)
        T, lmax, freqs = d["tmatrix"], d["lmax"], d["frequencies"]
        n_fold = geo["rotation_order"] if geo["rotation_order"] > 1 else None
        mirrors = geo["mirror_planes_deg"] or None
        print(f"\nT-matrix conformance ({args.tmat.name}, "
              f"{len(freqs)} frequencies):")
        print(f"{'f (THz)':>10} " + " ".join(
            f"{k:>14}" for k in point_group_report(
                T[0], lmax, n_fold=n_fold, mirror_phi0_deg=mirrors)))
        for jj in range(len(freqs)):
            rep = point_group_report(T[jj], lmax, n_fold=n_fold,
                                     mirror_phi0_deg=mirrors)
            print(f"{freqs[jj] / 1e12:10.4f} " +
                  " ".join(f"{v:14.3e}" for v in rep.values()))


if __name__ == "__main__":
    main()
