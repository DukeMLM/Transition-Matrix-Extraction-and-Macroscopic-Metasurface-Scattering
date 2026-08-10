r"""Validate the full CST extraction pipeline against an independent
non-analytic reference: the treams example TiO2 cylinder.

Reference: library/cylinder_tio2.tmat.h5 (downloaded from the treams
repository) -- radius 250 nm, height 300 nm, n = 2.5 (eps = 6.25), vacuum
embedding, lmax = 4, computed with JCMsuite (rotationally-symmetric 2D FEM,
built-in multipole expansion).  A completely independent solver AND
independent extraction method from this package's CST 3D tetrahedral FEM +
plane-wave/least-squares path, which is what makes the comparison a real
pipeline validation rather than a self-consistency check.

Reference structure was verified offline before this run (loader round
trip): m-block-diagonality exactly 0 (czinfinity), reciprocity 6e-10 in OUR
convention, S unitary to 3.7e-5.  |T| ~ 2-2.9, a STRONG scatterer -- the
weak-scatterer dynamic-range noise floor that dominated the Mie-sphere
comparison (see session notes 2026-08-05) does not apply here.

Comparison frequencies (exact reference grid points):
    320.0 THz  -- mid-band, off-resonance
    393.6 THz  -- extinction resonance 2 (strongest feature in the band)
High-band-only choice (2026-08-05, runtime decision): restricting to the
upper band raises f_min, which shrinks the monitor sphere (mf 4.09 -> 3.07)
and with it the whole meshed domain (~0.4x volume) -- the measured full
4-frequency/full-band configuration ran ~24 min per illumination (~28 h
total, 1.1M cells), which missed the required by-morning turnaround.  The
240/265.6 THz low-band comparison can be run later as a second campaign if
wanted.

Mesh: deterministic non-adaptive (mesh_adaption=False) at 30/12 steps per
wavelength -- the configuration validated against both the adaptive mesh and
the exact Mie solution earlier (requires the CellsPerWavelengthPolicy fix in
builder.set_tet_mesh, 2026-08-05).

Run on the CST machine:
    python validate_cylinder_tio2.py            # C4v-reduced (default), ~10 solves
    python validate_cylinder_tio2.py --full     # unreduced base path, 72 solves
    python validate_cylinder_tio2.py --compare-only [--full]  # reuse extracted h5

Mode note (2026-08-06): the C4v-REDUCED run is the default, adopted at
~00:45 when the measured unreduced pace (~10.3 min/illumination even on the
shrunk high-band domain) projected past the required by-morning finish.  The
cylinder is C-infinity-v, so C4v is exact for it.  Interpretation: a match
against the reference validates the illuminations->extraction path AND the
symmetry-reduction machinery jointly; if it were to disagree, the unreduced
--full run discriminates which is at fault.  8 of the 72 unreduced
illuminations are already cached in the __n2 run directory for that case.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst_tmatrix import vswf                                     # noqa: E402
from cst_tmatrix.config import ExtractionConfig, LIBRARY_ROOT    # noqa: E402
from cst_tmatrix.pipeline import (ScattererPlan, extract_tmatrix,  # noqa: E402
                                  recommended_monitor_factor, C0)
from cst_tmatrix.storage import load_tmatrix                     # noqa: E402
from cst_tmatrix.tmatrix_solve import (reciprocity_check,        # noqa: E402
                                       passivity_check)

# --- the reference and what we match against it --------------------------
REFERENCE = LIBRARY_ROOT / "cylinder_tio2.tmat.h5"
OUTPUT_FULL = LIBRARY_ROOT / "cylinder_tio2_cst.tmat.h5"
OUTPUT_C4V = LIBRARY_ROOT / "cylinder_tio2_cst_c4v.tmat.h5"

RADIUS_UM = 0.250
HEIGHT_UM = 0.300
EPS = 6.25                                  # n = 2.5, non-dispersive
LMAX = 4                                    # match the reference exactly
FREQS_THZ = np.array([320.0, 393.6])

R_CIRC_M = float(np.hypot(RADIUS_UM, HEIGHT_UM / 2) * 1e-6)   # 291.5 nm

MAX_CPUS = 6
MESH_STEPS = (30, 12)


def build(h):
    from cst_tmatrix.cst_driver import builder
    builder.define_material(h, "tio2_n2p5", eps=EPS)
    builder.new_component(h, "scatterer")
    builder.build_cylinder(h, RADIUS_UM, -HEIGHT_UM / 2, HEIGHT_UM / 2,
                           material="tio2_n2p5", name="cyl")


def run_extraction(use_c4v: bool):
    from cst_tmatrix.cst_driver import CSTSession

    freqs_hz = FREQS_THZ * 1e12
    mf = recommended_monitor_factor(R_CIRC_M, freqs_hz, LMAX)
    mode = "C4v-reduced" if use_c4v else "unreduced (full base path)"
    print(f"r_circ = {R_CIRC_M*1e6:.4f} um, lmax = {LMAX}, "
          f"auto monitor_factor = {mf:.3f}, mode = {mode}")
    plan = ScattererPlan(
        # distinct name per mode => distinct run directory, so the
        # index-keyed illumination caches of the two modes can never mix
        name="cylinder_tio2_cst_c4v" if use_c4v else "cylinder_tio2_cst",
        build=build,
        r_circ_m=R_CIRC_M,
        freqs_hz=freqs_hz,
        lmax=LMAX,
        monitor_factor=mf,
        length_unit="um",
        freq_unit="THz",
        mesh_steps_per_wave=MESH_STEPS,
        metadata={"geometry": "cylinder r=250nm h=300nm",
                  "material": "TiO2 n=2.5 (eps=6.25)",
                  "purpose": "pipeline validation vs treams/JCMsuite"},
    )
    if use_c4v:
        # exact for a C-infinity-v body; validated machinery (C4v tests)
        extraction = ExtractionConfig(
            lmax=LMAX, symmetry_n_fold=4,
            symmetry_mirror_phi0_deg=(0.0, 45.0, 90.0, 135.0))
    else:
        extraction = ExtractionConfig(lmax=LMAX)
    session = CSTSession()
    T, diags, out = extract_tmatrix(
        session, plan,
        extraction=extraction,
        solver_kwargs={"max_cpus": MAX_CPUS,
                       "hardware_acceleration": False,
                       "mesh_adaption": False},
        library_path=OUTPUT_C4V if use_c4v else OUTPUT_FULL)
    print(f"\nextraction written to {out}")
    return out


def compare(cst_path):
    ref = load_tmatrix(REFERENCE)
    cst = load_tmatrix(cst_path)
    lmax = cst["lmax"]
    assert ref["lmax"] == lmax == LMAX

    pol, ns, ms = vswf.mode_list(lmax)
    off_m = np.abs(ms[:, None] - ms[None, :]) > 0
    a_pw = vswf.plane_wave_coefficients(0.0, 0.0, "theta", lmax)

    print("\n" + "=" * 78)
    print("CST pipeline vs treams/JCMsuite reference (per frequency)")
    print("=" * 78)
    print(f"{'f[THz]':>8} {'|T_ref|':>8} {'|T_cst|':>8} {'rel diff':>10} "
          f"{'m-offdiag':>10} {'recip':>10} {'max sv(S)':>10} {'ext ratio':>10}")
    worst = 0.0
    for i, f_thz in enumerate(FREQS_THZ):
        j = int(np.argmin(np.abs(ref["frequencies"] - f_thz * 1e12)))
        assert abs(ref["frequencies"][j] - f_thz * 1e12) < 1e6, \
            f"{f_thz} THz is not on the reference grid"
        T_r, T_c = ref["tmatrix"][j], cst["tmatrix"][i]
        rel = np.linalg.norm(T_c - T_r) / np.linalg.norm(T_r)
        worst = max(worst, rel)
        moff = np.linalg.norm(T_c[off_m]) / np.linalg.norm(T_c)
        rec = reciprocity_check(T_c, lmax)
        smax, _uni = passivity_check(T_c)
        # optical-theorem proxy at normal incidence, ratio CST/reference
        ext_r = -np.real(np.vdot(a_pw, T_r @ a_pw))
        ext_c = -np.real(np.vdot(a_pw, T_c @ a_pw))
        print(f"{f_thz:>8.1f} {np.linalg.norm(T_r):>8.4f} "
              f"{np.linalg.norm(T_c):>8.4f} {rel:>10.3e} {moff:>10.3e} "
              f"{rec:>10.3e} {smax:>10.6f} {ext_c/ext_r:>10.4f}")

    # dominant-entry comparison at the strongest resonance
    i, f_thz = len(FREQS_THZ) - 1, FREQS_THZ[-1]
    j = int(np.argmin(np.abs(ref["frequencies"] - f_thz * 1e12)))
    T_r, T_c = ref["tmatrix"][j], cst["tmatrix"][i]
    idx = np.unravel_index(np.argsort(-np.abs(T_r), axis=None)[:6], T_r.shape)
    print(f"\nsix largest reference entries at {f_thz} THz:")
    print(f"{'(pol,l,m) <- (pol,l,m)':>26} {'T_ref':>24} {'T_cst':>24} "
          f"{'rel':>9}")
    for r, c in zip(*idx):
        lab = (f"({pol[r]},{ns[r]},{ms[r]:+d}) <- "
               f"({pol[c]},{ns[c]},{ms[c]:+d})")
        e = abs(T_c[r, c] - T_r[r, c]) / abs(T_r[r, c])
        print(f"{lab:>26} {T_r[r, c]:>24.4e} {T_c[r, c]:>24.4e} {e:>9.2e}")

    print("\n" + "=" * 78)
    print(f"WORST per-frequency relative Frobenius difference: {worst:.3e}")
    print("Reference-only context: JCMsuite's own convergence and the l<=4 "
          "truncation\nput a floor well below any CST discretization error; "
          "differences here are\nours, not the reference's.")
    print("=" * 78)


if __name__ == "__main__":
    use_c4v = "--full" not in sys.argv
    if "--compare-only" not in sys.argv:
        out = run_extraction(use_c4v)
    else:
        out = OUTPUT_C4V if use_c4v else OUTPUT_FULL
    compare(out)
