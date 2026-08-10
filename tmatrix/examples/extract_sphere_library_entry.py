"""Full library extraction: dielectric sphere, 3 frequencies, validated
against Mie theory.  This is the first real library entry AND the CST-in-
the-loop validation of the pipeline (a sphere is the one object whose T is
known exactly).

Estimated cost: lmax=3 -> N=30 modes -> 23 directions x 2 pols = 46 FD
solves.  Start it and let it run; every completed illumination is cached in
the local run dir, so it can be interrupted and resumed freely.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst_tmatrix.config import ExtractionConfig
from cst_tmatrix.pipeline import ScattererPlan, extract_tmatrix, C0
from cst_tmatrix.tmatrix_solve import sphere_reference_T
from cst_tmatrix.cst_driver import CSTSession
from cst_tmatrix.cst_driver import builder

RADIUS_UM = 40.0
EPS = 4.0
FREQS_THZ = np.array([0.5, 0.75, 1.0])
LMAX = 3


def build(h):
    builder.define_material(h, "diel_eps4", eps=EPS)
    builder.new_component(h, "scatterer")
    builder.build_sphere(h, RADIUS_UM, material="diel_eps4")


plan = ScattererPlan(
    name="sphere_eps4_r40um", build=build, r_circ_m=RADIUS_UM * 1e-6,
    freqs_hz=FREQS_THZ * 1e12, lmax=LMAX,
    metadata={"geometry": {"shape": "sphere", "radius": RADIUS_UM},
              "geometry_units": {"radius": "um"},
              "material": {"name": f"dielectric eps={EPS}",
                           "relative_permittivity": EPS,
                           "relative_permeability": 1.0}})

session = CSTSession()
T, diags, h5_path = extract_tmatrix(
    session, plan, ExtractionConfig(lmax=LMAX, illumination_factor=1.5))

print(f"\nSaved: {h5_path}")
print(f"{'f (THz)':>8} {'residual':>10} {'cond(A)':>8} {'max sv(S)':>10} "
      f"{'inc.dev':>8} {'vs Mie':>10}")
for jj, f_thz in enumerate(FREQS_THZ):
    k = 2 * np.pi * f_thz * 1e12 / C0
    T_mie = sphere_reference_T(LMAX, k * RADIUS_UM * 1e-6, np.sqrt(EPS))
    mie_err = np.max(np.abs(T[jj] - T_mie)) / np.max(np.abs(T_mie))
    print(f"{f_thz:8.3f} {diags['residual'][jj]:10.2e} "
          f"{diags['cond'][jj]:8.1f} {diags['s_max_sv'][jj]:10.6f} "
          f"{diags['incident_deviation_max'][jj]:8.1e} {mie_err:10.2e}")
print("\n'vs Mie' is the relative max-element error against the analytic "
      "T-matrix — the pipeline acceptance metric.")
