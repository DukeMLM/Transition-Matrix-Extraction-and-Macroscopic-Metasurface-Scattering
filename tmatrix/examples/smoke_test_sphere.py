"""Smoke test: ONE CST solve validates the entire chain.

Builds a dielectric sphere (eps=4, radius 40 um), runs a single plane-wave
FD solve at 0.75 THz, exports E/H on the monitor sphere, and checks:

  1. incident-coefficient deviation: the measured regular-wave coefficients
     must match the analytic plane-wave expansion up to one complex scale
     (validates units, geometry, point-file format, parsing, projection);
  2. physics: measured scattered coefficients vs Mie prediction
     f_pred = T_mie @ a_measured (validates the whole extraction physics).

Expected: deviation ~< 1e-2 (FEM accuracy), Mie mismatch ~< a few %.
Requires a Python environment with the CST Studio Suite interface (see
README) and a licensed CST installation:
  python examples/smoke_test_sphere.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst_tmatrix import vswf
from cst_tmatrix.config import RunPaths, LOCAL_RUN_ROOT, ExtractionConfig
from cst_tmatrix.pipeline import ScattererPlan, build_project, C0
from cst_tmatrix.quadrature import quadrature_for_monitor, unit_vectors
from cst_tmatrix.tmatrix_solve import sphere_reference_T
from cst_tmatrix.cst_driver import CSTSession
from cst_tmatrix.cst_driver import builder, monitors, excitation, export

# --- scenario --------------------------------------------------------------
RADIUS_UM = 40.0
EPS = 4.0
FREQ_THZ = 0.75
LMAX = 3
THETA_I, PHI_I, POL = 0.7, 1.1, "theta"

f_hz = FREQ_THZ * 1e12
k = 2 * np.pi * f_hz / C0
r_circ_m = RADIUS_UM * 1e-6
r_mon_m = 1.5 * r_circ_m
x_size = k * r_circ_m
m_rel = np.sqrt(EPS)
print(f"size parameter ka = {x_size:.3f}, lmax = {LMAX}, "
      f"monitor radius = {r_mon_m*1e6:.1f} um")


def build(h):
    builder.define_material(h, "diel_eps4", eps=EPS)
    builder.new_component(h, "scatterer")
    builder.build_sphere(h, RADIUS_UM, material="diel_eps4")


plan = ScattererPlan(name="smoke_sphere", build=build, r_circ_m=r_circ_m,
                     freqs_hz=np.array([f_hz]), lmax=LMAX)

session = CSTSession()
paths = RunPaths(plan.name, LOCAL_RUN_ROOT).ensure()
h, freqs_proj, r_mon_proj = build_project(session, plan, paths,
                                          ExtractionConfig(lmax=LMAX))

# quadrature + point file
th_q, ph_q, w_q, qshape = quadrature_for_monitor(LMAX, k, r_mon_m)
pts_m = r_mon_m * unit_vectors(th_q, ph_q)
pts_proj = pts_m / 1e-6                       # project unit um
point_file = export.write_point_file(paths.export_dir / "sphere_points.txt",
                                     pts_proj)
print(f"quadrature grid {qshape[0]} x {qshape[1]} = {len(th_q)} points")

# one illumination, one solve
excitation.set_plane_wave(h, THETA_I, PHI_I, POL)
h.save()
print("running FD solver ...")
h.run_solver()
print("solver done; exporting fields ...")

E, ZH = export.get_EH_on_points(
    h, monitors.efield_tree_path(freqs_proj[0]),
    monitors.hfield_tree_path(freqs_proj[0]),
    paths.export_dir, "smoke", point_file, pts_proj)

a_meas, f_meas = vswf.separate_surface_field(E, ZH, LMAX, k, r_mon_m,
                                             th_q, ph_q, w_q)

# check 1: incident against analytic expansion (one complex scale allowed)
a_nom = vswf.plane_wave_coefficients(THETA_I, PHI_I, POL, LMAX)
scale = (a_meas @ np.conj(a_nom)) / np.vdot(a_nom, a_nom).real
dev = np.linalg.norm(a_meas - scale * a_nom) / np.linalg.norm(a_meas)
print(f"\n[1] incident check: scale = {abs(scale):.4f} * exp({np.angle(scale):+.4f}j)")
print(f"    relative deviation from analytic plane wave = {dev:.3e}  (want <~1e-2)")

# check 2: scattered against Mie prediction
T_mie = sphere_reference_T(LMAX, x_size, m_rel)
f_pred = T_mie @ a_meas
num = np.linalg.norm(f_meas - f_pred)
den = np.linalg.norm(f_pred)
print(f"[2] Mie check: |f_meas - T_mie a_meas| / |T_mie a_meas| = {num/den:.3e}"
      f"  (want <~5e-2)")
print("\nSmoke test complete." if dev < 0.05 and num / den < 0.1 else
      "\nSMOKE TEST FAILED — inspect deviations above before proceeding.")
