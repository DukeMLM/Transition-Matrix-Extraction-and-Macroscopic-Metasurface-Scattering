"""T-matrix extraction from a manually prepared CST project.

For geometries that are more convenient to build interactively than to
script, prepare a CST project that contains ONLY the scatterer geometry and
its materials, then let the pipeline do everything else.  Requirements on
the template project (verified automatically where possible):

  - geometry centered at the origin (the multipole origin is (0,0,0));
  - no ports, waveguide ports, or periodic/unit-cell boundaries;
  - nothing extending to the domain boundary (no substrate slab): the
    monitor sphere must lie in homogeneous background;
  - built in the length and frequency units declared below.

The template file itself is never modified: it is copied into the run
directory and all extraction settings (frequency range, vacuum background,
open boundaries, mesh, monitors, solver) are applied to the copy.

Edit the CAPITALIZED settings below, then run:
  python examples/extract_from_template.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst_tmatrix.config import ExtractionConfig
from cst_tmatrix.pipeline import ScattererPlan, extract_tmatrix
from cst_tmatrix.cst_driver import CSTSession

# --------------------------------------------------------------------------
# Settings for this extraction
# --------------------------------------------------------------------------
TEMPLATE = r"C:\path\to\your_scatterer.cst"   # the prepared CST project
NAME = "my_scatterer"                          # library entry name
R_CIRC_M = 50e-6            # circumscribing-sphere radius, meters
FREQS_HZ = np.array([0.6e12, 0.8e12, 1.0e12])  # extraction frequencies, Hz
LENGTH_UNIT = "um"          # length unit the template was built in
FREQ_UNIT = "THz"           # frequency unit of the template
LMAX = None                 # None: Wiscombe criterion at the highest frequency

plan = ScattererPlan(
    name=NAME, template=TEMPLATE, r_circ_m=R_CIRC_M, freqs_hz=FREQS_HZ,
    lmax=LMAX, length_unit=LENGTH_UNIT, freq_unit=FREQ_UNIT,
    metadata={"geometry": {"shape": "user-defined (template)"},
              "material": {"name": "see template project"}})

session = CSTSession()
T, diags, h5_path = extract_tmatrix(session, plan)

print(f"\nSaved: {h5_path}")
print(f"{'f (THz)':>8} {'residual':>10} {'cond(A)':>8} {'max sv(S)':>10} "
      f"{'reciprocity':>11} {'inc.dev':>8}")
for jj, f_hz in enumerate(FREQS_HZ):
    print(f"{f_hz/1e12:8.3f} {diags['residual'][jj]:10.2e} "
          f"{diags['cond'][jj]:8.1f} {diags['s_max_sv'][jj]:10.6f} "
          f"{diags['reciprocity'][jj]:11.2e} "
          f"{diags['incident_deviation_max'][jj]:8.1e}")
print("\nAll diagnostics are stored with the T-matrix in the output file "
      "(group /computation/analysis).  Healthy values: residual and "
      "reciprocity at the solver accuracy (1e-3 .. 1e-2), max sv(S) <= 1, "
      "incident deviation < ~1e-2.")
