"""CST Studio Suite driver layer.

Execution-mode policy (see cst_python_reference/Claude_Summary.md):
- Mode 1 (add_to_history) for everything that defines the model: units,
  frequency range, background, boundaries, materials, geometry, monitors,
  plane wave, solver settings.  Survives history rebuilds.
- Mode 2 (direct proxy getters) for reading state and FarfieldCalculator
  list evaluation.
- Mode 3 (execute_vba_code) for ASCIIExport of field monitors (VBA-only
  object; export must not pollute the model history).
- Solver runs through the native m3d.run_solver() (raises RuntimeError).
"""

from .session import CSTSession, ProjectHandle
