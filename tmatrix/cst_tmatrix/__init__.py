"""cst_tmatrix — T-matrix extraction library for CST Studio Suite.

Workflow: CST (plane-wave excitation, full-wave solve, field/farfield export)
+ Python (VSWF projection, incident-coefficient computation, least-squares
solve for T, HDF5 storage).

CST simulation files are written to a local, non-synchronized working
directory (see config.RUN_ROOT); only code and the final T-matrix files
belong in a synchronized or version-controlled folder.
"""

__version__ = "0.1.0"

from . import mie, quadrature, storage, tmatrix_solve, vswf  # noqa: E402,F401
from .config import ExtractionConfig, FrequencyPlan, RunPaths  # noqa: F401
from .pipeline import ScattererPlan, extract_tmatrix  # noqa: F401
from .storage import load_tmatrix, save_tmatrix  # noqa: F401
