"""T-matrix extraction, aggregation and Floquet retrieval for metasurfaces.

Three subpackages, in pipeline order:

  tmatrix.extraction    CST near fields -> VSWF projection -> *.tmat.h5
  tmatrix.aggregation   a single-scatterer T-matrix -> periodic / finite array
                        S-parameters (Foldy-Lax, block-Bloch supercells)
  tmatrix.retrieval     the inverse direction: Floquet S-parameters -> T

Shared, convention-free machinery lives at the top level so that no module has
to re-derive it:

  tmatrix.units         c in both unit conventions, lambda <-> f
  tmatrix.numerics      small array reductions used by every gate/report
  tmatrix.plotting      matplotlib Agg setup, THz twin axis, parabolic minima
  tmatrix.results_io    readers for the run-output files (npz / CSV)
  tmatrix.paths         where the data directories live
  tmatrix.cst_env       where the CST python libraries live

Physical conventions are pinned repository-wide and documented in
retrieval/HANDOFF.md: e^{-i omega t}, outgoing h^(1), tmat.h5 mode ordering
(tmatrix.aggregation.vswf), Jones index 0 = TE / 1 = TM with rows = receive,
and a POSITIVE Bloch sign.  CST exports use e^{+j omega t} and are conjugated
on load.
"""

__version__ = "0.1.0"

__all__ = ["units", "numerics", "plotting", "results_io", "paths", "cst_env"]
