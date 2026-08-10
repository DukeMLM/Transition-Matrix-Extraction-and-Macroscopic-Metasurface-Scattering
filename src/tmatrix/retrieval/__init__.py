"""Inverse direction: Floquet S-parameters -> the single-scatterer T-matrix.

The conventions this package is pinned to are normative for the whole
repository and are documented in retrieval/HANDOFF.md; do not re-derive them.

Library modules:

  sparams_oblique   oblique-incidence S-parameters, TE/TM Jones convention
  bloch_lattice     Bloch-phased lattice sum C(k, k_par)
  precompute_C      cached C over the (frequency, angle) grid
  parametrize       the symmetry-constrained T parametrization
  forward           the forward model the fit differentiates
  fit               least-squares retrieval
  observability     what the measurement set can and cannot resolve
  deembed           CST port de-embedding and the label/hypothesis machinery

Command-line entry points (`python -m tmatrix.retrieval.<name>`):

  precompute_C, real_retrieval, gate_study, synthetic_test,
  validate_against_reference, treams_oblique_validate,
  cst_campaign (plan/build), cst_solve (solve/extract)

`fastfull` is a deliberately separate subpackage: it implements the
rank-optimized multimode retrieval of FAST_FULL_TMATRIX_WHEEL_PROPOSAL.md and
imports these conventions rather than modifying them.
"""
