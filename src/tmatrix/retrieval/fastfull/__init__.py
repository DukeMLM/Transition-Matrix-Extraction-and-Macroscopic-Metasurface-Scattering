"""Fast full-T-matrix retrieval from a rank-optimized multimode Floquet cell.

Implementation of FAST_FULL_TMATRIX_WHEEL_PROPOSAL.md.  This package is
deliberately SEPARATE from the validated one-pitch specular retrieval
(retrieval/*.py): that pipeline is complete and its conventions are pinned by
a measured CST campaign (see retrieval/RESULTS.md, HANDOFF.md).  Nothing here
modifies it; the conventions are IMPORTED from it and re-verified by gates.

Milestone map (proposal par. 14)
-------------------------------
  M1  physical rank / SNR / cost designer, no CST   -> lattice, transforms,
                                                       symmetry, jacobian,
                                                       cost, design
  M2  Ewald C + blind synthetic full loop           -> (not yet built)
  M3  empty-cell CST calibration                    -> (not yet built)
  M4  analytic Mie-sphere calibration               -> (not yet built)
  M5  wheel + independent encoding                  -> (not yet built)
  M6  broadband + truncation                        -> (not yet built)
  M7  standards-complete tmat.h5 deliverable        -> (not yet built)

Inherited normative conventions (retrieval/HANDOFF.md; do not re-derive)
-----------------------------------------------------------------------
* e^{-i omega t}, outgoing h^(1), tmat.h5 mode order (aggregation/vswf.py).
* Jones index order 0 = TE, 1 = TM; rows = receive, columns = incident.
* Polarization basis: e_TE = (-sin ph, cos ph, 0),
  e_TM = (k_z_hat cos ph, k_z_hat sin ph, -sin th), evaluated separately for
  each propagation direction (sparams_oblique.pol_basis).
* Bloch sign POSITIVE: C(k, k_par) = sum_{R != 0} A(R) e^{+i k_par . R}.
* CST data is e^{+j omega t}: conjugate on load (deembed.conj_cst).

The one genuinely NEW convention this package introduces is the flux
normalization that maps CST power waves to the repository's unit-field VSWF
amplitudes; it is derived and gated in `transforms.py`.
"""
__all__ = ["lattice", "transforms", "symmetry", "jacobian", "cost", "design"]
