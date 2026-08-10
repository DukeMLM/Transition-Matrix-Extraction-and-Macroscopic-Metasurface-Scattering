"""Forward direction: a single-scatterer T-matrix -> array S-parameters.

Library modules (import these):

  vswf          vector spherical wave functions, the mode basis and the
                projector; defines the tmat.h5 mode ordering
  tmat_io       reader for the tmat.h5 files the extraction stage writes
  translate     translation matrices and the Gaussian-tapered lattice sum C
  aggregate     Foldy-Lax solve of (I - C T) for the self-consistent field
  sparams       0th-order plane-wave S-parameters and cross sections
  supercell     block-Bloch aggregation of a heterogeneous supercell
  ewald_supercell   Ewald-accelerated lattice sums for the supercell blocks
  mirror        image-source construction for a ground plane

Command-line entry points (`python -m tmatrix.aggregation.<name>`):

  run_demo, run_case, run_supercell, run_mirror_demo     compute
  plot_results, plot_case, plot_supercell, plot_comparison,
  plot_cst_comparison, plot_experiment_summary, plot_figure_slide,
  compare_cases, error_budget, jones_xy                  report
  treams_case, treams_reference, treams_supercell,
  validate_synthetic_treams                              independent check
  cst_packed_reference                                   read a packed .cst
"""
