"""Stage 1-2 of the pipeline: CST near fields -> VSWF projection -> tmat.h5.

  extract_cst_near_fields     illuminate the isolated cell with a set of
                              plane waves and export E/H on a spherical
                              monitor
  compute_t_matrix_projection project those fields onto VSWFs and solve
                              F = T A in least squares

These are the in-repo templates of the extraction project; the benchmark
tmat.h5 files under test/ were produced by them.
"""
