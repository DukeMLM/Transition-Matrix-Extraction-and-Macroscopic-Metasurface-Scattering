"""Post-processing stages that consume stored T-matrices (docs/manual.tex,
Sec. "Planned extensions").  The extraction pipeline (pipeline.py) is not
involved here — everything in this subpackage starts from a tmat.h5 file
already on disk, plus (for symmetry.geometry_point_group) a CAD export.

Modules
-------
symmetry     : point-group selection-rule checks on an extracted T-matrix,
               and a standalone CAD/point-cloud point-group detector.
metasurface  : T-matrix -> forward-scattering amplitude -> periodic-array
               specular S-parameters (S11/S21), for comparison against a
               direct CST unit-cell (Floquet boundary) simulation.
"""
