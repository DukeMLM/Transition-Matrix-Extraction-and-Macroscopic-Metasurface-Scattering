"""Mode-1 VBA history blocks for model setup.

VBA syntax rules (project reference, SCHEMA v1.5): no parentheses on method
calls, space before arguments, ALL parameter values passed as quoted strings
(matches CST's own history generation).
"""

from __future__ import annotations

from .session import ProjectHandle


def _f(x) -> str:
    """Format a number as a quoted VBA argument."""
    return f'"{x:.12g}"' if isinstance(x, (int, float)) else f'"{x}"'


def set_units(h: ProjectHandle, length: str = "um", frequency: str = "THz",
              time: str = "ps"):
    h.add_history("define units", f"""With Units
  .SetUnit "Length", {_f(length)}
  .SetUnit "Frequency", {_f(frequency)}
  .SetUnit "Time", {_f(time)}
  .SetUnit "Temperature", "K"
End With""")


def set_frequency_range(h: ProjectHandle, fmin: float, fmax: float):
    """fmin/fmax in PROJECT frequency units.  Must precede the solver run."""
    h.add_history("define frequency range",
                  f"Solver.FrequencyRange {_f(fmin)}, {_f(fmax)}")


def set_background_vacuum(h: ProjectHandle, padding: float):
    """Normal (vacuum) background with uniform padding in project length
    units.  The background material is set to vacuum explicitly, so
    that the configuration is complete in the history.

    Note: .ApplyInAllDirections "True" does NOT
    broadcast a single explicitly-set face (e.g. XminSpace alone) to the
    other five -- that broadcast is a GUI-only convenience (the "Distance"
    box in Background Properties writes all six faces on OK/Apply); scripted
    VBA must set all six explicitly. Confirmed via Boundary.GetCalculationBox()
    after the old (XminSpace-only) history block: the resulting domain was
    padded ONLY on the minus-X face, with Y/Z at exactly the structure's own
    bounding box (zero padding) -- present in EVERY prior extraction, not
    just the ones that visibly failed. It went unnoticed at modest
    monitor_factor because CST's own default "expanded open" boundary
    behavior happened to provide enough incidental buffer; at
    monitor_factor=18 (~59 um of intended clearance) it did not, and the
    monitor sphere measurement was corrupted in Y/Z."""
    h.add_history("define background", f"""With Background
  .Reset
  .Type "normal"
  .Epsilon "1.0"
  .Mu "1.0"
  .ApplyInAllDirections "True"
  .XminSpace {_f(padding)}
  .XmaxSpace {_f(padding)}
  .YminSpace {_f(padding)}
  .YmaxSpace {_f(padding)}
  .ZminSpace {_f(padding)}
  .ZmaxSpace {_f(padding)}
End With""")


def set_open_boundaries(h: ProjectHandle, min_distance_per_wavelength: float = 8,
                        reflection: float = 1e-4, pml_layers: int = 4):
    """Open (add space) on all six faces, no symmetry planes (arbitrary
    illumination directions forbid symmetry reduction)."""
    h.add_history("define boundaries", f"""With Boundary
  .Xmin "expanded open"
  .Xmax "expanded open"
  .Ymin "expanded open"
  .Ymax "expanded open"
  .Zmin "expanded open"
  .Zmax "expanded open"
  .Xsymmetry "none"
  .Ysymmetry "none"
  .Zsymmetry "none"
  .Layer {_f(pml_layers)}
  .Reflection {_f(reflection)}
  .MinimumLinesDistance "5"
  .MinimumDistancePerWavelength {_f(min_distance_per_wavelength)}
End With""")


def define_material(h: ProjectHandle, name: str, eps: float | complex = 1.0,
                    mu: float = 1.0, tand: float | None = None,
                    tand_freq: float | None = None, sigma: float | None = None,
                    folder: str = ""):
    """Normal isotropic material.  Use tand+tand_freq (project frequency
    units) OR sigma (S/m), not both."""
    lines = [f'  .Name {_f(name)}']
    if folder:
        lines.append(f'  .Folder {_f(folder)}')
    lines += ['  .Type "Normal"',
              f'  .Epsilon {_f(eps.real if isinstance(eps, complex) else eps)}',
              f'  .Mu {_f(mu)}']
    if tand is not None:
        lines += [f'  .TanD {_f(tand)}', '  .TanDGiven "True"',
                  '  .TanDModel "ConstTanD"']
        if tand_freq is not None:
            lines.append(f'  .TanDFreq {_f(tand_freq)}')
    if sigma is not None:
        lines.append(f'  .Sigma {_f(sigma)}')
    body = "With Material\n  .Reset\n" + "\n".join(lines) + "\n  .Create\nEnd With"
    h.add_history(f"define material: {name}", body)


def new_component(h: ProjectHandle, name: str = "scatterer"):
    h.add_history(f"new component: {name}", f'Component.New {_f(name)}')


def build_sphere(h: ProjectHandle, radius: float, material: str = "PEC",
                 name: str = "sph", component: str = "scatterer",
                 center=(0.0, 0.0, 0.0)):
    """Analytic sphere primitive (Segments 0 = exact surface for tet mesh)."""
    h.add_history(f"define sphere: {component}:{name}", f"""With Sphere
  .Reset
  .Name {_f(name)}
  .Component {_f(component)}
  .Material {_f(material)}
  .Axis "z"
  .CenterRadius {_f(radius)}
  .TopRadius "0"
  .BottomRadius "0"
  .Center {_f(center[0])}, {_f(center[1])}, {_f(center[2])}
  .Segments "0"
  .Create
End With""")


def build_brick(h: ProjectHandle, xr, yr, zr, material: str = "PEC",
                name: str = "brick", component: str = "scatterer"):
    h.add_history(f"define brick: {component}:{name}", f"""With Brick
  .Reset
  .Name {_f(name)}
  .Component {_f(component)}
  .Material {_f(material)}
  .Xrange {_f(xr[0])}, {_f(xr[1])}
  .Yrange {_f(yr[0])}, {_f(yr[1])}
  .Zrange {_f(zr[0])}, {_f(zr[1])}
  .Create
End With""")


def build_cylinder(h: ProjectHandle, radius: float, zmin: float, zmax: float,
                   material: str = "PEC", name: str = "cyl",
                   component: str = "scatterer", axis: str = "z"):
    h.add_history(f"define cylinder: {component}:{name}", f"""With Cylinder
  .Reset
  .Name {_f(name)}
  .Component {_f(component)}
  .Material {_f(material)}
  .Axis {_f(axis)}
  .Outerradius {_f(radius)}
  .Innerradius "0"
  .Zrange {_f(zmin)}, {_f(zmax)}
  .Segments "0"
  .Create
End With""")


def set_tet_mesh(h: ProjectHandle, steps_per_wave_near: int = 15,
                 steps_per_wave_far: int = 6, curvature_order: int = 2):
    """Tetrahedral mesh with curved elements; densities above CST defaults
    (13/4) because multipole projection is more sensitive than S-parameters.

    Note: MeshSettings.CellsPerWavelengthPolicy
    defaults to "automatic", under which CST computes its OWN density and
    silently ignores StepsPerWaveNear/Far entirely; with
    the default policy, StepsPerWaveNear/Far 15/6 -> 30/12 -> 50/20 all gave
    the identical cell count CST's automatic density picked; with the policy
    below, the same three settings correctly gave 82175 / 599205 / 2331812
    cells.  Without this line, set_tet_mesh has never actually controlled
    mesh density."""
    h.add_history("set mesh type tetrahedral", f"""Mesh.MeshType "Tetrahedral"
With MeshSettings
  .SetMeshType "Tet"
  .Set "CellsPerWavelengthPolicy", "cellsperwavelength"
  .Set "StepsPerWaveNear", {_f(steps_per_wave_near)}
  .Set "StepsPerWaveFar", {_f(steps_per_wave_far)}
  .Set "CurvatureOrder", {_f(curvature_order)}
End With""")
