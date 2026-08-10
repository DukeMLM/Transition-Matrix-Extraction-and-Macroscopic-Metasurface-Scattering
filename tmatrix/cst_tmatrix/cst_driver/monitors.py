"""E/H field monitors and farfield monitors at explicit frequency lists.

Deterministic names are used ("e-field (f=X)") so result-tree paths are
predictable: '2D/3D Results\\E-Field\\e-field (f=X) [pw]' for plane-wave
excitation.  Frequencies are in PROJECT frequency units.
"""

from __future__ import annotations

from .session import ProjectHandle


def _fmt(x: float) -> str:
    """CST style frequency label: trim trailing zeros (10.0 -> '10')."""
    s = f"{x:.10g}"
    return s


def monitor_name(field: str, freq: float) -> str:
    prefix = {"Efield": "e-field", "Hfield": "h-field",
              "Farfield": "farfield"}[field]
    return f"{prefix} (f={_fmt(freq)})"


def define_field_monitors(h: ProjectHandle, freqs, fields=("Efield", "Hfield")):
    """One monitor per (field, frequency) with deterministic names."""
    for field in fields:
        for f in freqs:
            name = monitor_name(field, f)
            h.add_history(f"define monitor: {name}", f"""With Monitor
  .Reset
  .Domain "Frequency"
  .FieldType "{field}"
  .Name "{name}"
  .Frequency "{_fmt(f)}"
  .Create
End With""")


def define_farfield_monitors(h: ProjectHandle, freqs):
    for f in freqs:
        name = monitor_name("Farfield", f)
        h.add_history(f"define monitor: {name}", f"""With Monitor
  .Reset
  .Domain "Frequency"
  .FieldType "Farfield"
  .Name "{name}"
  .Frequency "{_fmt(f)}"
  .EnableNearfieldCalculation "True"
  .Create
End With""")


def efield_tree_path(freq: float, excitation: str = "pw") -> str:
    return f"2D/3D Results\\E-Field\\{monitor_name('Efield', freq)} [{excitation}]"


def hfield_tree_path(freq: float, excitation: str = "pw") -> str:
    return f"2D/3D Results\\H-Field\\{monitor_name('Hfield', freq)} [{excitation}]"


def farfield_tree_path(freq: float, excitation: str = "pw") -> str:
    return f"Farfields\\{monitor_name('Farfield', freq)} [{excitation}]"
