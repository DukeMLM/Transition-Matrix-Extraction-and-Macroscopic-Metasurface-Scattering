"""Frequency-domain solver configuration and execution.

Solver choice rationale (see README): the frequency-domain solver (tetrahedral,
adaptive) is the NAMM-prescribed engine and supports plane-wave excitation
via FDSolver.Stimulation "Plane Wave", "1".  Monitor frequencies are forced
as explicit "Single" samples plus AddMonitorSamples True, so every T-matrix
frequency is an exact solver sample (never interpolated).
"""

from __future__ import annotations

from .session import ProjectHandle


def _f(x) -> str:
    return f'"{x:.12g}"'


def configure_fd_solver(h: ProjectHandle, monitor_freqs, adaptation_freq=None,
                        accuracy: float = 1e-4, order: str = "Second",
                        mesh_adaption: bool = True,
                        max_cpus: int | None = None,
                        hardware_acceleration: bool | None = None):
    """Configure FDSolver for plane-wave scattering at the given monitor
    frequencies (project units).

    adaptation_freq : frequency for adaptive mesh refinement; default =
        max(monitor_freqs) (finest mesh requirement, per NAMM practice of
        sizing everything at f_max).
    max_cpus : if given, pin the solver to this many CPU threads
        (UseParallelization True + LimitCPUs True + MaxCPUs n; without the
        LimitCPUs flag CST ignores MaxCPUs and uses all cores).  None
        leaves the machine's CST defaults untouched.
    hardware_acceleration : if given, emit HardwareAcceleration True/False
        (GPU offload; vba-undoc command, syntax as recorded in histories).
        None leaves the machine default untouched.
    """
    freqs = list(monitor_freqs)
    if adaptation_freq is None:
        adaptation_freq = max(freqs)
    # a fresh MWS project starts with the Time Domain solver active;
    # run_solver() runs the ACTIVE solver, so this switch is mandatory
    h.add_history("change solver type",
                  'ChangeSolverType "HF Frequency Domain"')
    sample_lines = [
        f'  .AddSampleInterval {_f(f)}, {_f(f)}, "1", "Single", '
        f'"{"True" if abs(f - adaptation_freq) < 1e-12 else "False"}"'
        for f in freqs]
    if all(abs(f - adaptation_freq) >= 1e-12 for f in freqs):
        sample_lines.append(
            f'  .AddSampleInterval {_f(adaptation_freq)}, '
            f'{_f(adaptation_freq)}, "1", "Single", "True"')
    accel_lines = []
    if max_cpus is not None:
        accel_lines += ['  .UseParallelization "True"',
                        '  .LimitCPUs "True"',
                        f'  .MaxCPUs "{int(max_cpus)}"']
    if hardware_acceleration is not None:
        accel_lines.append(
            f'  .HardwareAcceleration '
            f'"{"True" if hardware_acceleration else "False"}"')
    body = "\n".join([
        "With FDSolver",
        "  .Reset",
        '  .SetMethod "Tetrahedral", "General purpose"',
        '  .Stimulation "Plane Wave", "1"',
        f'  .AccuracyTet {_f(accuracy)}',
        f'  .OrderTet "{order}"',
        '  .Type "Auto"',
        f'  .MeshAdaptionTet "{"True" if mesh_adaption else "False"}"',
        '  .ResetSampleIntervals "all"',
        *sample_lines,
        '  .AddMonitorSamples "True"',
        '  .SetOpenBCTypeTet "Default"',
        '  .StoreAllResults "False"',
        *accel_lines,
        "End With"])
    h.add_history("define FD solver parameters", body)


def configure_adaptation(h: ProjectHandle, min_passes: int = 3,
                         max_passes: int = 8, max_delta_s: float = 0.01):
    """Tighten tetrahedral adaptive refinement beyond the S-parameter
    default (0.02) — multipole content converges slower than S11."""
    h.add_history("define mesh adaptation", f"""With MeshAdaption3D
  .SetType "HighFrequencyTet"
  .MinPasses {_f(min_passes)}
  .MaxPasses {_f(max_passes)}
  .SetAdaptionStrategy "ExpertSystem"
  .MaxDeltaS {_f(max_delta_s)}
End With""")


def set_mesh_adaptation(h: ProjectHandle, enabled: bool):
    """Toggle FDSolver.MeshAdaptionTet without touching any other solver
    setting.

    The flag is evaluated on every .Start; it is not a one-shot "adapt on
    the first solve" switch.  Note that disabling it after an adapted
    solve does NOT retain the adapted mesh -- the next solve reverts to
    the base density (verified against CST 2026), so toggling it
    mid-extraction produces columns computed on different meshes.  Choose
    the mesh strategy before the illumination loop: either adaptation on
    for every solve, or (the pipeline default) adaptation off with an
    adequate fixed density.
    """
    h.add_history("toggle mesh adaptation",
                  f'FDSolver.MeshAdaptionTet '
                  f'"{"True" if enabled else "False"}"')


def run_fd_solver(h: ProjectHandle, timeout: float | None = None):
    """Run via the native API (raises on failure with CST messages)."""
    h.run_solver(timeout=timeout)
    return h.solver_info()
