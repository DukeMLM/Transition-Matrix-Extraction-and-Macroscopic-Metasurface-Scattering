"""CST session and project lifecycle with defensive error handling.

The error-handling rules below were established empirically against CST
Studio Suite 2026:

- History commands are not idempotent.  The intermittent client-side
  proxy exception ("'RemoteObjectMethod' object has no attribute
  '__name__'") is raised AFTER the submitted block has executed;
  resubmitting the same block then fails with a name collision.  Errors
  are therefore verified against the project message log rather than
  retried.
- In quiet mode, `prj.get_messages()` returns structured {'text','type'}
  entries; each history block and solver run is checked for new
  type=='ERROR' entries against a pre-call baseline.
- Solver runs execute inside the `quiet_mode_enabled()` context, which
  converts blocking dialog boxes into RuntimeError plus messages.
- The native per-call timeouts of the scripting interface are used where
  available; no thread- or process-level timeout machinery is added.
"""

from __future__ import annotations

from pathlib import Path


class CSTError(RuntimeError):
    """Raised when a CST operation fails; message includes CST's ERROR
    messages when available."""


def _parse_messages(raw):
    """Normalize prj.get_messages() to [{'text','type'}] (CST 2026 returns
    dicts already; strings tolerated defensively)."""
    out = []
    for m in raw or []:
        if isinstance(m, dict):
            out.append({"text": str(m.get("text", "")),
                        "type": str(m.get("type", "UNKNOWN"))})
        else:
            out.append({"text": str(m), "type": "UNKNOWN"})
    return out


class ProjectHandle:
    """Wraps a cst.interface Project + its Model3D proxy (+ owning DE)."""

    def __init__(self, prj, de=None):
        self.prj = prj
        self.de = de
        self.m3d = prj.model3d

    # -- messages -----------------------------------------------------------
    def _message_count(self) -> int:
        try:
            raw = self.prj.get_messages()
            return len(raw) if raw else 0
        except Exception:                                 # noqa: BLE001
            return 0

    def _errors_since(self, baseline: int):
        """New type=='ERROR' messages after the first `baseline` entries."""
        try:
            raw = self.prj.get_messages()
        except Exception:                                 # noqa: BLE001
            return []
        return [m for m in _parse_messages(raw)[baseline:]
                if m["type"] == "ERROR"]

    def _messages(self):
        """All current messages as a printable string (for error reports)."""
        try:
            msgs = _parse_messages(self.prj.get_messages())
            return "; ".join(f"[{m['type']}] {m['text']}" for m in msgs) \
                or "<none>"
        except Exception:                                 # noqa: BLE001
            return "<unavailable>"

    # -- history (Mode 1) ---------------------------------------------------
    def add_history(self, label: str, vba: str, timeout: int | None = None):
        """Submit one complete history block (Mode 1), error-checked against
        the message log.

        History commands are non-idempotent — this NEVER retries. The known
        flaky COM proxy error ("... has no attribute '__name__'") is raised
        AFTER the block has executed (verified against CST 2026; see
        module docstring):
        it is treated as success iff no new ERROR messages appeared.
        """
        baseline = self._message_count()
        try:
            if timeout is not None:
                self.m3d.add_to_history(label, vba, timeout=timeout)
            else:
                self.m3d.add_to_history(label, vba)
        except Exception as e:                            # noqa: BLE001
            errs = self._errors_since(baseline)
            if "__name__" in str(e) and not errs:
                return                    # block executed; client-side glitch
            detail = "; ".join(m["text"] for m in errs) or str(e)
            raise CSTError(
                f"History block '{label}' failed: {detail}\n"
                f"(History commands are non-idempotent — do not retry "
                f"as-is.)\nVBA was:\n{vba}") from e
        errs = self._errors_since(baseline)
        if errs:
            detail = "; ".join(m["text"] for m in errs)
            raise CSTError(
                f"History block '{label}' rejected by CST: {detail}\n"
                f"VBA was:\n{vba}")

    # -- Mode 3 -------------------------------------------------------------
    def execute_vba(self, body: str, timeout: int | None = None):
        """Run VBA via the schematic interpreter (Mode 3 — no history entry,
        results preserved; sanctioned for post-processing like ASCIIExport).
        `body` is the statement list; wrapped in Sub Main automatically."""
        code = "Sub Main\n" + body + "\nEnd Sub"
        try:
            if timeout is not None:
                self.prj.schematic.execute_vba_code(code, timeout=timeout)
            else:
                self.prj.schematic.execute_vba_code(code)
        except Exception as e:                            # noqa: BLE001
            raise CSTError(
                f"execute_vba_code failed: {e}\n"
                f"CST messages: {self._messages()}\nVBA was:\n{code}") from e

    # -- solver -------------------------------------------------------------
    def run_solver(self, timeout: float | None = None):
        """Blocking solver run inside a quiet-mode context (dialogs become
        RuntimeError + messages); failures re-raise with CST's ERROR texts."""
        baseline = self._message_count()
        ctx = None
        if self.de is not None:
            qm = getattr(self.de, "quiet_mode_enabled", None)
            if callable(qm):
                ctx = qm()
        try:
            if ctx is not None:
                with ctx:
                    self.m3d.run_solver(timeout=timeout)
            else:
                self.m3d.run_solver(timeout=timeout)
        except Exception as e:                            # noqa: BLE001
            errs = self._errors_since(baseline)
            detail = "; ".join(m["text"] for m in errs) or self._messages()
            raise CSTError(
                f"Solver run failed: {e}\nCST ERROR messages: {detail}") from e

    def solver_info(self):
        try:
            return self.m3d.get_solver_run_info()
        except Exception:                                 # noqa: BLE001
            return {}

    # -- misc ---------------------------------------------------------------
    def save(self, path: str | Path | None = None, include_results=False):
        if path is None:
            self.prj.save(allow_overwrite=True)
        else:
            self.prj.save(str(path), include_results=include_results,
                          allow_overwrite=True)


class CSTSession:
    """Owns a DesignEnvironment.  Use as a context manager or call close()."""

    def __init__(self, new_instance: bool = False):
        import cst.interface as ci
        # Mode 2 setters / History-flagged getters need this once per DE
        # instance (does NOT survive a DE restart); harmless otherwise.
        try:
            ci.Model3D.allow_history_commands()
        except Exception:                                 # noqa: BLE001
            pass
        if new_instance:
            self.de = ci.DesignEnvironment.new()          # starts quiet
        else:
            self.de = ci.DesignEnvironment.connect_to_any_or_new()
        try:
            self.de.set_quiet_mode(True)                  # belt and braces
        except Exception:                                 # noqa: BLE001
            pass

    def new_mws_project(self, cst_file: str | Path) -> ProjectHandle:
        """Create a fresh MWS project saved at cst_file (must be a LOCAL,
        non-synced path — see config.LOCAL_RUN_ROOT)."""
        cst_file = Path(cst_file)
        cst_file.parent.mkdir(parents=True, exist_ok=True)
        # a project already open at this path locks the file and makes the
        # save-as below fail — close it first (results are expendable here:
        # this call means "start this scatterer from scratch")
        try:
            old = self.de.get_open_project(str(cst_file))
            if old is not None:
                old.close()
        except Exception:                                 # noqa: BLE001
            pass
        prj = self.de.new_mws()
        h = ProjectHandle(prj, self.de)
        try:
            h.save(cst_file)
        except Exception as e:                            # noqa: BLE001
            try:
                prj.close()                # don't leave an untitled orphan
            except Exception:              # noqa: BLE001
                pass
            raise CSTError(
                f"Could not save new project to {cst_file} — is the file "
                f"open in another CST instance or locked?: {e}") from e
        return h

    def open_project(self, cst_file: str | Path) -> ProjectHandle:
        # get_open_project raises (rather than returning None) when the
        # project is not currently open in this DE instance -- same
        # defensive pattern as new_mws_project's use of the same call.
        try:
            prj = self.de.get_open_project(str(cst_file))
        except Exception:                                 # noqa: BLE001
            prj = None
        if prj is None:
            prj = self.de.open_project(str(cst_file))
        return ProjectHandle(prj, self.de)

    def close(self):
        try:
            self.de.close()
        except Exception:                                 # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # leave CST open by default: closing kills all open projects
        return False
