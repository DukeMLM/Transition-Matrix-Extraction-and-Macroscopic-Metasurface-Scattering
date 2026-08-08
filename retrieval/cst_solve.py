"""Sequential, checkpointed, resumable CST solve driver for the par. 7
oblique-incidence Floquet campaign (INVERSE_TMATRIX_FROM_FLOQUET.md par. 7;
HANDOFF.md "Immediate next steps" item 6).

THIS MODULE IS THE ONLY PLACE IN THE REPOSITORY THAT LAUNCHES A CST SOLVER.
cst_campaign.py deliberately refuses --solve (its SOLVE_ORDER_NOTE); this
module implements the missing half and enforces par. 7's ordering IN CODE.
The single solver call site is _launch_solver() -- exactly one
`.FDSolver.Start()` statement exists in this file; every other code path
(--plan, --extract-only) touches no solver at all.

Enforced order (par. 7; refusal is structural, not documentary)
---------------------------------------------------------------
stage 1  "closure"      struct_th00_ph00 + empty_th00
         gate: deembed.closure_normal, de-embedded complex S at theta = 0
         (both CST modes, S11 and S21) vs aggregation/results/
         periodic_results.npz, <= 5e-3 (deembed.CLOSURE_GATE).
         WHAT IT PROVES: the pinned-cellpad domain, the e^{+j omega t} ->
         e^{-i omega t} conjugation, the "inward" scan-angle convention and
         the empty-cell division together reproduce the validated
         normal-incidence physics.  The pinned cellpad changes the mesh of
         the validated run_v3 configuration, so this closure must be
         RE-EARNED before any oblique solve.
         PRODUCT: the residual spectrum IS the real fit sigma.  It is
         written to results/fit_sigma_from_closure.npz (well-known path)
         and replaces the 3e-3 placeholder everywhere downstream (fit
         weights, observability lambda, the synthetic noise gates).

stage 2  "acceptance"   struct_th60_ph22p5 + empty_th60
         (the angle is NOT the doc-literal (30, 0); see below)
         gate: the channel-dictionary acceptance -- exactly one of
         deembed.label_hypotheses()'s 8 discrete TE/TM label hypotheses must
         be identified against the forward model evaluated with the
         REFERENCE T at (theta = 60 deg, phi = 22.5 deg, direction = -1),
         using validate_against_reference's chi2 likelihood-ratio statistic
         (all 8 channels x the whole band, not just the largest residual):
         ACCEPT iff z_margin >= z_min AND chi2_reduced <= chi2_max.
         The doc-literal "max within 1e-2" verdict is computed and reported
         alongside for the record, but does not decide the gate.
         WHAT IT PROVES: CST Floquet mode numbers are mapped to the par. 2
         Jones basis unambiguously, i.e. the measured channels mean what the
         forward model thinks they mean.  Fail-safe by construction: an
         unresolved hypothesis family REFUSES rather than mis-picks.

         WHICH FAMILY  [campaign measurement, 2026-08-07]: the base
         8-member port-gauge family REFUSED (0 winners; chi2_reduced 658.9
         here and 2763.8 at theta=0).  Channel by channel the data matched
         the reference-T forward model to ~2e-3 EVERYWHERE except the S11
         TM co-pol entry, where |d| = 2|S| -- a pure SIGN.  No port gauge
         can produce it: a mode sign s_a multiplies S11[a,b] by s_a*s_b, so
         a co-polar diagonal is invariant, which is exactly WHY 0 of 8
         passed.  It is the documented par.-2 basis convention
         (sparams_oblique derives e_TM^(r) = -x_hat; forward.py's selftest
         already encodes S11[TM,TM] = -stored S11) meeting CST's waveguide
         port convention, which reuses one transverse mode pattern for both
         directions.  deembed.label_hypotheses(extended=True) adds r11_tm,
         a sign on the TM RECEIVE ROW of the S11 block applied AFTER the
         swap; with it, chi2_reduced -> 2.49 and 1.14 respectively.  This
         gate therefore runs on the EXTENDED 16-member family, and reports
         the base-8 verdict alongside so the refusal stays in the record.
         (deembed.apply_hypothesis is NOT an involution for the 4 members
         with swap=True and r11_tm=-1 -- use inverse=True to undo it.)

         WHY NOT (30, 0)  [verifier measurement, 2026-08-07]: with the
         C4v-PROJECTED reference T -- an honest model of a real C4v cell --
         the two cross-sign hypotheses are EXACTLY degenerate at every
         phi in {0, 45} deg, because a cross-sign error flips exactly the
         entries that vanish identically on a mirror plane.  The apparent
         separation at (30, 0) is the reference file's own ~0.3 %
         C4v-violation noise and will not survive contact with real CST
         data.  Physical cross-sign separations: (30, 22.5) 3.52e-3,
         (45, 22.5) 7.11e-3, (60, 22.5) 1.06e-2 -- only (60, 22.5) clears
         z_min = 5 at sigma = 3e-3.  The mode-swap half of the family is
         unambiguous at every angle (separation >= 0.546).  The residual
         cross-sign ambiguity is harmless exactly where it is unmeasurable.

         SIGMA COUPLING: the chi2 test is run at the sigma MEASURED by the
         stage-1 closure (results/fit_sigma_from_closure.npz), never at the
         3e-3 placeholder.  If that file is absent the gate REFUSES by name.
         z_margin is reported at 1x, 2x and 3x the measured sigma so the
         operator can see how close the verdict is to flipping.

stage 3  "rest"         every remaining run (11 structure + 3 empty + the
         perturbed empty repeat).  No gate of its own; the perturbed empty
         feeds par. 7's noise-floor calibration only.  The model-free
         mirror-plane cross-pol check that the acceptance stage gave up by
         moving off phi = 0 is run opportunistically here, on
         struct_th60_ph00 against the acceptance stage's own empty_th60.

A later stage cannot be started unless every earlier stage's runs hold valid
checkpoints AND that stage's gate is recorded as passed.  --force bypasses
the refusal and prints a loud warning naming the exact gate being bypassed.

Storage conventions (checkpoints)
---------------------------------
* S-parameters are stored RAW, exactly as CST delivers them, i.e. in the
  e^{+j omega t} convention -- NOT conjugated at storage time.  The
  checkpoint records this in its `convention` field and in solve_status.json.
  Consumers apply deembed.conj_cst (or go through deembed.deembed_blocks /
  map_cst_labels, which conjugate internally) before any physics.  Storing
  raw keeps the checkpoint a faithful record of the solve and keeps the
  single conjugation point inside deembed.py.
* Two grids are stored: `S_raw` on CST's native adaptive sweep grid
  (`f_raw_THz`), and `S_grid` on the 49-point tmat target grid
  (`f_grid_THz`, from the manifest), interpolated with
  deembed.interp_to_grid (separate Re/Im np.interp -- the
  build_saw_unitcell.py ~line 317 pattern).  Both are RAW-convention.
* Entry order is the manifest's expected_s_tree (= cst_campaign.
  expected_s_tree()): SZmax(a),Zmax(b) then SZmin(a),Zmax(b), a,b in {1,2}
  -- reflection block first, then transmission block, receive index a
  outer, incident index b inner.
* Empty-cell integrity (par. 7 per-angle analytic empty check) is run for
  every empty_* run at extraction time (deembed.check_empty_phase via
  deembed.map_cst_labels) and recorded in solve_status.json.  The
  phase-direction verdict is the hard one: after conj_cst,
  arg(S21_empty) must ADVANCE with f in the e^{-i omega t} convention.
  A negative slope means the conjugation direction or the "inward"
  convention is wrong, and the run is recorded as failing its integrity
  check.

Resume
------
A run whose cst_runs/<runid>/solve_status.json says status == "ok" and whose
cst_runs/<runid>/solve_result.npz loads with 8 entries on the 49-point grid
is SKIPPED.  --redo <runid[,runid...]> invalidates specific checkpoints.  A
checkpoint produced from a different manifest hash is reported as STALE but
still skipped (never spend solver hours implicitly); --redo is the explicit
way to re-solve it.  A failed run is recorded with its error text and does
not abort the campaign, but its stage is then reported INCOMPLETE and the
next stage stays blocked.

Modes
-----
--plan (DEFAULT)   print the full ordered execution plan and exit.  Touches
                   neither CST nor the solver, works on a machine with no
                   CST installed (every CST import is lazy).
--stage {closure,acceptance,rest,all}   actually solve, in order.
--runs a,b,c       solve a specific subset (still sequential, still gated).
--extract-only     re-read results from already-solved projects (needs
                   cst.results only -- no DesignEnvironment, no solver).
--no-extract       solve only, leave extraction for a later --extract-only.

Exit codes: 0 = all requested runs complete and all evaluated gates passed;
1 = a gate refused or failed; 2 = one or more runs failed; 3 = environment
problem (missing projects / CST not importable); 4 = solver watchdog timeout
with a solver thread that did not return (session left for the operator).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import cst_campaign
import deembed

HERE = Path(__file__).parent
RUNS_DIR_DEFAULT = HERE / "cst_runs"
RESULTS_DIR = HERE / "results"

LOG_NAME = "solve_log.txt"
CHECKPOINT_NPZ = "solve_result.npz"
STATUS_JSON = "solve_status.json"
GATE_JSON = {"closure": "gate_closure.json",
             "acceptance": "gate_acceptance.json"}

# The well-known path the main session / fit.py / observability.py read to
# replace the 3e-3 placeholder sigma (par. 7: "whatever residual this closure
# yields IS the true sigma for the fit weights").
SIGMA_NAME = "fit_sigma_from_closure.npz"
SIGMA_NPZ = RESULTS_DIR / SIGMA_NAME
CLOSURE_LABEL = "closure_campaign_normal"


def results_dir_for(runs_dir):
    """Where the closure gate writes its artifacts (fit sigma, closure npz,
    figure).  The real campaign (default --runs-dir) writes the well-known
    retrieval/results/ paths downstream code reads; a NON-default --runs-dir
    (scratch / rehearsal trees) writes <runs_dir>/results so a rehearsal can
    never overwrite the campaign's sigma."""
    runs_dir = Path(runs_dir)
    if runs_dir.resolve() == RUNS_DIR_DEFAULT.resolve():
        return RESULTS_DIR
    return runs_dir / "results"

CHECKPOINT_VERSION = 1
RAW_CONVENTION = ("RAW CST e^{+j omega t}: NOT conjugated at storage time; "
                  "apply deembed.conj_cst before any physics")

STAGES = ("closure", "acceptance", "rest")

# --- the two gated angles.  Every runid below is DERIVED from these two
# --- pairs, so retargeting a stage is one edit here.
CLOSURE_THETA, CLOSURE_PHI = 0.0, 0.0
# Retargeted 2026-08-07 from the doc-literal (30, 0) on the verifier's
# measurement: with the C4v-PROJECTED reference T (an honest model of a real
# C4v cell) the two cross-sign label hypotheses are EXACTLY degenerate at
# every phi in {0, 45} deg, because a cross-sign error flips entries that
# vanish identically on a mirror plane.  The separation seen at (30, 0) is
# the reference file's own ~0.3 % C4v-violation noise, which a real CST cell
# will not reproduce.  Measured physical cross-sign separations:
# (30,22.5) 3.52e-3 -> z 1.99;  (45,22.5) 7.11e-3 -> z 3.99;
# (60,22.5) 1.06e-2 -> z 5.92 -- only (60, 22.5) clears z_min = 5 at
# sigma = 3e-3.  The mode-swap half is unambiguous everywhere
# (separation >= 0.546).
ACCEPTANCE_THETA, ACCEPTANCE_PHI = 60.0, 22.5


def struct_runid(theta_deg, phi_deg):
    """cst_campaign's runid rule, reused so derived ids cannot drift from
    the manifest."""
    return (f"struct_th{cst_campaign._fmt_angle(theta_deg)}"
            f"_ph{cst_campaign._fmt_angle(phi_deg)}")


def empty_runid(theta_deg):
    return f"empty_th{cst_campaign._fmt_angle(theta_deg)}"


STAGE_RUNS = {
    "closure": (struct_runid(CLOSURE_THETA, CLOSURE_PHI),
                empty_runid(CLOSURE_THETA)),
    "acceptance": (struct_runid(ACCEPTANCE_THETA, ACCEPTANCE_PHI),
                   empty_runid(ACCEPTANCE_THETA)),
}

# Moving the acceptance off a mirror plane costs the reference-model-FREE
# orientation check (deembed.check_mirror_plane_crosspol is only meaningful
# at phi in {0, 45}).  It is not dropped: the same-theta mirror-plane
# structure run is checked opportunistically whenever its checkpoint exists
# (it is a stage-'rest' run, and shares the acceptance stage's empty).
MIRROR_CHECK_STRUCT = struct_runid(ACCEPTANCE_THETA, 0.0)
MIRROR_CHECK_EMPTY = empty_runid(ACCEPTANCE_THETA)
MIRROR_CHECK_JSON = "check_mirror_crosspol.json"

STAGE_GATE_DESC = {
    "closure": ("deembed.closure_normal <= %g vs aggregation/results/"
                "periodic_results.npz (complex, both CST modes); its "
                "residual IS the fit sigma" % deembed.CLOSURE_GATE),
    "acceptance": ("channel dictionary at (theta=%g, phi=%g, direction=-1): "
                   "chi2 likelihood-ratio over 8 channels x band against the "
                   "forward model with the reference T, over the EXTENDED "
                   "16-member hypothesis family, at the MEASURED closure "
                   "sigma (z_margin >= z_min and chi2_reduced <= chi2_max)"
                   % (ACCEPTANCE_THETA, ACCEPTANCE_PHI)),
    "rest": ("none (perturbed empty feeds noise-floor calibration only); "
             "the opportunistic model-free mirror-plane cross-pol check on "
             + MIRROR_CHECK_STRUCT + " runs here"),
}

ACCEPTANCE_TOL = 1e-2       # the doc-literal "max" statistic, reported only
ACCEPTANCE_STATISTIC = "chi2"
# "extended" = deembed.label_hypotheses(extended=True), 16 members: the 8
# port gauges x the r11_tm sign on the TM RECEIVE ROW of the S11 block.
# The campaign MEASURED that the base 8 cannot describe the data (0 of 8
# passed, chi2_reduced 658.9 at (60, 22.5) / 2763.8 at theta=0), because a
# port gauge multiplies S11[a,b] by s_a*s_b and therefore CANNOT touch a
# co-polar diagonal -- and the discrepancy is exactly a sign on the S11 TM
# co-pol entry.  With r11_tm = -1: 658.9 -> 2.49 and 2763.8 -> 1.14.  That
# is the documented par.-2 convention (sparams_oblique derives
# e_TM^(r) = -x_hat; forward.py's selftest already encodes
# S11[TM,TM] = -stored S11), not a solver artefact.  The base-8 verdict is
# still computed and logged so the refusal that produced this change stays
# legible from the log alone.
ACCEPTANCE_FAMILY = "extended"
DIRECTION_CST = -1          # HANDOFF: campaign/CST/treams illumination
N_ENTRIES = 8

# How the closure's residual spectrum is reduced to the single per-complex-
# observable noise scale the chi2 test needs.
SIGMA_REDUCTION = "RMS over all compared channels and the whole band"


class SolveError(RuntimeError):
    pass


# ===========================================================================
# Logging
# ===========================================================================

class Logger:
    """Line-buffered timestamped log to cst_runs/solve_log.txt + stdout, so
    the main session can `tail -f` it while the driver runs for hours."""

    def __init__(self, path=None, echo=True):
        self.path = Path(path) if path else None
        self.echo = echo
        self.fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def __call__(self, msg=""):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        if self.echo:
            print(line, flush=True)
        if self.fh is not None:
            self.fh.write(line + "\n")
            self.fh.flush()

    def raw(self, msg=""):
        """Print without a timestamp (banners, tables)."""
        if self.echo:
            print(msg, flush=True)
        if self.fh is not None:
            self.fh.write(msg + "\n")
            self.fh.flush()

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


# ===========================================================================
# Manifest / hashing / stage bookkeeping
# ===========================================================================

def load_manifest(runs_dir):
    p = Path(runs_dir) / "campaign_manifest.json"
    if not p.exists():
        raise SolveError(
            f"no campaign manifest at {p} -- run "
            f"`python cst_campaign.py` (dry-run, the default) first")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def manifest_hash(manifest):
    """sha256 (16 hex) of the manifest with the volatile `created` timestamp
    removed, so regenerating the dry-run does not invalidate checkpoints for
    a physically identical campaign."""
    m = {k: v for k, v in manifest.items() if k != "created"}
    blob = json.dumps(m, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def run_hash(run):
    blob = json.dumps(run, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def runs_by_id(manifest):
    return {r["runid"]: r for r in manifest["runs"]}


def stage_of(runid):
    for st, ids in STAGE_RUNS.items():
        if runid in ids:
            return st
    return "rest"


def stage_run_order(manifest):
    """{stage: [runid, ...]} in the order they must be solved.  Structure
    before empty inside the gated stages is irrelevant (the gate needs both)
    but the par. 7 text lists structure first, so keep that."""
    ids = [r["runid"] for r in manifest["runs"]]
    out = {}
    for st in STAGES[:2]:
        missing = [r for r in STAGE_RUNS[st] if r not in ids]
        if missing:
            raise SolveError(f"manifest lacks stage-{st} runs {missing} -- "
                             f"this driver requires the par. 7 campaign")
        out[st] = list(STAGE_RUNS[st])
    claimed = set(out["closure"]) | set(out["acceptance"])
    out["rest"] = [i for i in ids if i not in claimed]
    return out


# ===========================================================================
# Checkpoints
# ===========================================================================

def status_path(runs_dir, runid):
    return Path(runs_dir) / runid / STATUS_JSON


def checkpoint_path(runs_dir, runid):
    return Path(runs_dir) / runid / CHECKPOINT_NPZ


def read_status(runs_dir, runid):
    p = status_path(runs_dir, runid)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def checkpoint_state(runs_dir, run, man_hash, redo=()):
    """Inspect a run's checkpoint without touching CST.

    Returns dict(exists, valid, stale, status, detail).  valid == True means
    "skip on resume".  stale == True means the checkpoint was produced from a
    different manifest hash: it is reported loudly but still skipped (never
    spend solver hours implicitly) -- use --redo to force."""
    rid = run["runid"]
    st = read_status(runs_dir, rid)
    npz = checkpoint_path(runs_dir, rid)
    out = dict(exists=bool(st is not None or npz.exists()), valid=False,
               stale=False, status=(st or {}).get("status"), detail="")
    if rid in redo:
        out["detail"] = "forced by --redo"
        return out
    if st is None:
        out["detail"] = "no solve_status.json"
        return out
    if st.get("status") != "ok":
        out["detail"] = f"status = {st.get('status')!r}"
        return out
    if not npz.exists():
        out["detail"] = "status ok but solve_result.npz missing"
        return out
    try:
        with np.load(npz, allow_pickle=False) as z:
            n_entries = len(z["entries"])
            shape = tuple(z["S_grid"].shape)
    except Exception as e:
        out["detail"] = f"solve_result.npz unreadable: {e}"
        return out
    if n_entries != N_ENTRIES or shape[0] != N_ENTRIES:
        out["detail"] = (f"checkpoint has {n_entries} entries / S_grid "
                         f"{shape}, expected {N_ENTRIES}")
        return out
    out["valid"] = True
    out["stale"] = bool(st.get("manifest_sha256") != man_hash)
    out["detail"] = (f"ok, {shape[1]} target-grid points"
                     + (f"; STALE (built from manifest "
                        f"{st.get('manifest_sha256')}, current {man_hash})"
                        if out["stale"] else ""))
    return out


def load_raw_blocks(runs_dir, runid, grid="target"):
    """Reconstruct the {S-tree entry -> (f_THz, S_raw)} dict that
    deembed.deembed_blocks / deembed.map_cst_labels consume, from a
    checkpoint.  S is RAW CST (e^{+j omega t}); deembed conjugates."""
    p = checkpoint_path(runs_dir, runid)
    if not p.exists():
        raise SolveError(f"no checkpoint for {runid} at {p}")
    with np.load(p, allow_pickle=False) as z:
        entries = [str(e) for e in z["entries"]]
        if grid == "target":
            f = np.asarray(z["f_grid_THz"], dtype=float)
            S = np.asarray(z["S_grid"], dtype=complex)
        elif grid == "raw":
            f = np.asarray(z["f_raw_THz"], dtype=float)
            S = np.asarray(z["S_raw"], dtype=complex)
        else:
            raise ValueError("grid must be 'target' or 'raw'")
    return {e: (f.copy(), S[i].copy()) for i, e in enumerate(entries)}


# ===========================================================================
# CST access (lazy -- --plan must work on a machine with no CST)
# ===========================================================================

def _import_cst_interface():
    """Import cst.interface lazily.  Never called by --plan or
    --extract-only."""
    lib = cst_campaign.CST_PYTHON_LIB
    if lib not in sys.path:
        sys.path.insert(0, lib)
    try:
        import cst.interface as cstint          # noqa: PLC0415
        return cstint
    except ImportError as e:
        raise SolveError(
            f"cst.interface is not importable (looked in {lib}).  Solving "
            f"needs CST on this machine; --plan and --extract-only do not."
            f"  Original error: {e}") from e


def open_design_environment(log):
    """ONE DesignEnvironment for the whole session (StartMode.ExistingOrNew),
    closed by the caller's finally."""
    cstint = _import_cst_interface()
    log("opening CST DesignEnvironment (StartMode.ExistingOrNew)...")
    env = cstint.DesignEnvironment(
        mode=cstint.DesignEnvironment.StartMode.ExistingOrNew)
    log(f"  [ok] DesignEnvironment connected (pid {getattr(env, 'pid', '?')})")
    return env


def _safe_messages(project, limit=4000):
    """CST message window, tolerant of the GBK-codec crash documented in
    nir/cst_helpers.get_messages_safe."""
    try:
        msg = project.get_messages()
    except Exception:
        return ""
    if msg is None:
        return ""
    msg = str(msg)
    return msg if len(msg) <= limit else msg[-limit:]


# ---------------------------------------------------------------------------
# THE solver call site
# ---------------------------------------------------------------------------

def _launch_solver(project):
    """Start the frequency-domain solver on an OPEN project and block until
    it finishes.  Returns the wall time in seconds.

    *** THIS FUNCTION CONTAINS THE ONLY SOLVER LAUNCH IN THE REPOSITORY. ***
    It is the validated pattern of aggregation/cst_direct/
    build_saw_unitcell.py:271 (m3d.FDSolver.Start()), unchanged.  Callers
    must guarantee that no other solve is in flight -- a single license, one
    solver at a time, strictly sequential.
    """
    m3d = project.model3d
    t0 = time.time()
    m3d.FDSolver.Start()                      # <-- THE solver launch
    return time.time() - t0


def _solve_blocking_or_watchdog(project, timeout_min, log):
    """Run _launch_solver, optionally under a wall-clock watchdog.

    timeout_min falsy (the DEFAULT) -> the plain blocking call on the main
    thread, byte-for-byte the validated build_saw_unitcell.py path.

    timeout_min > 0 -> the same single call runs on a worker thread and the
    main thread joins with a deadline.  On expiry the driver logs loudly,
    makes a best-effort abort attempt through several candidate CST entry
    points, then waits a grace period.  THIS WATCHDOG PATH IS UNTESTED (no
    solve may be launched from this agent session): it is opt-in only, and
    the default path does not use threads at all.

    Returns (wall_s, timeout_state) with timeout_state in
    {None, "aborted", "stuck"}.
    """
    if not timeout_min:
        return _launch_solver(project), None

    box = {}

    def _worker():
        try:
            box["wall"] = _launch_solver(project)
        except BaseException as e:            # noqa: BLE001 - re-raised below
            box["exc"] = e

    t0 = time.time()
    th = threading.Thread(target=_worker, name="cst-fdsolver", daemon=True)
    th.start()
    th.join(timeout=float(timeout_min) * 60.0)
    if not th.is_alive():
        if "exc" in box:
            raise box["exc"]
        return box.get("wall", time.time() - t0), None

    log(f"  [FATAL] solver watchdog: still running after {timeout_min} min "
        f"(--timeout-min); attempting a best-effort abort")
    for desc, call in (
            ("model3d.FDSolver.Abort()",
             lambda: project.model3d.FDSolver.Abort()),
            ("model3d.Solver.Abort()",
             lambda: project.model3d.Solver.Abort()),
            ("model3d.abort_solver()",
             lambda: project.model3d.abort_solver())):
        try:
            call()
            log(f"  [warn] abort attempt via {desc}: accepted")
            break
        except Exception as e:                # noqa: BLE001
            log(f"  [warn] abort attempt via {desc} failed: {e}")
    th.join(timeout=120.0)
    if th.is_alive():
        log("  [FATAL] solver thread did not return after the abort grace "
            "period -- the CST session state is unknown.  This driver will "
            "NOT start another solve and will NOT close the "
            "DesignEnvironment; intervene in the CST GUI.")
        return time.time() - t0, "stuck"
    log("  [warn] solver aborted; the run is recorded as timed out")
    return time.time() - t0, "aborted"


# ===========================================================================
# Extraction
# ===========================================================================

def _mesh_info(project_path):
    """Cheap, offline mesh/solver diagnostics from the project's Result
    folder: the CADSurf bounding-box z-extent (should reproduce the
    manifest's L_expected_um) and the solver output messages."""
    proj_dir = Path(project_path).with_suffix("")
    info = dict(L_domain_um=None, n_tets=None, output_messages=None,
                result_dir=str(proj_dir / "Result"))
    log_tet = proj_dir / "Result" / "log.tet"
    try:
        info["L_domain_um"] = float(
            deembed.parse_domain_z_extent(log_tet))
    except Exception:
        pass
    try:
        txt = log_tet.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"[Tt]etrahedra[^0-9]{0,20}(\d[\d ,]*)", txt)
        if m:
            info["n_tets"] = int(m.group(1).replace(",", "").replace(" ", ""))
    except Exception:
        pass
    out_json = proj_dir / "Result" / "output.json"
    try:
        msgs = json.loads(out_json.read_text(encoding="utf-8"))
        info["output_messages"] = "; ".join(
            m.get("message", "") for m in msgs.get("messages", []))[:2000]
    except Exception:
        pass
    return info


def extract_run(run, manifest, log):
    """Read the 8 par. 7 S-tree entries from a solved project and build the
    checkpoint payload.  RAW CST spectra (no conjugation here).

    Returns (payload_dict, extra_status_dict)."""
    entries = list(run.get("expected_s_tree")
                   or cst_campaign.expected_s_tree())
    if len(entries) != N_ENTRIES:
        raise SolveError(f"{run['runid']}: manifest lists {len(entries)} "
                         f"S-tree entries, expected {N_ENTRIES}")
    raw = deembed.read_project_sparams(run["project_path"], entries)

    f0 = np.asarray(raw[entries[0]][0], dtype=float)
    n_raw = len(f0)
    grids_differed = False
    S_raw = np.empty((N_ENTRIES, n_raw), dtype=complex)
    for i, e in enumerate(entries):
        f_i, S_i = raw[e]
        f_i = np.asarray(f_i, dtype=float)
        S_i = np.asarray(S_i, dtype=complex)
        if len(f_i) != n_raw or not np.allclose(f_i, f0, rtol=0, atol=1e-9):
            grids_differed = True
            S_i = deembed.interp_to_grid(f_i, S_i, f0)
        S_raw[i] = S_i
    if grids_differed:
        log(f"  [warn] {run['runid']}: the 8 S-tree entries did not share "
            f"one frequency grid; all were Re/Im-interpolated onto the "
            f"SZmax(1),Zmax(1) grid before storage")

    f_grid = np.asarray(manifest["target_grid_THz"], dtype=float)
    S_grid = np.empty((N_ENTRIES, len(f_grid)), dtype=complex)
    for i in range(N_ENTRIES):
        S_grid[i] = deembed.interp_to_grid(f0, S_raw[i], f_grid)

    payload = dict(entries=np.array(entries),
                   f_raw_THz=f0, S_raw=S_raw,
                   f_grid_THz=f_grid, S_grid=S_grid,
                   runid=np.array(run["runid"]),
                   kind=np.array(run["kind"]),
                   theta_deg=float(run["theta_deg"]),
                   phi_deg=float(run["phi_deg"]),
                   accuracy_tet=np.array(str(run["accuracy_tet"])),
                   direction=DIRECTION_CST,
                   convention=np.array(RAW_CONVENTION),
                   checkpoint_version=CHECKPOINT_VERSION)
    extra = dict(n_raw_points=int(n_raw),
                 raw_grid_THz=[float(f0[0]), float(f0[-1])],
                 raw_grids_differed=bool(grids_differed),
                 convention=RAW_CONVENTION)
    return payload, extra


def empty_integrity(run, payload, manifest, log):
    """par. 7 per-angle analytic empty-cell check for an empty_* run.

    Runs deembed.map_cst_labels (which internally conjugates and calls
    deembed.check_empty_phase per co-pol mode) plus its co-pol degeneracy /
    cross-pol integrity numbers.  The phase-direction verdict is reported
    prominently: after conj_cst, arg(S21_empty) must ADVANCE with f."""
    entries = [str(e) for e in payload["entries"]]
    blocks = {e: (payload["f_grid_THz"], payload["S_grid"][i])
              for i, e in enumerate(entries)}
    lm = deembed.map_cst_labels(run["theta_deg"], run["phi_deg"],
                                empty_blocks=blocks,
                                L_expected_um=manifest.get("L_expected_um"))
    chk = lm["empty_checks"]
    ok = True
    for a in (1, 2):
        c = chk[f"mode{a}"]
        verdict = "ADVANCES (correct)" if c["sign_ok"] else "RETREATS (WRONG)"
        tag = "ok" if c["sign_ok"] else "FATAL"
        log(f"  [{tag}] {run['runid']} mode {a}: arg(S21_empty) {verdict} "
            f"with f; slope {c['slope_rad_per_THz']:+.6f} rad/THz "
            f"(expected {c['slope_expected_rad_per_THz']:+.6f}), "
            f"L_fit = {c['L_fit_um']:.6f} um "
            f"(L_expected = {manifest.get('L_expected_um'):.6f} um), "
            f"rel_err = {c['rel_err']:.3e}, "
            f"max||S21|-1| = {c['mag_dev']:.3e}, "
            f"max|S11_empty| = {c['s11_max']:.3e}")
        if not c["sign_ok"]:
            log(f"  [FATAL] {run['runid']}: a RETREATING empty phase means "
                f"the e^{{+j omega t}} -> e^{{-i omega t}} conjugation "
                f"direction or the par. 7 'inward' scan-angle convention is "
                f"wrong.  DO NOT de-embed against this run.")
        ok = ok and bool(c["passed"])
    log(f"  [{'ok' if chk['crosspol_max'] < 1e-3 else 'warn'}] "
        f"{run['runid']}: empty co-pol degeneracy "
        f"{chk['copol_degeneracy']:.3e}, empty cross-pol max "
        f"{chk['crosspol_max']:.3e} (both should be at the solver floor)")
    return dict(empty_checks=chk, empty_integrity_passed=bool(ok))


# ===========================================================================
# Per-run driver
# ===========================================================================

def process_run(run, stage, env, manifest, man_hash, runs_dir, log, args):
    """Solve (unless --extract-only) and extract (unless --no-extract) ONE
    run, then write its checkpoint + status.  Never raises for a run-level
    failure: returns a status dict with status != 'ok' instead, so one bad
    run cannot abort the campaign."""
    rid = run["runid"]
    d = Path(runs_dir) / rid
    d.mkdir(parents=True, exist_ok=True)
    project_path = Path(run["project_path"])
    t_run = time.time()
    status = dict(runid=rid, kind=run["kind"], stage=stage,
                  theta_deg=run["theta_deg"], phi_deg=run["phi_deg"],
                  accuracy_tet=run["accuracy_tet"],
                  project_path=str(project_path),
                  manifest_sha256=man_hash, run_sha256=run_hash(run),
                  checkpoint_version=CHECKPOINT_VERSION,
                  convention=RAW_CONVENTION,
                  direction=DIRECTION_CST,
                  started=datetime.now().isoformat(timespec="seconds"),
                  finished=None, wall_time_s=None, solve_wall_s=None,
                  extract_wall_s=None, solver_messages=None, mesh=None,
                  error=None, status="pending",
                  solver_launched=bool(not args.extract_only),
                  timeout_min=(float(args.timeout_min)
                               if args.timeout_min else None))

    def finish(state, error=None):
        status["status"] = state
        status["error"] = error
        status["finished"] = datetime.now().isoformat(timespec="seconds")
        status["wall_time_s"] = round(time.time() - t_run, 2)
        (d / STATUS_JSON).write_text(json.dumps(status, indent=2),
                                     encoding="utf-8")
        return status

    if not project_path.exists():
        msg = (f"project does not exist: {project_path}.  This driver does "
               f"NOT create projects (that would need a second "
               f"DesignEnvironment).  Run `python cst_campaign.py --build` "
               f"first, then re-run this driver -- it will resume here.")
        log(f"  [FATAL] {rid}: {msg}")
        return finish("project_missing", msg)

    # ---- solve ------------------------------------------------------------
    if not args.extract_only:
        log(f"  [solve] {rid}: opening {project_path.name} and starting the "
            f"FD solver (SEQUENTIAL, one solver at a time)")
        project = None
        try:
            project = env.open_project(str(project_path.resolve()))
            wall, tstate = _solve_blocking_or_watchdog(
                project, args.timeout_min, log)
            status["solve_wall_s"] = round(float(wall), 1)
            status["solver_messages"] = _safe_messages(project)
            if tstate == "stuck":
                status["session_unsafe"] = True
                try:
                    project.save()
                except Exception:
                    pass
                return finish("timeout_stuck",
                              f"solver watchdog expired after "
                              f"{args.timeout_min} min and the solver thread "
                              f"did not return")
            if tstate == "aborted":
                try:
                    project.save()
                    project.close()
                except Exception:
                    pass
                return finish("timeout",
                              f"solver aborted by the --timeout-min "
                              f"watchdog after {args.timeout_min} min")
            log(f"  [ok] {rid}: FD solver finished in "
                f"{status['solve_wall_s']:.0f} s")
            project.save()
            project.close()
        except SolveError:
            raise
        except Exception as e:                    # noqa: BLE001
            log(f"  [FATAL] {rid}: solver failed: {e}")
            if project is not None:
                try:
                    status["solver_messages"] = _safe_messages(project)
                    project.save()
                    project.close()
                except Exception:
                    pass
            return finish("solve_failed", f"{type(e).__name__}: {e}")
    else:
        log(f"  [extract-only] {rid}: no solver launched; reading results "
            f"from the existing project")

    status["mesh"] = _mesh_info(project_path)
    if status["mesh"]["L_domain_um"] is not None:
        L_exp = manifest.get("L_expected_um")
        log(f"  [ok] {rid}: mesh log z-extent "
            f"{status['mesh']['L_domain_um']:.6f} um "
            f"(manifest L_expected {L_exp:.6f} um, "
            f"diff {(status['mesh']['L_domain_um'] - L_exp) * 1e3:+.2f} nm)"
            + (f", {status['mesh']['n_tets']} tets"
               if status["mesh"]["n_tets"] else ""))

    if args.no_extract:
        log(f"  [warn] {rid}: --no-extract; checkpoint NOT written, so this "
            f"run stays 'incomplete' until `--extract-only` is run")
        return finish("solved_no_extract")

    # ---- extract ----------------------------------------------------------
    t_ex = time.time()
    try:
        payload, extra = extract_run(run, manifest, log)
    except Exception as e:                        # noqa: BLE001
        log(f"  [FATAL] {rid}: result extraction failed: {e}")
        return finish("extract_failed", f"{type(e).__name__}: {e}")
    status.update(extra)
    status["extract_wall_s"] = round(time.time() - t_ex, 2)

    if run["kind"].startswith("empty"):
        try:
            status.update(empty_integrity(run, payload, manifest, log))
        except Exception as e:                    # noqa: BLE001
            log(f"  [FATAL] {rid}: empty-cell integrity check failed: {e}")
            status["empty_integrity_passed"] = False
            status["empty_check_error"] = f"{type(e).__name__}: {e}"

    np.savez(d / CHECKPOINT_NPZ, **payload)
    log(f"  [ok] {rid}: checkpoint -> {d / CHECKPOINT_NPZ} "
        f"({N_ENTRIES} entries; RAW e^{{+j omega t}}, "
        f"{status['n_raw_points']} native + {len(payload['f_grid_THz'])} "
        f"target-grid points)")
    if status.get("empty_integrity_passed") is False:
        return finish("ok_empty_check_failed")
    return finish("ok")


# ===========================================================================
# Gates
# ===========================================================================

def _write_gate(runs_dir, stage, payload, log):
    p = Path(runs_dir) / GATE_JSON[stage]
    payload = dict(payload)
    payload["created"] = datetime.now().isoformat(timespec="seconds")
    p.write_text(json.dumps(payload, indent=2, default=str),
                 encoding="utf-8")
    log(f"  [ok] gate record -> {p}")
    return payload


def read_gate(runs_dir, stage):
    p = Path(runs_dir) / GATE_JSON.get(stage, "")
    if stage not in GATE_JSON or not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _empty_sign_gate(runs_dir, struct_id, empty_id, theta, phi, manifest,
                     log):
    """Shared prelude of both gates: load checkpoints, run the empty-cell
    convention checks, refuse on a retreating phase."""
    raw = load_raw_blocks(runs_dir, struct_id)
    emp = load_raw_blocks(runs_dir, empty_id)
    lm = deembed.map_cst_labels(theta, phi, empty_blocks=emp,
                                L_expected_um=manifest.get("L_expected_um"))
    chk = lm["empty_checks"]
    for a in (1, 2):
        if not chk[f"mode{a}"]["sign_ok"]:
            raise deembed.DeembedError(
                f"empty-cell phase-slope SIGN check FAILED for mode {a} at "
                f"theta={theta} (slope "
                f"{chk[f'mode{a}']['slope_rad_per_THz']:+.4f} rad/THz): the "
                f"conjugation direction or the 'inward' convention is wrong "
                f"-- refusing to de-embed")
    f, S11, S21 = deembed.deembed_blocks(raw, emp)
    log(f"  [ok] empty-cell convention checks pass at theta={theta:g} "
        f"(both modes advance; copol degeneracy "
        f"{chk['copol_degeneracy']:.2e}, empty cross-pol "
        f"{chk['crosspol_max']:.2e})")
    return f, S11, S21, chk


def gate_closure(runs_dir, manifest, man_hash, log):
    """par. 7 stage-1 gate.  Its residual spectrum IS the fit sigma."""
    res_dir = results_dir_for(runs_dir)
    sigma_npz = res_dir / SIGMA_NAME
    log.raw("")
    log("GATE 1 (closure): de-embedded complex S at theta = 0 vs "
        "aggregation/results/periodic_results.npz")
    log(f"        gate {deembed.CLOSURE_GATE:g}; the residual spectrum "
        f"BECOMES the fit sigma")
    f, S11, S21, chk = _empty_sign_gate(
        runs_dir, *STAGE_RUNS["closure"], CLOSURE_THETA, CLOSURE_PHI,
        manifest, log)

    xp = deembed.check_mirror_plane_crosspol(S11, S21, CLOSURE_PHI)
    log(f"  [{'ok' if xp['passed'] else 'warn'}] structure cross-pol at "
        f"(0, 0): max {xp['max_crosspol']:.3e} "
        f"(mirror-plane check, tol 5e-3)")

    S11_by_pol = {"mode1": S11[0, 0], "mode2": S11[1, 1]}
    S21_by_pol = {"mode1": S21[0, 0], "mode2": S21[1, 1]}
    res_dir.mkdir(parents=True, exist_ok=True)
    cl = deembed.closure_normal(f, S11_by_pol, S21_by_pol,
                                label=CLOSURE_LABEL,
                                out_npz=res_dir / f"{CLOSURE_LABEL}.npz",
                                out_fig=res_dir / f"fig_{CLOSURE_LABEL}.png")

    # Convention diagnostics: the gate is decided by the as-is residual
    # ONLY.  If it fails, these say whether a pure sign/conjugation
    # convention would have closed it -- information, never a silent fix.
    with np.load(deembed.REF_NPZ) as z:
        f_ref = z["freq"] / 1e12
        r11, r21 = z["S11"], z["S21"]
    alt = {}
    for name, m11, m21 in (
            ("as_is", 1.0, 1.0), ("S11_sign_flipped", -1.0, 1.0),
            ("both_sign_flipped", -1.0, -1.0)):
        w = 0.0
        for pol in S11_by_pol:
            g11 = deembed.interp_to_grid(f, S11_by_pol[pol], f_ref)
            g21 = deembed.interp_to_grid(f, S21_by_pol[pol], f_ref)
            w = max(w, float(np.max(np.abs(m11 * g11 - r11))),
                    float(np.max(np.abs(m21 * g21 - r21))))
        alt[name] = w
    w = 0.0
    for pol in S11_by_pol:
        g11 = deembed.interp_to_grid(f, S11_by_pol[pol], f_ref)
        g21 = deembed.interp_to_grid(f, S21_by_pol[pol], f_ref)
        w = max(w, float(np.max(np.abs(np.conj(g11) - r11))),
                float(np.max(np.abs(np.conj(g21) - r21))))
    alt["conjugated"] = w

    # sigma_channels: the per-observable residual spectra, so the acceptance
    # gate (and the fit) can take a true RMS over channels AND band instead
    # of the conservative per-frequency channel-max in `sigma`.
    chan_names = sorted(cl["residuals"])
    chan_res = np.stack([cl["residuals"][k] for k in chan_names])
    np.savez(sigma_npz, f_THz=cl["f_THz"], sigma=cl["sigma"],
             sigma_channels=chan_res, channel_names=np.array(chan_names),
             sigma_rms=float(np.sqrt(np.mean(chan_res ** 2))),
             worst=cl["worst"], gate=cl["gate"], passed=cl["passed"],
             source=np.array("cst_solve.py gate_closure "
                             + " / ".join(STAGE_RUNS["closure"])),
             runids=np.array(list(STAGE_RUNS["closure"])),
             manifest_sha256=np.array(man_hash),
             created=np.array(datetime.now().isoformat(timespec="seconds")),
             note=np.array("par. 7: this residual spectrum REPLACES the "
                           "3e-3 placeholder sigma in the fit weights, the "
                           "observability lambda and the noise gates"))

    log(f"  closure worst complex residual = {cl['worst']:.4e} "
        f"(gate {cl['gate']:g}): "
        f"{'PASS' if cl['passed'] else 'FAIL'}")
    log(f"  fit sigma spectrum: min {float(cl['sigma'].min()):.3e}, "
        f"median {float(np.median(cl['sigma'])):.3e}, "
        f"max {float(cl['sigma'].max()):.3e} over "
        f"{len(cl['sigma'])} frequencies")
    log(f"  scalar sigma for the acceptance chi2 ({SIGMA_REDUCTION}): "
        f"{float(np.sqrt(np.mean(chan_res ** 2))):.3e} "
        f"over {chan_res.shape[0]} channels x {chan_res.shape[1]} freqs "
        f"(replaces the 3e-3 placeholder)")
    for k in sorted(cl["mag_dev"]):
        log(f"    {k}: magnitude dev {cl['mag_dev'][k]:.3e}, "
            f"phase dev {cl['phase_dev_rad'][k]:.3e} rad")
    log(f"  convention diagnostics (as_is decides the gate): "
        + ", ".join(f"{k} {v:.3e}" for k, v in alt.items()))
    if not cl["passed"]:
        best = min(alt, key=alt.get)
        if best != "as_is" and alt[best] <= cl["gate"]:
            log(f"  [FATAL] the closure FAILS as measured, but the "
                f"'{best}' convention would have passed ({alt[best]:.3e}) "
                f"-- this indicts a sign/conjugation convention, NOT the "
                f"physics.  Adjudicate before touching downstream code; "
                f"this driver refuses to apply a silent fix.")
    log(f"  [ok] fit sigma -> {sigma_npz}")
    log(f"  [ok] closure arrays -> {cl['out_npz']}")
    log(f"  [ok] closure figure -> {cl['out_fig']}")

    return _write_gate(runs_dir, "closure", dict(
        stage="closure", passed=bool(cl["passed"]),
        worst=float(cl["worst"]), gate=float(cl["gate"]),
        sigma_npz=str(sigma_npz), closure_npz=cl["out_npz"],
        figure=cl["out_fig"],
        sigma_min=float(cl["sigma"].min()),
        sigma_median=float(np.median(cl["sigma"])),
        sigma_max=float(cl["sigma"].max()),
        sigma_rms=float(np.sqrt(np.mean(chan_res ** 2))),
        sigma_reduction=SIGMA_REDUCTION,
        sigma_channels=chan_names,
        mag_dev=cl["mag_dev"], phase_dev_rad=cl["phase_dev_rad"],
        convention_diagnostics=alt,
        empty_checks=chk, structure_crosspol=xp,
        runs=list(STAGE_RUNS["closure"]),
        manifest_sha256=man_hash), log)


def _closure_channel_residuals(res_dir):
    """The per-observable residual spectra written by deembed.closure_normal
    (res_S11_mode1, ... in <CLOSURE_LABEL>.npz).  Used when the sigma npz
    itself predates the sigma_channels field."""
    p = Path(res_dir) / f"{CLOSURE_LABEL}.npz"
    if not p.exists():
        return None, None
    try:
        with np.load(p, allow_pickle=False) as z:
            names = sorted(k for k in z.files if k.startswith("res_"))
            if not names:
                return None, None
            return (np.stack([np.asarray(z[k], dtype=float) for k in names]),
                    [n[4:] for n in names])
    except Exception:                                 # noqa: BLE001
        return None, None


def measured_sigma(runs_dir, log, verbose=True):
    """The per-complex-observable noise scale MEASURED by the stage-1
    closure, for the acceptance chi2 test.

    REFUSES (SolveError) when results/fit_sigma_from_closure.npz is absent:
    the acceptance verdict genuinely depends on the measured sigma, so
    silently falling back to the 3e-3 placeholder would make the gate report
    a significance it has not earned.

    Reduction: RMS over all compared channels AND the whole band of
    |S_de - S_ref|.  Sources, in order of preference:
      1. `sigma_channels` in the sigma npz (per-observable residual spectra);
      2. the res_* arrays of the closure npz written alongside it -- used
         when the sigma npz was written before sigma_channels existed;
      3. `sigma` alone = the per-frequency channel-MAX; its RMS is
         CONSERVATIVE (larger sigma -> smaller z) and is labelled as such.
    """
    res_dir = results_dir_for(runs_dir)
    p = res_dir / SIGMA_NAME
    if not p.exists():
        raise SolveError(
            f"the acceptance chi2 test needs the MEASURED closure sigma and "
            f"{p} does not exist.  Run `--stage closure` first.  This driver "
            f"REFUSES to fall back to the 3e-3 placeholder: the verdict "
            f"depends on sigma, so a placeholder would manufacture "
            f"significance.")
    with np.load(p, allow_pickle=False) as z:
        keys = set(z.files)
        per_freq = np.asarray(z["sigma"], dtype=float)
        closure_passed = bool(z["passed"]) if "passed" in keys else None
        worst = (float(z["worst"]) if "worst" in keys
                 else float(per_freq.max()))
        res = names = None
        provenance = None
        if "sigma_channels" in keys:
            res = np.asarray(z["sigma_channels"], dtype=float)
            names = ([str(s) for s in z["channel_names"]]
                     if "channel_names" in keys else [])
            provenance = "sigma_channels in " + SIGMA_NAME
    if res is None:
        res, names = _closure_channel_residuals(res_dir)
        if res is not None:
            provenance = (f"res_* arrays of {CLOSURE_LABEL}.npz "
                          f"({SIGMA_NAME} predates sigma_channels)")
    if res is not None:
        sigma = float(np.sqrt(np.mean(res ** 2)))
        reduction = (f"{SIGMA_REDUCTION}: RMS over {res.shape[0]} channels "
                     f"x {res.shape[1]} frequencies"
                     + (f" ({', '.join(names)})" if names else ""))
    else:
        sigma = float(np.sqrt(np.mean(per_freq ** 2)))
        reduction = (f"{SIGMA_REDUCTION}: RMS over {len(per_freq)} "
                     f"frequencies of the per-frequency channel-MAX "
                     f"residual -- CONSERVATIVE (larger sigma lowers z); "
                     f"no per-channel residuals were found")
        provenance = "sigma (per-frequency channel-max) -- fallback"
    out = dict(sigma=sigma, reduction=reduction, provenance=provenance,
               source=str(p), closure_passed=closure_passed,
               closure_worst=worst,
               sigma_per_freq_min=float(per_freq.min()),
               sigma_per_freq_median=float(np.median(per_freq)),
               sigma_per_freq_max=float(per_freq.max()))
    if verbose:
        log(f"  [ok] measured sigma = {sigma:.4e}  [{provenance}]")
        log(f"       reduction: {reduction}")
        log(f"       closure worst {worst:.3e}, per-frequency sigma "
            f"{out['sigma_per_freq_min']:.2e} .. "
            f"{out['sigma_per_freq_max']:.2e} "
            f"(median {out['sigma_per_freq_median']:.2e})")
        if closure_passed is False:
            log("  [warn] the closure that produced this sigma did NOT pass "
                "its own gate; the acceptance verdict inherits that doubt")
    return out


def mirror_crosspol_check(runs_dir, manifest, log, write=True):
    """Model-FREE label-orientation check, kept alive after the acceptance
    stage moved off a mirror plane.

    deembed.check_mirror_plane_crosspol is only meaningful at
    phi in {0, 45} deg, where the physical cross-pol vanishes identically
    (doc par. 3), so measured cross-pol above tolerance indicts the label
    ORIENTATION without any reference model.  MIRROR_CHECK_STRUCT is a
    stage-'rest' run sharing the acceptance stage's empty, so this runs
    opportunistically whenever both checkpoints exist."""
    have = [checkpoint_path(runs_dir, r).exists()
            for r in (MIRROR_CHECK_STRUCT, MIRROR_CHECK_EMPTY)]
    if not all(have):
        miss = [r for r, h in zip((MIRROR_CHECK_STRUCT, MIRROR_CHECK_EMPTY),
                                  have) if not h]
        log(f"  [warn] model-free mirror-plane cross-pol check NOT yet "
            f"available: no checkpoint for {miss} (it is a stage-'rest' "
            f"run; this check re-runs automatically after stage 'rest')")
        return dict(available=False, missing=miss,
                    struct=MIRROR_CHECK_STRUCT, empty=MIRROR_CHECK_EMPTY)
    raw = load_raw_blocks(runs_dir, MIRROR_CHECK_STRUCT)
    emp = load_raw_blocks(runs_dir, MIRROR_CHECK_EMPTY)
    _, S11, S21 = deembed.deembed_blocks(raw, emp)
    xp = deembed.check_mirror_plane_crosspol(S11, S21, 0.0)
    out = dict(available=True, struct=MIRROR_CHECK_STRUCT,
               empty=MIRROR_CHECK_EMPTY, theta_deg=ACCEPTANCE_THETA,
               phi_deg=0.0, note=("model-free: physical cross-pol vanishes "
                                  "identically on phi in {0,45}; excess "
                                  "indicts the label ORIENTATION, no "
                                  "reference model involved"),
               **{k: (float(v) if isinstance(v, float) else v)
                  for k, v in xp.items()})
    log(f"  [{'ok' if xp['passed'] else 'FATAL'}] model-free mirror-plane "
        f"cross-pol on {MIRROR_CHECK_STRUCT} (theta={ACCEPTANCE_THETA:g}, "
        f"phi=0): max {xp['max_crosspol']:.3e} (tol 5e-3) -- "
        f"{'consistent with' if xp['passed'] else 'INDICTS'} the label "
        f"orientation")
    if write:
        p = Path(runs_dir) / MIRROR_CHECK_JSON
        p.write_text(json.dumps(dict(out,
                                     created=datetime.now().isoformat(
                                         timespec="seconds")), indent=2,
                                default=str), encoding="utf-8")
        log(f"  [ok] mirror-plane check record -> {p}")
    return out


def _label_hypotheses(extended=True):
    """deembed.label_hypotheses, tolerant of a deembed that predates the
    extended family (the flag was added 2026-08-07)."""
    try:
        return deembed.label_hypotheses(extended=extended), extended
    except TypeError:
        return deembed.label_hypotheses(), False


def _marginal_dimension(z_table, winner):
    """Which hypothesis coordinate separates the best-fitting hypothesis
    from its closest rival -- i.e. which dimension the verdict hangs on.

    `winner` may be None (the helper leaves it None on a REFUSAL, which is
    exactly when this diagnostic matters most); the best-fitting hypothesis
    is then taken from the z_table, where z == 0 by construction.

    Returns (differing_keys, rival_hypothesis, rival_z, best_hypothesis) or
    (None, None, None, None)."""
    if not z_table:
        return None, None, None, None
    best = winner or min(z_table, key=lambda r: r[1])[0]
    rivals = [(h, z) for h, z in z_table if dict(h) != dict(best)]
    if not rivals:
        return None, None, None, best
    h2, z2 = min(rivals, key=lambda r: r[1])
    keys = sorted(k for k in set(best) | set(h2)
                  if best.get(k, 1) != h2.get(k, 1))
    return keys, h2, z2, best


def _fallback_channel_dictionary_acceptance(fm, band, S11_m, S21_m,
                                            theta_deg, phi_deg, tol,
                                            direction, log):
    """MAX-statistic-only stand-in for validate_against_reference.
    channel_dictionary_acceptance, used ONLY when that module is not
    importable.  It CANNOT do the chi2 likelihood-ratio test the gate is
    specified on, so a run that lands here must REFUSE, not pass: the caller
    marks the result unusable.  Same forward model (reference T at
    (theta, phi, direction)) and the same 8 hypotheses, pooled over the 8
    channels and the band."""
    ia = fm.angle_index(float(theta_deg), float(phi_deg))
    T_ref = fm.data.T
    S11_p = np.empty((2, 2, len(band)), dtype=complex)
    S21_p = np.empty((2, 2, len(band)), dtype=complex)
    for p, i in enumerate(band):
        S = fm.predict(T_ref[i], i, [ia], direction)[0]
        S11_p[..., p] = S[0]
        S21_p[..., p] = S[1]
    hyps, ext = _label_hypotheses(ACCEPTANCE_FAMILY == "extended")
    table = []
    for hyp in hyps:
        h11, h21 = deembed.apply_hypothesis(S11_m, S21_m, hyp)
        table.append((hyp, max(float(np.max(np.abs(h11 - S11_p))),
                               float(np.max(np.abs(h21 - S21_p))))))
    resids = np.array([r for _, r in table])
    order = np.argsort(resids)
    out = dict(angle_index=int(ia), theta_deg=float(theta_deg),
               phi_deg=float(phi_deg), band=list(band), tol=float(tol),
               band_mode="pooled", table=table,
               family=("extended" if ext else "base"),
               n_hypotheses=len(hyps),
               n_winners=int(np.sum(resids <= tol)),
               best_residual=float(resids[order[0]]),
               second_residual=float(resids[order[1]]),
               margin=float(resids[order[1]] / tol),
               direction=int(direction), error=None, hypothesis=None,
               residual=None, passed=False)
    try:
        try:
            hyp, r = deembed.select_hypothesis(S11_m, S21_m, S11_p, S21_p,
                                               tol, extended=ext)
        except TypeError:
            hyp, r = deembed.select_hypothesis(S11_m, S21_m, S11_p, S21_p,
                                               tol)
        out.update(hypothesis=hyp, residual=float(r), passed=True)
    except deembed.DeembedError as e:
        out.update(passed=False, error=str(e))
    for hyp, r in table:
        log(f"    swap={str(hyp['swap']):<5s} s11_cross={hyp['s11_cross']:+d} "
            f"s21_cross={hyp['s21_cross']:+d} "
            f"r11_tm={hyp.get('r11_tm', 1):+d} : {r:.4e}"
            + ("   <-- within tol" if r <= tol else ""))
    return out


def gate_acceptance(runs_dir, manifest, man_hash, log, tol=ACCEPTANCE_TOL):
    """par. 7 stage-2 gate: the channel-dictionary acceptance at
    (ACCEPTANCE_THETA, ACCEPTANCE_PHI) = (60, 22.5) deg, direction = -1,
    decided by the chi2 likelihood-ratio statistic at the MEASURED closure
    sigma.  See the module docstring for why the angle is not the
    doc-literal (30, 0)."""
    log.raw("")
    log(f"GATE 2 (acceptance): channel dictionary at "
        f"(theta={ACCEPTANCE_THETA:g}, phi={ACCEPTANCE_PHI:g}), "
        f"direction={DIRECTION_CST}, statistic={ACCEPTANCE_STATISTIC}")
    log(f"        (NOT the doc-literal (30, 0): with the C4v-projected "
        f"reference T the cross-sign hypotheses are exactly degenerate on "
        f"every phi in {{0, 45}} mirror plane -- module docstring)")

    # sigma FIRST: the gate refuses by name if the closure has not measured
    # it, before any expensive forward evaluation.
    sig = measured_sigma(runs_dir, log)
    sigma = sig["sigma"]

    f, S11, S21, chk = _empty_sign_gate(
        runs_dir, *STAGE_RUNS["acceptance"],
        ACCEPTANCE_THETA, ACCEPTANCE_PHI, manifest, log)

    # phi = 22.5 is NOT a mirror plane -- check_mirror_plane_crosspol does
    # not apply here.  Report the measured cross-pol as information only
    # (at phi = 22.5 it is PHYSICAL and expected to be nonzero: these are
    # the only angles delivering all 8 observables, doc par. 3), and keep
    # the model-free check alive on the same-theta mirror-plane run.
    xoff = max(float(np.max(np.abs(S11[0, 1]))),
               float(np.max(np.abs(S11[1, 0]))),
               float(np.max(np.abs(S21[0, 1]))),
               float(np.max(np.abs(S21[1, 0]))))
    log(f"  [ok] structure cross-pol at ({ACCEPTANCE_THETA:g}, "
        f"{ACCEPTANCE_PHI:g}): max {xoff:.3e} -- INFORMATION ONLY "
        f"(phi=22.5 is not a mirror plane, so nonzero cross-pol is "
        f"physical and is exactly what makes this angle decisive)")
    mirror = mirror_crosspol_check(runs_dir, manifest, log)

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from forward import ForwardModel                       # noqa: PLC0415
    fm = ForwardModel()
    ia = fm.angle_index(ACCEPTANCE_THETA, ACCEPTANCE_PHI)
    band = [i for i in range(fm.nf) if fm.have[i, ia]]
    if not band:
        raise SolveError(
            f"the Bloch-sum cache has no frequency at angle index {ia} "
            f"({ACCEPTANCE_THETA:g}, {ACCEPTANCE_PHI:g}) -- run "
            f"precompute_C.py before this gate")
    log(f"  forward model: {len(band)} cached frequencies at angle index "
        f"{ia}; reference T = tmat.h5 /T")

    S11_m = S11[..., band]
    S21_m = S21[..., band]

    helper = None
    helper_err = None
    try:
        from validate_against_reference import (            # noqa: PLC0415
            channel_dictionary_acceptance)
        helper = channel_dictionary_acceptance
        log("  [ok] using validate_against_reference."
            "channel_dictionary_acceptance")
    except Exception as e:                                  # noqa: BLE001
        helper_err = f"{type(e).__name__}: {e}"

    base_res = None
    if helper is not None:
        try:
            res = helper(fm, band, S11_m, S21_m,
                         theta_deg=ACCEPTANCE_THETA, phi_deg=ACCEPTANCE_PHI,
                         tol=tol, direction=DIRECTION_CST,
                         band_mode="pooled",
                         statistic=ACCEPTANCE_STATISTIC, sigma=sigma,
                         family=ACCEPTANCE_FAMILY, verbose=True)
            pred = (res.get("S11_pred"), res.get("S21_pred"))
            res = {k: v for k, v in res.items()
                   if k not in ("S11_pred", "S21_pred")}
            # The base-8 verdict, for the record: it is the refusal that
            # forced the move to the extended family, and keeping it in the
            # log makes the reason legible without this docstring.
            try:
                b = helper(fm, band, S11_m, S21_m,
                           theta_deg=ACCEPTANCE_THETA,
                           phi_deg=ACCEPTANCE_PHI, tol=tol,
                           direction=DIRECTION_CST, band_mode="pooled",
                           statistic=ACCEPTANCE_STATISTIC, sigma=sigma,
                           family="base",
                           pred=(None if pred[0] is None else pred),
                           verbose=False)
                base_res = dict(
                    family="base", n_hypotheses=b.get("n_hypotheses"),
                    passed=bool(b.get("passed")),
                    hypothesis=b.get("hypothesis"),
                    chi2_reduced=b.get("chi2_reduced"),
                    z_margin=b.get("z_margin"),
                    n_winners=b.get("n_winners"),
                    best_residual=b.get("best_residual"),
                    error=b.get("error"))
            except Exception as e:                    # noqa: BLE001
                base_res = dict(family="base", error=f"not computed: {e}")
        except TypeError as e:
            helper = None
            helper_err = f"signature mismatch: {e}"

    if helper is None:
        # The fallback implements the doc-literal MAX statistic only.  It
        # CANNOT decide this gate, which is specified on chi2 at the
        # measured sigma, so the verdict is forced to REFUSE.
        log("  " + "!" * 70)
        log(f"  [FATAL] validate_against_reference."
            f"channel_dictionary_acceptance is UNUSABLE ({helper_err}).")
        log(f"  [FATAL] The local fallback implements the doc-literal MAX "
            f"statistic ONLY -- it cannot compute the chi2 likelihood ratio "
            f"this gate is specified on, and at phi={ACCEPTANCE_PHI:g} the "
            f"MAX verdict is NOT a substitute.  REFUSING rather than "
            f"silently downgrading the statistic.")
        log(f"  [FATAL] Fix the import, then re-run "
            f"`--stage acceptance` (the runs are already checkpointed, so "
            f"no solver time is spent).")
        log("  " + "!" * 70)
        res = _fallback_channel_dictionary_acceptance(
            fm, band, S11_m, S21_m, ACCEPTANCE_THETA, ACCEPTANCE_PHI,
            tol, DIRECTION_CST, log)
        res["max_statistic_passed"] = bool(res.get("passed"))
        res["passed"] = False
        res["error"] = (f"chi2 statistic unavailable ({helper_err}); the "
                        f"MAX-only fallback cannot decide this gate")

    # ---- base-8 verdict, reported alongside for the record ---------------
    log(f"  [record] hypothesis family = {res.get('family')} "
        f"({res.get('n_hypotheses')} members: the 8 port gauges x the "
        f"r11_tm sign on the S11 TM receive row)")
    if base_res is not None and "passed" in base_res:
        log(f"  [record] BASE-8 port-gauge verdict (does NOT decide this "
            f"gate): {'accept' if base_res['passed'] else 'REFUSE'}, "
            f"chi2_reduced "
            f"{base_res.get('chi2_reduced', float('nan')):.3f}, z_margin "
            f"{base_res.get('z_margin', float('nan')):.2f}")
        log(f"           A chi2_reduced >> 1 there is the campaign's "
            f"measured proof that no port gauge can describe the data: a "
            f"mode sign multiplies S11[a,b] by s_a*s_b and cannot touch a "
            f"co-polar diagonal, while the discrepancy IS a sign on the "
            f"S11 TM co-pol entry.")
    elif base_res is not None:
        log(f"  [record] BASE-8 verdict unavailable: {base_res.get('error')}")

    # ---- max-statistic verdict, reported alongside for the record --------
    max_passed = res.get("max_statistic_passed")
    if max_passed is None:
        max_passed = (res.get("n_winners") == 1)
    log(f"  [record] doc-literal MAX verdict (informational, does NOT "
        f"decide this gate): {res.get('n_winners')} of 8 hypotheses within "
        f"tol={tol:g} -> {'accept' if max_passed else 'refuse'}"
        f"; best {res.get('best_residual', float('nan')):.3e}, next-best "
        f"{res.get('second_residual', float('nan')):.3e}")

    # ---- sigma sensitivity: how close is the verdict to flipping? --------
    z1 = res.get("z_margin")
    z2 = res.get("z_margin_2sigma")
    z3 = res.get("z_margin_3sigma")
    z_min = res.get("z_min")
    if z1 is not None:
        log(f"  [record] sigma sensitivity of the chi2 verdict "
            f"(z_min = {z_min}):")
        log(f"             z_margin @ 1x sigma ({sigma:.3e}) = {z1:.2f}"
            f"   {'PASS' if z_min is None or z1 >= z_min else 'FAIL'}")
        if z2 is not None:
            log(f"             z_margin @ 2x sigma ({2 * sigma:.3e}) = "
                f"{z2:.2f}   "
                f"{'PASS' if z_min is None or z2 >= z_min else 'FAIL'}")
        if z3 is not None:
            log(f"             z_margin @ 3x sigma ({3 * sigma:.3e}) = "
                f"{z3:.2f}   "
                f"{'PASS' if z_min is None or z3 >= z_min else 'FAIL'}")
        log(f"             chi2_reduced = "
            f"{res.get('chi2_reduced', float('nan')):.3f} "
            f"(expect ~1; gated only from above at "
            f"{res.get('chi2_max')})")

    # ---- which coordinate the verdict actually hangs on -------------------
    marg_keys, rival, rival_z, best_hyp = _marginal_dimension(
        res.get("z_table"), res.get("hypothesis"))
    if marg_keys:
        w = best_hyp
        changes = ", ".join(
            f"{k}: {w.get(k, 1)} -> {rival.get(k, 1)}" for k in marg_keys)
        marginal = (z_min is not None and rival_z is not None
                    and rival_z < 2.0 * z_min)
        log(f"  [{'CAUTION' if marginal else 'record'}] the verdict is "
            f"decided by the {marg_keys} dimension: the closest rival "
            f"differs ONLY in ({changes}) at z = {rival_z:.2f}"
            + (f" = {rival_z / z_min:.2f}x z_min" if z_min else ""))
        if marginal:
            log(f"  [CAUTION] that is a thin margin.  At 2x the measured "
                f"sigma the same comparison gives z = "
                f"{(z2 if z2 is not None else rival_z / 2):.2f}, which "
                f"would {'REFUSE' if (z2 or rival_z / 2) < (z_min or 0) else 'still pass'}"
                f".  Treat any conclusion that depends on {marg_keys} as "
                f"provisional, and note that a cross-sign error is "
                f"UNOBSERVABLE (hence harmless) at every phi in {{0, 45}} "
                f"angle -- it flips entries that vanish identically there.")

    if res.get("passed"):
        log(f"  [ok] ACCEPTED ({ACCEPTANCE_STATISTIC}, family "
            f"{res.get('family')}, at the measured sigma {sigma:.3e}): "
            f"{res['hypothesis']} -- z_margin "
            f"{z1 if z1 is None else round(z1, 2)} >= z_min {z_min}, "
            f"chi2_reduced {res.get('chi2_reduced', float('nan')):.3f} <= "
            f"{res.get('chi2_max')}")
    else:
        log(f"  [FATAL] channel-dictionary acceptance REFUSED "
            f"(fail-safe by design): {(res.get('error') or '').splitlines()[0]}")
        if z1 is not None and z_min is not None and z1 < z_min:
            log(f"  [FATAL] the runner-up hypothesis is only {z1:.2f} sigma "
                f"away (need {z_min}), differing in {marg_keys}.  At the "
                f"measured sigma {sigma:.3e} the family is UNDECIDABLE in "
                f"that coordinate at this angle.  Correct fail-safe "
                f"behaviour, not a bug.  Mitigations, in order: (a) if the "
                f"unresolved coordinate is a CROSS-SIGN "
                f"(s11_cross/s21_cross), the ambiguity is UNOBSERVABLE and "
                f"therefore harmless on every phi in {{0,45}} angle -- fit "
                f"on mirror-plane angles, or pin the sign with the "
                f"model-free {MIRROR_CHECK_STRUCT} cross-pol check; "
                f"(b) improve sigma (the closure residual is the input -- "
                f"tighten AccuracyTet or the mesh); (c) add a decisive "
                f"angle: the cross-sign separation grows with theta at "
                f"phi=22.5 (measured 3.52e-3 / 7.11e-3 / 1.06e-2 at "
                f"theta = 30 / 45 / 60).")
        if res.get("chi2_reduced") is not None and \
                res.get("chi2_max") is not None and \
                res["chi2_reduced"] > res["chi2_max"]:
            log(f"  [FATAL] chi2_reduced = {res['chi2_reduced']:.2f} > "
                f"{res['chi2_max']}: the WINNER itself does not fit, so no "
                f"label verdict is trustworthy regardless of margin.  That "
                f"is the signature of a MISSING hypothesis dimension, not "
                f"of noise -- it is exactly how the campaign found r11_tm "
                f"(base-8 chi2_reduced 658.9 at this angle).  Check whether "
                f"the residual is concentrated in ONE channel: a single "
                f"channel at |d| = 2|S| is a sign convention, and the "
                f"family needs extending again.")

    return _write_gate(runs_dir, "acceptance", dict(
        stage="acceptance", passed=bool(res.get("passed")),
        theta_deg=ACCEPTANCE_THETA, phi_deg=ACCEPTANCE_PHI,
        direction=DIRECTION_CST, statistic=ACCEPTANCE_STATISTIC,
        family=res.get("family"), n_hypotheses=res.get("n_hypotheses"),
        base_family_verdict=base_res,
        marginal_dimension=marg_keys, marginal_rival=rival,
        marginal_rival_z=rival_z, best_fit_hypothesis=best_hyp,
        tol=float(tol),
        sigma=sigma, sigma_source=sig,
        z_min=z_min, chi2_max=res.get("chi2_max"),
        z_margin=z1, z_margin_2sigma=z2, z_margin_3sigma=z3,
        chi2_reduced=res.get("chi2_reduced"),
        D_best=res.get("D_best"), D_second=res.get("D_second"),
        n_obs=res.get("n_obs"),
        max_statistic_passed=bool(max_passed),
        n_winners=res.get("n_winners"), hypothesis=res.get("hypothesis"),
        residual=res.get("residual"),
        best_residual=res.get("best_residual"),
        second_residual=res.get("second_residual"),
        margin=res.get("margin"), error=res.get("error"),
        n_freqs=len(band), angle_index=res.get("angle_index"),
        table=[(h, r) for h, r in res.get("table", [])],
        D_table=[(h, d) for h, d in res.get("D_table", [])],
        z_table=[(h, z) for h, z in res.get("z_table", [])],
        helper=("validate_against_reference" if helper is not None
                else f"REFUSED -- chi2 unavailable ({helper_err})"),
        empty_checks=chk, structure_crosspol_offplane=xoff,
        mirror_crosspol=mirror,
        runs=list(STAGE_RUNS["acceptance"]),
        manifest_sha256=man_hash), log)


GATE_FUNCS = {"closure": gate_closure, "acceptance": gate_acceptance}


# ===========================================================================
# Ordering enforcement
# ===========================================================================

def stage_status(runs_dir, manifest, man_hash, redo=()):
    """Complete picture of every stage: which runs are checkpointed, whether
    the stage is complete, and the recorded gate verdict."""
    order = stage_run_order(manifest)
    by_id = runs_by_id(manifest)
    out = {}
    for st in STAGES:
        ids = order[st]
        states = {rid: checkpoint_state(runs_dir, by_id[rid], man_hash, redo)
                  for rid in ids}
        gate = read_gate(runs_dir, st)
        out[st] = dict(
            runs=ids, states=states,
            n_done=sum(1 for s in states.values() if s["valid"]),
            complete=all(s["valid"] for s in states.values()) and bool(ids),
            gate=gate,
            gate_needed=st in GATE_JSON,
            gate_passed=(True if st not in GATE_JSON
                         else bool(gate and gate.get("passed"))))
    return out


def blockers(stage, sstat):
    """The list of unmet par. 7 prerequisites for `stage` (empty = allowed)."""
    out = []
    for earlier in STAGES[:STAGES.index(stage)]:
        s = sstat[earlier]
        if not s["complete"]:
            out.append(f"stage '{earlier}' INCOMPLETE "
                       f"({s['n_done']}/{len(s['runs'])} runs checkpointed)")
        elif s["gate_needed"] and not s["gate_passed"]:
            g = s["gate"]
            if not g:
                verdict = "not evaluated"
            elif g.get("error"):
                verdict = f"FAILED ({str(g['error']).splitlines()[0]})"
            elif g.get("worst") is not None:
                verdict = (f"FAILED (worst {g['worst']:.3e} vs gate "
                           f"{g.get('gate')})")
            else:
                verdict = (f"FAILED ({g.get('n_winners')} of 8 hypotheses "
                           f"within {g.get('tol')})")
            out.append(f"stage '{earlier}' gate {verdict}: "
                       f"{STAGE_GATE_DESC[earlier]}")
    return out


def enforce_order(stage, sstat, force, log):
    """Refuse to start `stage` unless par. 7's prerequisites hold."""
    b = blockers(stage, sstat)
    if not b:
        return True
    if force:
        log.raw("")
        log("!" * 74)
        log(f"!!! --force: STARTING STAGE '{stage}' WITH UNMET par. 7 "
            f"PREREQUISITES !!!")
        for x in b:
            log(f"!!!   BYPASSING: {x}")
        log("!!! The par. 7 ordering exists because a phase-reference error "
            "masquerades")
        log("!!! as a T-matrix error.  Results produced past this point are "
            "NOT gated.")
        log("!" * 74)
        log.raw("")
        return True
    log(f"[FATAL] REFUSING to start stage '{stage}': par. 7 requires")
    for x in b:
        log(f"  - {x}")
    log("  Solve the earlier stages first (--stage closure, then "
        "--stage acceptance), or pass --force to bypass (loudly).")
    return False


# ===========================================================================
# Plan
# ===========================================================================

def print_plan(runs_dir, manifest, man_hash, sstat, log, requested=None,
               force=False):
    by_id = runs_by_id(manifest)
    log.raw("=" * 78)
    log.raw(f"CST SOLVE PLAN -- campaign {manifest['campaign_id']}, "
            f"{len(manifest['runs'])} runs")
    log.raw(f"  runs dir      : {runs_dir}")
    log.raw(f"  manifest hash : {man_hash}  (doc: {manifest['doc']})")
    log.raw(f"  domain        : L_expected = {manifest['L_expected_um']:.6f} "
            f"um, direction = {manifest['angles_direction']}, "
            f"Z_PAD = {manifest['z_pad_um']} um")
    log.raw(f"  storage       : {RAW_CONVENTION}")
    log.raw(f"  MODE          : PLAN ONLY -- no CST process is started and "
            f"no solver is launched")
    log.raw("=" * 78)

    for n, st in enumerate(STAGES, 1):
        s = sstat[st]
        b = blockers(st, sstat)
        log.raw("")
        log.raw(f"STAGE {n}/3  '{st}'  ({len(s['runs'])} runs)")
        log.raw(f"  gate      : {STAGE_GATE_DESC[st]}")
        if st in GATE_JSON:
            if s["gate"] is None:
                gtxt = "not evaluated"
            else:
                g = s["gate"]
                gtxt = ("PASSED" if g.get("passed") else "FAILED")
                if st == "closure" and g.get("worst") is not None:
                    gtxt += (f" (worst {g['worst']:.3e} vs gate "
                             f"{g.get('gate')})")
                if st == "acceptance":
                    if g.get("z_margin") is not None:
                        fam = (g.get("family")
                               or "base (record predates the extension)")
                        gtxt += (f" (family {fam}, "
                                 f"{g.get('n_hypotheses') or 8} members; "
                                 f"z_margin {g['z_margin']:.2f} vs z_min "
                                 f"{g.get('z_min')}, chi2_reduced "
                                 f"{g.get('chi2_reduced', float('nan')):.2f} "
                                 f"vs {g.get('chi2_max')}")
                        if g.get("marginal_dimension"):
                            gtxt += (f", marginal in "
                                     f"{g['marginal_dimension']}")
                        gtxt += ")"
                        if g.get("hypothesis"):
                            gtxt += f"  winner {g['hypothesis']}"
                    else:
                        gtxt += (f" ({g.get('n_winners')} of "
                                 f"{g.get('n_hypotheses', 8)} hypotheses "
                                 f"within {g.get('tol')})")
            log.raw(f"  gate state: {gtxt}")
        log.raw(f"  progress  : {s['n_done']}/{len(s['runs'])} runs hold "
                f"valid checkpoints"
                + ("  [STAGE COMPLETE]" if s["complete"] else ""))
        if b:
            log.raw(f"  BLOCKED BY: " + ("; ".join(b))
                    + ("   (--force would bypass)" if not force
                       else "   (--force GIVEN: would proceed)"))
        else:
            log.raw(f"  prerequisites: satisfied -- this stage may start")
        log.raw(f"    {'#':>2} {'runid':<22} {'kind':<15} {'th':>4} "
                f"{'phi':>5} {'project':<9} {'checkpoint':<12} action")
        for i, rid in enumerate(s["runs"], 1):
            run = by_id[rid]
            cs = s["states"][rid]
            proj = "present" if Path(run["project_path"]).exists() \
                else "MISSING"
            if cs["valid"]:
                cp = "STALE" if cs["stale"] else "ok"
            elif cs["exists"]:
                cp = str(cs["status"] or "partial")[:12]
            else:
                cp = "-"
            if cs["valid"] and not cs["stale"]:
                action = "skip (resume)"
            elif cs["valid"] and cs["stale"]:
                action = "skip (stale; --redo to re-solve)"
            elif proj == "MISSING":
                action = "BLOCKED: run `cst_campaign.py --build`"
            else:
                action = f"SOLVE + extract ({cs['detail']})"
            if requested is not None and rid not in requested:
                action = "not requested"
            log.raw(f"    {i:>2} {rid:<22} {run['kind']:<15} "
                    f"{run['theta_deg']:>4g} {run['phi_deg']:>5g} "
                    f"{proj:<9} {cp:<12} {action}")

    log.raw("")
    log.raw("ORDERING (enforced in code, not merely documented):")
    log.raw("  closure  -> acceptance : blocked until both closure runs are "
            "checkpointed AND")
    log.raw("                           the <= %g complex closure passes."
            % deembed.CLOSURE_GATE)
    log.raw("                           Its residual becomes the fit sigma "
            "AND the sigma of the")
    log.raw("                           acceptance chi2 test -- the two "
            "gates are genuinely coupled.")
    log.raw("  acceptance -> rest     : blocked until both acceptance runs "
            "are checkpointed AND")
    log.raw("                           the chi2 test at (%g, %g) resolves "
            "the label family"
            % (ACCEPTANCE_THETA, ACCEPTANCE_PHI))
    log.raw("                           (z_margin >= z_min, chi2_reduced "
            "<= chi2_max) at that sigma.")
    log.raw("  --force bypasses a refusal and prints the bypassed gate by "
            "name.")
    log.raw("")
    sigma_npz = results_dir_for(runs_dir) / SIGMA_NAME
    log.raw(f"measured-sigma coupling: {sigma_npz}")
    try:
        sig = measured_sigma(runs_dir, log, verbose=False)
        log.raw(f"  [EXISTS] sigma for the acceptance chi2 = "
                f"{sig['sigma']:.4e}   [{sig['provenance']}]")
        log.raw(f"           {sig['reduction']}")
    except SolveError:
        log.raw("  [not yet written] -- the acceptance gate will REFUSE by "
                "name until stage 'closure' writes it")
    except Exception as e:                            # noqa: BLE001
        log.raw(f"  [unreadable: {e}]")
    mirror_json = Path(runs_dir) / MIRROR_CHECK_JSON
    log.raw(f"model-free mirror-plane cross-pol check: {MIRROR_CHECK_STRUCT} "
            f"vs {MIRROR_CHECK_EMPTY} (stage 'rest')")
    log.raw(f"  -> {mirror_json}"
            + ("  [EXISTS]" if mirror_json.exists()
               else "  [pending: needs the stage-'rest' checkpoint]"))
    log.raw("=" * 78)


# ===========================================================================
# Summary
# ===========================================================================

def print_summary(runs_dir, manifest, man_hash, results, sstat, log):
    log.raw("")
    log.raw("=" * 78)
    log.raw("SUMMARY")
    log.raw(f"  {'runid':<22} {'stage':<11} {'status':<22} {'solve s':>9} "
            f"{'extract s':>9}")
    for rid, stt in results.items():
        log.raw(f"  {rid:<22} {stt.get('stage', ''):<11} "
                f"{str(stt.get('status')):<22} "
                f"{(stt.get('solve_wall_s') if stt.get('solve_wall_s') is not None else float('nan')):>9.1f} "
                f"{(stt.get('extract_wall_s') if stt.get('extract_wall_s') is not None else float('nan')):>9.2f}")
    log.raw("")
    for st in STAGES:
        s = sstat[st]
        gtxt = "-" if st not in GATE_JSON else (
            "not evaluated" if s["gate"] is None else
            ("PASSED" if s["gate"].get("passed") else "FAILED"))
        log.raw(f"  stage {st:<11} {s['n_done']}/{len(s['runs'])} runs "
                f"checkpointed, gate: {gtxt}")
    log.raw("=" * 78)


# ===========================================================================
# CLI
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="par. 7 CST solve driver: sequential, checkpointed, "
                    "resumable, gate-ordered.  THE ONLY SOLVER LAUNCH SITE "
                    "IN THE REPOSITORY.")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR_DEFAULT)
    ap.add_argument("--plan", action="store_true",
                    help="print the ordered execution plan and exit "
                         "(DEFAULT; touches neither CST nor the solver)")
    ap.add_argument("--stage", choices=("closure", "acceptance", "rest",
                                        "all"),
                    help="solve a stage (par. 7 order enforced)")
    ap.add_argument("--runs", type=str, default=None,
                    help="comma-separated runids to solve (still "
                         "sequential, still gate-checked)")
    ap.add_argument("--force", action="store_true",
                    help="bypass a gate refusal (prints a loud warning "
                         "naming the bypassed gate)")
    ap.add_argument("--redo", type=str, default="",
                    help="comma-separated runids whose checkpoints are "
                         "invalidated and re-solved")
    ap.add_argument("--timeout-min", type=float, default=0.0,
                    help="per-run solver watchdog in minutes (0 = disabled, "
                         "the DEFAULT and the validated blocking path; "
                         "the watchdog path is opt-in and untested)")
    ap.add_argument("--no-extract", action="store_true",
                    help="solve only; do not read results or write "
                         "checkpoints")
    ap.add_argument("--extract-only", action="store_true",
                    help="re-read results from already-solved projects; "
                         "launches no solver and opens no DesignEnvironment")
    args = ap.parse_args(argv)

    if args.no_extract and args.extract_only:
        print("[FATAL] --no-extract and --extract-only are mutually "
              "exclusive")
        return 3

    runs_dir = Path(args.runs_dir)
    plan_only = args.plan or (args.stage is None and args.runs is None
                              and not args.extract_only)
    log = Logger(None if plan_only else runs_dir / LOG_NAME)

    try:
        manifest = load_manifest(runs_dir)
    except SolveError as e:
        print(f"[FATAL] {e}")
        log.close()
        return 3
    man_hash = manifest_hash(manifest)
    by_id = runs_by_id(manifest)
    order = stage_run_order(manifest)
    redo = tuple(x for x in args.redo.split(",") if x)
    for rid in redo:
        if rid not in by_id:
            print(f"[FATAL] --redo: unknown runid {rid!r}")
            log.close()
            return 3

    # ---- what was requested ------------------------------------------------
    requested = None
    stages_to_run = []
    if args.runs:
        ids = [x for x in args.runs.split(",") if x]
        bad = [x for x in ids if x not in by_id]
        if bad:
            print(f"[FATAL] --runs: unknown runid(s) {bad}")
            log.close()
            return 3
        requested = ids
        stages_to_run = sorted({stage_of(x) for x in ids},
                               key=STAGES.index)
    elif args.stage:
        stages_to_run = (list(STAGES) if args.stage == "all"
                         else [args.stage])
        requested = [r for st in stages_to_run for r in order[st]]
    elif args.extract_only:
        stages_to_run = list(STAGES)
        requested = [r["runid"] for r in manifest["runs"]]

    sstat = stage_status(runs_dir, manifest, man_hash, redo)

    if plan_only:
        print_plan(runs_dir, manifest, man_hash, sstat, log,
                   requested=requested, force=args.force)
        if requested is not None:
            log.raw("")
            log.raw(f"REQUESTED ({len(requested)} runs): "
                    f"{', '.join(requested)}")
            rc = 0
            for st in stages_to_run:
                b = blockers(st, sstat)
                if b and not args.force:
                    log.raw(f"  stage '{st}': WOULD REFUSE -- "
                            + "; ".join(b))
                    rc = 1
                elif b and args.force:
                    log.raw(f"  stage '{st}': would proceed under --force, "
                            f"BYPASSING: " + "; ".join(b))
                else:
                    log.raw(f"  stage '{st}': would proceed "
                            f"(prerequisites satisfied)")
            log.raw("  (plan mode: nothing was solved)")
            log.close()
            return rc
        log.close()
        return 0

    # ---- execution ---------------------------------------------------------
    log.raw("")
    log.raw("#" * 78)
    log(f"cst_solve.py starting: stages {stages_to_run}, "
        f"{len(requested)} run(s) requested")
    log(f"  runs dir {runs_dir}; manifest {man_hash}")
    log(f"  mode: "
        + ("EXTRACT-ONLY (no solver, no DesignEnvironment)"
           if args.extract_only else
           ("SOLVE (no extraction)" if args.no_extract
            else "SOLVE + extract")))
    log(f"  SEQUENTIAL: one solver at a time (single license)")
    if args.timeout_min:
        log(f"  [warn] --timeout-min {args.timeout_min}: the watchdog path "
            f"runs the solver on a worker thread and is UNTESTED; the "
            f"default (0) uses the validated blocking call")
    log.raw("#" * 78)

    env = None
    results = {}
    rc_runs = 0
    session_unsafe = False
    try:
        for st in stages_to_run:
            todo = [r for r in order[st]
                    if requested is None or r in requested]
            if not todo:
                continue
            sstat = stage_status(runs_dir, manifest, man_hash, redo)
            if not enforce_order(st, sstat, args.force, log):
                rc_runs = max(rc_runs, 1)
                break
            pending = [r for r in todo
                       if not sstat[st]["states"][r]["valid"]]
            log.raw("")
            log.raw("=" * 78)
            log(f"STAGE '{st}': {len(todo)} run(s) in scope, "
                f"{len(pending)} to {'extract' if args.extract_only else 'solve'}, "
                f"{len(todo) - len(pending)} already checkpointed (skipped)")
            log(f"  gate after this stage: {STAGE_GATE_DESC[st]}")
            log.raw("=" * 78)

            for rid in todo:
                cs = checkpoint_state(runs_dir, by_id[rid], man_hash, redo)
                if cs["valid"]:
                    log(f"  [skip] {rid}: valid checkpoint ({cs['detail']})")
                    if cs["stale"]:
                        log(f"  [warn] {rid}: checkpoint is STALE relative "
                            f"to the current manifest; NOT re-solved "
                            f"automatically -- use --redo {rid}")
                    continue
                # Open the ONE DesignEnvironment lazily, and only for a run
                # that can actually be solved: a missing project must not
                # cost a CST startup.
                if (env is None and not args.extract_only
                        and Path(by_id[rid]["project_path"]).exists()):
                    try:
                        env = open_design_environment(log)
                    except SolveError as e:
                        log(f"[FATAL] {e}")
                        log.close()
                        return 3
                stt = process_run(by_id[rid], st, env, manifest, man_hash,
                                  runs_dir, log, args)
                results[rid] = stt
                if stt["status"] not in ("ok",):
                    rc_runs = max(rc_runs, 2)
                if stt.get("session_unsafe"):
                    session_unsafe = True
                    break
            if session_unsafe:
                break

            # ---- stage gate ---------------------------------------------
            sstat = stage_status(runs_dir, manifest, man_hash, redo)
            if st not in GATE_FUNCS:
                if not sstat[st]["complete"]:
                    log(f"  [FATAL] stage '{st}' INCOMPLETE "
                        f"({sstat[st]['n_done']}/{len(sstat[st]['runs'])} "
                        f"runs checkpointed)")
                # The model-free orientation check the acceptance stage gave
                # up by moving off phi = 0 lives here; run it whenever its
                # checkpoints have appeared.
                if not args.no_extract:
                    try:
                        mirror_crosspol_check(runs_dir, manifest, log)
                    except Exception as e:            # noqa: BLE001
                        log(f"  [warn] model-free mirror-plane cross-pol "
                            f"check raised {type(e).__name__}: {e}")
                continue
            if not sstat[st]["complete"]:
                log(f"  [FATAL] stage '{st}' INCOMPLETE "
                    f"({sstat[st]['n_done']}/{len(sstat[st]['runs'])} runs "
                    f"checkpointed): the gate CANNOT be evaluated and the "
                    f"next stage stays blocked")
                rc_runs = max(rc_runs, 2)
                continue
            if args.no_extract:
                log(f"  [warn] --no-extract: stage '{st}' gate skipped "
                    f"(no checkpoints written)")
                continue
            try:
                g = GATE_FUNCS[st](runs_dir, manifest, man_hash, log)
            except Exception as e:                     # noqa: BLE001
                log(f"  [FATAL] stage '{st}' gate raised "
                    f"{type(e).__name__}: {e}")
                _write_gate(runs_dir, st,
                            dict(stage=st, passed=False,
                                 error=f"{type(e).__name__}: {e}",
                                 manifest_sha256=man_hash), log)
                rc_runs = max(rc_runs, 1)
                break
            if not g["passed"]:
                log(f"  [FATAL] stage '{st}' gate FAILED -- later stages "
                    f"stay blocked (use --force only after adjudicating)")
                rc_runs = max(rc_runs, 1)
                if not args.force:
                    break
    finally:
        if env is not None:
            if session_unsafe:
                log("[FATAL] leaving the DesignEnvironment OPEN: a solver "
                    "thread did not return.  Close CST manually once the "
                    "solve state is clear.")
            else:
                try:
                    env.close()
                    log("[ok] DesignEnvironment closed")
                except Exception as e:                 # noqa: BLE001
                    log(f"[warn] DesignEnvironment close failed: {e}")

    sstat = stage_status(runs_dir, manifest, man_hash, redo)
    print_summary(runs_dir, manifest, man_hash, results, sstat, log)

    rc = rc_runs
    if session_unsafe:
        rc = 4
    if rc == 0:
        missing = [r for r in (requested or [])
                   if not checkpoint_state(runs_dir, by_id[r], man_hash
                                           )["valid"]]
        if missing and not args.no_extract:
            log(f"[FATAL] requested runs without a valid checkpoint: "
                f"{missing}")
            rc = 2
        for st in stages_to_run:
            if st in GATE_JSON and sstat[st]["complete"] \
                    and not sstat[st]["gate_passed"]:
                rc = max(rc, 1)
    log(f"exit code {rc}")
    log.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
