"""Optimize the coded cell against nuisance-marginalized information.

Reviewer recommendation 6 (`review.md`, 2026-08-07 15:45): maximize the
weakest direction of the Schur complement

    F_T = J_c^T J_c - J_c^T J_eta (J_eta^T J_eta + Q_eta)^-1 J_eta^T J_c

rather than sigma_40(H).  Gate A established why: the coded-cell inverse is
rank 40/40 with an exact noise-free recovery, but a port-plane offset is
99.98 % collinear with a change in T, so the design that maximizes raw
information need not be the design that separates T from calibration.

The run does three things:

  1. audits the incumbents (`small@8`, `generic@8`) so the starting point is
     on record;
  2. searches for a cell maximizing the MARGINALIZED objective, with the
     real Ewald C in the loop and a passive D4h ensemble.  NOTE: while
     TARGET_CONDITIONED_PRIOR is set, the ensemble's loss grid was calibrated
     against the reserved reference wheel, so this branch is NOT
     target-independent and nothing selected here can close Gate A or E;
  3. re-audits the winner against the incumbent under the same nuisance
     classes, and re-runs the Gate A blind recovery on it so the improvement
     is expressed in T error and not only in a singular value.

Run:  python -m fastfull.opt_marginalized [--samples 300] [--lam 8]
"""
import argparse
import binascii
import json
import os
import shutil
import sys
import time

import hashlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "aggregation"))
sys.path.insert(0, os.path.dirname(HERE))

from vswf import ModeBasis                                    # noqa: E402
from tmat_io import TMatrixData                               # noqa: E402

from . import design as dz                                    # noqa: E402
from . import symmetry as sym                                 # noqa: E402
from . import nuisance as nz                                  # noqa: E402
from . import lattice as lt                                   # noqa: E402
from . import transforms as xf                                # noqa: E402
from . import ewald as ew                                     # noqa: E402
from . import synthetic as sy                                 # noqa: E402
from . import m1_study as ms                                  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(HERE), "results", "fastfull")

LINEAGE_CONDITIONED = "target_conditioned"
LINEAGE_INDEPENDENT = "target_independent"

# Declared ensemble scale (dimensionless Frobenius norm of T).
# Fixed a priori so the search never reads the reference wheel.
ENSEMBLE_FRO = 0.25
# ---------------------------------------------------------------- WARNING
# TARGET-CONDITIONED PRIOR.  This loss grid was chosen so that the draws
# bracket the 8 um REFERENCE WHEEL's measured absorption.  The proposal
# (par. 7.3) reserves that T for benchmarking and forbids using it to choose
# priors, so this branch is DEVELOPMENT-ONLY and is not the
# target-independent ensemble the design contract requires.  It consumes part
# of the Gate-E ground truth, and no selection made with it can close Gate A
# or Gate E.  A genuinely independent anchor -- from declared Au dispersion,
# geometry-scale bounds and causal sector weights rather than from the
# answer -- is required before any candidate is selected on this basis, and
# a different untouched T must then be reserved for validation.
#
# The grid is also calibrated at the REFERENCE norm ||T||_F = 0.113992,
# where draw absorption is 3.5e-4 / 7.0e-4 / 1.4e-3 against the wheel's
# 5.515e-4.  The optimizer draws at ENSEMBLE_FRO = 0.25, where the same
# grid gives 5.8e-4 .. 3.0e-3, mean 1.69e-3 = 3.06x the wheel.  The
# production ensemble is therefore NOT wheel-matched either; the gate below
# measures the production configuration and reports that factor rather than
# a favourable one.
TARGET_CONDITIONED_PRIOR = True
LOSS_GRID = (0.0025, 0.005, 0.01)
STRESS_LOSS = 0.05                    # reported, never used for selection
WHEEL_ABSORPTION_8UM = 5.5147e-4      # target-derived; see the WARNING

# Hard-coded incumbents USED TO BYPASS THE ARCHIVE FILTER ENTIRELY: they
# entered the leaderboard as lineage-free source constants, and if one won the
# selected record was stamped with whatever lineage the current run used.  A
# geometry with no verified origin is exactly what must NOT be promotable, so
# each now carries an explicit declared lineage and is filtered like any
# archived proposal.
#
# Both are declared TARGET-CONDITIONED, and deliberately so.  They were chosen
# by the M1 design study, whose objective was evaluated with the reserved
# reference wheel in the loop; I cannot produce a hash-bound proof that their
# selection was independent of it.  Declaring them conditioned is the
# direction that cannot overclaim: it costs an independent run two starting
# points, whereas declaring them independent would smuggle target-derived
# geometry across the boundary the proposal (par. 7.3) draws.  Promoting
# either requires a rerun under a declared-independent prior, not an edit
# here.
INCUMBENT_DESIGNS = {
    "small@8": dz.Design(10.8121, 7.2371, 92.75, 19.68, -0.0098, -0.4930),
    "generic@8": dz.Design(13.7913, 13.5664, 89.74, 49.16, -0.5000, -0.5000),
}
INCUMBENT_LINEAGE = {
    "small@8": LINEAGE_CONDITIONED,
    "generic@8": LINEAGE_CONDITIONED,
}
INCUMBENTS = INCUMBENT_DESIGNS          # audited/reported everywhere; see
                                        # eligible_incumbents() for selection

# ARCHIVE of every candidate any previous run has proposed.  A fresh
# stochastic search must not be allowed to "win" with a point worse than one
# already known: the 04:58 run labelled a cell scoring 5.637 the winner while
# the archived 04:24 cell scored 5.924 on the SAME ensemble, simply because
# `search` resamples and never reconsiders old points.  Every archived cell is
# re-evaluated under the current ensemble and competes with the search output.
# Entries here are HAND-TRANSCRIBED at printed (rounded) precision and are
# therefore NOT trustworthy: the 04:24 candidate's true alpha was
# 63.4185055507 deg, not the 56.22 transcribed below, and that error alone
# suppressed the true point by 5.84% and reversed a leaderboard.  They are
# kept only as a labelled fallback; `candidate_registry.json` is the
# authority and every run appends its full-precision candidates to it.
_TRANSCRIBED = {
    "cand-0424(rounded)": dict(p1_um=6.7421, p2_um=11.5720, gamma_deg=91.36,
                               alpha_deg=56.22, f1=0.4600, f2=0.0336),
    "cand-0458(rounded)": dict(p1_um=10.8598, p2_um=7.0968, gamma_deg=95.23,
                               alpha_deg=6.07, f1=-0.2505, f2=0.2728),
}
REGISTRY = os.path.join(OUT_DIR, "candidate_registry.json")
RUNS_DIR = os.path.join(OUT_DIR, "runs")


def _lineage():
    return (LINEAGE_CONDITIONED if TARGET_CONDITIONED_PRIOR
            else LINEAGE_INDEPENDENT)


def _sha_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Files whose contents determine a run's numbers.  Hashed BEFORE the study
# executes and verified unchanged at exit: hashing at the end certifies
# whatever is on disk then, which in earlier cycles was not the code that
# ran.  The list deliberately includes the data inputs and the aggregation
# modules, not just this package.
SNAPSHOT_FILES = [
    os.path.join(HERE, m) for m in
    ("nuisance.py", "symmetry.py", "synthetic.py", "design.py", "jacobian.py",
     "transforms.py", "lattice.py", "ewald.py", "cost.py", "coupling.py",
     "m1_study.py", "opt_marginalized.py")
] + [
    os.path.join(os.path.dirname(HERE), m) for m in
    ("parametrize.py", "sparams_oblique.py", "bloch_lattice.py",
     "precompute_C.py", "forward.py")
] + [
    os.path.join(HERE, "..", "..", "aggregation", "vswf.py"),
    os.path.join(HERE, "..", "..", "aggregation", "tmat_io.py"),
    os.path.join(HERE, "..", "..", "aggregation", "translate.py"),
    ms.REF_TMAT,
    os.path.join(os.path.dirname(HERE), "results",
                 "fit_sigma_from_closure.npz"),
]


def snapshot_inputs():
    """{path: sha256} for every file that determines the numbers."""
    import platform
    import numpy
    import scipy
    snap = {}
    root = os.path.dirname(os.path.dirname(HERE))
    for p in SNAPSHOT_FILES:
        ap = os.path.abspath(p)
        if not os.path.exists(ap):
            continue
        try:
            key = os.path.relpath(ap, root)
        except ValueError:
            # On Windows relpath RAISES across drives, so a snapshotted input
            # on another mount used to kill the run outright.  Fall back to
            # the absolute path: identity is what matters, not prettiness.
            key = ap
        snap[key] = _sha_file(ap)
    vers = ["py" + platform.python_version(), "np" + numpy.__version__,
            "sp" + scipy.__version__]
    for name in ("treams", "h5py"):
        try:
            m = __import__(name)
            vers.append(name + getattr(m, "__version__", "?"))
        except Exception:
            vers.append(name + "-absent")
    snap["__env__"] = " ".join(vers)
    return snap


# Snapshot taken at IMPORT time (populated at the bottom of the module, once
# every helper exists).  `run()` compares against it, so a file edited after
# import but before run() -- which would otherwise be hashed in its NEW state
# while the OLD module object keeps executing -- is DETECTED rather than
# silently certified.  This does not make the run hermetic: the modules
# already imported are still the old ones, so the only safe response is to
# refuse and let the caller restart in a fresh process.
_IMPORT_SNAPSHOT = None


def assert_snapshot_matches_import(snap):
    """Refuse to run when disk no longer matches what this process imported."""
    if _IMPORT_SNAPSHOT is None:
        return
    changed = sorted(k for k in set(_IMPORT_SNAPSHOT) | set(snap)
                     if k != "__env__"
                     and _IMPORT_SNAPSHOT.get(k) != snap.get(k))
    if changed:
        raise RuntimeError(
            "these inputs changed after this process imported them: %s; the "
            "imported modules are still the OLD ones, so the run would "
            "certify bytes it did not execute -- restart in a fresh process"
            % ", ".join(changed))


def _snapshot_hash(snap):
    return hashlib.sha256(
        json.dumps(snap, sort_keys=True).encode()).hexdigest()


def design_key(design):
    """Canonical full-precision identity of a GEOMETRY.

    Two records of the same cell proposed by different runs are the same
    candidate.  Without this the archive grew without bound: every run
    republished the same deterministic search geometry under a new run
    prefix, the archive fingerprint moved, the next run derived a fresh id,
    and the cycle never reached the completed-identity refusal.
    """
    return _canon_hash({k: repr(v) for k, v in sorted(
        canonical_design(design).items())})


def canonical_design(design):
    """THE canonical parameter representation.

    `design_key` normalized signed zero but `freeze_archive` fingerprinted the
    RAW record dictionary, so two archives holding the same geometry as
    `alpha=+0.0` and `alpha=-0.0` shared a geometry key and still produced
    different fingerprints -- identical selector geometry, different archive
    epoch and run id.  Both now go through this function.

    Only EXACT symmetries are applied: `+0.0`/`-0.0` are the same float, and
    alpha is taken mod 360 because a rotation by a full turn is the identity.
    Nothing that is merely near-equivalent is folded in here.
    """
    d = dict(design.to_dict() if hasattr(design, "to_dict") else design)
    out = {}
    for k, v in d.items():
        v = float(v) + 0.0                       # kills -0.0
        if k == "alpha_deg":
            v = v % 360.0 + 0.0
        elif k in ("f1", "f2"):
            # Bloch fractions are reciprocal-lattice periodic: f and f+1 give
            # the same primitive-cell phase, so f=-0.5 and +0.5 are one point.
            # Folded into the half-open zone [-0.5, 0.5).
            v = (v + 0.5) % 1.0 - 0.5 + 0.0
        out[k] = v
    return out


def make_record(design, origin, run_id, snap_hash, cfg_hash,
                proposal_lineage=None, proposal_proof=None):
    """One candidate record.

    `lineage` is the EVALUATION lineage -- the prior this run used.
    `proposal_lineage` is where the geometry CAME FROM and is immutable: an
    archived cell that wins under a later run keeps the lineage of the run
    that proposed it.  Stamping the current lineage on an archived design is
    how a target-conditioned proposal could be laundered into an independent
    manifest.
    """
    rec = dict(design=design.to_dict(), origin=origin, run_id=run_id,
               snapshot_sha256=snap_hash, config_sha256=cfg_hash,
               target_conditioned=bool(TARGET_CONDITIONED_PRIOR),
               lineage=_lineage(),
               proposal_lineage=proposal_lineage or _lineage())
    if proposal_proof is not None:
        rec["proposal_proof"] = proposal_proof
    return rec


def _proof(source, parent_run_id, parent_record, design, lineage,
           parent_output_root=None, parent_record_digest=None):
    """A resolvable provenance citation for a `selected` record.

    `record_digest` and the parent's output root are included for run-backed
    sources.  Naming a run and a record was not enough: the id identifies the
    parent's INPUTS, so a parent whose candidate bytes were replaced still
    answered to the same citation.  `record_digest` existed but had no call
    site, which is exactly the gap this closes.
    """
    d = dict(source=source, parent_run_id=str(parent_run_id),
             parent_record=str(parent_record),
             design_key=design_key(design), proposal_lineage=lineage)
    if parent_output_root is not None:
        d["parent_output_root"] = parent_output_root
    if parent_record_digest is not None:
        d["parent_record_digest"] = parent_record_digest
    return d


ORIGIN_PREFIXES = ("search", "polish", "selected", "transcribed")
DESIGN_FIELDS = ("p1_um", "p2_um", "gamma_deg", "alpha_deg", "f1", "f2")


def _design_ok(d):
    """The six-field finite design schema, checked BEFORE any selector
    deserializes it.

    `_record_ok` used to check only that a `design` key existed.  A
    self-consistent completed run carrying `design={}` therefore verified,
    and the selector then died with `KeyError: 'p1_um'` -- a malformed
    directory could take down the next run rather than being skipped.
    """
    if not isinstance(d, dict) or set(d) != set(DESIGN_FIELDS):
        return False
    for f in DESIGN_FIELDS:
        v = d[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
        if not np.isfinite(v):
            return False
    if not (d["p1_um"] > 0.0 and d["p2_um"] > 0.0):
        return False
    if not (1.0 < d["gamma_deg"] < 179.0):
        return False
    if not (abs(d["f1"]) <= 1.0 and abs(d["f2"]) <= 1.0):
        return False
    try:
        dz.Design.from_dict(d)
    except Exception:
        return False
    return True


FRESH_ORIGINS = ("search", "polish")


PROOF_SOURCES = ("same_run", "archive", "incumbent", "transcribed")


def _record_ok(rec, shelf_lineage):
    """Local schema for ONE record.  Provenance RESOLUTION is separate --
    see `resolve_selected_parent` -- because it needs the whole run."""
    need = ("design", "origin", "run_id", "snapshot_sha256", "config_sha256",
            "target_conditioned", "lineage", "proposal_lineage")
    if not all(k in rec for k in need):
        return False
    if not _design_ok(rec["design"]):
        return False
    origin = rec["origin"]
    if not isinstance(origin, str) or not origin.startswith(ORIGIN_PREFIXES):
        return False
    if rec["lineage"] != shelf_lineage:
        return False
    prop = rec["proposal_lineage"]
    if prop not in (LINEAGE_CONDITIONED, LINEAGE_INDEPENDENT):
        return False
    if prop == LINEAGE_CONDITIONED and shelf_lineage == LINEAGE_INDEPENDENT:
        return False
    if origin.startswith(FRESH_ORIGINS) and prop != rec["lineage"]:
        # a fresh proposal cannot claim a lineage its run did not use
        return False
    want_cond = (shelf_lineage == LINEAGE_CONDITIONED)
    return bool(rec["target_conditioned"]) == want_cond


MAX_PROVENANCE_DEPTH = 16


def _check_incumbent(name, key, lineage):
    """Terminal check for an incumbent citation.  Shared, because
    `_walk_provenance` used to accept ANY nested proof whose source string
    said `incumbent` as a root without looking at the named constant -- so a
    parent whose own selected record cited an invalid incumbent was treated
    as a valid ancestor even though full verification rejects that parent."""
    if name not in INCUMBENT_DESIGNS:
        return "incumbent proof names an unknown incumbent %r" % name
    if design_key(INCUMBENT_DESIGNS[name]) != key:
        return "incumbent %r has a different geometry" % name
    if INCUMBENT_LINEAGE.get(name) != lineage:
        return "incumbent %r has a different declared lineage" % name
    return None


def _check_transcribed(name, key, lineage):
    """Terminal check for a transcription citation.  Shared, as above."""
    if name not in _TRANSCRIBED:
        return "transcription proof names an unknown entry %r" % name
    try:
        if design_key(dz.Design.from_dict(_TRANSCRIBED[name])) != key:
            return "transcribed entry %r has a different geometry" % name
    except Exception as exc:
        return "transcribed entry %r is undecodable: %s" % (name, exc)
    if lineage != LINEAGE_CONDITIONED:
        return "a transcribed fallback is target-conditioned by definition"
    return None


def _walk_provenance(run_id, record, key, lineage, runs_dir, seen=None,
                     proof=None):
    """Follow a citation to a FRESH search/polish root, or say why it fails.

    Each hop opens the named run directory, verifies its manifest and artifact
    hashes, finds the named record, and requires it to carry the same
    canonical geometry and proposal lineage.  A record whose origin is
    `search`/`polish` terminates the walk; a `selected` record is followed to
    ITS cited parent.  Cycles and runaway depth are refusals, not hangs.
    """
    seen = set() if seen is None else seen
    while True:
        if len(seen) >= MAX_PROVENANCE_DEPTH:
            return "provenance chain exceeds %d hops" % MAX_PROVENANCE_DEPTH
        if (run_id, record) in seen:
            return "provenance chain is cyclic at %s/%s" % (run_id[:12],
                                                            record)
        seen.add((run_id, record))
        rd = os.path.join(runs_dir, run_id)
        if not os.path.isdir(rd):
            return "cited parent run %s does not exist" % run_id[:12]
        man, shelf = _verify_structural(rd, run_id)
        if man is None:
            return "cited parent run %s does not verify (%s)" % (run_id[:12],
                                                                 shelf)
        parent = shelf.get(record)
        if not isinstance(parent, dict):
            return "cited record %r is not in parent run %s" % (record,
                                                                run_id[:12])
        # BYTE-LEVEL BINDING: the citation must name the parent's exact
        # published outputs and the exact record it read, not merely a run id
        # and a record name.
        # EVERY run-backed hop is bound, not only the first.  The previous
        # version validated the incoming proof and then set `proof = None`
        # without ever adopting the intermediate parent's own proof, so a
        # middle record could carry a deliberately wrong parent root and
        # record digest and the walk still succeeded.
        if proof is not None:
            want_root = proof.get("parent_output_root")
            if want_root is not None:
                got_root = output_root(rd, run_id)
                if got_root != want_root:
                    return ("parent %s published outputs %s, but the citation "
                            "names %s" % (run_id[:12], str(got_root)[:12],
                                          str(want_root)[:12]))
            want_rec = proof.get("parent_record_digest")
            if want_rec is not None and record_digest(parent) != want_rec:
                return ("parent record %s/%s does not match the digest the "
                        "citation names" % (run_id[:12], record))
        try:
            if design_key(dz.Design.from_dict(parent["design"])) != key:
                return "parent %s/%s carries a different geometry" % (
                    run_id[:12], record)
        except Exception as exc:
            return "parent design is undecodable: %s" % exc
        if parent.get("proposal_lineage") != lineage:
            return "parent %s/%s has a different proposal lineage" % (
                run_id[:12], record)
        if parent["origin"].startswith(FRESH_ORIGINS):
            return None                       # reached a fresh proposal root
        pp = parent.get("proposal_proof")
        if not isinstance(pp, dict):
            return "parent %s/%s is a selection with no proof" % (
                run_id[:12], record)
        # THE PROOF'S OWN FIELDS COME FIRST.  Checking the terminal NAME
        # before them let a child accept a parent that full verification
        # rejects for contradictory `design_key`/`proposal_lineage`: the name
        # lookup succeeded and returned before the contradiction was reached.
        if pp.get("design_key") != key:
            return "parent %s/%s cites a different geometry than it carries" \
                % (run_id[:12], record)
        if pp.get("proposal_lineage") != lineage:
            return "parent %s/%s cites a different proposal lineage" % (
                run_id[:12], record)
        # A TERMINAL IS CHECKED, NOT TAKEN ON ITS WORD.
        if pp.get("source") == "incumbent":
            return _check_incumbent(pp.get("parent_record"), key, lineage)
        if pp.get("source") == "transcribed":
            return _check_transcribed(pp.get("parent_record"), key, lineage)
        proof = pp              # ADOPT it: the next hop is bound by THIS
        if pp.get("source") == "same_run":
            if pp.get("parent_run_id") != run_id:
                return "same_run proof in %s does not name its own run" % \
                    run_id[:12]
            if not isinstance(pp.get("parent_record_digest"), str):
                return ("same_run proof in %s carries no record digest"
                        % run_id[:12])
            record = pp.get("parent_record", "")
        elif pp.get("source") == "archive":
            # an intermediate ARCHIVE hop must agree with THAT parent's own
            # archive body, not merely name something
            pbody = man.get("archive_body")
            if not isinstance(pbody, dict) or key not in pbody:
                return "parent %s cites an archive entry absent from its " \
                       "own archive body" % run_id[:12]
            pe = pbody[key]
            if (pp.get("parent_run_id") != str(pe.get("first_run"))
                    or pp.get("parent_record") != pe.get("name")):
                return "parent %s archive citation disagrees with its own " \
                       "body" % run_id[:12]
            for f in ("parent_output_root", "parent_record_digest"):
                if not isinstance(pp.get(f), str) or not pp[f]:
                    return ("archive proof in %s carries no %s"
                            % (run_id[:12], f))
            run_id = pp.get("parent_run_id", "")
            record = pp.get("parent_record", "")
        else:
            return "parent proof has an unknown source %r" % pp.get("source")
        if not isinstance(record, str) or not record:
            return "parent proof names no record"


def _verify_structural(run_dir, rid):
    """Verify a PARENT run FULLY except for recursive descent.

    `_resolve=False` used to skip the archive-invariant check and the local
    proof validation as well, not just the recursion.  A middle ancestor
    whose archive fingerprint contradicted its own body therefore failed FULL
    verification while still legitimising a child that cited it -- the child
    was admitted and the parent rejected.  Only the descent is suppressed
    now; every local invariant runs, because the caller walks the chain
    itself and recursing here would redo the same work exponentially.
    """
    return verify_completed_run(run_dir, expect_run_id=rid, _descend=False)


def resolve_selected_parent(rec, shelf, archive_body, rid, runs_dir=None,
                            descend=True, seen=None):
    """RESOLVE a selected record's parent, or say why it cannot be resolved.

    A `proposal_proof` used to be syntactic: nonempty strings that agreed with
    the child's own design.  Nothing resolved the named parent, so a record
    citing `does-not-exist/imaginary` verified and its geometry entered an
    independent freeze.  Worse, SAME-lineage selections needed no proof at
    all, so an independent selected record carrying an arbitrary geometry --
    unrelated to its own run's search, its polish shortlist, or the archive --
    also verified.

    Every selected record must now name exactly one existing source and match
    it on canonical geometry and proposal lineage:

      * `same_run`   -- a search/polish record in THIS run's shelf;
      * `archive`    -- an entry of the frozen archive body persisted in the
                        run's own config, by geometry key;
      * `incumbent`  -- a declared incumbent constant;
      * `transcribed`-- a labelled `(rounded)` fallback.
    """
    proof = rec.get("proposal_proof")
    if not isinstance(proof, dict):
        return "selected record carries no proposal_proof"
    src = proof.get("source")
    if src not in PROOF_SOURCES:
        return "proposal_proof source %r is not one of %s" % (src,
                                                              PROOF_SOURCES)
    need = ["parent_run_id", "parent_record", "design_key",
            "proposal_lineage"]
    # RUN-BACKED CITATIONS MUST BIND BYTES.  These two were written only when
    # a caller happened to supply them and compared only when present, so a
    # citation could silently downgrade to the old run-id/record-name
    # semantics -- which is exactly the binding that was claimed to be closed.
    if src == "archive":
        need += ["parent_output_root", "parent_record_digest"]
    elif src == "same_run":
        need += ["parent_record_digest"]
    for f in need:
        if not isinstance(proof.get(f), str) or not proof[f]:
            return "proposal_proof is missing %s" % f
    try:
        child = design_key(dz.Design.from_dict(rec["design"]))
    except Exception as exc:
        return "selected design is undecodable: %s" % exc
    if proof["design_key"] != child:
        return "proposal_proof names a different geometry than it carries"
    if proof["proposal_lineage"] != rec["proposal_lineage"]:
        return "proposal_proof contradicts the record's proposal lineage"

    if src == "same_run":
        if proof["parent_run_id"] != rid:
            return "same_run proof does not name this run"
        parent = shelf.get(proof["parent_record"])
        if not isinstance(parent, dict):
            return ("same_run parent %r is not in this run's shelf"
                    % proof["parent_record"])
        if not parent.get("origin", "").startswith(FRESH_ORIGINS):
            return "same_run parent is not a search/polish record"
        try:
            pk = design_key(dz.Design.from_dict(parent["design"]))
        except Exception as exc:
            return "same_run parent design is undecodable: %s" % exc
        if pk != child:
            return "same_run parent carries a different geometry"
        if parent.get("proposal_lineage") != rec["proposal_lineage"]:
            return "same_run parent has a different proposal lineage"
        if record_digest(parent) != proof["parent_record_digest"]:
            return "same_run parent record does not match its cited digest"
        return None

    if src == "archive":
        # AN ARCHIVE CITATION MUST REACH A REAL PARENT RUN.  The previous
        # version looked the child key up in the child's OWN persisted
        # `archive_body` and compared the proof to that entry -- a hash over a
        # claim makes the claim immutable, not true.  A self-consistent
        # artifact naming `first_run=does-not-exist` with a ghost record and
        # no parent directory verified and entered an independent freeze.
        #
        # The body is still checked (it is what the run says it saw), but the
        # citation is then RESOLVED against the parent run directory on disk,
        # and the chain is followed to a fresh search/polish root.
        if not isinstance(archive_body, dict):
            return "archive proof but the run persisted no archive body"
        entry = archive_body.get(child)
        if not isinstance(entry, dict):
            return "archive proof names a geometry absent from the frozen "                   "archive"
        if entry.get("proposal_lineage") != rec["proposal_lineage"]:
            return "archive entry has a different proposal lineage"
        if proof["parent_run_id"] != str(entry.get("first_run")):
            return "archive proof names a parent run the archive disagrees "                   "with"
        if proof["parent_record"] != entry.get("name"):
            return "archive proof names a parent record the archive "                   "disagrees with"
        # the body's own entry must be internally consistent too
        try:
            if design_key(dz.Design.from_dict(entry["design"])) != child:
                return "archive entry's design does not match its own key"
        except Exception as exc:
            return "archive entry design is undecodable: %s" % exc
        # DESCENT IS OPTIONAL; THE LOCAL CHECKS ABOVE ARE NOT.  This
        # parameter was threaded all the way here and then never consulted,
        # so every structural ancestor check re-walked the whole suffix of the
        # chain, and a two-run archive cycle recursed to RecursionError
        # instead of stopping at MAX_PROVENANCE_DEPTH.  Exactly ONE outer
        # walker owns the `seen` set and the hop budget.
        if not descend:
            return None
        if runs_dir is None:
            return ("cannot resolve an archive citation without a runs "
                    "directory")
        return _walk_provenance(proof["parent_run_id"], proof["parent_record"],
                                child, rec["proposal_lineage"], runs_dir,
                                seen=seen if seen is not None else set(),
                                proof=proof)

    if src == "incumbent":
        return _check_incumbent(proof["parent_record"], child,
                                rec["proposal_lineage"])

    return _check_transcribed(proof["parent_record"], child,
                              rec["proposal_lineage"])


REQUIRED_ARTIFACTS = ("result.json", "candidates.json")
RECEIPTS = "receipts"


def output_root(run_dir, rid):
    """Content digest of what a run PUBLISHED, recomputed from the bytes.

    `run_id` hashes the snapshot and config -- the run's INPUTS.  It says
    nothing about which outputs were admitted, and the artifact digests lived
    as mutable labels inside `manifest.json` with nothing anchoring the
    manifest itself.  A probe therefore rewrote every candidate geometry, the
    selected proof key, the result winner and the leaderboard, refreshed the
    two artifact labels, and the run verified under the identical id.

    The output root is computed over the ACTUAL bytes of every required
    artifact plus the manifest with its own artifact labels removed (so the
    labels cannot be traded against the root).  It is written into the
    completion marker and appended to `runs/receipts.jsonl`, and verification
    requires all three to agree.

    THIS IS TAMPER-EVIDENCE, NOT TAMPER-PROOFING.  Anyone who can rewrite the
    artifacts can also rewrite the marker and the journal; what the scheme
    buys is that a partial edit -- the realistic accident and the realistic
    probe -- cannot pass, and that a child citing a parent binds the parent's
    exact bytes.  A signing key would be needed for more, and there is none.
    """
    parts = {}
    for nm in REQUIRED_ARTIFACTS:
        p = os.path.join(run_dir, nm)
        if not os.path.exists(p):
            return None
        parts[nm] = _sha_file(p)
    mf = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(mf):
        return None
    try:
        with open(mf) as fh:
            man = json.load(fh)
    except Exception:
        return None
    if not isinstance(man, dict):
        return None
    body = {k: v for k, v in man.items() if k != "artifacts"}
    return _canon_hash(dict(run_id=rid, artifacts=parts,
                            manifest=_canon_hash(body)))


def record_digest(rec):
    """Content digest of ONE candidate record, so a proof can cite bytes."""
    return _canon_hash(rec)


RECEIPT_LEASE_S = 120.0          # a publish takes milliseconds; this is slack
RECEIPT_INIT_GRACE_S = 5.0       # a freshly created lock may briefly be empty
_PUBLISH_HOOK = None             # test hook: fires inside the fenced section


def marker_owner(runs_dir, root):
    """The run id of a STRUCTURALLY VALID published run whose marker names
    `root`, or None.

    This is the authoritative-owner check, and it must be RECEIPT-INDEPENDENT.
    The previous version asked `verify_completed_run(...,
    allow_missing_receipt=True)`, which exempts only an *absent* receipt -- so
    in the one state it existed for (a present but unreadable receipt) it
    returned False, and `append_receipt` would then happily install a FOREIGN
    run id over a real published run's root, leaving that run permanently
    unverifiable.  `receipt_mode="skip"` ignores the receipt entirely.
    """
    if not os.path.isdir(runs_dir):
        return None
    for nm in os.listdir(runs_dir):
        if nm.startswith(".") or nm == RECEIPTS:
            continue
        rd = os.path.join(runs_dir, nm)
        mk = os.path.join(rd, "complete")
        if not os.path.isdir(rd) or not os.path.exists(mk):
            continue
        try:
            with open(mk) as fh:
                parts = fh.read().split()
        except Exception:
            continue
        if len(parts) < 2 or parts[1] != root:
            continue
        man, _ = verify_completed_run(rd, _descend=False,
                                      receipt_mode="skip")
        if man is not None:
            return nm
    return None


def _lock_state(lock, lease):
    """(token, stale) for a reservation file.

    A parse failure is NOT proof of staleness: a lock created with `O_EXCL`
    is briefly EMPTY before its body lands, and treating that as age infinity
    let a contender steal a reservation microseconds old.  An unparseable lock
    is judged by its mtime against a short initialization grace instead.
    """
    try:
        with open(lock) as fh:
            row = json.load(fh)
        tok = row["token"]
        age = time.time() - float(row["ts"])
        owner = row.get("owner")
    except Exception:
        try:
            age = time.time() - os.stat(lock).st_mtime
        except OSError:
            return None, False
        return None, age > max(lease, RECEIPT_INIT_GRACE_S)
    if owner == os.getpid():
        return tok, False        # our own live process: never steal from it
    return tok, age > lease


def _acquire(lock, lease):
    """Take the reservation and return our FENCING TOKEN, or None.

    Stealing is done by renaming the stale lock aside and then re-reading it:
    if the content is not the stale row we judged, we lost an ABA race (the
    owner released and a new publisher took it between our read and our
    rename) and we put it back rather than proceeding.
    """
    token = binascii.hexlify(os.urandom(12)).decode()
    body = json.dumps(dict(token=token, owner=os.getpid(), ts=time.time()))
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        seen, stale = _lock_state(lock, lease)
        if not stale:
            return None
        aside = lock + ".stale.%s" % token
        try:
            os.rename(lock, aside)
        except OSError:
            return None
        got, _ = _lock_state(aside, lease)
        if got != seen:
            # ABA: we grabbed a DIFFERENT lock than the one we judged stale
            try:
                os.rename(aside, lock)
            except OSError:
                pass
            return None
        os.unlink(aside)                     # tombstones are not left behind
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            return None
    with os.fdopen(fd, "w") as fh:
        fh.write(body)
        fh.flush()
        os.fsync(fh.fileno())
    return token


def _still_ours(lock, token):
    """Fencing check: is the reservation still the one we took?"""
    got, _ = _lock_state(lock, RECEIPT_LEASE_S)
    return got == token


def _release(lock, token):
    """Release ONLY our own reservation.

    An expired owner used to unlink unconditionally in its `finally`, which
    deleted its successor's live lock.
    """
    if _still_ours(lock, token):
        try:
            os.unlink(lock)
        except OSError:
            pass


def append_receipt(runs_dir, rid, root, _lease=None):
    """Publish ONE immutable receipt for `root`, under a fenced reservation.

    Seven designs were needed here, each closing the previous one's hole.

      1. shared append-only JSONL: unlocked, so one truncated fragment made
         every run in the namespace unreadable;
      2. keyed by run id: two same-identity workers (whose roots differ
         because `result.json` carries wall-clock `search_seconds`) wrote
         contradictory rows and the RENAME WINNER stopped verifying;
      3. per-root check-then-`os.rename`: returned success for any existing
         path, silently adopting a stale receipt as ours;
      4. per-root `O_CREAT|O_EXCL` on the FINAL path then the body: exclusive
         but not crash-atomic -- a torn receipt made every retry raise;
      5. reservation + fsynced temp + atomic replace: the final receipt could
         no longer be torn, but a crash left the LOCK forever;
      6. a timestamp lease: recovered from crashes, but was not MUTUAL
         EXCLUSION.  A freshly created lock is briefly empty and was stolen as
         "age infinity"; an ABA interleaving let a contender rename a lock
         that had already been replaced by a live one; nothing fenced the
         original owner, so a paused publisher could resume and overwrite the
         thief with BOTH returning success; the orphan sweep deleted an active
         publisher's temp; an expired owner unlinked its successor's lock; and
         reclamation of a torn final receipt happened OUTSIDE the reservation,
         so two recoverers could both proceed and one could unlink the other's
         valid receipt.

    Now: acquisition returns a FENCING TOKEN, everything -- including reading
    and reclaiming a torn receipt -- happens inside the reservation, and the
    token is re-checked immediately before the replace and again on release.
    A publisher that lost its reservation aborts without writing and without
    unlinking anything.

    NOT claimed: cross-process liveness detection.  A lock naming *this*
    process is never stolen, but a lock naming another live process that has
    exceeded the lease will be.  Fencing is what makes that safe -- the
    dispossessed owner cannot publish -- rather than the lease being correct.
    """
    lease = RECEIPT_LEASE_S if _lease is None else float(_lease)
    d = os.path.join(runs_dir, RECEIPTS)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "%s.json" % root)
    lock = path + ".lock"
    body = json.dumps(dict(run_id=rid, output_root=root), sort_keys=True)

    for _ in range(8):
        token = _acquire(lock, lease)
        if token is None:
            got = read_receipt(runs_dir, root)
            if got == rid:
                return path              # a concurrent publisher did our work
            if got is not None and got != "unreadable":
                raise RuntimeError(
                    "a receipt for output root %s already exists and names "
                    "%r, not %s" % (root[:12], got, rid[:12]))
            time.sleep(0.01)
            continue
        try:
            # EVERYTHING that inspects or repairs the receipt is now inside
            # the reservation, so two recoverers cannot both proceed.
            got = read_receipt(runs_dir, root)
            if got == rid:
                return path                              # idempotent
            if got is not None and got != "unreadable":
                raise RuntimeError(
                    "a receipt for output root %s already exists and names "
                    "%r, not %s; refusing to publish over it"
                    % (root[:12], got, rid[:12]))
            if got == "unreadable":
                owner = marker_owner(runs_dir, root)
                if owner is not None and owner != rid:
                    raise RuntimeError(
                        "the receipt for output root %s is unreadable and a "
                        "valid published run (%s) owns it; %s may not repair "
                        "it" % (root[:12], owner[:12], rid[:12]))
                os.unlink(path)
            # Sweep orphan temp bodies for THIS root.  We hold the
            # reservation, so any other temp for it is abandoned -- which is
            # what makes this safe.  The previous version swept during
            # acquisition, outside any reservation, and so could delete an
            # ACTIVE publisher's temp.
            for nm in os.listdir(d):
                if (nm.startswith("%s.json." % root) and nm.endswith(".tmp")
                        and token not in nm):
                    try:
                        os.unlink(os.path.join(d, nm))
                    except OSError:
                        pass
            if _PUBLISH_HOOK is not None:
                _PUBLISH_HOOK(token)
            tmp = path + ".%s.tmp" % token
            fh = open(tmp, "w")
            try:
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                fh.close()
            if not _still_ours(lock, token):
                # FENCED OUT: our reservation was taken while we worked.
                os.unlink(tmp)
                raise RuntimeError(
                    "lost the reservation for output root %s while "
                    "publishing; not writing" % root[:12])
            os.replace(tmp, path)           # atomic: never a torn receipt
            return path
        finally:
            _release(lock, token)
    raise RuntimeError("could not publish a receipt for output root %s"
                       % root[:12])


def read_receipt(runs_dir, root):
    """The run id recorded for `root`, None if absent, or `"unreadable"`.

    The body used to be trusted for its `run_id` alone, so its stored
    `output_root` could name a different root entirely and nothing noticed.
    The schema is now exact and the stored root must equal both the requested
    root and the one in the filename.
    """
    path = os.path.join(runs_dir, RECEIPTS, "%s.json" % root)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            row = json.load(fh)
    except Exception:
        return "unreadable"
    if not isinstance(row, dict) or set(row) != {"run_id", "output_root"}:
        return "unreadable"
    if not all(isinstance(row[k], str) and row[k] for k in row):
        return "unreadable"
    if row["output_root"] != root:
        return "unreadable"
    return row["run_id"]


def _canon(obj):
    """The ONE serialization used for hashing, so a hash can be recomputed
    from what was stored rather than merely compared to a stored label."""
    return json.dumps(obj, sort_keys=True, default=str)


def _canon_hash(obj):
    return hashlib.sha256(_canon(obj).encode()).hexdigest()


def derive_run_id(snap, cfg):
    """Run identity RECOMPUTED from the manifest bodies."""
    return hashlib.sha256(
        (_canon_hash(snap) + _canon_hash(cfg)).encode()).hexdigest()


def lineage_of_config(cfg):
    """Lineage DERIVED from the hashed config, never from a free label.

    `target_conditioned` is inside `cfg`, so it is covered by `cfg_hash` and
    therefore by the run id.  A manifest that merely *claims* a lineage
    cannot move a target-conditioned run into an independent selection.
    """
    if not isinstance(cfg, dict) or "target_conditioned" not in cfg:
        return None
    return (LINEAGE_CONDITIONED if bool(cfg["target_conditioned"])
            else LINEAGE_INDEPENDENT)


RESULT_SCHEMA_VERSION = 1
# Every scientific field the result MUST carry, with its required type.  The
# verifier previously bound identity, conditioning and the winner geometry and
# left everything around them open: a result with `winner_source` naming a
# candidate that does not exist, a fabricated leaderboard and
# `stress_audit={fake: NaN}` still verified.  The headline geometry was proved
# and the claimed experiment was not.
RESULT_FIELDS = {
    "schema_version": int, "run_id": str, "snapshot_sha256": str,
    "config_sha256": str, "snapshot": dict, "config": dict,
    "winner": dict, "winner_source": str, "target_conditioned_prior": bool,
    "evidence_status": str,
    "lam_um": float, "sigma": float, "seed": int, "n_samples": int,
    "polish": int, "n_ensemble": int, "search_seconds": float,
    "ensemble_fro": float, "loss_grid": list, "stress_loss": float,
    "leaderboard": list, "comparison": dict, "audits": dict,
    "stress_audit": dict, "ensemble_diversity": dict, "gate_a": dict,
    "constraints": dict, "nuisance_classes": list, "generator": str,
    "q_eta": float, "gate_a_schema_version": int,
}
# Three DISTINCT states.  The producer wrote `gate-candidate` for every
# independent prior while the verifier derived `screening-only` whenever Gate
# A was skipped, so the first independent skip-gate screening run would have
# quarantined its own otherwise valid stage.  Worse, "gate_a is non-empty"
# was treated as a pass: a dictionary full of FAILED recoveries derived
# `gate-candidate`.  Status is now computed in one place from explicit gate
# execution and explicit pass criteria.
# Three DISTINCT states.  The producer wrote `gate-candidate` for every
# independent prior while the verifier derived `screening-only` whenever Gate
# A was skipped, so the first independent skip-gate screening run would have
# quarantined its own otherwise valid stage.  Status is now computed in one
# place from explicit execution and explicit criteria.
#
# NOTE ON THE STRONGEST NAME.  It is `error-screen-passed`, NOT
# `gate-passed`.  What is checked below is a blind-recovery error screen over
# the declared candidate and perturbation sets.  Proposal Gate A also demands
# rank 40, useful-direction SNR above 10, noise-free and basin stability,
# passivity, and binding to frozen trial identities -- none of which the
# saved report currently carries, so none of which can be verified here.
# Calling a two-threshold screen a passed acceptance gate would be a false
# claim; the name says exactly what was tested.
EVIDENCE_STATUSES = ("screening-only", "error-screen-attempted",
                     "error-screen-passed", "custom-screen-passed")
# The version-1 producer's fixed comparison.  The candidate set used to be
# read from the artifact's own config and, when absent, not checked at all --
# so a hand-authored result naming one invented candidate could reach the
# strongest status.  `error-screen-passed` now REQUIRES this exact set; a
# passing screen over any other set gets `custom-screen-passed`, which
# deliberately carries no production-protocol implication.
GATE_A_PROTOCOL_CANDIDATES = ("small@8", "winner")
GATE_A_SCHEMA_VERSION = 1
GATE_A_MAX_FRO_ERR = 0.05          # proposal par. 7.3: p90 global dT < 5%
GATE_A_MAX_BLOCK_ERR = 0.05        # ...and dominant-sector
# What the screen does NOT establish, recorded next to the criteria so it
# cannot be quietly forgotten when the name is read.
GATE_A_UNVERIFIED = ("rank-40 identifiability", "useful-direction SNR > 10",
                     "noise-free and basin stability", "passivity",
                     "frozen trial/holdout identities")
GATE_A_REQUIRED_MODEL_FIELDS = ("fro_err", "fro_err_worst", "block_err_worst",
                                "dS_rank", "multistart_unique",
                                "position_in_bracket")


def gate_a_verdict(gate_a, cfg=None, expect_candidates=None):
    """(ran, passed, why) for a blind-recovery ERROR SCREEN report.

    The previous version iterated whatever candidate and model names the
    result happened to supply and tested two thresholds, so a one-cell report
    naming an invented model with errors `-1` and `-2` returned
    `(True, True)`.  Negative errors are impossible, an arbitrary subset is
    not the declared experiment, and a missing field is not a pass.

    The report must now cover the DECLARED candidate set and, for each, the
    declared perturbation families, with every required field present, finite
    and non-negative.
    """
    if not isinstance(gate_a, dict) or not gate_a:
        return False, False, "no gate_a report"
    if expect_candidates is None:
        # An absent declaration used to skip the check entirely, so an
        # artifact that simply omitted the key was never compared to anything.
        return True, False, ("the hashed config declares no gate_a candidate "
                             "set, so the screen's protocol cannot be "
                             "identified")
    got, want = set(gate_a), set(expect_candidates)
    if got != want:
        return True, False, ("gate_a covers %s, not the declared candidate "
                             "set %s" % (sorted(got), sorted(want)))
    want_models = set(sy.ERROR_MODELS)
    for nm, entry in gate_a.items():
        if not isinstance(entry, dict):
            return True, False, "gate_a[%s] is not an object" % nm
        models = entry.get("models")
        if not isinstance(models, dict) or not models:
            return True, False, "gate_a[%s] reports no models" % nm
        if set(models) != want_models:
            return True, False, ("gate_a[%s] reports models %s, not the "
                                 "declared %s" % (nm, sorted(models),
                                                  sorted(want_models)))
        for mn, m in models.items():
            if not isinstance(m, dict):
                return True, False, "gate_a[%s][%s] is not an object" % (nm,
                                                                         mn)
            for f in GATE_A_REQUIRED_MODEL_FIELDS:
                if f not in m:
                    return True, False, "gate_a[%s][%s] omits %s" % (nm, mn,
                                                                     f)
            # PRESENCE IS NOT VALIDATION.  These three were required only to
            # exist, so `dS_rank=None`, `multistart_unique=False` and
            # `position_in_bracket=None` passed a complete-looking report.
            if not isinstance(m["dS_rank"], int) or isinstance(m["dS_rank"],
                                                               bool):
                return True, False, "gate_a[%s][%s] dS_rank is not an int" % (
                    nm, mn)
            if m["dS_rank"] < 1:
                return True, False, ("gate_a[%s][%s] dS_rank = %r is not a "
                                     "real perturbation" % (nm, mn,
                                                            m["dS_rank"]))
            if not isinstance(m["multistart_unique"], bool):
                return True, False, ("gate_a[%s][%s] multistart_unique is not "
                                     "a bool" % (nm, mn))
            if not m["multistart_unique"]:
                return True, False, ("gate_a[%s][%s] reports a non-unique "
                                     "multistart basin" % (nm, mn))
            pib = m["position_in_bracket"]
            if not _is_num(pib) or not np.isfinite(pib):
                return True, False, ("gate_a[%s][%s] position_in_bracket is "
                                     "not a finite number" % (nm, mn))
            for f in ("fro_err", "fro_err_worst", "block_err_worst"):
                v = m[f]
                if not _is_num(v) or not np.isfinite(v):
                    return True, False, "gate_a[%s][%s] %s is not a finite " \
                        "number" % (nm, mn, f)
                if v < 0.0:
                    return True, False, ("gate_a[%s][%s] %s = %.4g is "
                                         "negative; an error cannot be"
                                         % (nm, mn, f, v))
            if m["fro_err_worst"] < m["fro_err"]:
                return True, False, ("gate_a[%s][%s] worst error is below "
                                     "its own median" % (nm, mn))
            for f, lim in (("fro_err_worst", GATE_A_MAX_FRO_ERR),
                           ("block_err_worst", GATE_A_MAX_BLOCK_ERR)):
                if m[f] > lim:
                    return True, False, ("gate_a[%s][%s] %s = %.4g exceeds "
                                         "%.4g" % (nm, mn, f, m[f], lim))
    if set(gate_a) != set(GATE_A_PROTOCOL_CANDIDATES):
        return True, "custom", ("all declared models within thresholds, but "
                                "over %s rather than the version-%d protocol "
                                "set %s"
                                % (sorted(gate_a), GATE_A_SCHEMA_VERSION,
                                   sorted(GATE_A_PROTOCOL_CANDIDATES)))
    return True, True, "all declared models within the screen thresholds"


def derive_evidence_status(lineage, cfg, gate_a, expect_candidates=None):
    """THE evidence status.  Producer and verifier both call this."""
    ran, passed, _ = gate_a_verdict(gate_a, cfg, expect_candidates)
    if cfg.get("skip_gate_a", False):
        ran = False
    if lineage != LINEAGE_INDEPENDENT or not ran:
        return "screening-only"
    if passed == "custom":
        return "custom-screen-passed"
    return "error-screen-passed" if passed else "error-screen-attempted"


def _is_num(v):
    """A real number, NOT a bool.  `bool` is a subclass of `int`, so
    `schema_version=True` used to satisfy an `int` requirement."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _typed(v, ty):
    if ty is bool:
        return isinstance(v, bool)
    if ty is int:
        return isinstance(v, int) and not isinstance(v, bool)
    if ty is float:
        return _is_num(v)
    return isinstance(v, ty)


# Fields inside a metric report that legitimately hold a string.  Kept as an
# explicit allowlist so a new one has to be declared, not discovered.
METRIC_STRING_FIELDS = ("label", "track", "model", "name", "source",
                        "latent_id", "production_row", "stress_row",
                        "generator")


def _metrics_ok(obj, path="", depth=0):
    """Reported metrics must be finite NUMBERS in a declared container.

    The earlier check walked dicts and lists and returned None for anything
    that was not a number, so arbitrary non-numeric leaves passed silently.
    """
    if depth > 12:
        return "nested too deeply at %s" % path
    if isinstance(obj, bool) or obj is None:
        return None
    if isinstance(obj, str):
        # A METRIC LEAF MAY NOT BE PROSE.  Allowing arbitrary strings meant
        # `p10="fabricated"` and free-text audit values passed the "every
        # number is finite" check vacuously.  Strings survive only as dict
        # keys (handled below) or in the declared allowlist.
        if path.rsplit(".", 1)[-1].split("[")[0] in METRIC_STRING_FIELDS:
            return None
        return "string where a metric belongs at %s" % path
    if isinstance(obj, (int, float)):
        return None if np.isfinite(obj) else "non-finite value at %s" % path
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                return "non-string key at %s" % path
            why = _metrics_ok(v, "%s.%s" % (path, k), depth + 1)
            if why:
                return why
        return None
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            why = _metrics_ok(v, "%s[%d]" % (path, i), depth + 1)
            if why:
                return why
        return None
    return "unsupported %s at %s" % (type(obj).__name__, path)


_BASIS_CACHE = {}


def _d4h_basis():
    """Cached 40-coefficient basis, built only when a paired block needs it."""
    if "B" not in _BASIS_CACHE:
        _BASIS_CACHE["B"] = sym.build_d4h_reciprocity_basis(
            ModeBasis.standard(3), verify_numeric=False)[0]
    return _BASIS_CACHE["B"]


def _paired_block_ok(p, cfg, nm):
    """A paired block must be RECOMPUTABLE, not merely well-shaped.

    Three versions were needed.  The first trusted a Boolean: a stub of
    `{n_pairs, paired_by_latent_id: True}` passed with no rows behind it.  The
    second checked each row's shape, positivity, id syntax and
    `ratio = stress/production` -- but every summary was still self-asserted,
    so a probe could rewrite all the latent ids and loss labels and set the
    aggregates to contradictory values (`p10="fabricated"`, a negative
    degradation count) and the run still verified.

    Now: the latent ids are REGENERATED from the hashed seed and ensemble
    size, the loss labels must lie on the hashed grid at the stratum their
    pair index implies, both sides' T-row hashes are required and must be
    distinct per side, and every aggregate is recomputed from the rows.
    """
    need = ("n_pairs", "p10", "p50", "p90", "worst", "best", "n_degraded",
            "pairs", "rows_bound_to_ensemble", "paired_by_latent_id",
            "worst_unpaired")
    for f in need:
        if f not in p:
            return "stress audit %s omits %s" % (nm, f)
    if not (p["rows_bound_to_ensemble"] and p["paired_by_latent_id"]):
        return "stress audit %s is not bound to the ensemble" % nm
    rows = p["pairs"]
    n_expected = int(cfg["n_ensemble"]) if "n_ensemble" in cfg else None
    if not isinstance(rows, list) or len(rows) != p["n_pairs"]:
        return "stress audit %s has the wrong row count for %r pairs" % (
            nm, p["n_pairs"])
    if n_expected is not None and p["n_pairs"] != n_expected:
        return "stress audit %s reports %r pairs, config says %r" % (
            nm, p["n_pairs"], n_expected)
    grid = [float(x) for x in cfg.get("loss_grid", [])]
    if not grid:
        return "stress audit %s: the hashed config declares no loss grid" % nm
    # REBUILD BOTH ENSEMBLES.  Row hashes previously needed only to be
    # nonempty, unique and disjoint strings -- `p0`/`s0` passed -- so nothing
    # tied a reported row to the T that was actually evaluated.  They are now
    # regenerated from the hashed seed, grid, norm and stress loss, and both
    # the per-row hashes AND the aggregate ensemble hashes must match.
    want_ids = want_prod = want_stress = None
    try:
        B = _d4h_basis()
        ens_p, ens_s, pair_loss = build_paired_ensembles(
            B, int(cfg["seed"]), n_expected, grid=tuple(grid),
            stress_loss=float(cfg["stress_loss"]),
            target_fro=float(cfg["ensemble_fro"]))
        want_ids = paired_ensemble_ids(B, int(cfg["seed"]), n_expected)
        want_prod = ensemble_row_ids(ens_p)
        want_stress = ensemble_row_ids(ens_s)
        for key, arr in (("ensemble_sha256", ens_p),
                         ("stress_ensemble_sha256", ens_s)):
            if key in cfg:
                got = hashlib.sha256(
                    np.ascontiguousarray(arr).tobytes()).hexdigest()
                if got != cfg[key]:
                    return ("stress audit %s: the rebuilt %s does not match "
                            "the hashed config" % (nm, key))
    except KeyError as exc:
        return "stress audit %s: the config omits %s" % (nm, exc)
    except Exception as exc:
        return "stress audit %s: cannot rebuild the ensembles (%s)" % (nm,
                                                                       exc)
    prod_rows, stress_rows, ratios = [], [], []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            return "stress audit %s row %d is not an object" % (nm, i)
        for f in ("pair_id", "latent_id", "loss", "production", "stress",
                  "ratio", "production_row", "stress_row"):
            if f not in row:
                return "stress audit %s row %d omits %s" % (nm, i, f)
        if row["pair_id"] != i:
            return "stress audit %s row %d is misnumbered" % (nm, i)
        for f in ("latent_id", "production_row", "stress_row"):
            v = row[f]
            if not (isinstance(v, str) and len(v) == 64
                    and all(c in "0123456789abcdef" for c in v)):
                return "stress audit %s row %d has no sha256 %s" % (nm, i, f)
        for f, want in (("latent_id", want_ids),
                        ("production_row", want_prod),
                        ("stress_row", want_stress)):
            if want is not None and row[f] != want[i]:
                return ("stress audit %s row %d %s is not the one the hashed "
                        "configuration produces" % (nm, i, f))
        for f in ("loss", "production", "stress", "ratio"):
            v = row[f]
            if not _is_num(v) or not np.isfinite(v) or v <= 0.0:
                return "stress audit %s row %d has a bad %s" % (nm, i, f)
        if grid and abs(row["loss"] - grid[i % len(grid)]) > 1e-12:
            return ("stress audit %s row %d loss %.6g is not the hashed "
                    "grid's stratum %d (%.6g)"
                    % (nm, i, row["loss"], i % len(grid), grid[i % len(grid)]))
        if abs(row["ratio"] - row["stress"] / row["production"]) > 1e-9:
            return "stress audit %s row %d ratio is inconsistent" % (nm, i)
        prod_rows.append(row["production_row"])
        stress_rows.append(row["stress_row"])
        ratios.append(float(row["ratio"]))
    for lbl, seq in (("latent", [r["latent_id"] for r in rows]),
                     ("production-row", prod_rows),
                     ("stress-row", stress_rows)):
        if len(set(seq)) != len(seq):
            return "stress audit %s repeats a %s hash" % (nm, lbl)
    if set(prod_rows) & set(stress_rows):
        return ("stress audit %s uses one T-row hash on both sides; the two "
                "ensembles differ by construction" % nm)
    # RECOMPUTE every aggregate from the rows
    arr = np.array(ratios)
    checks = (("p10", float(np.percentile(arr, 10))),
              ("p50", float(np.percentile(arr, 50))),
              ("p90", float(np.percentile(arr, 90))),
              ("worst", float(arr.min())), ("best", float(arr.max())),
              ("n_degraded", int((arr < 1.0).sum())))
    for f, want in checks:
        got = p[f]
        if not _is_num(got) or not np.isfinite(got):
            return "stress audit %s %s is not a finite number" % (nm, f)
        if abs(float(got) - want) > 1e-9 * max(1.0, abs(want)):
            return "stress audit %s %s=%r does not match the rows (%.6g)" % (
                nm, f, got, want)
    wu = p["worst_unpaired"]
    want_wu = (min(r["stress"] for r in rows)
               / min(r["production"] for r in rows))
    if not _is_num(wu) or abs(float(wu) - want_wu) > 1e-9 * max(1.0, want_wu):
        return "stress audit %s worst_unpaired does not match the rows" % nm
    if "by_loss" not in p:
        return "stress audit %s omits by_loss" % nm
    if True:
        strata = {}
        for row in rows:
            strata.setdefault("%.4g" % row["loss"], []).append(row["ratio"])
        if set(p["by_loss"]) != set(strata):
            return "stress audit %s by_loss strata do not match the rows" % nm
        for k, v in p["by_loss"].items():
            vals = strata[k]
            if (v.get("n") != len(vals)
                    or abs(v.get("median", 0) - float(np.median(vals))) > 1e-9
                    or abs(v.get("worst", 0) - min(vals)) > 1e-9
                    or abs(v.get("best", 0) - max(vals)) > 1e-9):
                return "stress audit %s by_loss[%s] contradicts the rows" % (
                    nm, k)
    return None


def _archive_body_ok(body, cfg):
    """Recompute the archive invariants the config merely LABELS.

    `archive_sha256`, `archive_n` and `archive_lineages` were written by the
    run and never recomputed, so a deliberately wrong fingerprint label sat
    next to a fabricated body without contradiction.
    """
    claimed = [k for k in ("archive_sha256", "archive_n", "archive_lineages")
               if k in cfg]
    if body is None:
        # An absent body used to satisfy the check whenever `archive_n` was
        # 0, so a run could claim `archive_sha256="definitely-wrong"` and an
        # arbitrary lineage set with nothing to contradict them -- and the
        # claimed empty epoch could not be replayed.
        if not claimed:
            return None                        # nothing claimed, nothing owed
        return "the config makes archive claims (%s) with no body to check "\
               "them against" % ", ".join(claimed)
    if not isinstance(body, dict):
        return "archive body is not an object"
    for k, v in body.items():
        if not isinstance(v, dict) or "design" not in v:
            return "archive entry %s is malformed" % k[:8]
        if not _design_ok(v["design"]):
            return "archive entry %s has an invalid design" % k[:8]
        try:
            if design_key(dz.Design.from_dict(v["design"])) != k:
                return "archive entry %s is filed under the wrong key" % k[:8]
        except Exception as exc:
            return "archive entry %s is undecodable: %s" % (k[:8], exc)
        if v.get("proposal_lineage") not in (LINEAGE_CONDITIONED,
                                             LINEAGE_INDEPENDENT):
            return "archive entry %s has no valid proposal lineage" % k[:8]
    fp = _canon_hash({k: dict(design=canonical_design(v["design"]),
                              proposal_lineage=v["proposal_lineage"])
                      for k, v in body.items()})
    if "archive_sha256" in cfg and cfg["archive_sha256"] != fp:
        return "archive fingerprint %s does not match the body (%s)" % (
            str(cfg["archive_sha256"])[:12], fp[:12])
    if "archive_n" in cfg and int(cfg["archive_n"]) != len(body):
        return "archive_n=%r contradicts a body of %d entries" % (
            cfg["archive_n"], len(body))
    want = sorted(set(v["proposal_lineage"] for v in body.values()))
    if "archive_lineages" in cfg and sorted(cfg["archive_lineages"]) != want:
        return "archive_lineages contradicts the body"
    return None


def _result_ok(res, man, cfg, shelf, sel, lin):
    """Closed schema for the scientific result -- exact fields, exact types,
    DERIVED status, and every claim compared against the hashed config."""
    if set(res) != set(RESULT_FIELDS):
        extra = sorted(set(res) - set(RESULT_FIELDS))
        missing = sorted(set(RESULT_FIELDS) - set(res))
        return ("result.json field set is not exact (extra %s, missing %s)"
                % (extra[:4], missing[:4]))
    for f, ty in RESULT_FIELDS.items():
        if not _typed(res[f], ty):
            return "result.json %s is %s, expected %s" % (
                f, type(res[f]).__name__, ty.__name__)
    if res["gate_a_schema_version"] != GATE_A_SCHEMA_VERSION:
        return "gate_a schema version %r is not %d" % (
            res["gate_a_schema_version"], GATE_A_SCHEMA_VERSION)
    if res["schema_version"] != RESULT_SCHEMA_VERSION:
        return "result schema version %r is not %d" % (res["schema_version"],
                                                       RESULT_SCHEMA_VERSION)
    # EVIDENCE STATUS IS DERIVED, not claimed.  A target-conditioned run could
    # simply write `gate-candidate`; it must now equal what the hashed lineage
    # and the actual gate execution imply.
    want_status = derive_evidence_status(lin, cfg, res["gate_a"],
                                         cfg.get("gate_a_candidates"))
    if res["evidence_status"] != want_status:
        _, _, why_g = gate_a_verdict(res["gate_a"], cfg,
                                     cfg.get("gate_a_candidates"))
        return ("evidence_status %r is not the derived %r (lineage %s; %s)"
                % (res["evidence_status"], want_status, lin, why_g))
    # EVERY duplicated configuration value must agree with the hashed config
    # The parity loop used to `continue` when the config lacked the key, so
    # `generator` -- reported but never stored -- was never compared, and the
    # test fixture (which did store it) masked the production-shape gap.
    for f, cf in (("seed", "seed"), ("n_samples", "samples"),
                  ("polish", "polish"), ("n_ensemble", "n_ensemble"),
                  ("lam_um", "lam_um"), ("sigma", "sigma"),
                  ("stress_loss", "stress_loss"),
                  ("ensemble_fro", "ensemble_fro"),
                  ("generator", "generator"), ("loss_grid", "loss_grid"),
                  ("q_eta", "q_eta"), ("constraints", "constraints")):
        if cf not in cfg:
            return "the hashed config omits %s, so the result cannot be " \
                   "checked against it" % cf
        av, bv = res[f], cfg[cf]
        if isinstance(av, dict) and isinstance(bv, dict):
            same = (set(av) == set(bv)
                    and all(float(av[k]) == float(bv[k]) for k in av))
        elif isinstance(av, list) or isinstance(bv, list):
            same = [float(x) for x in av] == [float(x) for x in bv]
        elif _is_num(av) and _is_num(bv):
            same = float(av) == float(bv)
        else:
            same = av == bv
        if not same:
            return "result %s=%r contradicts the hashed config %r" % (f, av,
                                                                      bv)
    if sel["origin"] != "selected:%s" % res["winner_source"]:
        return ("winner_source %r does not match the selected record origin "
                "%r" % (res["winner_source"], sel["origin"]))
    for i, row in enumerate(res["leaderboard"]):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return "leaderboard row %d is malformed" % i
        nm, dd, val = row
        if not isinstance(nm, str) or not _design_ok(dd):
            return "leaderboard row %d has no valid name/design" % i
        if not _is_num(val) or not np.isfinite(val):
            return "leaderboard row %d has a non-finite objective" % i
    wkey = design_key(dz.Design.from_dict(res["winner"]))
    named = [r for r in res["leaderboard"] if r[0] == res["winner_source"]]
    if not named:
        return "winner_source %r is not on the leaderboard" % res[
            "winner_source"]
    if design_key(dz.Design.from_dict(named[0][1])) != wkey:
        return ("the leaderboard row named by winner_source is a different "
                "geometry")
    if max(r[2] for r in res["leaderboard"]) != named[0][2]:
        return ("winner_source does not hold the best objective on its own "
                "leaderboard")
    for f in ("comparison", "audits", "ensemble_diversity", "constraints",
              "nuisance_classes", "leaderboard"):
        if not res[f]:
            return "result.json %s is empty" % f
    # the nuisance families are part of the claim, not a free label
    if list(res["nuisance_classes"]) != list(nz.DEFAULT_CLASSES):
        return ("result.json nuisance_classes %r is not the declared set"
                % (res["nuisance_classes"],))
    if not cfg.get("skip_gate_a", False) and not res["gate_a"]:
        return "gate_a was not skipped but the result reports none"
    if not res["stress_audit"]:
        return "result.json stress_audit is empty"
    for nm, st in res["stress_audit"].items():
        if not isinstance(st, dict):
            return "stress audit %s is not an object" % nm
        why = _paired_block_ok(st.get("paired") or {}, cfg, nm)
        if why:
            return why
    # NOT the leaderboard: it has its own exact row schema above, where a
    # name string is required rather than forbidden.
    for f in ("comparison", "audits", "stress_audit", "gate_a",
              "ensemble_diversity"):
        why = _metrics_ok(res[f], f)
        if why:
            return "result.json has a %s" % why
    return None


def verify_completed_run(run_dir, expect_run_id=None, _descend=True,
                        receipt_mode="require"):
    """Return (manifest, records) for a run that is provably complete, else
    (None, reason).

    A `complete` file is not evidence, and neither is a manifest that merely
    *declares* hashes.  Two earlier versions of this function were bypassable:

      * the first checked only that `complete` and `candidates.json` existed
        and then trusted the record's self-declared lineage;
      * the second iterated `manifest["artifacts"]`, so an EMPTY artifact map
        passed vacuously -- a directory with no `result.json`, no snapshot or
        config body, and arbitrary hash strings was admitted.

    Everything below must now agree, and every hash is RECOMPUTED:

      * the directory name (or `expect_run_id`, for a hidden staging path)
        is the run id, and the marker contains exactly that id;
      * the manifest carries the full `snapshot` and `config` BODIES, and
        their canonical hashes reproduce the declared `*_sha256` values;
      * the run id is rederived from those two hashes and must match;
      * the artifact set is EXACTLY `REQUIRED_ARTIFACTS`, each present and
        hash-matching;
      * `result.json` parses and names this run;
      * lineage is DERIVED from the hashed config and the manifest label must
        agree with it;
      * every record carries the manifest's run/config/snapshot identity.

    Any malformed shape -- a list where a mapping belongs, a null record,
    unparseable JSON -- is a REJECTION with a diagnostic, never an exception:
    one bad directory must not be able to take down all selection.
    """
    rid = expect_run_id or os.path.basename(os.path.normpath(run_dir))
    try:
        return _verify_completed_run(run_dir, rid, _descend,
                                    receipt_mode)
    except Exception as exc:                       # malformed shapes, not bugs
        return None, "malformed run (%s: %s)" % (type(exc).__name__, exc)


def _verify_completed_run(run_dir, rid, _descend=True,
                          receipt_mode="require"):
    mk = os.path.join(run_dir, "complete")
    mf = os.path.join(run_dir, "manifest.json")
    for nm in ("complete", "manifest.json") + REQUIRED_ARTIFACTS:
        if not os.path.exists(os.path.join(run_dir, nm)):
            return None, "missing %s" % nm
    with open(mk) as fh:
        marker = fh.read().split()
    if not marker or marker[0] != rid:
        return None, "marker does not name this run"
    with open(mf) as fh:
        man = json.load(fh)
    if not isinstance(man, dict):
        return None, "manifest is not an object"
    if man.get("run_id") != rid:
        return None, "manifest run_id does not match the run"

    # -- the manifest must be SELF-DESCRIBING: bodies, not just labels
    snap, cfg = man.get("snapshot"), man.get("config")
    if not isinstance(snap, dict) or not isinstance(cfg, dict):
        return None, "manifest omits the snapshot or config body"
    if _canon_hash(snap) != man.get("snapshot_sha256"):
        return None, "snapshot body does not reproduce its declared hash"
    if _canon_hash(cfg) != man.get("config_sha256"):
        return None, "config body does not reproduce its declared hash"
    if derive_run_id(snap, cfg) != rid:
        return None, "run id is not derivable from the snapshot and config"

    # -- the artifact set is closed, not whatever the manifest chose to list
    arts = man.get("artifacts")
    if not isinstance(arts, dict):
        return None, "artifacts is not an object"
    if set(arts) != set(REQUIRED_ARTIFACTS):
        return None, ("artifact set is %s, not the required %s"
                      % (sorted(arts), sorted(REQUIRED_ARTIFACTS)))
    for nm, h in arts.items():
        p = os.path.join(run_dir, nm)
        if not os.path.exists(p) or _sha_file(p) != h:
            return None, "artifact %s does not match its manifest hash" % nm

    # -- THE OUTPUT ROOT: recomputed from bytes, and matched against both
    # the marker and the append-only journal
    root = output_root(run_dir, rid)
    if root is None:
        return None, "cannot compute the output root"
    if len(marker) < 2:
        return None, "marker carries no output root"
    if marker[1] != root:
        return None, ("published outputs do not match the completion marker "
                      "(%s vs %s); the artifacts changed after completion"
                      % (root[:12], marker[1][:12]))
    # A RECEIPT IS MANDATORY for a published run.  Treating its absence as
    # "nothing to contradict" made the whole scheme fail open: deleting the
    # journal and re-signing the marker let a rewritten run verify under the
    # same input-derived id.  Hidden staging has no receipt yet, so it -- and
    # only it -- may skip this.
    # THREE receipt modes, because two were not enough.
    #   "require"       -- a published run must carry a matching receipt;
    #   "allow_missing" -- hidden staging, which has none yet.  It means
    #                      ABSENCE ONLY: a receipt that IS present must still
    #                      match, or a stage could self-verify against a
    #                      receipt naming another run and then be published;
    #   "skip"          -- the receipt is not consulted at all.  This exists
    #                      for `marker_owner`, which must decide who OWNS a
    #                      root precisely when that root's receipt is
    #                      unreadable.  Using "allow_missing" there made the
    #                      check return False in the one state it was added
    #                      for, so a FOREIGN run id could overwrite a valid
    #                      published run's receipt.
    if receipt_mode not in ("require", "allow_missing", "skip"):
        return None, "unknown receipt mode %r" % receipt_mode
    if receipt_mode != "skip":
        recorded = read_receipt(os.path.dirname(os.path.normpath(run_dir)),
                                root)
        if recorded is None:
            if receipt_mode == "require":
                return None, ("no receipt for output root %s; a published run "
                              "must carry one" % root[:12])
        elif recorded == "unreadable":
            return None, "the receipt for this output root is unreadable"
        elif recorded != rid:
            return None, ("the receipt for this output root names run %s, "
                          "not %s" % (str(recorded)[:12], rid[:12]))

    # -- the result must exist and name this run
    with open(os.path.join(run_dir, "result.json")) as fh:
        res = json.load(fh)
    if not isinstance(res, dict) or res.get("run_id") != rid:
        return None, "result.json does not name this run"
    # ...and must agree with the manifest on WHAT WAS RUN.  Checking only the
    # id let a result carrying foreign snapshot/config bodies stay admissible
    # as long as its artifact hash was refreshed.
    for f in ("snapshot_sha256", "config_sha256"):
        if res.get(f) != man.get(f):
            return None, "result.json %s disagrees with the manifest" % f
    # ...and equality of two hash LABELS is not parity.  A three-key result
    # carrying foreign `snapshot`/`config` bodies, a false conditioning claim
    # and a nonsensical winner verified as long as those two labels matched
    # and the artifact byte hash was refreshed.  The stored bodies must be
    # the manifest's bodies, and the scientific claims must be present and
    # consistent.
    if res.get("snapshot") != snap or res.get("config") != cfg:
        return None, "result.json carries a foreign snapshot or config body"
    if not _design_ok(res.get("winner")):
        return None, "result.json winner is not a valid design"

    # -- lineage is derived from the HASHED config, not from a free label
    lin = lineage_of_config(cfg)
    if lin is None:
        return None, "config declares no target_conditioned flag"
    if man.get("lineage") != lin:
        return None, ("manifest lineage %r contradicts the hashed config (%s)"
                      % (man.get("lineage"), lin))
    if bool(res.get("target_conditioned_prior")) != (lin == LINEAGE_CONDITIONED):
        return None, "result.json contradicts the hashed config on lineage"

    with open(os.path.join(run_dir, "candidates.json")) as fh:
        book = json.load(fh)
    if not isinstance(book, dict) or set(book) != {lin}:
        return None, "candidates shelf disagrees with the derived lineage"
    shelf = book[lin]
    if not isinstance(shelf, dict) or not shelf:
        return None, "candidate shelf is empty or not an object"
    for nm, rec in shelf.items():
        if not isinstance(rec, dict):
            return None, "record %s is not an object" % nm
        if not _record_ok(rec, lin):
            return None, "record %s fails schema/provenance" % nm
        if (rec.get("run_id") != rid
                or rec.get("config_sha256") != man.get("config_sha256")
                or rec.get("snapshot_sha256") != man.get("snapshot_sha256")):
            return None, "record %s carries a foreign identity" % nm
    # -- EVERY selected record must resolve to an existing parent.  A proof
    # that is merely well-formed is not provenance.
    sel = [r for r in shelf.values() if r["origin"].startswith("selected")]
    if len(sel) != 1:
        return None, "shelf holds %d selected records, expected 1" % len(sel)
    # LOCAL INVARIANTS ALWAYS.  Only the recursive descent is optional.
    why = _archive_body_ok(man.get("archive_body"), cfg)
    if why is not None:
        return None, why
    why = resolve_selected_parent(
        sel[0], shelf, man.get("archive_body"), rid,
        runs_dir=os.path.dirname(os.path.normpath(run_dir)),
        descend=_descend)
    if why is not None:
        return None, "unresolvable provenance: %s" % why
    why = _result_ok(res, man, cfg, shelf, sel[0], lin)
    if why is not None:
        return None, why
    try:
        same = (design_key(dz.Design.from_dict(res["winner"]))
                == design_key(dz.Design.from_dict(sel[0]["design"])))
    except Exception as exc:
        return None, "winner is undecodable: %s" % exc
    if not same:
        return None, "result.json winner is not the selected candidate"
    return man, shelf


def iter_completed(runs_dir):
    """(run_id, manifest, records) for every VERIFIED complete run."""
    if not os.path.isdir(runs_dir):
        return
    for rid in sorted(os.listdir(runs_dir)):
        rd = os.path.join(runs_dir, rid)
        if not os.path.isdir(rd) or rid.startswith("."):
            continue          # staging directories are hidden and skipped
        man, recs = verify_completed_run(rd)
        if man is not None:
            yield rid, man, recs
        elif os.environ.get("FASTFULL_VERBOSE_ADMISSION"):
            print("  skipped %s: %s" % (rid[:12], recs), flush=True)


def load_registry(lineage=None, include_fallbacks=True, runs_dir=None):
    """Eligible candidates, DERIVED by scanning COMPLETED run directories.

    There is no shared mutable index to lose updates or race on: a run's
    candidates become visible only when its own directory carries a
    `complete` marker, and the marker is written last.  A crashed run leaves
    an incomplete directory whose candidates are never admitted.
    """
    want = lineage or _lineage()
    runs = runs_dir or RUNS_DIR
    out = {}
    if include_fallbacks and want == LINEAGE_CONDITIONED:
        for nm, d in _TRANSCRIBED.items():
            out[nm] = dz.Design.from_dict(d)
    eligible = ([LINEAGE_INDEPENDENT] if want == LINEAGE_INDEPENDENT
                else [LINEAGE_INDEPENDENT, LINEAGE_CONDITIONED])
    for rid, man, recs in iter_completed(runs):
        if man["lineage"] not in eligible:
            continue          # lineage decided by the MANIFEST, not the record
        for nm, rec in recs.items():
            out[nm] = dz.Design.from_dict(rec["design"])
    return out


def write_run_candidates(run_dir, records, lineage=None):
    """Write this run's candidates INSIDE its own immutable directory."""
    lin = lineage or _lineage()
    for nm, rec in records.items():
        if not _record_ok(rec, lin):
            raise ValueError("record %r fails the schema/lineage check" % nm)
    path = os.path.join(run_dir, "candidates.json")
    if os.path.exists(path):
        raise ValueError("run directory already has candidates: %s" % path)
    with open(path, "w") as fh:
        json.dump({lin: records}, fh, indent=1)
    return path


class _ExclusiveLock(object):
    """Interprocess lock built on O_CREAT|O_EXCL, which is atomic on both
    Windows and POSIX.  Held across the whole scan-through-replace."""

    def __init__(self, path, timeout=30.0, poll=0.02):
        self.path, self.timeout, self.poll, self.fd = path, timeout, poll, None

    def __enter__(self):
        import errno
        import time as _t
        deadline = _t.monotonic() + self.timeout
        while True:
            try:
                self.fd = os.open(self.path,
                                  os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return self
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                if _t.monotonic() > deadline:
                    # a stale lock must not wedge the pipeline forever
                    try:
                        os.unlink(self.path)
                    except OSError:
                        pass
                    deadline = _t.monotonic() + self.timeout
                _t.sleep(self.poll)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            try:
                os.unlink(self.path)
            except OSError:
                pass
        return False


def rebuild_index(runs_dir=None, out_path=None, _attempts=4):
    """Derived, NON-SELECTING metadata, rebuilt through the same verified
    iterator the selector uses.

    Concurrency.  Three versions were needed.  Unique temp names stopped two
    threads from destroying each other's file but not from publishing a STALE
    book.  A pre-replace rescan narrowed the window without closing it: a
    stale rebuild could pass its rescan, write its temp file, read the current
    index, and only then be overtaken -- its `os.replace` still overwrote the
    newer generation.  Read-check-replace is not atomic, so the whole
    scan-through-replace is now held under an exclusive lock, and publication
    is VERIFIED afterwards against `iter_completed` rather than assumed.

    Selection never depended on this file -- it scans run directories -- but
    the index could silently omit completed candidates and mislead a reader.
    """
    runs = runs_dir or RUNS_DIR
    path = out_path or REGISTRY
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last = None
    for _ in range(max(1, int(_attempts))):
        with _ExclusiveLock(path + ".lock"):
            book = {"_note": "DERIVED, NON-AUTHORITATIVE. Rebuilt from "
                             "verified complete runs; selection reads run "
                             "directories, not this file."}
            seen = []
            for rid, man, recs in iter_completed(runs):
                book.setdefault(man["lineage"], {}).update(recs)
                seen.append(rid)
            book["_generation"] = sorted(seen)
            tmp = path + ".%d.%s.tmp" % (
                os.getpid(), binascii.hexlify(os.urandom(6)).decode())
            with open(tmp, "w") as fh:
                json.dump(book, fh, indent=1)
            os.replace(tmp, path)
            last = book
        # VERIFY the publication, outside the lock: if a run completed while
        # we held it, the book is already behind and we simply go again.
        if sorted(r for r, _, _ in iter_completed(runs)) == last["_generation"]:
            return last
    raise RuntimeError("could not publish a current candidate index after %d "
                       "attempts; runs are completing faster than the index "
                       "can be rebuilt" % _attempts)


def append_registry(records, lineage=None):
    raise RuntimeError("append_registry is removed: candidates live in "
                       "per-run directories and the index is rebuilt from "
                       "completed runs (see write_run_candidates / "
                       "rebuild_index)")


def _source_hash():
    """SHA-256 over the modules that determine a run's numbers.

    The artifact previously recorded a loss grid that the source no longer
    used, with nothing to detect the drift.  Binding a source hash makes an
    artifact self-identifying even when the log has been overwritten.
    """
    import hashlib
    h = hashlib.sha256()
    for mod in ("nuisance.py", "symmetry.py", "synthetic.py", "design.py",
                "jacobian.py", "transforms.py", "lattice.py", "ewald.py",
                "opt_marginalized.py"):
        p = os.path.join(HERE, mod)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def _pieces(design, k, modes, cons):
    lat = design.lattice()
    o = lt.enumerate_orders(lat, k, f_bloch=(design.f1, design.f2),
                            kz_min_frac=cons.kz_min_frac,
                            wood_margin=cons.wood_margin)
    if not o.n_retained:
        return None
    ch = lt.ChannelSet(o)
    return (ch, xf.build_A(k, ch, modes), xf.build_W(k, ch, modes),
            ew.lattice_sum_C(lat, k, modes, lat.bloch(design.f1, design.f2)))


def audit(design, k, modes, B, T, sigma, cons, label):
    p = _pieces(design, k, modes, cons)
    if p is None:
        print("  %s: infeasible" % label)
        return None
    ch, A, W, C = p
    info = nz.marginalized_information(W, A, B, T, C, sigma, ch)
    kzs = np.unique(np.round(ch.kz / ch.k, 6))
    print("  %s  %d channels, %d orders, kz/k spread %.3f (%d distinct)"
          % (label, ch.n, ch.n // 4, kzs.max() - kzs.min(), len(kzs)))
    print(nz.format_audit(info, indent=4))
    info["n_channels"] = ch.n
    info["kz_spread"] = float(kzs.max() - kzs.min())
    info["kz_values"] = kzs.tolist()
    return info


def gate_a(design, k, modes, B, T, sigma, cons, seed=12345):
    return sy.run_candidate(design, k, modes, B, T, sigma, cons,
                            n_trials=2, seed=seed, n_multistart=1)


def eligible_incumbents(want_lineage):
    """Incumbents admissible for SELECTION under `want_lineage`.

    Source-code constants are not exempt from the evidence boundary.  Both
    declared incumbents are target-conditioned, so an independent run gets an
    empty dict here and must find its own starting point.  They are still
    audited and reported -- only their eligibility to be SELECTED is gated.
    """
    ok = eligible_proposals(want_lineage)
    return {nm: d for nm, d in INCUMBENT_DESIGNS.items()
            if INCUMBENT_LINEAGE.get(nm, LINEAGE_CONDITIONED) in ok}


def eligible_proposals(want_lineage):
    """Which PROPOSAL lineages may be selected by a run using `want_lineage`.

    An independent run may select only independent proposals.  A
    target-conditioned development run may look at both, because it is
    already disqualified from closing a gate.
    """
    if want_lineage == LINEAGE_INDEPENDENT:
        return (LINEAGE_INDEPENDENT,)
    return (LINEAGE_INDEPENDENT, LINEAGE_CONDITIONED)


def freeze_archive(runs_dir, want_lineage=None, pin=None):
    """ONE verified, LINEAGE-FILTERED, DEDUPLICATED archive snapshot.

    The archive is a selector input: an archived cell may beat the fresh
    search and become the run's selected winner.  Three defects are closed
    here.

    * It was loaded AFTER the run id was computed, so one id could publish
      different winners depending on which other runs finished first.
    * It ignored lineage entirely.  With `TARGET_CONDITIONED_PRIOR=False` a
      conditioned candidate still entered the frozen selector, and because
      the selected record was stamped with the CURRENT lineage, a conditioned
      proposal could be republished as independent.  Hashing the archive made
      that contamination reproducible; it did not make it eligible.  Proposal
      lineage now filters admission and travels with the record.
    * It keyed on RECORD NAME, so the same geometry republished by a later
      run counted as a new candidate.  Every run then saw a bigger archive,
      derived a new id, reran the same seeded search, and published more
      duplicates -- an endless chain that never reached the completed-identity
      refusal while archive cost grew linearly.  The fingerprint is now over
      the set of unique GEOMETRIES (`design_key`), so a run that discovers no
      new geometry has the same identity as its predecessor and is refused.

    `pin`, when given, asserts the expected archive fingerprint and raises on
    any mismatch, so a campaign can be reproduced against a fixed epoch.

    Returns (designs, provenance, fingerprint), keyed by canonical geometry.
    """
    want = want_lineage or _lineage()
    ok_props = eligible_proposals(want)
    designs, prov = {}, {}
    for rid, man, recs in iter_completed(runs_dir):
        for nm, rec in recs.items():
            prop = rec.get("proposal_lineage", man["lineage"])
            if prop not in ok_props:
                continue                       # NOT eligible: never selected
            # canonicalize the OBJECT, not only its hash: archives whose
            # entries differ only as alpha=0 vs 360 hashed identically while
            # the selector still evaluated and stored the raw floats, so one
            # run identity could evaluate slightly different representations.
            d = dz.Design.from_dict(canonical_design(
                dz.Design.from_dict(rec["design"])))
            key = design_key(d)
            if key in prov:
                # same geometry, already held: keep the FIRST proposal and
                # record that it was seen again, so republishing cannot move
                # the fingerprint
                prov[key]["seen_in"] = sorted(set(prov[key]["seen_in"]
                                                  + [rec["run_id"]]))
                continue
            designs[key] = d
            prov[key] = dict(name=nm, origin=rec["origin"],
                             proposal_lineage=prop, design=rec["design"],
                             first_run=rec["run_id"], seen_in=[rec["run_id"]],
                             output_root=output_root(
                                 os.path.join(runs_dir, rid), rid),
                             record_digest=record_digest(rec))
            if any(o["name"] == nm for kk, o in prov.items() if kk != key):
                prov[key]["name"] = "%s@%s" % (nm, rec["run_id"][:8])
    if LINEAGE_CONDITIONED in ok_props:
        for nm, d0 in _TRANSCRIBED.items():           # labelled, unselectable
            d = dz.Design.from_dict(canonical_design(dz.Design.from_dict(d0)))
            key = design_key(d)
            if key not in prov:
                designs[key] = d
                prov[key] = dict(name=nm, origin="transcribed",
                                 proposal_lineage=LINEAGE_CONDITIONED,
                                 design=d0, first_run=None, seen_in=[])
    # the fingerprint covers the GEOMETRY SET and each entry's proposal
    # lineage -- deliberately NOT the run ids that republished it
    # fingerprint the CANONICAL body, not the first record's raw dictionary
    for k, v in prov.items():
        v["canonical_design"] = canonical_design(v["design"])
    fp = _canon_hash({k: dict(design=v["canonical_design"],
                              proposal_lineage=v["proposal_lineage"])
                      for k, v in prov.items()})
    if pin is not None and pin != fp:
        raise RuntimeError("archive pin %s does not match the frozen archive "
                           "%s (%d candidates); refusing to run against an "
                           "archive the caller did not expect"
                           % (pin[:12], fp[:12], len(prov)))
    # The selector map used to be `{record_name: design}`, so two verified
    # runs carrying DIFFERENT geometries under the same record name silently
    # overwrote one another: `archive_sha256`/`archive_n` then certified a
    # candidate the leaderboard never evaluated.  Labels are now made unique
    # by appending the geometry key, guaranteeing a bijection with `prov`.
    named = {}
    for k, v in prov.items():
        label = v["name"]
        if label in named:
            label = "%s#%s" % (v["name"], k[:8])
        named[label] = designs[k]
        v["label"] = label
    if len(named) != len(prov):
        raise RuntimeError("selector map (%d) does not match the hashed "
                           "archive (%d)" % (len(named), len(prov)))
    return named, prov, fp


def build_paired_ensembles(B, seed, n_pairs, grid=None,
                           stress_loss=None, target_fro=None):
    """The production ensemble pair.  THE one place this is constructed.

    Latent coefficient vectors are drawn ONCE; production maps draw i at grid
    stratum `i % len(grid)` and stress maps THE SAME draw at `stress_loss`.
    Pair i therefore differs only in the Hermitian loss multiplier, so a
    per-pair change is attributable to absorption.

    This is a module-level helper because the synthetic gate previously
    reimplemented it -- grouping sequential RNG draws by loss, where
    production cycles loss over pre-drawn latents.  The two constructions
    produced different ensembles (hashes `87246d...` vs `a00a63...`, max entry
    difference 6.094e-4), so the test was not measuring the production
    configuration it claimed to measure.  Callers get the same object the
    optimizer uses or they get nothing.

    Returns (production, stress, pair_loss).  `pair_loss[i]` is the
    production loss of pair i and row i of both arrays is pair i.  Call
    `paired_ensemble_ids(B, seed, n_pairs)` for the shared latent hashes.
    """
    grid = LOSS_GRID if grid is None else grid
    stress_loss = STRESS_LOSS if stress_loss is None else stress_loss
    target_fro = ENSEMBLE_FRO if target_fro is None else target_fro
    n_pairs = int(n_pairs)
    c_draws = sym.latent_draws(B, np.random.default_rng(seed), n_pairs)
    pair_loss = [grid[i % len(grid)] for i in range(n_pairs)]
    draw = lambda i, L: sym.random_passive_d4h(
        B, None, n_draw=1, target_fro=target_fro, loss_factor=L,
        c_draws=[c_draws[i]])[0]
    ens = np.concatenate([draw(i, pair_loss[i]) for i in range(n_pairs)])
    ens_stress = np.concatenate([draw(i, stress_loss) for i in range(n_pairs)])
    return ens, ens_stress, pair_loss


def paired_ensemble_ids(B, seed, n_pairs):
    """Latent hash of each pair -- the identity production and stress SHARE.

    Regenerated from the same seed rather than threaded through, so a caller
    cannot accidentally pass ids belonging to a different construction.
    """
    c = sym.latent_draws(B, np.random.default_rng(seed), int(n_pairs))
    return [hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            for x in c]


def ensemble_row_ids(ens):
    """Hash of each ensemble row, for binding a report to the array."""
    return [hashlib.sha256(np.ascontiguousarray(T).tobytes()).hexdigest()
            for T in np.atleast_3d(np.asarray(ens))]


def paired_stress_stats(prod_rows, stress_rows, pair_loss=None,
                        pair_ids=None, prod_ids=None, stress_ids=None,
                        expect_prod=None, expect_stress=None, unbound=False,
                        key="sigma_marg"):
    """PER-PAIR deltas, not a ratio of two independent worst cases.

    `worst(stress) / worst(production)` compares two different draws whenever
    the two bottlenecks differ, which is the normal case: a direct probe found
    per-pair ratios spanning 0.959..1.021 while the single scalar read 0.97,
    hiding both a +2.1% improvement and a -4.1% degradation.

    The contract is STRICT, in three stages, because each weaker version was
    demonstrably insufficient:

      * an early version took `min(len(prod), len(stress))`, so two production
        rows against one stress row were silently reported as one valid pair;
      * equal LENGTHS still could not prove that row i is the same latent
        draw on both sides -- reordered stress rows were accepted and
        mislabelled.  Two distinct identities are needed and both are checked.
        `prod_ids`/`stress_ids` are the per-draw T hashes REPORTED by
        `design.evaluate`; `expect_prod`/`expect_stress` are the hashes of the
        arrays actually passed to it.  Requiring these to agree elementwise
        binds report row i to array row i on each side.  `pair_ids` are the
        LATENT draw hashes, which are what production row i and stress row i
        genuinely share -- their T matrices differ by construction, so
        comparing T hashes ACROSS the two sides could never establish
        pairing;
      * value validation was asymmetric: production denominators had to be
        positive while zero or negative stress values were accepted, and a
        NaN loss label passed straight into stratification.

    Absolute values and the production loss of each pair are persisted, and
    ratios are reported by baseline-loss stratum as well as pooled, because
    the pairs sit at different 20x/10x/5x multipliers.
    """
    if len(prod_rows) != len(stress_rows) or not prod_rows:
        raise ValueError("paired statistics need equal, nonzero row counts; "
                         "got %d production and %d stress rows"
                         % (len(prod_rows), len(stress_rows)))
    n = len(prod_rows)
    supplied = [x is not None for x in (pair_ids, prod_ids, stress_ids,
                                        expect_prod, expect_stress)]
    if any(supplied) and not all(supplied):
        # A BOUND CLAIM IS ALL-OR-NOTHING.  Each list used to be optional
        # independently, and the reported flag was computed from the
        # production side alone -- so supplying production ids and no stress
        # ids returned `rows_bound_to_ensemble=True` with nothing binding the
        # stress rows at all.  An unbound diagnostic is still available: pass
        # `unbound=True` and none of the id lists.
        raise ValueError("a bound paired claim needs pair_ids, prod_ids, "
                         "stress_ids, expect_prod and expect_stress together; "
                         "got %s" % [bool(x) for x in supplied])
    bound = all(supplied)
    if not bound and not unbound:
        raise ValueError("refusing to report an unlabelled unpaired "
                         "statistic; supply the id lists or pass unbound=True")
    if bound:
        for side, got, want in (("production", prod_ids, expect_prod),
                                ("stress", stress_ids, expect_stress)):
            if len(got) != n or len(want) != n:
                raise ValueError("%s draw-id lists must have one entry per "
                                 "pair" % side)
            bad = [i for i in range(n) if got[i] != want[i]]
            if bad:
                raise ValueError("%s report rows %s are not the draws that "
                                 "were passed; the rows are not bound to the "
                                 "ensemble" % (side, bad[:5]))
        if len(pair_ids) != n:
            raise ValueError("pair_ids must have one entry per pair")
        if len(set(pair_ids)) != n:
            raise ValueError("latent ids must be unique; %d distinct for %d "
                             "pairs" % (len(set(pair_ids)), n))
        shaped = all(isinstance(x, str) and len(x) == 64
                     and all(c in "0123456789abcdef" for c in x)
                     for x in pair_ids)
        if not shaped:
            raise ValueError("latent ids must be sha256 hex digests")
    p = np.array([float(r[key]) for r in prod_rows])
    q = np.array([float(r[key]) for r in stress_rows])
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
        raise ValueError("paired statistics need finite values on both sides")
    if not (np.all(p > 0.0) and np.all(q > 0.0)):
        raise ValueError("paired statistics need positive values on BOTH "
                         "sides; got production min %.6g, stress min %.6g"
                         % (p.min(), q.min()))
    r = q / p
    pc = lambda x: float(np.percentile(r, x))
    if pair_loss is not None:
        loss = [float(x) for x in pair_loss]
        if len(loss) != len(r):
            raise ValueError("pair_loss must have one entry per pair")
        if not all(np.isfinite(L) and L > 0.0 for L in loss):
            raise ValueError("baseline loss labels must be finite and "
                             "positive; got %s" % loss)
    else:
        loss = [None] * len(r)
    ids = list(pair_ids) if bound else [None] * len(r)
    out = dict(n_pairs=len(r), p10=pc(10), p50=pc(50), p90=pc(90),
               worst=float(r.min()), best=float(r.max()),
               n_degraded=int((r < 1.0).sum()),
               rows_bound_to_ensemble=bound,
               paired_by_latent_id=bound,
               worst_unpaired=float(q.min() / p.min()),
               pairs=[dict(pair_id=i, latent_id=ids[i], loss=loss[i],
                           production=float(p[i]), stress=float(q[i]),
                           ratio=float(r[i]),
                           production_row=(prod_ids[i] if bound else None),
                           stress_row=(stress_ids[i] if bound else None))
                      for i in range(len(r))])
    if pair_loss is not None:
        strata = {}
        for i, L in enumerate(loss):
            strata.setdefault("%.4g" % L, []).append(float(r[i]))
        out["by_loss"] = {k: dict(n=len(v), median=float(np.median(v)),
                                  worst=float(min(v)), best=float(max(v)))
                          for k, v in sorted(strata.items())}
    return out


# Test hook fired immediately after the archive is frozen and before any
# optimization.  A concurrency gate must be able to guarantee that two
# workers freeze the SAME archive; without a synchronization point the
# scheduler may let one publish before the other freezes, giving two
# legitimate identities and a flaky test rather than a real race.
_AFTER_FREEZE_HOOK = None


def run(lam_um=8.0, samples=300, polish=2, seed=20260807, n_ens=6,
        out_dir=OUT_DIR, skip_gate_a=False, archive_pin=None):
    # SNAPSHOT FIRST: hash the code and data that are about to be executed,
    # not whatever is on disk ten minutes later.
    snap = snapshot_inputs()
    assert_snapshot_matches_import(snap)
    snap_hash = _canon_hash(snap)
    # FREEZE THE ARCHIVE from the REQUESTED namespace, before anything is
    # optimized.  `run(out_dir=...)` used to call a bare `load_registry()`,
    # which reads the repository-global RUNS_DIR: a temporary or independent
    # campaign would silently select global candidates while writing
    # elsewhere.  Everything downstream uses this frozen copy only.
    runs_root = os.path.join(out_dir, "runs")
    archive, archive_prov, archive_hash = freeze_archive(
        runs_root, want_lineage=_lineage(), pin=archive_pin)
    if _AFTER_FREEZE_HOOK is not None:
        _AFTER_FREEZE_HOOK()
    modes = ModeBasis.standard(3)
    B, _ = sym.build_d4h_reciprocity_basis(modes)
    data = TMatrixData(ms.REF_TMAT)
    i = int(np.argmin(np.abs(data.wavelength_um - lam_um)))
    k, T_ref = data.k_at(i), data.T[i]
    sigma = dz.measured_sigma(float(data.wavelength_um[i]))
    cons = dz.Constraints(kz_min_frac=0.2, wood_margin=0.05, n_orders_min=2,
                          n_orders_max=12,
                          area_max_um2=ms._area_max(lam_um,
                                                    dict(n_orders_max=12)),
                          dressing_max=0.5, deembed_sigma_min=0.5,
                          signal_min_sigma=3.0)
    # Target independence: the ensemble norm is a DECLARED constant, not
    # ||T_ref||.  Reading the reference wheel to set it -- which the first
    # version did -- makes the search target-dependent in exactly the way
    # par. 7.3 forbids, even though the draws themselves are random.
    ens, ens_stress, pair_loss = build_paired_ensembles(B, seed, n_ens)
    latent_ids = paired_ensemble_ids(B, seed, n_ens)
    n_pairs = len(pair_loss)
    div = sym.ensemble_diversity(B, ens)
    print("  ensemble: %d paired draws over loss grid %s; identity cosine "
          "max %.3f, pairwise max %.3f, effective rank %.2f/40"
          % (len(ens), LOSS_GRID, div["identity_cosine_max"],
             div["pairwise_cosine_max"], div["effective_rank"]), flush=True)

    cfg = dict(lam_um=float(lam_um), seed=int(seed), samples=int(samples),
               polish=int(polish), n_ensemble=int(len(ens)),
               ensemble_fro=float(ENSEMBLE_FRO), loss_grid=list(LOSS_GRID),
               stress_loss=float(STRESS_LOSS), skip_gate_a=bool(skip_gate_a),
               sigma=float(sigma), q_eta=0.0, generator="cayley",
               # the candidate set the error screen MUST cover, declared in
               # the hashed config so a report cannot narrow itself to a
               # convenient subset after the fact
               gate_a_candidates=([] if skip_gate_a
                                  else ["small@8", "winner"]),
               target_conditioned=bool(TARGET_CONDITIONED_PRIOR),
               # the SAME shape the result reports, so parity is a real
               # comparison: a sorted item list against a dict could never be
               # compared, which is why `generator` and `constraints` went
               # unchecked for several rounds
               constraints=dict(kz_min_frac=cons.kz_min_frac,
                                wood_margin=cons.wood_margin,
                                n_orders_min=cons.n_orders_min,
                                n_orders_max=cons.n_orders_max,
                                area_max_um2=cons.area_max_um2,
                                dressing_max=cons.dressing_max,
                                deembed_sigma_min=cons.deembed_sigma_min,
                                signal_min_sigma=cons.signal_min_sigma),
               ensemble_sha256=hashlib.sha256(
                   np.ascontiguousarray(ens).tobytes()).hexdigest(),
               stress_ensemble_sha256=hashlib.sha256(
                   np.ascontiguousarray(ens_stress).tobytes()).hexdigest(),
               # IDENTITY CARRIES THE GEOMETRY EPOCH ONLY.  Hashing the full
               # provenance body reintroduced identity churn: adding the same
               # geometry under a lexicographically earlier run left the
               # geometry fingerprint unchanged but moved the retained
               # name/first_run, so the next invocation derived a new id and
               # repeated the same deterministic search -- partly reopening
               # the chain deduplication was meant to close.  The provenance
               # body is published in the MANIFEST instead, where it is
               # auditable and recomputed against this fingerprint.
               archive_sha256=archive_hash,
               archive_n=len(archive_prov),
               archive_lineages=sorted(set(
                   v["proposal_lineage"] for v in archive_prov.values())))
    cfg_hash = _canon_hash(cfg)
    # FULL manifest hash as identity: snapshot (code + data + env) and the
    # complete configuration including both ensembles' bytes.  Truncating to
    # eight hex characters, and omitting skip_gate_a and the stress draws,
    # could alias distinct runs.
    run_id = derive_run_id(snap, cfg)

    # REFUSE A COMPLETED IDENTITY BEFORE SPENDING THE SEARCH.  The duplicate
    # check used to run after the full optimization, so a repeat cost ten
    # minutes before being told the answer already existed.
    _run_dir = os.path.join(out_dir, "runs", run_id)
    if verify_completed_run(_run_dir)[0] is not None:
        raise RuntimeError("run %s is already complete at %s; refusing to "
                           "recompute or overwrite it"
                           % (run_id[:12], _run_dir))

    print("lambda %.3f um, sigma %.4e, %d nuisance classes"
          % (data.wavelength_um[i], sigma, len(nz.DEFAULT_CLASSES)),
          flush=True)
    print("\n=== incumbents (audited at the reference T, reporting only) ===",
          flush=True)
    audits = {nm: audit(d, k, modes, B, T_ref, sigma, cons, nm)
              for nm, d in INCUMBENTS.items()}

    print("\n=== searching on the MARGINALIZED objective ===", flush=True)
    box = ms._box_for(lam_um, dict(n_orders_min=2, n_orders_max=12))
    t0 = time.time()
    best, polished = dz.search(np.array([k]), modes, B=B,
                               track="marginalized", n_samples=samples,
                               n_polish=polish, constraints=cons, box=box,
                               seed=seed, verbose=True, bloch_eval=2,
                               T_ensemble=ens, nuisance_classes=True)
    dt = time.time() - t0
    if best is None:
        print("no feasible design")
        return None
    print("  search %.1f s -> %r" % (dt, best), flush=True)
    # `best` is about to be replaced by whatever wins the archive comparison,
    # so keep the fresh search output and its polished shortlist now -- an
    # earlier version archived the SELECTED point under the name "search@...",
    # losing the only new information the run produced.
    search_best = best
    polished_designs = [d for _, d, _ in polished]

    # re-evaluate the archive on the CURRENT ensemble and keep the best point
    # overall, so a changed ensemble cannot silently demote a known optimum
    print("\n=== archive re-evaluation on the current ensemble ===",
          flush=True)
    def _obj(d):
        return dz.score(dz.evaluate(d, np.array([k]), modes, B=B,
                                    constraints=cons, T_ensemble=ens,
                                    nuisance_classes=True), "marginalized")
    board = [("search", best, _obj(best))]
    for nm, d in (list(archive.items())
                  + list(eligible_incumbents(_lineage()).items())):
        board.append((nm, d, _obj(d)))
    board.sort(key=lambda t: -t[2])
    for nm, d, v in board:
        print("  %-20s objective %.5g%s"
              % (nm, v, "   [rounded: reported, not selectable]"
                 if nm.endswith("(rounded)") else ""), flush=True)
    # A hand-transcribed point must never be SELECTED: its coordinates are
    # known to be wrong (the 04:24 alpha was 63.4185055507, not 56.22), so
    # letting it win would propagate the transcription error again.
    board = [row for row in board if not row[0].endswith("(rounded)")]
    search_obj = next(v for nm, _, v in board if nm == "search")
    if board[0][0] != "search":
        print("  ARCHIVE WINS: %s scores %.5g against this run's search "
              "output %.5g (%.0f%% better); reporting the archived point"
              % (board[0][0], board[0][2], search_obj,
                 100 * (board[0][2] / search_obj - 1.0)), flush=True)
    best_overall_name, best, best_obj = board[0]

    print("\n=== winner, audited under the same nuisance classes ===",
          flush=True)
    audits["marginalized_winner"] = audit(best, k, modes, B, T_ref, sigma,
                                          cons, "winner")

    print("\n=== objective comparison (ensemble; %s prior) ==="
          % ("TARGET-CONDITIONED" if TARGET_CONDITIONED_PRIOR
             else "target-independent"),
          flush=True)
    rows = {}
    for nm, d in list(INCUMBENTS.items()) + [("winner", best)]:
        r = dz.evaluate(d, np.array([k]), modes, B=B, constraints=cons,
                        T_ensemble=ens, nuisance_classes=True)
        f = r["per_freq"][0] if r["per_freq"] else {}
        m = f.get("marginalized", {})
        rows[nm] = dict(design=d.to_dict(), ok=r["ok"],
                        area=r["area_um2"], n_channels=f.get("n_channels"),
                        sigma_free=m.get("worst_sigma_free"),
                        sigma_marg=m.get("worst_sigma_marg"),
                        sigma_marg_per_obs=m.get("worst_sigma_marg_per_obs"),
                        n_obs=m.get("n_obs"),
                        loss=m.get("worst_loss"),
                        sigma_ratio=m.get("worst_sigma_ratio"),
                        cost_min=r["cost"]["t_campaign_min"],
                        obj_marg=r.get("objective_marginalized"),
                        obj_wheel=r.get("objective_wheel"))
        print("  %-10s area %7.1f um^2  ch %3s  sigma_free %8.4g  "
              "sigma_marg %8.4g  loss %6.1fx  cost %6.0f min"
              % (nm, rows[nm]["area"], rows[nm]["n_channels"],
                 rows[nm]["sigma_free"] or 0, rows[nm]["sigma_marg"] or 0,
                 rows[nm]["loss"] or 0, rows[nm]["cost_min"]), flush=True)

    print("\n=== over-loss PAIRED stress audit (loss %.4g -> %.3f, "
          "reported, NOT used for selection) ==="
          % (float(np.mean(pair_loss)), STRESS_LOSS), flush=True)
    stress = {}
    for nm, d in list(INCUMBENTS.items()) + [("winner", best)]:
        ev = lambda E: dz.evaluate(d, np.array([k]), modes, B=B,
                                   constraints=cons, T_ensemble=E,
                                   nuisance_classes=True)
        rp, rs = ev(ens), ev(ens_stress)
        mp = (rp["per_freq"][0] if rp["per_freq"] else {}).get(
            "marginalized", {})
        msr = (rs["per_freq"][0] if rs["per_freq"] else {}).get(
            "marginalized", {})
        pd_p = [{"sigma_marg": x} for x in
                mp.get("per_draw_sigma_marg", [])]
        pd_s = [{"sigma_marg": x} for x in
                msr.get("per_draw_sigma_marg", [])]
        id_p = mp.get("per_draw_id")
        id_s = msr.get("per_draw_id")
        st = dict(sigma_marg=msr.get("worst_sigma_marg"),
                  sigma_free=msr.get("worst_sigma_free"),
                  objective=rs.get("objective_marginalized"))
        if pd_p and pd_s:
            st["paired"] = paired_stress_stats(
                pd_p, pd_s, pair_loss=pair_loss, pair_ids=latent_ids,
                prod_ids=id_p, stress_ids=id_s,
                expect_prod=ensemble_row_ids(ens),
                expect_stress=ensemble_row_ids(ens_stress))
            p = st["paired"]
            print("  %-10s PAIRED sigma_marg ratio  p10 %.4f  p50 %.4f  "
                  "p90 %.4f   worst %.4f  best %.4f   (%d/%d pairs degraded)"
                  % (nm, p["p10"], p["p50"], p["p90"], p["worst"], p["best"],
                     p["n_degraded"], p["n_pairs"]), flush=True)
            print("  %-10s   the unpaired worst/worst scalar would read "
                  "%.4f -- it compares two different draws"
                  % ("", p["worst_unpaired"]), flush=True)
            for L, v in sorted(p.get("by_loss", {}).items()):
                print("  %-10s   baseline loss %-8s n %d  median %.4f  "
                      "worst %.4f" % ("", L, v["n"], v["median"], v["worst"]),
                      flush=True)
        else:
            print("  %-10s sigma_marg %8.4g (no per-draw rows; unpaired)"
                  % (nm, st["sigma_marg"] or 0.0), flush=True)
        stress[nm] = st

    print("\n=== Gate A blind recovery, incumbent vs winner ===", flush=True)
    ga = {}
    for nm, d in (() if skip_gate_a else
                  (("small@8", INCUMBENTS["small@8"]), ("winner", best))):
        r = gate_a(d, k, modes, B, T_ref, sigma, cons, seed=seed)
        ga[nm] = r
        if not r.get("models"):
            print("  %s: not full rank" % nm)
            continue
        print("  %-10s %s" % (nm, "  ".join(
            "%s %.2f%%" % (mm, 100 * v["fro_err"])
            for mm, v in r["models"].items())), flush=True)

    rec = dict(schema_version=RESULT_SCHEMA_VERSION,
               gate_a_schema_version=GATE_A_SCHEMA_VERSION,
               # DERIVED, by the SAME function the verifier uses.  Computing
               # it here independently is how the producer came to emit
               # `gate-candidate` for every independent prior while the
               # verifier derived `screening-only` for a skipped gate -- the
               # first independent screening run would have quarantined its
               # own valid stage.
               evidence_status=derive_evidence_status(
                   _lineage(), cfg, ga, cfg.get("gate_a_candidates")),
               run_id=run_id, snapshot_sha256=snap_hash, snapshot=snap,
               config_sha256=cfg_hash, config=cfg,
               target_conditioned_prior=bool(TARGET_CONDITIONED_PRIOR),
               stress_loss=float(STRESS_LOSS), stress_audit=stress,
               lam_um=float(data.wavelength_um[i]), sigma=float(sigma),
               nuisance_classes=list(nz.DEFAULT_CLASSES),
               ensemble_fro=float(ENSEMBLE_FRO), n_ensemble=int(len(ens)),
               loss_grid=list(LOSS_GRID),
               ensemble_diversity={kk: (vv.tolist()
                                        if hasattr(vv, "tolist") else vv)
                                   for kk, vv in div.items()},
               q_eta=0.0, polish=int(polish),
               constraints=dict(kz_min_frac=cons.kz_min_frac,
                                wood_margin=cons.wood_margin,
                                n_orders_min=cons.n_orders_min,
                                n_orders_max=cons.n_orders_max,
                                area_max_um2=cons.area_max_um2,
                                dressing_max=cons.dressing_max,
                                deembed_sigma_min=cons.deembed_sigma_min,
                                signal_min_sigma=cons.signal_min_sigma),
               search_seconds=dt, n_samples=samples, seed=seed,
               winner=best.to_dict(), winner_source=best_overall_name,
               leaderboard=[(nm, d.to_dict(), float(v))
                            for nm, d, v in board],
               generator="cayley", audits=audits, comparison=rows,
               gate_a=ga)
    # ---- COMMIT.  Everything is written into a private staging directory
    # and the run is published by a single atomic rename.  This makes the
    # commit exclusive (two processes racing on the same identity cannot
    # interleave writes; the loser's rename fails and it refuses), atomic
    # (a reader never sees a half-written run, and staging names begin with
    # "." so the verified iterator skips them), and retryable (a crash leaves
    # only staging, so the same identity can simply be run again).
    run_dir = os.path.join(runs_root, run_id)
    # UNIQUE staging: a previous failed attempt in this same process left
    # `.staging-<pid>-<id>` behind, and reusing that path made a same-process
    # retry fail on the candidates file that was already there.  os.urandom
    # makes each attempt its own directory, and `exist_ok=False` makes the
    # claim exclusive rather than merely likely.
    stage = os.path.join(runs_root, ".staging-%d-%s-%s"
                         % (os.getpid(), run_id[:8],
                            binascii.hexlify(os.urandom(6)).decode()))
    os.makedirs(stage)
    path = os.path.join(stage, "result.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1, default=ms._json_default)

    records = {"search_best@%s" % run_id[:12]:
               make_record(search_best, "search", run_id, snap_hash,
                           cfg_hash)}
    for j, d in enumerate(polished_designs):
        records["polished%d@%s" % (j, run_id[:12])] = make_record(
            d, "polish", run_id, snap_hash, cfg_hash)
    # An archived winner keeps the lineage of the run that PROPOSED it.
    # Stamping the current lineage here is how a target-conditioned proposal
    # could be republished as independent.
    _key = design_key(best)
    _won = [(kk, v) for kk, v in archive_prov.items() if kk == _key]
    _inc = [nm for nm, d in INCUMBENT_DESIGNS.items()
            if design_key(d) == _key]
    _fresh = [nm for nm, r in records.items()
              if design_key(dz.Design.from_dict(r["design"])) == _key]
    if _fresh:
        # THIS run proposed it: cite the search/polish record directly
        _prop = _lineage()
        _proof_d = _proof("same_run", run_id, _fresh[0], best, _prop,
                          parent_record_digest=record_digest(
                              records[_fresh[0]]))
    elif _won:
        kk, v = _won[0]
        _prop = v["proposal_lineage"]
        _src = "transcribed" if v["first_run"] is None else "archive"
        _proof_d = _proof(_src, v["first_run"], v["name"], best, _prop,
                          parent_output_root=v.get("output_root"),
                          parent_record_digest=v.get("record_digest"))
    elif _inc:
        _prop = INCUMBENT_LINEAGE.get(_inc[0], LINEAGE_CONDITIONED)
        _proof_d = _proof("incumbent", "incumbent-constant", _inc[0], best,
                          _prop)
    else:
        raise RuntimeError(
            "the selected design has no resolvable source: it is not this "
            "run's search or polish output, not in the frozen archive, and "
            "not a declared incumbent. Refusing to publish a candidate whose "
            "provenance cannot be shown.")
    if _prop == LINEAGE_CONDITIONED and _lineage() == LINEAGE_INDEPENDENT:
        raise RuntimeError(
            "the selected design %s was proposed under a target-conditioned "
            "prior and cannot be published in an independent run; this is an "
            "evidence-boundary violation, not a labelling detail"
            % best_overall_name)
    records["selected@%s" % run_id[:12]] = make_record(
        best, "selected:%s" % best_overall_name, run_id, snap_hash, cfg_hash,
        proposal_lineage=_prop, proposal_proof=_proof_d)
    cand_path = write_run_candidates(stage, records)

    snap_exit = snapshot_inputs()
    if snap_exit != snap:
        # clean up rather than leaving a poisoned stage behind
        shutil.rmtree(stage, ignore_errors=True)
        changed = sorted(k for k in set(snap) | set(snap_exit)
                         if snap.get(k) != snap_exit.get(k))
        raise RuntimeError("these inputs changed during the run: %s; "
                           "refusing to publish it" % ", ".join(changed))

    manifest = dict(run_id=run_id, snapshot_sha256=snap_hash, snapshot=snap,
                    config_sha256=cfg_hash, config=cfg, lineage=_lineage(),
                    archive_body={k: dict(name=v["name"],
                                          first_run=v["first_run"],
                                          proposal_lineage=(
                                              v["proposal_lineage"]),
                                          design=v["canonical_design"])
                                  for k, v in archive_prov.items()},
                    artifacts={os.path.basename(p): _sha_file(p)
                               for p in (path, cand_path)},
                    n_candidates=len(records))
    with open(os.path.join(stage, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, default=ms._json_default)
    # the marker carries the OUTPUT root as well as the input-derived id, so
    # a run identifies what it published and not only what it was given
    _root = output_root(stage, run_id)
    with open(os.path.join(stage, "complete"), "w") as fh:
        fh.write("%s %s" % (run_id, _root))

    # SELF-VERIFY BEFORE PUBLISHING.  The previous version called the
    # verifier on the staging path, watched it fail (the basename is not the
    # run id), executed `pass`, and published anyway -- so the only real
    # check happened after the rename, by which point an invalid directory
    # already occupied the identity.  `expect_run_id` lets the verifier judge
    # a hidden stage under the name it is about to take.
    man_chk, why = verify_completed_run(stage, expect_run_id=run_id,
                                        receipt_mode="allow_missing")
    if man_chk is None:
        quarantine = stage.replace(".staging-", ".rejected-")
        os.rename(stage, quarantine)
        raise RuntimeError("staged run %s failed its own verification (%s); "
                           "quarantined at %s and NOT published"
                           % (run_id[:12], why, os.path.basename(quarantine)))
    # THE RECEIPT IS WRITTEN BEFORE THE COMMIT.  It is keyed by the OUTPUT
    # ROOT, so a same-identity race loser writes a different file rather than
    # a conflicting row, and its orphan is inert because that root never
    # appears on disk.  Writing first closes the crash window that would
    # otherwise leave a published run with no receipt -- which is now fatal,
    # because a missing receipt used to make the whole scheme fail open.
    # A receipt collision must QUARANTINE THE STAGE, not surface after the
    # rename: an invalid directory in the final deterministic location would
    # occupy the identity and block a clean retry.
    try:
        append_receipt(runs_root, run_id, _root)
    except RuntimeError as exc:
        quarantine = stage.replace(".staging-", ".rejected-")
        os.rename(stage, quarantine)
        raise RuntimeError("%s; staged run quarantined at %s and NOT "
                           "published" % (exc, os.path.basename(quarantine)))
    try:
        os.rename(stage, run_dir)          # THE commit; atomic
    except OSError:
        # the race loser cleans up after itself, so its staging directory
        # cannot accumulate or block a later retry of the same identity
        shutil.rmtree(stage, ignore_errors=True)
        raise RuntimeError("run %s was already published by another process; "
                           "this attempt discarded its own staging copy"
                           % run_id[:12])
    man_chk, why = verify_completed_run(run_dir)
    if man_chk is None:
        raise RuntimeError("published run fails verification: %s" % why)
    path = os.path.join(run_dir, "result.json")



    # derived pointers last: if either fails the run is still complete
    latest = os.path.join(out_dir, "latest.json")
    tmp = latest + ".%d.%s.tmp" % (os.getpid(),
                                   binascii.hexlify(os.urandom(6)).decode())
    with open(tmp, "w") as fh:
        json.dump(dict(run_id=run_id, run_dir=os.path.relpath(run_dir,
                                                              out_dir)), fh)
    os.replace(tmp, latest)
    book = rebuild_index(runs_root,
                         os.path.join(out_dir, "candidate_registry.json"))
    print("run %s committed: %d candidates, %s"
          % (run_id[:12], len(records), path), flush=True)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lam", type=float, default=8.0)
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--polish", type=int, default=2)
    p.add_argument("--ens", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260807)
    a = p.parse_args(argv)
    run(lam_um=a.lam, samples=a.samples, polish=a.polish, n_ens=a.ens,
        seed=a.seed)
    return 0


_IMPORT_SNAPSHOT = snapshot_inputs()


if __name__ == "__main__":
    sys.exit(main())
