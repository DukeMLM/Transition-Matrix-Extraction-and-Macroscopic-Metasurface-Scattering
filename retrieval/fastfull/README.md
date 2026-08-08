# `retrieval/fastfull` — fast full T-matrix from a coded Floquet cell

Implementation of `retrieval/FAST_FULL_TMATRIX_WHEEL_PROPOSAL.md`.
**M1 is built and gated. M2 is implementation/screening in progress, not
complete** — the proposal defines M2 completion to include calibrated
perturbations and closure of Gate A, and neither exists: the perturbation
amplitudes are normalized hypotheticals, not measured physics. The Ewald
coupling agrees with the repository's validated tapered lattice sum to 1e-6
and the blind noisy recovery has been run. **Gates A, B, D, E and F remain
open; Gate C is unattempted.** Gate D additionally needs the separate
`aggregation/` repo-vs-treams discrepancy (~0.1 in S, opposite passivity
verdicts on the same T) resolved in a common basis. M3–M7 are not started.

This package is deliberately separate from the validated one-pitch specular
retrieval in `retrieval/*.py`. That pipeline is complete and its conventions
are pinned by a measured CST campaign (`RESULTS.md`, `HANDOFF.md`); nothing
here modifies it. The conventions are *imported* from it and re-verified by
gates — the central one being that the new flux-normalized transforms
reproduce `sparams_oblique`'s Jones blocks to 3e-16 in a single-order cell.

That gate establishes **consistency with the existing single-order analytic
model, not CST multimode correctness**: port fields, per-mode gauge and
labels, reference-plane phases and a diffractive cell are all Gate B at M3.
M1 claims no Gate A or Gate E verdict; see `M1_FINDINGS.md` §0 and §7.

## Modules

| module | what it owns |
|---|---|
| `lattice.py` | `Lattice2D` (rect/oblique/rotated), reciprocal vectors, fractional Bloch coordinates, shell enumeration; `enumerate_orders` with propagating mask, grazing cut and Rayleigh/Wood margins; `ChannelSet` (side × order × polarization) |
| `transforms.py` | flux-normalized `A` (30 × M_in) and `W` (M_out × 30), the CST-vs-physical TM gauge, `empty_modal_S`, `t_effective`, `deembed_lattice`, generic-track conditioning |
| `symmetry.py` | D4h = C4v × {E, σ_h} with σ_h derived in closed form and numerically; the 40-coefficient D4h ⊗ reciprocity basis; Cayley passive ensemble with a loss knob, plus `ensemble_diversity` |
| `coupling.py` | tapered real-space Bloch sum `C` for a general Bravais lattice, reusing the validated square-lattice assembly; convergence verdict that **refuses** rather than guesses |
| `ewald.py` | Ewald `C` via `treams` for the diffractive cells `coupling.py` refuses — Gate D's independent second implementation; eta-stability verdict; Born/dressing diagnostics |
| `jacobian.py` | analytic Jacobian `H = ∂vec(S)/∂c`, noise whitening, rank/SNR, invariant multipole-block recovery errors under the iid/systematic bracket |
| `cost.py` | CST wall-time / memory / RHS proxy anchored on the campaign's measured 64 s / 4-RHS solve |
| `design.py` | the par. 7 design problem: constraints, objectives, two-stage search, pooled multi-encoding evaluation |
| `synthetic.py` | M2 Gate A: calibrated structured error models and blind recovery — linear seed, C-continuation, LM; `recover_joint` fits T and the calibration together |
| `nuisance.py` | calibration tangents (independent rx/tx, incl. the receive-only TM-row fault), the Schur complement `F_T = J_c^T (I-P_eta) J_c`, and the finite `Calibration` map for joint recovery |
| `m1_study.py` | the M1 deliverable: runs the whole comparison and writes `results/fastfull/M1_DESIGN_STUDY.md` |
| `gate_a_run.py` | the M2 deliverable: blind recovery over the candidate set with a held-out encoding, writes `results/fastfull/GATE_A_STUDY.md` |
| `opt_marginalized.py` | the nuisance-marginalized search.  Each run is an immutable transaction: it is built in a private, uniquely-named `.staging-*` directory, **validated there under its future name**, and published by a single atomic `os.rename`; a failed stage is quarantined as `.rejected-*` and never occupies the identity.  Identity is the full SHA-256 of a start-of-run snapshot (22 code+data files plus library versions, re-verified at exit) **and the frozen candidate archive** — the archive is a selector input, so a run that saw a different archive is a different run, and it is read from the requested `out_dir` only.  The archive is **lineage-filtered** and **deduplicated by canonical geometry**, so republishing the same cell does not move the fingerprint and a run that finds nothing new is refused as a duplicate.  `archive_pin=` pins an epoch.  Provenance is *resolved*, not declared: `proposal_lineage` is mandatory, a fresh `search`/`polish` record must carry its own run's lineage, and **every** `selected` record must resolve to an existing parent — a same-run `search`/`polish` record, an archive entry **walked to a real parent run directory on disk and onward to a fresh search/polish root**, a declared incumbent, or a labelled transcription — matching on canonical geometry and proposal lineage.  Terminal citations are checked against the actual constant at **every** level of the chain (not just the top), an intermediate archive hop must agree with that parent's own archive body, and one cycle/depth budget spans the whole chain, so an invalid ancestor sinks its descendants.  The archive invariants are recomputed from the body, never read off labels.  Identity carries the geometry epoch only; the provenance body rides in the manifest so it cannot churn the run id.  A run id names INPUTS, so each run also publishes an `output_root` recomputed from the artifact bytes plus the label-stripped manifest, written into the completion marker and published as one write-once receipt per root at `runs/receipts/<root>.json`, MANDATORY for any published run; all three must agree (tamper-EVIDENCE, not tamper-proofing — there is no signing key).  Parent proofs MUST carry the parent's output root and record digest (`archive` citations both, `same_run` the record digest), and every run-backed hop adopts and checks the next proof — so a citation binds bytes at every level, not only the first.  Receipts are published under a **fenced** per-root reservation: acquisition returns a token that is re-checked immediately before the atomic replace and again on release, so a publisher that loses its reservation writes nothing and unlinks nothing.  Reading, repairing and sweeping all happen inside the reservation.  Seven crash boundaries are fault-injected in the gates and all recover with the receipt directory holding nothing but the receipt.  Cross-process liveness is NOT detected — fencing is what makes lease expiry safe.  An unreadable receipt is repaired only by its owner, identified by a receipt-independent check (`receipt_mode="skip"`); a foreign run id is refused.  An existing receipt is parsed and must already match or the publish raises and quarantines the stage, and `read_receipt` enforces an exact `{run_id, output_root}` body whose root must equal both the request and the filename.  Ancestor verification suppresses only recursive descent, never a local invariant.  The two hard-coded incumbents are declared **target-conditioned** and filtered like any proposal, so an independent run inherits neither.  Admission is by `verify_completed_run`, which recomputes every hash, rederives the run id from the stored snapshot and config bodies, requires exactly `{result.json, candidates.json}`, enforces a closed versioned result schema (exact field set, exact types — `bool` does not satisfy `int`; reported seed/samples/ensemble must equal the hashed config, `winner_source` must match the selected record's origin, the winner must appear on its own leaderboard, every reported metric must be finite, `evidence_status` is DERIVED by one function shared with the producer, over four states — `screening-only` / `error-screen-attempted` / `error-screen-passed` / `custom-screen-passed`, where the strongest name requires the version-1 protocol candidate set and any other passing set gets the weaker `custom-` name.  It is an ERROR SCREEN, deliberately not called a passed Gate A: it needs the declared candidate set and perturbation families all within 5% with finite non-negative errors, while `GATE_A_UNVERIFIED` records the five proposal criteria (rank 40, useful-direction SNR > 10, basin stability, passivity, frozen trial identities) it does NOT check; and each paired stress block must be RECOMPUTABLE — both ensembles REBUILT during verification from the hashed seed/grid/norm/stress-loss, every row hash required to equal the rebuilt row and the aggregate ensemble hashes to equal the config, loss labels on the hashed grid, `by_loss` required, and every aggregate recomputed from the rows), and **derives lineage from the hashed config** rather than a manifest label — so target-conditioned geometries cannot enter a target-independent selection.  Any malformed shape is a rejection with a diagnostic, never an exception.  `candidate_registry.json` is derived and non-selecting.  Includes a paired over-loss stress audit reporting **per-pair** p10/p50/p90 (same latent draw at both loss factors), reported and never used for selection |

## Running

```bash
python test_fastfull_core.py      # 27 gates
```

```bash
python test_fastfull_design.py    # 23 gates
```

```bash
python test_fastfull_ewald.py     # 14 gates (M2 coupling)
```

```bash
python test_fastfull_synthetic.py # 31 gates (M2 Gate A + provenance)
```

```bash
python -m fastfull.m1_study --samples 700
```

```bash
python -m fastfull.gate_a_run --trials 3
```

```bash
python -m fastfull.opt_marginalized --samples 300
```

```bash
python -m fastfull.design --evaluate 26.0,33.8,90,0,0.090,-0.460 --lam 20 --benchmark --kz-min 0 --wood-margin 0 --orders 1,64 --area-max 5000
```

## The one new convention: flux normalization

CST Floquet ports are power normalized; the repository's `plane_wave_coeffs`
uses unit electric-field amplitude. Unit CST modal amplitude corresponds to
a field amplitude `α = sqrt(2 Z0 k / (A_cell |k_z|))`, so

```
A[:, c]  = s_c   · α_c · plane_wave_coeffs(k̂_c, ê_c)
W[c', :] = s_c'  · (2πi / (A_cell |k_z,c'|)) / α_c' · ê_c'† · FF(k̂_c')
```

`Z0` cancels exactly between the two. When only one order propagates the two
normalizations collapse back to `2πi / (k A cosθ)` and the whole product
reduces to `sparams_oblique.jones_blocks` — that identity is gate (f).

`s` is the TE/TM gauge: `s(TE) = +1`, `s(TM) = propagation direction`. This
models a CST port mode as one fixed transverse pattern per (order,
polarization), used unchanged whichever way the wave crosses it, and it
reproduces entry-for-entry the θ→0 mapping table in the `sparams_oblique`
docstring — including the −1 on the S11 TM receive row whose omission took
the campaign's χ²_red from 658.9 to 2.49. A residual per-mode sign
(`mode_gauge`) is exposed and must be closed by the M3 port-field export, not
assumed; see `transforms.py`'s two documented caveats.

## Scope limits that are enforced in code, not just written down

* **C = 0 screening — now discharged.** `jacobian.py` still computes the
  bare Jacobian by default and says so, and `coupling.converged_C` still
  refuses on a diffractive cell. But `ewald.py` supplies a converged C
  for those cells, and `m1_study.verify_with_coupling` re-measures every
  winner with it: σ₄₀ moves ≤ 5 %, predicted error ≤ 0.4 %, rank stays
  40/40. A coding cell has ‖C T‖ = 0.17 against 5.1 for the campaign's
  2 µm cell, so it sits close to the Born/linear regime.
* **The error level is a systematic discrepancy, not a noise floor.** σ comes
  from the campaign's *measured* normal-incidence closure, interpolated at the
  design wavelength (2.8417e-3 at 8 µm, 3.1751e-3 at 20 µm). That residual is
  dominated by model error (`results/REAL_RETRIEVAL.md` §4.3), so it does not
  average down over the hundreds of modal entries a coded cell provides.
  Every recovery figure is reported as an iid / systematic **bracket**, the
  design objective divides by √n_obs, and no Gate E verdict is claimed from
  either end. Collapsing the bracket needs M3's per-channel covariance.
* **Reported metrics are basis invariant.** The 40 basis matrices come from
  eigenvectors of a degenerate projector, so per-coordinate statistics are
  meaningless; all errors are computed over invariant multipole blocks and
  gated against random rotations of the basis.
* **Target independence — currently BROKEN on the marginalized branch.** The
  C = 0 objective does not depend on T at all, and the ensemble norm is a
  declared constant rather than ‖T_ref‖. But the passive ensemble's LOSS
  GRID was calibrated against the reserved reference wheel, so that branch is
  target-conditioned and development-only (`TARGET_CONDITIONED_PRIOR`).
  `benchmark_reference` and `reference_recovery` remain separate entry points
  the search never calls. **Ensemble coverage is a separate, open
  problem**: an independent 32-draw ensemble places the actual wheel below
  every random draw, so the design ensemble is misspecified even though it is
  target-independent.
* **Cost proxy is a ranking tool.** Its exponents are knobs anchored on one
  measured point and extrapolated up to ~100× in dof; it ranks designs, it
  does not predict wall times.

## The controlling result

The calibration tangents are ~99.98 % collinear with T (smallest principal
angles under 1.2° for the phase, angular and port-plane families alike).
Consequences, all from **synthetic, same-forward-model** experiments — none
of these is a measurement (`M1_FINDINGS.md` §11):

* a joint fit of T **and** the calibration takes the dominant error class
  from 24.0 % to **1.16 %**, recovering the port-plane offset to 0.4666 µm
  against a true 0.4889;
* a **free** joint fit is catastrophic when that systematic is absent — iid
  data goes 5.4 % → 131 % — and no single prior width protects both cases;
* optimizing the marginalized objective without a de-embedding-condition
  constraint is gamed by a collective-resonance cell (Gate A error 284 %);
* the marginalized objective's apparent 28.6 % gain was mostly a degenerate
  passive ensemble (effective rank 1.44 of 40); with an exact-Cayley
  loss-grid generator at effective rank ~5.4 the gain is **1.9 %** and the
  winner still does not transfer to the wheel, so `small@8` stays incumbent;
* a single stochastic restart is not a selector — one run labelled a cell
  "winner" at objective 4.17 while an archived cell scored 5.81 on the same
  ensemble, so `opt_marginalized` now keeps an explicit candidate ARCHIVE and
  reports a leaderboard;
* across every run so far the training objective and blind recovery
  **disagree**: the cells that win the objective lose on the wheel;
* every completed search so far used a training prior 13x-89x more
  absorptive than the wheel, so **all candidate rankings are provisional**;
  the corrected grid is moreover **target-conditioned** (it was calibrated
  against the reserved reference T) and its production ensemble is still
  3.06x the wheel's absorption, so nothing selected with it can close Gate A
  or Gate E;
* the constrained marginalized winner improves the objective 32 % and the
  T-only recovery not at all, because a T-only estimator cannot collect
  nuisance-marginalized information.

So **calibration-model uncertainty currently dominates the error budget**,
and encoding performance is conditional on the calibrated nuisance
distribution — the encoding is not irrelevant (the same experiment shows a
2.5× difference between cells). That makes M3's reference-plane-shift,
mesh-ladder and empty-repeat measurements a precondition, not a refinement.
`small@8` remains the operational incumbent; the marginalized winner improves
the search metric but is worse on the reference wheel in every stored
measurement.
