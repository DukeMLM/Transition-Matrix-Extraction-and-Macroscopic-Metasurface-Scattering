# Novelty Analysis: Where Does This Pipeline Stand vs. Prior Art?

*Prepared 2026-08-06. Literature map assembled by a four-track survey (treams/KIT
ecosystem; extraction front-ends; large finite-array solvers; T-matrix design
programs), then synthesized. Machine-collected citations — spot-verify the exact
references before using them in a manuscript.*

## 1. The honest starting point

The physics from T-matrix to S-parameters is the textbook chain — Waterman (1965)
T-matrices, Foldy–Lax (1945/51) multiple scattering, plane-wave projection of the
0th order. Any two correct implementations must agree (ours agrees with treams to
2.7e-4 and with direct CST to 0.3% — that identity is the correctness anchor, not
a novelty deficit). **No novelty claim is available at the level of the
formulation, and none should be attempted.** Differences in *how* the operators
are computed (Ewald vs. tapered-Richardson lattice sums, analytic vs. projected
translation operators) are numerical-methods choices, not contributions.

## 2. COVERED — already published, do not claim

| Claim | Covering work |
|---|---|
| Solver-agnostic T-matrix extraction (plane-wave illuminations → VSWF projection) | Fruhnert et al., Beilstein J. Nanotechnol. 8, 614 (2017); Demésy et al., JOSA A (2018); Garcia-Santiago et al., PRB (2019) |
| Lattice sums → plane-wave S-matrix → substrate/film stacking, oblique incidence | Beutel et al., JOSA B 38, 1782 (2021); treams, Comput. Phys. Commun. 297, 109076 (2024) |
| Standardized T-matrix data format + database | Asadova et al., JQSRT (2025), tmat.h5, 42-author standard; Daphona database (arXiv:2602.02101) |
| Finite arrays on layered substrates via Foldy–Lax | SMUTHI, JQSRT 273, 107846 (2021) (Sommerfeld coupling); MSTM v4 (2022); FHMSTM, npj Metamaterials (2025) — 5024 pillars on substrate |
| Large finite arrays + adjoint gradient design | Skarda et al., npj Comput. Mater. 8, 78 (2022) — 645λ×645λ, distributed T-matrix; dreams, APL Photonics (2026) — JAX autodiff, clusters and lattices |
| The bare chain "microscopic solver → T → aggregation → macroscopic response" | GPM (Bertrand/Vynck/Lalanne, JOSA A 2020); KIT homogenization (Zerulla et al., Adv. Opt. Mater. 2023); molecular Hyper-T-matrix (Adv. Mater. 2023) |

## 3. PARTIALLY COVERED — exists, with named limitations

- **Commercial-solver ingestion.** The tmat.h5 consortium ships reference
  extraction scripts for COMSOL, JCMsuite, ONELAB, nanobem, ADDA, MEEP, SMARTIES
  — **CST/FIT is absent**. The Duke coupled-dipole line (Pulido-Mancera et al.,
  PRB 96, 235402, 2017) extracts from CST/HFSS but **truncates at dipole order**.
- **Substrate + finite arrays.** SMUTHI's per-particle T-matrices come from
  NFM-DS (axisymmetric-friendly), with **documented trouble for flat particles
  hugging interfaces — exactly the MIM/patch regime**; and it has no design loop.
- **Design loops.** dreams differentiates **analytic Mie spheres only** (arbitrary
  shapes explicitly need a differentiable solver); Skarda/Fan is **free-space
  only** (no substrate/ground plane).
- **Extracted-T-matrix quality.** The 2026 AAA pole-compression work
  (arXiv:2602.18414) handles frequency sampling but **explicitly not noise,
  reciprocity, passivity, or symmetry enforcement** — nobody publishes error
  models or constraint enforcement for solver-extracted T-matrices.

## 4. OPEN — the defensible lanes, in build order

**Lane 1 — Validated CST/FIT → tmat.h5 extraction with quality enforcement.**
No published CST reference extraction exists; the consortium repo lacks it.
Add what nobody publishes: broadband (single-run time-domain) extraction plus
noise filtering and reciprocity/passivity/symmetry enforcement on extracted
matrices. Directly leverages the working pipeline in this repo (including the
convention pinning done here: the N-wave radial i-factor, reciprocity permutation
verified against the file's own stored metric). Incremental but citable and
uncontested; a natural contribution to the tmat.h5 ecosystem.

**Lane 2 — Ground-plane-aware extraction for MIM meta-atoms feeding both
periodic and finite aggregation.** The intersection (reflection-coupled /
image-corrected per-site T-matrices from a commercial solver) × (flat metallic
atoms on ground planes — where SMUTHI documentedly fails) × (the same extracted
object driving periodic S-parameter stacks AND heterogeneous finite arrays) is
claimed by no surveyed work. The PEC-image machinery in `mirror.py` (validated by
machine-exact unitarity of a lossless array over PEC) is the seed; the full
version needs the Sommerfeld R-matrix of the manual's Stage 2. Defensible because
it attacks a *documented failure mode* of the incumbent, not a gap of
convenience.

**Lane 3 — Design-by-library around a non-differentiable commercial solver.**
A closed loop: CST-extracted T-matrix library + ML surrogate/interpolation over
it + finite-array S-parameter targets with per-site heterogeneity, ideally
experimentally validated. dreams needs analytic differentiable T-matrices;
Daphona's ML operates on single-scatterer data — the loop over *extracted,
non-analytic* atoms with *array-level* targets is open. Highest impact and the
best fit to this group's surrogate-modeling assets — but dreams (Dec 2025) and
Daphona (Feb 2026) are moving toward it; position claims against them explicitly,
and speed matters.

Also open (algorithmic, higher risk): **recursive hierarchical cluster-T
renormalization** (aggregate a patch → effective cluster T → aggregate clusters)
inside a design loop — TERMS (JQSRT 2022) stops at one cluster level with no
optimization. This is what "multi-scale" should ultimately mean for this program.

## 5. Anti-claims — framings that will not survive review

- "T-matrix metasurface S-parameters" (Beutel 2021 / treams 2024)
- "Finite heterogeneous arrays via Foldy–Lax" (treams, SMUTHI, FHMSTM)
- "T-matrix extraction from full-wave solvers" (Fruhnert 2017 and successors)
- "T-matrix + machine learning" (Daphona 2026)

Novelty lives only in the **combination** — CST extraction + ground-plane
coupling + quality enforcement + closed design loop — and Lane 1 → 2 → 3 is the
defensible build order.

## 6. What this repo already contributes toward those lanes

- A working, convention-verified CST extraction consumer and validation harness
  (Lane 1's launch pad; the harness itself — Mie/optical theorem at 1e-15,
  reciprocity vs. stored metrics, treams and direct-CST closure at ≤0.3% — is the
  quality-enforcement groundwork).
- Finite-array Foldy–Lax with per-site T-matrices and arbitrary in-plane
  positions (`aggregate.py`), plus the periodic lattice-sum path (`translate.py`).
- PEC-image ground-plane coupling (`mirror.py`) — Lane 2's special case, with the
  unitarity test that any Sommerfeld generalization must also pass.
- The web viewer (`webview/`) and diagnostics tooling for extracted-file QA.
