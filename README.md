# Fast Full T-Matrix Retrieval from Floquet S-Parameters

**This branch is the fast-T-matrix trail.** It carries the experimental inverse
route only: recovering an isolated, symmetry-constrained T-matrix from multimode
Floquet S-parameters, with explicit calibration and covariance gates.

The forward pipeline — T-matrix extraction, array aggregation, heterogeneous
supercells and their CST benchmarks — is the subject of the **`dary`** branch
and is not developed here. See [`dary`](../../tree/dary) for the aggregation
study, its results tree, `experiment.md` and `OPEN_QUESTIONS.md`.

> **Why aggregation code is still here.** The retrieval modules import
> `tmatrix.aggregation.{vswf, tmat_io, translate, aggregate, sparams, mirror}`
> (48 module-level imports across `src/tmatrix/retrieval/` and
> `tests/retrieval/`), and `fastfull/opt_marginalized.py` sha256-hashes three of
> those source files as run provenance. Three de-embedding and validation gates
> also read `aggregation/results/periodic_results.npz`,
> `aggregation/results/treams_reference.npz` and `aggregation/cst_direct/run_v3`.
> The shared package and those inputs are therefore retained; the aggregation
> *study* — its results sweeps, CST supercell projects and narrative — is not.
> The aggregation drivers left in the package (`run_supercell.py`,
> `compare_cases.py`, the plotters) belong to `dary` and will not find their
> output directories on this branch.

---

## Status

The retrieval study asks whether a periodic CST experiment can recover the
complete isolated `lmax = 3` T-matrix without the conventional enclosing-field
projection. The stored matrix is 30×30; D4h symmetry plus reciprocity reduce it
to 40 independent complex coefficients per frequency. A deliberately
lower-symmetry cell and complete open-order Floquet S-matrix are used to expose
otherwise dark multipole sectors.

The current development incumbent is `small@8`, a six-order/24-channel design
near 8 µm. The errors below come from same-forward-model synthetic probes whose
families were normalized to a specular-derived discrepancy level; they are not
measured diffractive-channel covariance. The evidence is therefore encouraging
but still screening-level:

| check | current result | interpretation |
|---|---:|---|
| ideal noise-free D4h recovery | rank 40/40; numerical-error T recovery | algebraic identifiability demonstrated |
| iid perturbation | 5.15% global T error | approximately at, but not below, the 5% gate |
| mode-mixing perturbation | 4.27% | passes this synthetic stress direction only |
| smooth-angular / reference-plane perturbations | 19.88% / 23.99% | calibration aliases remain dominant |
| adversarial perturbation | 40.59% | robust recovery not demonstrated |
| nuisance-orthogonal weakest margin | about 3.5 vs required >10 | useful-direction SNR gate open |
| cost proxy | 78.5 min vs 49.1 min reference | matched-speed gate open |
| same approach at 20 µm | 117% iid / 801% systematic error | current per-frequency design is a no-go |

The strongest physics result is a conditional nuisance-model proof of concept:
jointly fitting the *correct* reference-plane offset reduced the `small@8`
error from 23.99% to 1.16%. In contrast, fitting uncalibrated nuisance freedom
raised the iid error from 5.4% to 130.6%. The next discriminating experiment is
therefore a frozen, held-out comparison of the full 40-D baseline against an
independently calibrated Au/geometry sector covariance and a shared-pole,
passive-residue prior with a nonzero full-space residual. No independent
physics-prior advantage is claimed yet.

Start with:

* [`retrieval/FAST_FULL_TMATRIX_WHEEL_PROPOSAL.md`](retrieval/FAST_FULL_TMATRIX_WHEEL_PROPOSAL.md) — method, milestones, and stop/go gates
* [`src/tmatrix/retrieval/fastfull/README.md`](src/tmatrix/retrieval/fastfull/README.md) — implementation and test entry points
* [`retrieval/results/fastfull/M1_DESIGN_STUDY.md`](retrieval/results/fastfull/M1_DESIGN_STUDY.md) — rank/SNR/cost screen
* [`retrieval/results/fastfull/GATE_A_STUDY.md`](retrieval/results/fastfull/GATE_A_STUDY.md) — blind synthetic stress study
* [`review.md`](review.md) — adversarial review history and unresolved evidence boundaries

Validation entry points, in the `cst_inference` environment:

```bash
pytest -k fastfull
```

or one at a time (each is also a standalone script with its own PASS/FAIL
table):

```bash
python tests/retrieval/test_fastfull_core.py
```

```bash
python tests/retrieval/test_fastfull_design.py
```

```bash
python tests/retrieval/test_fastfull_ewald.py
```

```bash
python tests/retrieval/test_fastfull_synthetic.py
```

The last one is search-heavy: about 8 minutes on the reviewed machine.

This Git branch contains the retrieval source, tests, compact FastFull reports,
and small campaign metadata. Unpacked CST databases and the bulk generated
result tree remain local and are excluded by `.gitignore`.

---

## Conventions (read this before touching anything)

All Stage-3 code follows the [tmat.h5 standard](https://doi.org/10.48550/arXiv.2404.10399)
exactly as declared by the data file:

* time dependence **e^(−iωt)**, outgoing radial functions **h_l^(1)**
* orthonormal spherical harmonics with **Condon–Shortley** phase
* Jackson-normalized VSWFs: `X = LY/√(l(l+1))`, `M = z_l X`, `N = (1/k)∇×M`
  — note N's radial term carries a factor **i** in this convention
* parity basis (TE/"magnetic" = M-waves, TM/"electric" = N-waves)
* `f = T a` (the file calls `f` "p"); mode order read from `/modes`, never assumed
* lmax = 3 → n = 2·lmax·(lmax+2) = 30 modes

---

## Repository layout

```
src/tmatrix/
  retrieval/                  Floquet-S-to-isolated-T inverse route
    fastfull/                 D4h basis, coded-cell design, Ewald and recovery
  aggregation/                shared forward machinery the retrieval code imports
  extraction/                 CST near fields -> VSWF projection -> *.tmat.h5
  units.py numerics.py plotting.py results_io.py paths.py cst_env.py

tests/retrieval/              ladder, parametrize, de-embedding, fastfull gates
tests/aggregation/            VSWF and translation layers, the manual's 6.5.5
tests/test_suites.py          the pytest front end that runs them all

retrieval/                    inverse-route outputs and reports
  FAST_FULL_TMATRIX_WHEEL_PROPOSAL.md   feasibility hypothesis and gates
  HANDOFF.md                  the normative conventions; read before editing
  results/fastfull/           compact M1/M2 screening artifacts
  cst_runs/*.json             curated campaign metadata (raw CST trees stay local)

INVERSE_TMATRIX_FROM_FLOQUET.md   the inverse-route concept note
review.md, review_response.md     adversarial review log and replies

aggregation/                  only the inputs the retrieval gates read
  results/                    periodic_results.npz, treams_reference.npz, ...
  cst_direct/run_v3/          the de-embedding reference run
  NOVELTY.md                  prior-art map (cited by the concept note, Lane 1/3)
  IMPLEMENTATION_GUIDE.md     forward-pipeline tutorial (cited at SS7.4)

test/, ref/                   benchmark inputs and the operational manual
```
