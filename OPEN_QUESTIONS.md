# Open questions

Things this study raised and did **not** settle. Each entry names the specific
experiment that would close it, so it can be picked up cold.

---

## 1. Does arrangement *symmetry* affect accuracy, or only pair geometry?

**Status: BOTH, IN A HIERARCHY. Three experiments; each single-predictor answer
is falsified on its own, and each axis is now measured by a matched control.**

* At ρ = 0.944, changing the symmetry class does not rescue a cell —
  [`results_2x2_ACDB_l3`](aggregation/results_2x2_ACDB_l3/REPORT.md).
* ρ is not sufficient: `e,b;c,a` has a *lower* ρ than `a,d;b,c` (0.809 against
  0.854) and is **5.2× worse** —
  [`results_2x2_EBCA_l3`](aggregation/results_2x2_EBCA_l3/REPORT.md).
* Symmetry is a real second axis: `e,b;g,a` has the **identical** worst pair
  (A–B), ρ (0.8092) and gap (1.5265 µm) as `e,b;c,a`, and is *tighter* on all
  three other 8 µm pairs — so every distance metric ranks it same-or-worse. Its
  mismatch is 10.0 % against 25.0 %, and it is **3.4× better**.

* But on the ρ axis the matched control comes out **backwards**: `c,a;g,f` holds
  the mismatch at 7.1 %, identical to `e,c;f,a`, and raises ρ from 0.674 to
  0.719 — and is **1.9× better**, not worse.

| cell | ρ | diag mm | all mm | MSE, complex S21 |
|---|---|---|---|---|
| `a,b;c,d` | 0.944 | 27.3 % | 18.8 % | 0.1654 |
| `a,c;d,b` | 0.944 | 20.0 % | 18.8 % | 0.1206 |
| `e,b;c,a` | 0.809 | 25.0 % | 20.0 % | 0.0612 |
| **`e,b;g,a`** | **0.809** | **10.0 %** | **10.0 %** | **0.0182** |
| `a,d;b,c` | 0.854 | 9.1 % | 9.1 % | 0.0118 |
| `e,c;f,a` | 0.674 | 7.1 % | 7.1 % | 0.0028 |
| **`c,a;g,f`** | **0.719** | **7.1 %** | **7.1 %** | **0.0015** |
| **`e,a;f,c`** | **0.652** | **7.7 %** | **7.7 %** | **0.0011** |

**Neither predictor orders these eight alone**, and together they hold in
**22 of 23 decidable pairs** — the one exception being `e,c;f,a` against
`c,a;g,f` above, where the cell that is better-or-equal on both predictors
scores worse. Four benchmarks sit at exactly ρ = 0.809 — atom G alone
(0.00094), `a,b;b,a` (0.00367), `e,b;g,a` (0.0182), `e,b;c,a` (0.0612) — the
same convergence ratio, 65× apart, which is the compact statement of why ρ
alone cannot be an acceptance test.

**Where the predictors stop working is now bounded.** They order the four cells
above 0.01 cleanly and fail to resolve the four below 0.003. Those four span
only 0.0011–0.0028 while their constituent atoms alone score 0.00028–0.00107,
so the cells sit within a small factor of their own input floor and the ranking
among them is set by whichever narrow collective mode each happens to host.
Two alternatives are ruled out for the exception: it is not input quality —
`c,a;g,f`'s atoms are individually *worse* on average (0.00076 against 0.00060)
— and it is not a grid artifact, since the ordering survives on the 97-point
grid and on the magnitude metric.

Still open: what sets that floor. `e,a;f,c` reaches mean |ΔS21| 0.023, inside
the spread of its own four atoms measured alone (0.016–0.030), so at the good
end the residual looks like the input T-matrices rather than the aggregation —
but no experiment separates the two contributions directly (see §4), and until
one does, "the aggregation adds nothing" is an inference from coincident
magnitudes, not a measurement.

The single-atom series is untouched by this and is now seven points; E, F and G
were extracted after the claim was made, cost no full-wave time (the packed
sweep already held runs 1, 8, 7), and land on the curve: ρ = 0.539 / 0.584 /
0.629 / 0.719 / 0.809 / 0.899 / 0.989 gives dip errors +0.2 / −0.3 / −0.4 /
−0.7 / −1.0 / −4.1 / −18.7 %, monotone throughout.

The record of the first experiment follows.

`a,c;d,b` reproduces `a,b;c,d`'s pair geometry exactly (same worst pair B–D,
same ρ = 0.944, same 0.448 µm gap) in a different symmetry class, and it
reconstructs like `a,b;c,d`, not like `a,d;b,c`:

| cell | ρ | MSE, complex S21 | mean \|ΔS21\| | deepest dip misplaced by |
|---|---|---|---|---|
| `a,b;c,d` | 0.944 | 0.1654 | 0.308 | −5.51 µm |
| **`a,c;d,b`** | **0.944** | **0.1207** | **0.244** | **+4.92 µm** |
| `a,d;b,c` | 0.854 | 0.0118 | 0.080 | +0.11 µm |

The two cells sharing ρ = 0.944 are within 1.4× of each other; the one with
ρ = 0.854 is 10–14× better than either. So within these four atoms symmetry is
not what made `a,d;b,c` accurate. Both failing cells misplace their deepest
transmission dip by ~5 µm in *opposite* directions, so this is a truncation
error, not a correctable bias.

Two qualifications were recorded at the time: the residual 0.1207-vs-0.1654 gap
between the two ρ = 0.944 cells is unexplained (27 %, but second-order against
the 10× step), and the falsification is of the *diagonal-mirror* symmetry metric
quoted below — under an all-mirrors metric `a,b;c,d` and `a,c;d,b` tie at
18.8 %, so that variant predicts what was observed and is untested.

> **Superseded.** This section originally concluded "ρ predicts accuracy". The
> `e,b;c,a` run above shows that is too strong: ρ = 0.809 scores 0.0612, five
> times worse than ρ = 0.854. The claim that survives is the one stated at the
> top — ρ bounds the error at fixed composition and does not order cells against
> each other — which makes §3 below a *more* live defect, not less: the guard it
> specifies would have passed `e,b;c,a` without a warning.

The original framing is kept below for the record.

---

### Original statement of the question

The two four-atom cells built from the same A, B, C, D disagree with full-wave by
very different amounts:

| cell | worst pair | ρ = (aᵢ+aⱼ)/d | nearest-mirror mismatch | mean \|ΔS21\| vs CST |
|---|---|---|---|---|
| `a,b;c,d` | B–D | 0.944 | 27 % | 0.308 |
| `a,d;b,c` | A–D | 0.854 | 9 % | 0.083 |

Two explanations fit equally well, and they are **confounded** in this pair of
cells, because putting B and D on a diagonal is simultaneously what makes
`a,d;b,c` nearly mirror-symmetric *and* what widens its tightest gap:

* **Geometry.** Accuracy is set by the worst pair's translation convergence
  ratio ρ; `a,d;b,c` simply has a better one.
* **Symmetry.** In a more symmetric arrangement the same pair geometry recurs,
  so the same coupling error recurs and can partially cancel between equivalent
  pairs. This would make symmetric cells *appear* better without the solver
  being any more correct.

### The discriminating experiment

The 24 assignments of four distinct atoms to the four sites collapse under the
D4 symmetry of the site square to **exactly three** distinct cells. The third
one breaks the tie:

| cell | diagonal pairs | worst pair | ρ | mirror mismatch |
|---|---|---|---|---|
| `a,b;c,d` | A–D, B–C | B–D | 0.944 | 27 % |
| `a,d;b,c` | A–C, D–B | A–D | 0.854 | 9 % |
| **`a,c;d,b`** | **A–B, C–D** | **B–D** | **0.944** | **20 %** |

`a,c;d,b` has the *same* worst pair and therefore the *same* ρ as `a,b;c,d`, but
a different symmetry class. So:

* if it lands near `a,b;c,d` (≈ 0.3), **geometry is the driver** and symmetry is
  a red herring;
* if it lands nearer `a,d;b,c` (≈ 0.08), **symmetry matters on its own** and the
  ρ story is incomplete.

Run it with:

```bash
python -m tmatrix.aggregation.cst_supercell.build_2x2_supercell --pair ACDB --only supercell
```

```bash
python -m tmatrix.aggregation.cst_supercell.read_supercell_results --run aggregation/cst_supercell/runs_ACDB/supercell/supercell.cst --empty aggregation/cst_supercell/runs/empty/empty.cst --out aggregation/results_2x2_ACDB_l3
```

plus the matching `run_supercell --site A -4 4 --site C 4 4 --site D -4 -4
--site B 4 -4`. About 1–2 h of CST; it reuses the existing empty-cell run and
completes the set of three arrangements either way.

### What was already tried and did not work

* **lmax convergence as an internal proxy.** An invalid translation should not
  converge in multipole order, so the lmax 3→4→5 spread ought to flag the bad
  cells. It does not: `a,d;b,c` (widest gap, best CST agreement) has the
  *largest* spread. The test is confounded because the l = 4, 5 rows of these
  T-matrices sit at the extraction noise floor and the lattice sum amplifies
  them, so the spread mostly measures noise amplification, not geometry.
* **A pure-geometry residual test** of the truncated re-expansion. The version
  written was buggy — it ignored the source radius, and returned residuals that
  *grew* with lmax, which is unphysical. Not used. A correct version would need
  an extended source (e.g. a ring of dipoles at radius aᵢ) rather than a point
  multipole, because a point multipole's series converges everywhere and cannot
  exhibit the failure.

---

## 2. Why does cross-polarization vanish on square sites?

**Status: empirical result, no derivation.**

With four *distinct* atoms the cell's point group is C1, so nothing forbids
cross-polarization at normal incidence — yet \|t_xy\| ≤ 3×10⁻¹² in all four
arrangements (including `e,b;c,a`, a different atom set), across the whole band. It is a genuine cancellation, not an
insensitivity of the method:

| configuration | \|t_xy\| |
|---|---|
| four distinct atoms on the square sites | ≤ 2×10⁻¹² |
| same atoms, one site moved to (2.5, −5.5) | 1.0×10⁻² |
| same square sites, rectangular 16 × 20 lattice | 1.0×10⁻¹⁰ |

So it tracks the **square arrangement of the four sites**, not the lattice and
not the code. No symmetry argument for it is offered here. Worth deriving: if
there is a theorem that a planar array of individually C4v scatterers on a
square site set has a diagonal normal-incidence Jones matrix regardless of
composition, it would be worth stating, and it constrains what these cells can
be designed to do (they can be made birefringent but not polarization-rotating).

The direct CST runs measure the same channel independently (mode 2 of the
Floquet port), so the data to check this against full-wave already exists in
`results_2x2_ABCD_l3/cst_direct_supercell.csv`.

---

## 3. The pipeline does not refuse when ρ is close to 1

**Status: known defect, fix specified, not implemented.**

`run_supercell.py` silently produced a 19 %-wrong answer for atom D, and a
0.31-mean-error answer for `a,b;c,d`, with no warning. The repository already
has a refuse-rather-than-guess policy for the Ewald η split
(`ewald_supercell.converged_W` returns `None` and lists reasons); the same
policy should cover the translation convergence ratio.

Concretely: compute ρ = max over 8 µm neighbour pairs of (aᵢ + aⱼ)/d, and warn
above ~0.85, refuse above ~0.95, with `--force` to override. The thresholds can
be calibrated on the fourteen benchmarked cases in this study. The blocker is
that aᵢ is not currently stored in the `tmat.h5` files — the circumscribing
radius would have to be added to the format, or passed on the command line.

**A ρ-only gate is not enough, and this is now measured, not suspected.**
`e,b;c,a` has ρ = 0.809 — below even the warning threshold — and puts the depth
on the wrong one of two collective modes, scoring 0.0612 against `a,d;b,c`'s
0.0118 at ρ = 0.854 (§1). So the guard as specified would have passed it
silently. It still catches everything it was built for (0.944 and 0.989 both
fail), so implement it as a floor — but it must be documented as a necessary
condition, not an acceptance test, and the reported diagnostic should include
the per-pair ρ list rather than only the maximum, so a user can see what the
single number is hiding.

---

## 4. Two error mechanisms are separated only qualitatively

**Status: partially quantified.**

`error_budget.py` separates input-T uncertainty from lattice amplification, and
shows the mid-band error is a systematic pole shift rather than noise. It does
**not** separate that from the translation-truncation error, which is the
dominant term once ρ → 1. A cell's total error is currently attributed by
inspection (which frequency window it sits in), not by a decomposition. A
version that propagates both sources and reports their relative size per
frequency would make the "which fix helps me" question answerable directly
rather than by argument.
