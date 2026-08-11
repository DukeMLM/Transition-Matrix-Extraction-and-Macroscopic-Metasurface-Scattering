# `e,b;c,a` — the counterexample: ρ is not sufficient

A cell built to be **looser** than the one cell that works, and **less**
symmetric than either cell that fails. If the convergence ratio ρ were what
decides accuracy, it should have been the best of the four. It is the second
worst.

```
        E  B          E at (−4, +4)   B at (+4, +4)
        C  A          C at (−4, −4)   A at (+4, −4)
```

| | |
|---|---|
| cell | 16 × 16 µm, atoms 8 µm apart, lmax 3, Ewald coupling, 25-point grid |
| worst pair | **A–B, ρ = 0.809, gap 1.526 µm** — the loosest of the four four-atom cells |
| diagonal pairs | A–E, B–C |
| mirror mismatch | **25.0 % diagonal, 20.0 % all-mirror** — the *most* asymmetric of the four |
| atoms | E (`saw_gold_wl10p30um`, scale 3.00), B (5.00), C (3.25), A (4.00) |
| direct CST | `../cst_supercell/runs_EBCA/`, 3603 s, empty-cell companion reused from `../cst_supercell/runs/empty/` |

## Why this cell was run

[`../../OPEN_QUESTIONS.md`](../../OPEN_QUESTIONS.md) §1 was settled by showing
that at ρ = 0.944 the symmetry class does not rescue a cell: `a,c;d,b`
reproduces `a,b;c,d`'s pair geometry exactly and scores with it. That left the
converse untested. The one accurate cell, `a,d;b,c`, is *also* the most nearly
mirror-symmetric of the three (9.1 % against 27.3 % and 20.0 %), so in the cell
that works, "low ρ" and "nearly symmetric" were still confounded.

Three more atoms extracted from the same sweep open the space from 3 distinct
cells to 105.
[`arrangement_predictors.py --search`](../../src/tmatrix/aggregation/arrangement_predictors.py)
finds exactly two that are simultaneously lower in ρ than `a,d;b,c` **and**
further from a mirror than either failing cell under *both* metrics. `e,b;c,a`
is one of them, and it reuses three of the four original atoms, so only one new
T-matrix enters. The two hypotheses were therefore a factor of ten apart:

| | predicted MSE |
|---|---|
| ρ drives accuracy | ≤ 0.012 (at or below `a,d;b,c`) |
| symmetry drives accuracy | 0.12 – 0.17 (with the failing cells) |

## Verdict: neither. ρ is necessary, not sufficient

| cell | ρ | diag mm | all mm | MSE, complex S21 | mean \|ΔS21\| |
|---|---|---|---|---|---|
| `a,b;c,d` | 0.944 | 27.3 % | 18.8 % | 0.1654 | 0.308 |
| `a,c;d,b` | 0.944 | 20.0 % | 18.8 % | 0.1206 | 0.244 |
| **`e,b;c,a`** | **0.809** | **25.0 %** | **20.0 %** | **0.0612** | **0.181** |
| `a,d;b,c` | 0.854 | 9.1 % | 9.1 % | 0.0118 | 0.080 |

**`e,b;c,a` has the lowest ρ of the four and is 5.2× worse than the cell at
ρ = 0.854.** The ρ-only rule predicted ≤ 0.012 and got 0.0612, so it is
falsified as a *sufficient* predictor. The symmetry rule predicted 0.12–0.17
and got 0.0612, so it does not carry the cell either — and symmetry does not
order the four any better than ρ does: `a,c;d,b` at 20.0 % scores 0.1206, worse
than `e,b;c,a` at 25.0 %. Each predictor gets exactly one inversion out of four.

The ranking does not depend on the grid or on how the diffracted channels are
counted:

| cell | ρ | 25-point grid | 97-point grid | λ > 16 µm only (single channel) |
|---|---|---|---|---|
| `a,b;c,d` | 0.944 | 0.1654 | 0.1699 | 0.4166 |
| `a,c;d,b` | 0.944 | 0.1206 | 0.1225 | 0.3170 |
| **`e,b;c,a`** | **0.809** | **0.0612** | **0.0631** | **0.1294** |
| `a,d;b,c` | 0.854 | 0.0118 | 0.0120 | 0.0274 |

## What actually goes wrong: the two collective modes swap

The failure is not a smooth loss of accuracy. Both the prediction and CST have
the same two collective resonances, at ≈ 16.3 µm and ≈ 19.7 µm — and the
prediction puts the depth on the wrong one:

| feature | T-matrix prediction | direct CST |
|---|---|---|
| ≈ 16.3 µm | **0.145 (deep)** | 0.444 (shallow) |
| ≈ 19.7 µm | 0.431 (shallow) | **0.097 (deep)** |

So the aggregation gets which mode radiates strongly exactly backwards. CST's
19.74 µm resonance is 2.4 µm red-shifted from B's own 8 µm-lattice resonance
(17.34 µm) and is followed by a near-total transparency window
(|S21| = 0.970, |S11| = 0.028 at 17.93 µm); the prediction reproduces neither.
The error is confined to 15.8 ≤ λ ≤ 20.0 µm, where |ΔS21| reaches 0.68, and is
below 0.14 everywhere else.

By contrast `a,d;b,c` has a single broad dip (0.050 at 17.53 µm, essentially B's
own resonance shifted by 0.2 µm) and the aggregation places it to 0.11 µm.

### Raising the multipole order does not repair it, and cannot test it here

| lmax | MSE S21 | mean \|ΔS21\| | deepest predicted dip | min A |
|---|---|---|---|---|
| 3 | 0.0612 | 0.181 | 0.339 @ 11.10 µm | +0.030 |
| 4 | 0.0538 | 0.180 | 0.338 @ 11.10 µm | **−0.031** |
| 5 | 0.0683 | 0.195 | 0.307 @ 11.53 µm | **−0.135** |

The answer does not converge — and the absorption goes negative at lmax 4 and 5,
which a passive sheet cannot do. This is the confound `OPEN_QUESTIONS` §1
already records: the l = 4, 5 rows of these T-matrices sit at the extraction
noise floor and the lattice sum amplifies them, so an lmax sweep on this data
measures noise amplification, not convergence. It is evidence that lmax is not
a usable internal diagnostic; it is *not* evidence that the ρ = 0.809 truncation
has converged.

## What survives

* **ρ bounds the error at fixed composition; it does not order cells against
  each other.** Within one atom set it still works: A, B, C, D give
  0.944 → 0.1654 / 0.1206 and 0.854 → 0.0118, monotone. Across atom sets it
  fails outright.
* **The single-atom series is untouched and is now seven points.** E, F and G
  were extracted after the claim was made and land on the curve without
  adjustment (`../results_E_ewald_l3`, `_F_`, `_G_` — the packed sweep already
  held runs 1, 8 and 7, so this cost no full-wave time):

  | atom | ρ = 2r/d | MSE S21 | mean \|ΔS21\| | dip error |
  |---|---|---|---|---|
  | E | 0.539 | 0.00028 | 0.016 | +0.2 % |
  | C | 0.584 | 0.00038 | 0.017 | −0.3 % |
  | F | 0.629 | 0.00066 | 0.023 | −0.4 % |
  | A | 0.719 | 0.00107 | 0.030 | −0.7 % |
  | G | 0.809 | 0.00094 | 0.028 | −1.0 % |
  | B | 0.899 | 0.00393 | 0.054 | −4.1 % |
  | D | 0.989 | 0.03507 | 0.149 | −18.7 % |

  Dip error is monotone in ρ across all seven; MSE is monotone but for a 12 %
  A/G inversion. Note that a *single atom* at ρ = 0.809 (G) is accurate to
  1 % — the same ρ that leaves `e,b;c,a` 5× wrong. Same convergence ratio,
  two orders of magnitude apart in error. Whatever the four-atom cells are
  limited by, the pair convergence ratio is not the whole of it.

## Consequence for the proposed guard

`OPEN_QUESTIONS` §3 specifies a refuse-when-ρ-is-close-to-1 guard: warn above
0.85, refuse above 0.95. **That guard would have passed `e,b;c,a` silently** —
ρ = 0.809 is below even the warning threshold — on an answer that puts the
depth on the wrong collective mode. A ρ-only gate is therefore not safe. It
still catches the cases it was built for (0.944 and 0.989 both fail), so it is
worth having as a floor, but it must not be presented as a sufficient
acceptance test.

## Provenance

The three new T-matrices carry no geometry, so the mapping to sweep rows was
measured, not assumed: each file's own 8 µm-lattice prediction was scored
against every one of the ten runs in `test/2x2/SAW_gold_noSub_packed.cst`.

| file | assignment | scale | r (µm) | runner-up |
|---|---|---|---|---|
| `saw_gold_wl10p30um` | **E**, run 1 | 3.00 | 2.15784 | 135× worse |
| `saw_gold_wl11p60um` | **F**, run 8 | 3.50 | 2.51748 | 37× worse |
| `saw_gold_wl14p90um` | **G**, run 7 | 4.50 | 3.23676 | 69× worse |

Each `wl` in a file name is that atom's own transmission dip on its own 8 µm
lattice, which is what makes the naming self-identifying.

## Checks that passed

| check | result |
|---|---|
| independent implementation (treams) | max \|ΔS\| = 7.9e-13 — the disagreement with CST is physics, not code |
| port-plane de-embedding | L = 56.8131 µm from the empty-cell run, branch-resolved |
| empty-cell noise floor | \|S21 e^{jkL} − 1\| ≤ 2.5e-3 below 27 THz |
| closed diffraction channels | zero power in every group while evanescent |
| passivity, CST | max \|S11\|² + \|S21\|² + higher = 0.9932 |
| passivity, prediction (lmax 3) | A ∈ [+0.030, +0.276] |
| cross-polarization | \|t_xy\| ≤ 2.8e-12 — `OPEN_QUESTIONS` §2 again, on a new atom set |

## Reproduce

```bash
python -m tmatrix.aggregation.arrangement_predictors --search
```

```bash
python -m tmatrix.aggregation.run_supercell --cell 16 --site test/2x2/saw_gold_wl10p30um_10to34THz.tmat.h5 -4 4 --site test/2x2/saw_gold_wl17p30um_10to34THz.tmat.h5 4 4 --site test/2x2/saw_gold_wl10p90um_10to34THz.tmat.h5 -4 -4 --site test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5 4 -4 --lmax 3 --out results_2x2_EBCA_l3
```

```bash
python -m tmatrix.aggregation.cst_supercell.build_2x2_supercell --pair EBCA --only supercell
```

```bash
python -m tmatrix.aggregation.cst_supercell.read_supercell_results --run aggregation/cst_supercell/runs_EBCA/supercell/supercell.cst --empty aggregation/cst_supercell/runs/empty/empty.cst --out aggregation/results_2x2_EBCA_l3
```

## The follow-up this points to

One control would separate what is left. `e,d;c,a` has **exactly** `a,d;b,c`'s
ρ (0.854) and its worst pair (A–D), still contains D, but scores 25.0 % / 25.0 %
on the mirror metrics against `a,d;b,c`'s 9.1 % / 9.1 %. It is the same design
as the first experiment — fix the pair geometry, change the symmetry class —
run at the *good* ρ instead of the bad one. About an hour of CST:

```bash
python -m tmatrix.aggregation.cst_supercell.build_2x2_supercell --pair EDCA --only supercell
```

If it lands near 0.06 as well, symmetry is a real second axis and the pair
`(ρ, mirror mismatch)` orders all five cells. If it stays near 0.012, then
neither predictor is the story and what changed here is the atom set, not the
arrangement.
