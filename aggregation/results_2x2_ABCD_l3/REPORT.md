# `a,b;c,d` — four *different* atoms in one cell

The first cell in this study with four distinct species. Method and the full
validation ladder are in
[`../results_2x2_super_l3/REPORT.md`](../results_2x2_super_l3/REPORT.md); the
matched arrangement of the same four atoms is
[`../results_2x2_ADBC_l3/REPORT.md`](../results_2x2_ADBC_l3/REPORT.md). This
file carries the physics shared by both four-atom cells plus what is specific to
this one.

```
        A  B          A at (−4, +4)   B at (+4, +4)
        C  D          C at (−4, −4)   D at (+4, −4)
```

| | |
|---|---|
| A | `saw_gold_wl13p10um`, scale 4.00, r = 2.877 µm |
| B | `saw_gold_wl17p30um`, scale 5.00, r = 3.596 µm |
| C | `saw_gold_wl10p90um`, scale 3.25, r = 2.338 µm |
| D | `saw_gold_wl23p50um`, scale 5.50, r = 3.956 µm |
| cell | 16 × 16 µm, atoms 8 µm apart, lmax 3, Ewald coupling, 120 × 120 solve |
| worst Eq. (57) margin | **B–D at +0.448 µm** — see below |

## What changes when all four atoms differ

The two-species checkerboard was C4v about one atom site. With four distinct
species that symmetry is gone, and two of its consequences go with it.

**The dark diffraction orders switch on.** The coherent basis sum of manual
Eq. (64) for the (±1,0)/(0,±1) family is `i(F₁ − F₂ + F₃ − F₄)`. In a
checkerboard F₁ = F₄ and F₂ = F₃, so it cancelled identically; with four
different atoms it does not:

| cell | power in the odd (n1+n2) orders | first diffraction at |
|---|---|---|
| a,b;b,a, a,c;c,a, b,c;c,b | ≤ 5×10⁻³¹ | 26.50 THz (λ 11.31 µm) |
| **a,b;c,d** | **5.7×10⁻²** | **18.74 THz (λ 16.00 µm)** |
| **a,d;b,c** | **4.7×10⁻²** | **18.74 THz** |

The four-atom cells therefore start diffracting 7.8 THz lower than any
two-species cell built from the same atoms — the dark window between the two
Rayleigh onsets is gone.

**The cell becomes birefringent.** Solving separately for x- and y-polarized
incidence gives the 0th-order Jones matrix. It stays diagonal, but its two
entries separate:

| cell | max ‖t_xx\| − \|t_yy‖ | where | mean | max \|t_xy\| |
|---|---|---|---|---|
| a,b;c,d | **0.603** | 18.74 µm (\|t_xx\| 0.286 vs \|t_yy\| 0.890) | 0.176 | 1.8×10⁻¹² |
| a,d;b,c | 0.168 | 15.78 µm | 0.037 | 1.5×10⁻¹² |

So `a,b;c,d` is a strong linear polarizer near 18.7 µm while `a,d;b,c`, built
from the *same four atoms*, is nearly isotropic. Arrangement alone controls the
anisotropy.

**Cross-polarization stays zero, and that is not an artifact.** \|t_xy\| ≤ 2×10⁻¹²
in both arrangements even though the point group is C1. Displacing one atom off
its square site to (2.5, −5.5) makes the same code return \|t_xy\| = 1.0×10⁻²,
and a rectangular 16 × 20 lattice with the sites left square still gives
1.0×10⁻¹⁰ — so the cancellation is a property of the *square arrangement of the
four sites*, not an insensitivity of the method. It is reported here as an
empirical result: no symmetry argument for it is offered, and the direct CST run
measures the same channel independently (mode 2 of the Floquet port).

## Validation

| check | result |
|---|---|
| independent treams implementation, complex S21 | max 6.9×10⁻¹⁵ |
| linear-solve residual | 2.0×10⁻¹⁵ |
| A = 1 − R − T | [+0.045, +0.208], non-negative everywhere |
| power into higher orders, maximum | 0.409 |

## The caveat this arrangement carries

Atom D is the largest of the four and only just satisfies manual Eq. (57) on an
8 µm lattice — on its own lattice it fails outright, see
[`../results_D_ewald_l3/REPORT.md`](../results_D_ewald_l3/REPORT.md). In a mixed
cell each atom's own species is 11.31 µm away, so what matters is the unlike
pairs at 8 µm:

| arrangement | 8 µm pairs | worst margin |
|---|---|---|
| **a,b;c,d** | A–B, C–D, A–C, **B–D** | **+0.448 µm** |
| a,d;b,c | A–D, B–C, A–B, D–C | +1.167 µm |

`a,b;c,d` puts the two largest atoms next to each other; `a,d;b,c` does not.
Stated properly, Eq. (57) is *satisfied* in both — what differs is the
convergence ratio of the addition theorem, rho = (a_i + a_j)/d, whose truncation
error falls like rho^lmax: **0.944 here against 0.854** for `a,d;b,c`. On the
evidence of the single-atom series this arrangement should be the less reliable
of the two, and the direct CST runs test that.

Note that rho and arrangement symmetry are confounded across these two cells;
[`../../OPEN_QUESTIONS.md`](../../OPEN_QUESTIONS.md) §1 records the third
arrangement that separates them.

## Against the direct CST supercell run

6 565 s (1.8 h), 63 adaptive frequency points, 4.2 M tetrahedral DOF. Built by
`build_2x2_supercell.py --pair ABCD`, same settings as every other cell, sharing
the `runs/empty` companion run for de-embedding.

| | MSE | max \|Δ\| | mean \|Δ\| |
|---|---|---|---|
| complex S21 | **0.1654** | 0.840 | 0.308 |
| complex S11 | 0.1660 | 0.833 | 0.308 |
| … S21 below 20 THz | **0.3807** | | 0.580 |
| … S21 above 20 THz | 0.0218 | | 0.126 |

| feature | this repo (refined grid) | direct CST | offset |
|---|---|---|---|
| D-derived transmission dip | 17.63 µm, \|S21\| 0.081 | 16.00 µm, 0.104 | **−10.2 %** |
| A/B-derived dip | 13.03 µm, \|S21\| 0.331 | 13.49 µm, 0.472 | +3.4 % |
| diffracted power, maximum | 0.409 | 0.388 | 5 % relative |
| dark-channel check | — | odd family **carries** 0.388 | as predicted |

**This is the worst-agreeing cell in the study**, and the error is concentrated
below 20 THz (MSE 0.381 against 0.022 above — a factor 17), which is exactly
where atom D resonates and where its rho = 0.944 pair coupling with B is least
converged. Above 20 THz, where the smaller atoms dominate, the agreement is
ordinary.

Compare `a,d;b,c` — the same four atoms rearranged so that D's 8 µm neighbour is
A (rho = 0.854) instead of B — which reaches MSE 0.0118 and places both dips to
under 1 %. Composition, density and lattice are identical; only which atom sits
next to which differs, and it costs a factor **14** in MSE.

The diffraction bookkeeping is nonetheless right even here: the odd orders that
the checkerboards extinguished carry 0.388 in the direct run against 0.409
predicted, confirming the symmetry-breaking prediction independently of the
0th-order disagreement.

## Files

Same layout as `../results_2x2_super_l3/`, plus `jones_xy.npz` — the 0th-order
\|t_xx\|, \|t_yy\|, \|t_xy\| for both four-atom arrangements across the band.
`../results_2x2_ABCD_fine/` holds the same sweep on a 4× refined grid.
