# `a,d;b,c` — the same four atoms, rearranged

The matched partner of [`../results_2x2_ABCD_l3/REPORT.md`](../results_2x2_ABCD_l3/REPORT.md):
identical composition, identical lattice, identical filling fraction, different
assignment of atoms to sites. The physics shared by both four-atom cells — the
odd orders switching on, the birefringence, the vanishing cross-polarization — is
described there; this file records what differs.

```
        A  D          A at (−4, +4)   D at (+4, +4)
        B  C          B at (−4, −4)   C at (+4, −4)
```

| | |
|---|---|
| cell | 16 × 16 µm, atoms 8 µm apart, lmax 3, Ewald coupling, 120 × 120 solve |
| worst Eq. (57) margin | **+1.167 µm** (A–D), against +0.448 µm for `a,b;c,d` |

## Arrangement is a design variable

Same four atoms, same 16 µm cell, same 8 µm spacing — and a different
metasurface:

| f (THz) | λ (µm) | a,b;c,d \|S21\| | a,d;b,c \|S21\| | \|Δ\| |
|---|---|---|---|---|
| 12 | 24.98 | 0.734 | 0.672 | 0.067 |
| 14 | 21.41 | 0.555 | 0.527 | 0.334 |
| 18 | 16.66 | 0.609 | 0.383 | 0.248 |
| 20 | 14.99 | 0.486 | 0.677 | 0.315 |
| 26 | 11.53 | 0.549 | 0.861 | **0.350** |
| 30 | 9.99 | 0.777 | 0.666 | 0.117 |

Rearranging the same constituents changes \|S21\| by up to **0.429** (mean 0.159).
Nothing about composition or density has changed; only which atom sits next to
which. This is the cleanest statement in the study of why aggregation is a
multiple-scattering solve and not a sum: `Σ_s T_s` is identical for the two
cells and cannot distinguish them.

Two further differences worth naming:

* **Anisotropy.** `a,b;c,d` reaches ‖t_xx\| − \|t_yy‖ = 0.603 near 18.7 µm;
  this arrangement reaches only 0.168 (mean 0.037 against 0.176). Put the two
  most similar atoms — B (scale 5.00) and D (scale 5.50) — on opposite corners
  and the cell is nearly isotropic; put them side by side and it is a strong
  linear polarizer.
* **Absorption.** This cell reaches A = 0.487 at 21.4 µm, the largest of any cell
  in the study, against 0.208 for `a,b;c,d`.

## Validation

| check | result |
|---|---|
| independent treams implementation, complex S21 | max 1.2×10⁻¹⁴ |
| linear-solve residual | 3.9×10⁻¹⁵ |
| A = 1 − R − T | [+0.048, +0.487], non-negative everywhere |
| power in the odd (n1+n2) orders | 4.7×10⁻² — bright, as for `a,b;c,d` |
| power into higher orders, maximum | 0.429 |
| max cross-polarized \|S\| | 1.5×10⁻¹² |

## Why this arrangement should be the more trustworthy one

Atom D only just satisfies manual Eq. (57) at 8 µm — satisfied, but with the
addition theorem's convergence ratio rho = (a_i + a_j)/d close to 1, so the
truncation error rho^lmax barely contracts. `a,b;c,d` places B and D adjacent
(rho = 0.944); this arrangement keeps them apart and its worst pair is A–D
(rho = 0.854), comparable to A–B in the `a,b;b,a` cell, which reconstructed
well. So `a,d;b,c` should agree better with CST than `a,b;c,d` does.

Caveat worth carrying: this arrangement is also the most nearly mirror-symmetric
of the three, because the pair it puts on a diagonal (B, D) is the pair closest
in size. Geometry and symmetry are therefore confounded in this comparison —
[`../../OPEN_QUESTIONS.md`](../../OPEN_QUESTIONS.md) §1 specifies the third
arrangement that separates them.

## Against the direct CST supercell run

17 469 s (4.9 h), 95 adaptive frequency points, ~5.7 M tetrahedral DOF — the
longest and heaviest solve in the study. Built by
`tmatrix.aggregation.cst_supercell.build_2x2_supercell --pair ADBC`, same
settings as every other cell, sharing the `runs/empty` companion run.

| | MSE | max \|Δ\| | mean \|Δ\| |
|---|---|---|---|
| complex S21 | **0.0118** | 0.347 | 0.080 |
| complex S11 | 0.0115 | 0.342 | 0.079 |
| R / T / A | — | 0.349 / 0.141 / 0.317 | 0.063 / 0.045 / 0.047 |
| … S21 below 20 THz | 0.0264 | | 0.139 |
| … S21 above 20 THz | **0.0020** | | 0.041 |

| feature | this repo (refined grid) | direct CST | offset |
|---|---|---|---|
| D-derived transmission dip | 17.38 µm, \|S21\| 0.114 | 17.53 µm, 0.050 | **+0.85 %** |
| A/B-derived dip | 13.03 µm, \|S21\| 0.247 | 13.08 µm, 0.230 | **+0.38 %** |
| transparency peak between them | 14.80 µm, \|S21\| 0.682 | 15.00 µm, 0.766 | +1.31 % |
| diffracted power, maximum | 0.429 | 0.415 | 3 % relative |

**The prediction made before either run finished holds.** `a,d;b,c` was expected
to land between the two-species cells (MSE 0.0007–0.0037) and `a,b;c,d`
(0.1654) on the strength of its better worst-pair convergence ratio, and it
does: **MSE 0.0118**, a factor **14** better than `a,b;c,d` from the same four
atoms (3.8× on the mean absolute error — MSE separates them further because
`a,b;c,d` fails by badly misplacing a few resonances rather than by drifting
across the band). Both of its dips are placed to under 1 %, against −10.2 % for
the D-derived dip in `a,b;c,d`.

Above 20 THz the agreement is MSE 0.0020 — indistinguishable from the
well-behaved two-species cells. The error is concentrated below 20 THz, where
atom D resonates and where its rho = 0.854 pair coupling is least converged. So
even with the widest gap of the three arrangements there is residual translation
error; it simply is not catastrophic.

Why the solve took 4.9 h against 1.8 h for `a,b;c,d`: 95 frequency points
against 63, and 5.7 M DOF against 4.2 M. The adaptive sweep put 16 points into
18–20 THz alone, because this cell's D-derived resonance sits almost on top of
the bright (±1,0) diffraction edge at 18.74 THz — a resonance on a Rayleigh edge
is the hardest thing an interpolating sweep can be asked to resolve. With
`FreqDistAdaptMode "Distributed"` the extra points also inflate the merged mesh,
so the two costs multiply rather than add.

## Files

Same layout as `../results_2x2_super_l3/`; `../results_2x2_ADBC_fine/` holds the
same sweep on a 4× refined grid, and the Jones-matrix data for both four-atom
arrangements is in `../results_2x2_ABCD_l3/jones_xy.npz`.
