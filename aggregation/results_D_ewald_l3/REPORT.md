# Atom D alone — where the method breaks, and why

This directory is a **documented failure**, kept because it is the most
informative negative result in the study. Atom D on its own 8 µm lattice sits at
the edge of the translation-addition theorem's useful convergence, and the
reconstruction fails in the way manual §6.6 anticipates.

| | |
|---|---|
| atom **D** | `saw_gold_wl23p50um_10to34THz.tmat.h5`, `scale` 5.50, r = 3.956 µm, CST run 3 |
| lattice | 8 µm square, the same as atoms A, B and C |
| stored diagnostics | residual 0.0045–0.0217, reciprocity 0.024–0.094, passivity **1.0489** |

## The condition, stated correctly

Manual Eq. (57): a conventional outgoing→regular spherical translation between
two scatterers requires ‖r_i − r_j‖ > a_i + a_j.

**That condition is satisfied here** — atom D has r = 3.956 µm, so 2a = 7.912 µm
against an 8 µm pitch. Nothing is formally invalid. What fails is the
*convergence rate*: the truncated addition theorem's error falls like

```
rho^lmax ,      rho = (a_i + a_j) / ‖r_i − r_j‖
```

and for D–D at 8 µm rho = **0.989**, so each extra multipole order buys 1 %. At
lmax 3 the series has essentially not started converging. An earlier draft of
this report said the condition was violated; it is not — it is marginally met,
which in practice is worse, because nothing warns you.

| atom | r (µm) | 2a (µm) | rho | rho³ | resonance, repo vs CST | mean \|ΔS21\| |
|---|---|---|---|---|---|---|
| C | 2.338 | 4.675 | 0.584 | 0.200 | 10.897 / 10.933 | 0.017 |
| A | 2.877 | 5.754 | 0.719 | 0.372 | 13.039 / 13.126 | 0.030 |
| B | 3.596 | 7.193 | 0.899 | 0.727 | 16.638 / 17.341 | 0.054 |
| **D** | **3.956** | **7.912** | **0.989** | **0.967** | **19.106 / 23.509 (−19 %)** | **0.149** |

The trend follows rho, and D fails outright: the array resonance
lands at 19.1 µm against 23.5 µm measured, a 19 % error, and the absorption goes
**negative** (A down to −0.114), which a passive sheet cannot do.

## Raising the multipole order does not repair it

Manual §6.6: *"If circumscribing spheres overlap, increasing L_i alone need not
repair the local expansion; use plane-wave-mediated coupling or a composite
cluster T-matrix."* Measured here:

| lmax | mean \|ΔS21\| vs CST | min A | dip position (CST: 23.509 µm) |
|---|---|---|---|
| 1 | 0.391 | +0.019 | 16.66 µm |
| 2 | 0.228 | +0.024 | 17.63 µm |
| **3** | **0.149** | −0.114 | 18.74 µm |
| 4 | 0.186 | −0.281 | 19.99 µm |
| 5 | 0.213 | −0.358 | 18.74 µm |

The error bottoms out at lmax 3 and then gets *worse*, while the passivity
violation grows monotonically. Two error sources move in opposite directions:
the translation truncation falls like 0.989^lmax — needing lmax of order 100 to
contract usefully — while the lattice sum amplifies the l = 4, 5 rows of T,
which sit at the extraction noise floor. The second wins long before the first
helps.

## Two independent codes agree on the wrong answer

This is the sharpest illustration in the study of what a cross-code check does
and does not buy. The independent treams implementation reproduces this
reconstruction to **1.2×10⁻¹⁵** on complex S21 — because treams uses the same
spherical addition theorem and inherits the same validity limit.

> Agreement with treams validates the **implementation**. Only the direct CST
> run validates the **physics**. A result can be numerically perfect and
> physically wrong, and this directory is what that looks like.

## What it does *not* invalidate

Atom D is fine as a constituent of a mixed cell, where its own species sits
11.31 µm away and its 8 µm neighbours are smaller atoms:

| pair at 8 µm | a_i + a_j | margin |
|---|---|---|
| A–B | 6.474 | +1.526 |
| A–C | 5.215 | +2.785 |
| A–D | 6.833 | +1.167 |
| B–C | 5.934 | +2.066 |
| **B–D** | 7.552 | **+0.448** |
| C–D | 6.294 | +1.706 |
| any same-species pair at 11.31 µm | ≤ 7.912 | ≥ +3.402 |

So the four-atom cells are usable, but with a caveat that distinguishes the two
arrangements: `a,b;c,d` puts B and D adjacent (margin +0.448 µm) while
`a,d;b,c` does not (worst +1.167 µm). See
[`../results_2x2_ABCD_l3/REPORT.md`](../results_2x2_ABCD_l3/REPORT.md) and
[`../results_2x2_ADBC_l3/REPORT.md`](../results_2x2_ADBC_l3/REPORT.md).

## The fix, if atom D on a dense lattice is ever needed

Manual §6.6 names the two options: plane-wave-mediated coupling (Theobald et al.)
or a composite cluster T-matrix that encloses the overlapping pair in one
circumscribing sphere. Neither is implemented here. A third practical option is
simply a larger pitch — at 10 µm the margin is +2.088 µm and D would behave like
atom A does at 8 µm.

## Files

`periodic_results.csv` / `.npz`, `floquet_orders.csv`, `treams_reference.npz`,
`cst_direct_reference.csv` (packed run 3), `run.json`, `fig1_sparams.png`,
`fig2_power.png`.
