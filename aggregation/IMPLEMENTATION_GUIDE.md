# Building a T-Matrix Array-Aggregation Pipeline from Scratch
### A rookie-facing implementation manual, with all the technical details

This document explains, step by step, how the code in this folder was built:
from an empty directory to a pipeline that takes one unit cell's T-matrix
(extracted from CST) and predicts the S-parameters of a whole metasurface —
validated to 0.3 % against a direct CST periodic simulation.

It is written for someone who knows electromagnetic basics (Maxwell, plane
waves, S-parameters) but has never touched vector spherical waves or
multiple-scattering theory. Nothing is assumed beyond that; everything else
is defined when first used. At the same time, no technical detail that
mattered in practice is left out — including the mistakes.

---

## 0. The big idea (why this works at all)

A metasurface simulation is expensive because the full structure has millions
of unknowns. But Maxwell's equations are **linear**, so the scattering of one
isolated unit cell ("atom") is completely characterized by a single linear
operator: give me the incoming field, I return the outgoing field. If we
expand fields in a good basis, that operator becomes an ordinary matrix — the
**transition matrix (T-matrix)**:

    f = T a

* `a` — coefficients of the *incoming* field in a basis of **regular** waves
  (finite at the origin),
* `f` — coefficients of the *outgoing* (scattered) field in a basis of
  **outgoing** waves (radiating, singular at the origin) — the tmat.h5
  format calls this vector `p`; this guide writes `f`,
* `T` — a small matrix that encodes *everything* about the atom's geometry
  and material at one frequency. Truncating the basis at maximum multipole
  order lmax = 3 (the file's choice) gives n = 2·lmax·(lmax+2) = 30 modes,
  so T is 30×30 here.

The expensive full-wave solver (CST) is used **once per atom** to find `T`.
After that, an array of thousands of atoms is pure linear algebra:

1. each atom is excited by the incident wave *plus* the outgoing waves of all
   other atoms, translated to its own origin (**Foldy–Lax equations**);
2. the summed outgoing waves of the whole array, projected onto plane waves,
   give S11 and S21.

The two deliverables of this project are exactly those two steps:
**aggregation** (`aggregate.py`, `translate.py`) and **S-parameter
extraction** (`sparams.py`). Everything else exists to make those two steps
*provably correct*.

The basis that makes this work is the **vector spherical wave functions
(VSWFs)** — the electromagnetic analogue of spherical harmonics. Section 2
builds them. But first, the most important non-mathematical lesson of the
whole project.

---

## 1. Rule zero: conventions are the whole game

A T-matrix is just a table of numbers. Those numbers are meaningless unless
you know *exactly* which basis functions, normalizations, and sign
conventions the extraction code used. Every textbook differs in at least one
of:

* time convention: fields ~ e^(−iωt) (physics) vs e^(+jωt) (engineering) —
  this flips every complex conjugate and turns h_l^(1) into h_l^(2)
  (h_l^(1,2) = spherical Hankel functions, the outgoing/incoming radial
  waves);
* Condon–Shortley phase: an extra (−1)^m inside the spherical harmonics;
* normalization of the vector harmonics (Jackson vs Tsang vs Mishchenko all
  differ by factors of i^±1, √(l(l+1)), √(4π));
* mode ordering and the meaning of "electric"/"magnetic".

Our input file is a standard `tmat.h5` and *declares* its convention in the
root attribute:

> time dependence exp(−iωt), outgoing radial functions h_l^(1),
> Jackson-normalized vector spherical waves with Condon–Shortley phase,
> parity basis; p = T a with modes as listed in /modes.

("Parity basis" means the modes are labeled TE/"magnetic" vs TM/"electric" —
the M and N waves of Section 2.3 — as opposed to the helicity basis, which
uses circular-polarization combinations of the two. The file's `p` is our
`f`.)

The implementation strategy that follows from rule zero:

1. **Write down one convention explicitly** (docstring of `vswf.py`) and
   implement *everything* — plane waves, far fields, translations, mirrors —
   in that single convention.
2. **Never trust a formula from a paper without a numerical test.** Every
   layer of this code has a test that would catch a wrong sign or a wrong
   factor of i (Section 7). Two of these tests failed during development and
   caught real convention bugs (Sections 2.4 and 6.2).
3. **Never assume the input file matches your convention — verify it.**
   Section 7.4 shows three checks (passivity, reciprocity, positive
   absorption) that confirm the file's T really is in the convention you
   implemented.

---

## 2. Layer 0 — the VSWF engine (`vswf.py`)

### 2.1 Scalar spherical harmonics and the two angular helper functions

We use orthonormal spherical harmonics with Condon–Shortley phase — exactly
what `scipy.special.sph_harm_y(l, m, θ, φ)` returns. From them we need two
derived angular functions for every (l, m):

    τ_lm(θ,φ) = ∂Y_lm/∂θ
    π_lm(θ,φ) = m·Y_lm / sin θ

`τ` is computed with an exact **ladder identity** (no finite differences!):

    ∂θ Y_lm = ½[ √((l−m)(l+m+1)) e^(−iφ) Y_{l,m+1}
               − √((l+m)(l−m+1)) e^(+iφ) Y_{l,m−1} ]

which follows from the angular-momentum raising/lowering operators
L± = e^(±iφ)(±∂θ + i·cotθ·∂φ). `π` is computed by direct division — safe
because Y_lm ~ sin^|m|θ near the poles, so the ratio is finite. The only
place division would blow up is *exactly at* θ = 0 or π, which matters
because the S-parameter formulas evaluate the far field exactly on-axis. The
fix is a **pole nudge**: clip θ into [ε, π−ε] with ε = 1e−9. The error this
introduces is O(ε²) ≈ 1e−18 — far below anything else.

### 2.2 Vector spherical harmonics X and Z

Jackson's transverse vector harmonic is built from the orbital angular
momentum operator L = −i r×∇:

    X_lm = L Y_lm / √(l(l+1))
         = [ −π_lm θ̂ − i τ_lm φ̂ ] / √(l(l+1))

and its 90°-rotated partner:

    Z_lm = r̂ × X_lm = [ i τ_lm θ̂ − π_lm φ̂ ] / √(l(l+1))

X and Z are orthonormal *tangential* vector fields on the sphere — that is
what later makes projections (Section 2.6) a simple angular integral.

**How we made sure these are right:** a one-off scratch check during
development computed L Y_lm by finite differences at random points and
compared against the implementation (agreement ~2e−10). It is not kept in
the committed suite — the curl test of Section 2.4 covers X and Z
indirectly and permanently.

### 2.3 The wave functions M and N

With radial functions z_l = j_l (regular, superscript (1)) or z_l = h_l^(1)
(outgoing, superscript (3)):

    M_lm^(c)(r) = z_l^(c)(kr) X_lm(θ,φ)                     ← "magnetic", TE
    N_lm^(c)(r) = (1/k) ∇ × M_lm^(c)                        ← "electric", TM
                = i·√(l(l+1)) · (z_l(kr)/kr) · Y_lm · r̂  +  ξ_l(kr) · Z_lm

(The superscript (c) selects the radial function — c = 1 regular, c = 3
outgoing — and is unrelated to the Hankel-function superscript in h_l^(1).)

where ξ_l(x) = [x·z_l(x)]′/x = z_l′(x) + z_l(x)/x.

### 2.4 War story #1: the missing factor of i

The first version of the code had the *textbook* (Tsang-style) N-wave:
`N = √(l(l+1))(z/x)Y r̂ + ξ Z` — **no i on the radial term**. The
finite-difference curl test (which checks N ≡ ∇×M/k mode by mode) failed at
the 30 % level, and the per-component diagnosis showed the radial part off by
*exactly* i while the tangential part was exact.

The reason: most references define their vector harmonics from ∇Y — real
θ-functions τ, π with the e^(imφ) factor split off (Tsang convention) —
while Jackson's X carries the −i inside L = −i r×∇. Chasing
the curl through Jackson's definition really does produce an i on the radial
term. The convention declared by the file forces Jackson's X — so the i
belongs there. **Lesson: implement the defining relation (N = ∇×M/k), test
against it numerically, and let the test — not the textbook — settle the
phase.**

### 2.5 Magnetic fields for free (duality)

With e^(−iωt), Maxwell gives H = ∇×E/(iωμ). Define the scaled magnetic field
H̃ ≡ i·Z₀·H = ∇×E/k. Then, because ∇×M = kN and ∇×N = kM:

    E = M  ⟹  H̃ = N          E = N  ⟹  H̃ = M

So the magnetic field of every mode is just *the other* mode — no extra
code. This "duality swap" is used everywhere (projections, plane waves).

### 2.6 Projecting a field onto regular waves (the workhorse)

Given E and H̃ sampled on a sphere of radius r₀, we recover the regular
expansion coefficients c of the field. Because X and Z are orthonormal and M
has no Z-component while N has no X-component:

    ⟨X_lm, E⟩  = c^M_lm · j_l(kr₀)        ⟨Z_lm, E⟩  = c^E_lm · ξ_l(kr₀)
    ⟨X_lm, H̃⟩ = c^E_lm · j_l(kr₀)        ⟨Z_lm, H̃⟩ = c^M_lm · ξ_l(kr₀)

(⟨·,·⟩ = ∫ conj(·)·(·) dΩ). Each coefficient is measured **twice** (once via
E, once via H̃); we combine them in least squares:

    c^M = (j_l·⟨X,E⟩ + ξ_l·⟨Z,H̃⟩) / (j_l² + ξ_l²)

This matters because j_l(kr₀) has zeros — with only the E-projection the
inversion would blow up at those radii; j and ξ never vanish simultaneously,
so the combined estimate is always well conditioned. (This is also exactly
what the CST extraction driver did, per its own metadata.)

The integral uses a **Gauss–Legendre × uniform-φ product grid**
(`sphere_quadrature`): GL in cosθ with n_θ nodes integrates polynomials up to
degree 2n_θ−1 exactly, and n_φ uniform azimuthal points kill all Fourier
modes |Δm| < n_φ exactly. A 16×32 grid (512 points) is exact far beyond our
l ≤ 3 content; the residual "aliasing" error from high-l field content is
~1e−6 (measured — see the lattice-sum quadrature-stability row of the test
table in Section 7). For speed, all projections are precomputed into two
matrices (X̄W, Z̄W) so each projection is a single GEMM (one dense
matrix–matrix product) — see `RegularProjector`.

### 2.7 Plane-wave expansion coefficients

The incident wave E = ê·e^(ik·r) must be expressed in regular VSWFs. Using
the scalar expansion e^(ik·r) = 4π Σ i^l j_l(kr) Y*_lm(k̂) Y_lm(r̂), the
Hermiticity of L, and the duality swap for the electric part:

    a^M_lm = 4π i^l     · X*_lm(k̂) · ê
    a^E_lm = 4π i^(l−1) · Z*_lm(k̂) · ê

The i^(l−1) (not i^(l+1)!) comes out of the H̃-based derivation:
H̃ = i k̂×ê e^(ik·r), and (k̂×ê)·X* = −ê·Z*. This exact sign chain was also
verified numerically: reconstruct ê·e^(ik·r) from the expansion at lmax = 12
and compare pointwise (3e−8 agreement, for E *and* H̃).

### 2.8 Far-field amplitude

For large argument h_l^(1)(x) → (−i)^(l+1) e^(ix)/x, and the curl in the far
zone becomes ik r̂×. Defining E_sca → F(r̂)·e^(ikr)/r:

    F(r̂) = (1/k) Σ_lm [ f^M_lm (−i)^(l+1) X_lm(r̂) + f^E_lm (−i)^l Z_lm(r̂) ]

Test: evaluate the outgoing fields directly at kr = 1e5 and compare (5e−5,
limited by the asymptotic expansion itself, scales as 1/(kr)).

### 2.9 The capstone test of Layer 0: a Mie sphere

For a homogeneous sphere the T-matrix is known analytically (Mie theory,
diagonal: T^E_l = −a_l, T^M_l = −b_l). The test builds the Mie T for a
lossless dielectric sphere, runs it through *our* plane-wave coefficients and
far-field code, and checks three numbers against each other:

* σ_ext from the **optical theorem**: σ_ext = (4π/k)·Im[ê*·F(k̂_inc)],
* σ_sca from integrating |F|² over the sphere,
* σ_sca from the classic Mie series (2π/k²)·Σ_l (2l+1)(|a_l|²+|b_l|²).

For a lossless sphere all three must agree exactly. They agree to **3e−15**
(machine precision). This one test pins simultaneously: the plane-wave
coefficients, both far-field phases, the optical-theorem sign (and therefore
the time convention), and the mode bookkeeping. If you build such a pipeline
yourself, *make this test exist before anything else downstream*.

---

## 3. Layer 1 — translation operators (`translate.py`)

### 3.1 What they are

Atom j scatters; atom i sees that scattered field as part of *its* incoming
field. The outgoing waves of j must therefore be re-expanded as **regular**
waves around i. That re-expansion is linear, so it is a matrix:

    V^(3)_ν(ρ − d) = Σ_μ A(d)[μ,ν] · V^(1)_μ(ρ),    valid for |ρ| < |d|

with d = r_source − r_target; V_ν stands for M or N as selected by the mode
index ν, and the superscripts (1)/(3) mean regular/outgoing as in
Section 2.3. The classical route to A(d) is the
translation-addition theorem (Stein/Cruzan formulas with Wigner/Gaunt
coefficients) — powerful but notoriously convention-fragile: one wrong
(−1)^m and everything downstream is silently wrong.

### 3.2 The convention-safe choice: numerical projection

We already have (Section 2.6) a machine that converts "field sampled on a
sphere" into "regular coefficients". So compute A(d) columnwise *from its
definition*: evaluate the E and H̃ fields of outgoing mode ν centered at d on
a sphere of radius r₀ < |d| around the target, and project. No Gaunt
coefficients, no new conventions — A(d) is exact by construction relative to
the same `vswf.py` that defines everything else. Cost: one batched GEMM per
displacement, ~30 modes × 512 points.

Two tests lock it:

* **Re-expansion test**: random outgoing combination at d, re-expanded with a
  generous row basis (lmax = 12), reproduces the field at interior points to
  2e−7. Important subtlety: the truncated addition theorem converges like
  (r/d)^(l − l_src), where l is the order of the regular re-expansion mode
  and l_src the order of the source mode — the error only starts falling
  once l exceeds l_src. So exactness is only visible with a *large* row
  basis at *small* test radius; at lmax = 3 the reconstruction error near
  the validity boundary is O(10 %) *and that is not a bug*, it is the
  truncation the whole T-matrix method lives with.
* **r₀-invariance**: A(d) is an expansion-coefficient object; computing it on
  spheres of different radii must give the same matrix (3e−7, the
  conditioning floor of the projection).

### 3.3 The rotation identity (the key speed trick)

For displacements rotated about z by φ, with our e^(imφ) azimuthal
convention:

    A(Rot_φ d)[μ,ν] = e^(i(m_ν − m_μ)φ) · A(d)[μ,ν]

Proof sketch: rotating a VSWF about z multiplies it by e^(−imφ); write the
translated field of the rotated configuration as a rotation of the original
translated field and match coefficients. Verified numerically to 4e−10.

Consequence: for any set of in-plane displacements, only **one projection per
distinct radius** is needed; all other angles are diagonal phase fix-ups.
On a square lattice with thousands of sites, that is a ~10× reduction
(number of distinct radii ≈ number of distinct integers expressible as
i²+j²). The same identity holds with a constant z-offset (needed for the
ground-plane image lattice, Section 6), because rotation about z leaves the
offset untouched.

---

## 4. Layer 2 — aggregation (`aggregate.py`)

### 4.1 Foldy–Lax: the self-consistent array equations

The *exciting* field of atom i = incident field + scattered fields of all
other atoms, each translated to i's origin:

    a^i = a_inc^i + Σ_{j≠i} A(R_j − R_i) f^j,        f^j = T^j a^j

Substituting f = Ta turns this into one global block-linear system:

    (I − A_blk · diag(T¹...T^N)) · (a¹...a^N) = (a_inc¹...a_inc^N)

`build_finite_system` assembles exactly this N·n × N·n matrix (n = 30 modes),
caching one translation matrix per distinct displacement radius and rotating
per pair. `solve_finite` solves it by dense LU. This is "aggregating each
atom's T-matrix into one array matrix" — deliverable 1, finite-array form.
Atoms may have *different* T-matrices (pass a list) and arbitrary in-plane
positions.

### 4.2 The infinite periodic array

At normal incidence every cell of an infinite lattice is excited
identically (Bloch phase = 1), so a^i ≡ a and the block system collapses to
one cell:

    a = a_inc + C·T·a   ⟹   a = (I − C·T)^(−1) a_inc,   f = T·a

with the **lattice sum** C = Σ_{R≠0} A(R). One 30×30 solve per frequency.
The "effective array T-matrix" is T_eff = T(I − CT)^(−1): it maps the bare
incident coefficients directly to the per-cell outgoing coefficients with
all multiple scattering resummed. (Check the algebra with the push-through
identity: T(I−CT)^(−1) = (I−TC)^(−1)T.)

### 4.3 The lattice sum, and war story #2: algebraic taper convergence

C is a sum of h_l-type terms decaying like 1/R over a 2D lattice — the
number of sites per ring grows like R, so the terms of the ring-sum do not
decay at all: the series is only **conditionally convergent**. The physical
value is the Abel limit (add infinitesimal absorption, let it → 0).

First attempt: multiply each term by a Gaussian taper w(R) = e^(−R²/Rc²) and
grow Rc until converged. The convergence test *failed*: changing Rc from
32 µm to 44 µm still moved C by ~2e−3 relative. Understanding why is a nice
little exercise in asymptotics. The tapered radial integral of the
propagating channel is

    ∫₀^∞ e^(ikR) e^(−R²/Rc²) dR = (√π Rc/2)e^(−k²Rc²/4) + i·Rc·D(kRc/2)

where D is the **Dawson function**. The first term is exponentially small —
that is the naive expectation. But D(x) ~ 1/(2x) + 1/(4x³) + ⋯, so the
second term approaches the Abel value i/k only **algebraically**, with an
error series in even powers of 1/(kRc). (Equivalent frequency-domain view:
the taper convolves the k-space spectrum with a Gaussian of width 2/Rc at a
finite distance k from the light-circle pole — a smooth-function error,
polynomial in the width.)

The fix follows directly from the error structure: since
C(Rc) = C∞ + c₂/Rc² + c₄/Rc⁴ + ⋯, evaluate C at three taper lengths
(kRc = 10, 14, 20) and **Richardson-extrapolate in 1/Rc²**. Because all
tapers share the same shell matrices A_s (only the scalar weights differ),
the two extra tapers are nearly free. Result: taper-set stability of the
extrapolated C is ~2e−6 entrywise. The evanescent (g ≠ 0) channels never
were a problem — for a deeply subwavelength lattice they converge
exponentially.

(Here g denotes the reciprocal-lattice vectors of the array — the
diffraction orders. For a subwavelength pitch every g ≠ 0 order is
evanescent, decaying exponentially away from the array plane.)

Two further practical notes:

* **Truncate generously**: sum real-space shells to 3.5·Rc_max, where the
  taper is e^(−12.25) ≈ 5e−6 — truncation must stay below the extrapolation
  accuracy, and it is *not* smooth in Rc, so it would poison Richardson.
* **Measure convergence entrywise**, normalizing each entry by
  (|C_entry| + radiative scale 2π/(kA)). C spans ~8 orders of magnitude
  (quasi-static near-field entries reach 1e6); a global max-abs metric only
  sees irrelevant relative jitter of the huge entries.

### 4.4 Sanity anchor for the whole layer

During the independent code-review pass, an ad-hoc end-to-end script (not
retained in the committed suite) pushed a lossless Mie-sphere array through
lattice sum → periodic solve → S-parameters: energy was conserved to 1e−7.
Energy conservation is *not* built into any single formula — it only
emerges if C, T, the solve, and the S-projection are all mutually
consistent — which makes this the cheapest brutal test of the whole layer.
(The committed suite covers the same ground with the lossless *mirror*
unitarity test of Section 6, which is even stricter.)

---

## 5. Layer 3 — S-parameters (`sparams.py`)

### 5.1 From a lattice of spherical waves to two plane waves

Every cell radiates the same far-field pattern F(r̂) (Section 2.8). Summing
e^(ik|r−R|)/|r−R| over a subwavelength lattice, only the 0th diffraction
order propagates; the ring-integral evaluation (u-substitution
s = √(u²+z²), Abel limit for the oscillatory tail) gives the classic
identity

    Σ_R e^(ik|r−R|)/|r−R|  →  (2πi/(A·k_z)) · e^(i k_z |z|)

so the array's radiation *is* a pair of plane waves with amplitude
(2πi/(A·k_z))·F(±ẑ). Adding the direct wave for transmission:

    S21 = 1 + (2πi/(k·A·cosθ)) · ê* · F(k̂_fwd)
    S11 =     (2πi/(k·A·cosθ)) · ê* · F(k̂_spec)

(at normal incidence cosθ = 1, k̂_fwd = +ẑ, k̂_spec = −ẑ). These are the
manual's equations with j → i — the manual's `+j` is only correct in the
physics convention; in a true e^(+jωt) engineering convention it must be −j.
For a finite array, replace A by N·A_cell and F by the phase-referenced sum
of per-cell patterns (at normal incidence with in-plane cells the phases are
all 1). Cross-polarization uses ê_cross in the same formulas; energy balance
is A = 1 − |S11_co|² − |S11_×|² − |S21_co|² − |S21_×|².

### 5.2 Cross sections as convention sentinels

For the single atom, `cross_sections` computes σ_ext (optical theorem),
σ_sca (|F|² integral) and σ_abs = σ_ext − σ_sca. For a passive scatterer
σ_abs ≥ 0 **only if the time convention of T matches the code** — if the
file were secretly e^(+jωt), T would be conjugated and σ_abs would come out
negative. On the demo file: positive across the band. Sentinel passed.

---

## 6. Bonus layer — ground plane by image theory (`mirror.py`)

The demo atom is an absorber *designed to sit over a metal ground plane*;
isolated, it has no resonance in the band (Section 8.2). Image theory adds
the ground plane exactly for a PEC mirror at z = −h:

1. **Image map.** The image of outgoing mode ν at the origin is an outgoing
   field at (0,0,−2h) with a per-mode sign: E_im(c_im+ρ) = D·V_ν(σρ) =
   s_ν·V_ν(ρ), D = diag(−1,−1,1), σ = diag(1,1,−1). The signs are computed
   *numerically* (evaluate both sides at random points, mode by mode) and
   turn out to be s = +(−1)^(l+m) for magnetic, −(−1)^(l+m) for electric.
   **War story #3:** the hand-guessed analytic pattern had the two
   polarizations swapped; the numerical determination caught it instantly.
   Guessing parity signs by hand is a mug's game — compute them.
2. **Coupling operator.** R_m = A((0,0,−2h))·diag(s): my own image excites
   me. Then f = (I − T·R_m)^(−1) T a_inc,tot — which is *precisely* the
   manual's substrate formula T_coupled = (I − T_iso R)^(−1) T_iso with the
   image operator playing R.
3. **Periodic case.** a = a_inc,tot + [C + C_im·diag(s)]·f, where C_im sums
   image translations over *all* lattice vectors including R = 0 (own
   image), using the same shell machinery with a z-offset.
4. **Excitation and S11.** The "incident" field over a mirror is the
   incoming wave plus its specular ground reflection (r_g = −e^(2ikh));
   the reflected amplitude assembles the ground term, the direct sheet, and
   the image sheet with its extra propagation phase e^(2ikh).

**The decisive test:** a *lossless* Mie-sphere array over a PEC mirror must
reflect everything: |S11|² + |S11_×|² = 1. Any sign or phase slip anywhere in
the image bookkeeping breaks this. Result: R = 1.000000. When a single
scalar test constrains this many moving parts at once, build it.

Physical caveat, stated honestly in the results: the demo geometry has
2h = 0.70 µm while the cell's circumscribing radius is 0.72 µm — the image
sits at the edge of the re-expansion's validity sphere (the
"Rayleigh-hypothesis" regime: the expansion is being used at distances
where its convergence is no longer guaranteed, because the image source
lies inside the cell's circumscribing sphere), and the real design's
dielectric spacer is not in the file, so the
mirror demo is *qualitative*: an absorption resonance appears at ~13 µm for
the design spacing and disappears when the spacing changes — the MIM physics
emerges, with expected quantitative shifts.

---

## 7. The validation pyramid (how you *know* it's right)

The tests were not an afterthought — each layer was frozen only when its
tests passed, so bugs always localized to the newest layer. Bottom to top:

| section | test | catches | result |
|---|---|---|---|
| 0 | FD curl: H̃ = ∇×E/k per mode | wrong N/M definition, the i factor | 2e−9 |
| 0 | plane-wave reconstruction (E and H̃) | wrong a^M/a^E phases | 3e−8 |
| 0 | far field vs direct at kr = 1e5 | wrong (−i)^l phases | 5e−5 |
| 0 | projection round trip | quadrature, LSQ combination | 1e−13 |
| 0 | **Mie: optical theorem = \|F\|² = Mie series** | *everything above at once* | 3e−15 |
| 1 | translation re-expansion (lmax 12) | projection-based A(d) | 2e−7 |
| 1 | A(d) r₀-invariance | hidden r₀ dependence | 3e−7 |
| 1 | rotation identity | phase convention of the shells trick | 4e−10 |
| 2 | lattice-sum taper/quadrature/r₀ stability | conditional-convergence handling | ≤2e−6 |
| 2 | lossless Mie array energy conservation (ad-hoc, review pass) | any C/T/solve/S inconsistency | 1e−7 |
| 3 | reciprocity S21 = S12 on the real T | asymmetric bookkeeping | 2e−4 |
| 3 | finite arrays → periodic limit | finite vs lattice-sum consistency | monotone ✓ |
| 6 | lossless mirror unitarity R = 1 | all image bookkeeping | exact |

Plus three *external* validations no internal test can replace:

* **treams** (independent Ewald-based multiple-scattering code, KIT): same
  tmat.h5, same lattice → complex S-parameters agree to 2.7e−4 (S21) /
  3.4e−4 (S11) at all 49 frequencies. Two codebases, two lattice-sum
  algorithms, one answer.
* **Feature fidelity**: a synthetic Lorentzian dipole resonance at 15 µm
  pushed through the pipeline produces a deep collective resonance at 18 µm
  (min |S21| = 0.265). treams reproduces both the position and the depth to
  3 decimals. (The 15 → 18 µm shift is *real physics* — dense-lattice dipole
  coupling — and a warning: don't "validate" a multiple-scattering code by
  checking that features stay where the single atom had them. Also note the
  *uncoupled* sheet is not a reference for strong scatterers: without the
  self-consistent radiative coupling it violates unitarity, |S21| ≈ 10.)
* **Direct CST periodic simulation** of the same free-standing array:
  agreement to |ΔS21| ≤ 0.0011, |ΔS11| ≤ 0.0029 over the whole band — this
  also bounds the error of the CST near-field extraction itself (~0.3 %).

### 7.4 Verifying the *input file's* convention

Before trusting T: (a) passivity — max singular value of S = I + 2T over
the band is 1.00007 ≤ 1 + extraction noise; (b) reciprocity — in this basis
T must satisfy T[μ,ν] = (−1)^(m_μ+m_ν) T[ν̄,μ̄] (bar = m → −m); our residual
(0.6–1.2 %) matches the file's own stored `reciprocity` diagnostic *exactly*,
proving we read the basis the way the writer meant it; (c) σ_abs > 0
(Section 5.2). Only after these three does the pipeline get to run.

---

## 8. The demo, and reading its physics honestly

### 8.1 Numbers

49 frequencies (λ = 8–20 µm), pitch 2 µm, x-polarized normal incidence.
Free-standing array: |S21| falls smoothly 0.992 → 0.942, |S11| rises
0.118 → 0.317, absorption ≤ 1.2 % — *featureless*.

### 8.2 Why featureless is correct, not a failure

The extracted T_iso itself grows monotonically toward 8 µm with no in-band
peak: the isolated resonator's dipole resonance lies *below* 8 µm. The
designed λc = 15 µm response is an MIM (metal–insulator–metal) resonance
that exists only with the ground plane — which Stage 1 of the extraction
deliberately removes. A disconnected subwavelength patch array far above its
resonance *must* transmit (quasi-static bound: sheet reflection
≈ kα/2A ≈ 0.1 at 15 µm, where α is the electric polarizability of one
cell, taken at the PEC-disk scale 16a³/3). Three
independent computations agreeing on the flat curves settle it. The mirror
extension (Section 6) then shows the absorber physics reappearing.

---

## 9. War story #4: getting the "real" reference out of CST

Asked for the true S-parameters, the archived `.cst` turned out to be a
48 KB model-only archive (no results), so a fresh periodic simulation was
scripted (`cst_direct/build_saw_unitcell.py`: unit-cell boundaries, Floquet
ports, FD solver, lossy gold). Two failures on the way — both instructive:

1. **False mesh convergence.** The first run finished in 27 s on a
   1372-cell mesh and gave |S11| ≈ 0.97. Refining the mesh 5× *did not
   change the answer* — which usually means "converged", but here both
   meshes were equally unable to represent the real problem. Never accept
   mesh convergence as proof when the answer violates a physical bound.
2. **The unit-cell bounding-box trap (the actual cause).** With
   `Boundary "unit cell"`, CST sizes the transverse period to the *geometry
   bounding box* unless told otherwise. The resonator's box is its 1.44 µm
   diameter, so CST simulated a 1.44 µm-pitch lattice of **touching** rings
   — a connected inductive mesh, which really does reflect ~97 %. The
   solver even hinted at it: "edges … treated as infinitely thin PEC wires"
   (the fused contacts). Diagnosis came from **control tests with known
   answers**: an almost-vacuum brick (S21 = 1 ✓), a full gold sheet
   (measured |S11| = 0.990–0.994 ✓), and a 1 µm gold patch — which behaved *identically* to
   the full sheet. A patch in a 75 %-open cell cannot mimic a sheet; the
   only explanation is that the cell had shrunk onto the patch. Fix:

       Boundary.UnitCellFitToBoundingBox "False"
       Boundary.UnitCellDs1 "2.0"
       Boundary.UnitCellDs2 "2.0"

   After the fix, the patch control transmits (|S21| ≈ 0.99) and the
   spoke-wheel run lands on the T-matrix prediction to 0.3 %.

The meta-lesson generalizes: **when a simulation disagrees with a physical
bound, stop tuning and start feeding it problems whose answers you already
know.** The three-control ladder (empty / solid / simple-inclusion) costs
minutes and localizes setup bugs that no amount of mesh refinement reveals.

---

## 10. Numerical engineering notes (why it's fast enough)

* Everything vectorized to GEMMs: projections are `(batch × 3·N_pts) @
  (3·N_pts × modes)` products; VSWF evaluation batches all modes over all
  points.
* Shell grouping (Section 3.3): one projection per distinct lattice radius
  ⇒ ~10× fewer projections; Richardson tapers share shells ⇒ 3 tapers for
  ~1× cost.
* Finite arrays: translation cache per distinct radius; 13×13 array = 5070
  unknowns, ~seconds per frequency.
* Full 49-frequency periodic demo: ~8 min single-threaded-ish NumPy;
  mirror variant ~8 min per mirror height (image shells not cached across
  h).
* Parameter choices, with reasons: r₀ = 0.8 µm (must be < pitch; large
  enough that j₃(kr₀) is not tiny); quadrature 16×32 (exact to degree 31,
  aliasing floor ~1e−6); kRc = (10, 14, 20) (leading taper error (2/kRc)² ≈
  4 % per taper, ~1e−6 after Richardson); shell truncation 3.5·Rc_max.

---

## 11. Pitfall checklist (tape this to your monitor)

1. e^(−iωt) vs e^(+jωt): decide once; it flips h⁽¹⁾↔h⁽²⁾, conjugates T,
   and the optical theorem's sign is your detector.
2. Jackson X-based N-waves carry an **i on the radial term**. Test
   N = ∇×M/k numerically; don't copy Tsang into a Jackson pipeline.
3. `np.vdot(a, b)` conjugates the **first** argument.
4. Read the mode list (l, m, polarization) from the file; never assume the
   ordering.
5. π_lm = mY/sinθ is finite at the poles but 0/0 numerically — nudge θ by
   1e−9 before on-axis evaluations.
6. Free-space 2D lattice sums are conditionally convergent; a Gaussian
   taper converges only **algebraically** (Dawson tail) — Richardson in
   1/Rc², and keep truncation below the extrapolation error.
7. Convergence metrics on C must be entrywise; Frobenius is blinded by the
   1e6 quasi-static entries.
8. The truncated addition theorem converges like (r/d)^(l−l_src): don't
   "fix" it with quadrature; it's truncation, and at lmax = 3 with
   pitch/diameter = 1.39 it is the method's real accuracy limit.
9. Image/parity signs: compute numerically; verify with a lossless-mirror
   unitarity test.
10. Uncoupled ("dilute") sheet formulas are unphysical for strong
    scatterers — only the Foldy–Lax solve respects unitarity.
11. CST `"unit cell"` boundaries fit the geometry bounding box:
    `UnitCellFitToBoundingBox "False"` + explicit `UnitCellDs1/Ds2`, and
    run empty/sheet/patch controls before believing any new periodic setup.
12. Identical answers on two meshes prove nothing if both meshes are wrong
    the same way; physical bounds outrank convergence checks.

---

## 12. Extending the pipeline

* **Oblique incidence**: multiply the per-shell azimuthal phase sums in
  `assemble_shell_sum` by Bloch factors e^(ik∥·R) (the shell machinery is
  ready for it, but the current code hard-assumes k∥ = 0), add the 1/cosθ
  in the S-parameter prefactor, and evaluate F at the tilted
  specular/forward directions.
* **Real substrate instead of PEC mirror**: replace diag(s)-image with the
  Sommerfeld-integral reflection operator R (the manual's Stage 2); the
  aggregation structure is unchanged.
* **Mixed atoms / disorder**: `build_finite_system` already accepts
  per-site T-matrices and arbitrary in-plane positions.
* **Higher lmax**: everything is lmax-generic; the file's lmax = 3 is the
  binding limit (Wiscombe suggests L = 5 at the band edge — worth an
  extraction-side convergence pass at pitch/diameter this tight).

## 13. File map

| file | role |
|---|---|
| `vswf.py` | conventions + VSWF engine + plane waves + far field + projector |
| `translate.py` | translation operators, shells, tapered+Richardson lattice sums |
| `aggregate.py` | Foldy–Lax finite solve, periodic solve, effective array T |
| `sparams.py` | S11/S21, energy balance, cross sections |
| `mirror.py` | PEC ground plane: parity signs, image lattice, mirrored S11 |
| `tmat_io.py` | tmat.h5 reader |
| `test_vswf.py`, `test_translate.py`, `test_mirror.py`, `test_feature_fidelity.py` | the validation pyramid |
| `run_demo.py`, `run_mirror_demo.py`, `plot_results.py`, `plot_cst_comparison.py` | drivers and figures |
| `treams_reference.py`, `validate_synthetic_treams.py` | independent treams cross-checks |
| `cst_direct/build_saw_unitcell.py`, `cst_direct/control_tests.py` | direct CST reference + setup controls |
| `REPORT.md` | results summary; this guide covers the *how* |
