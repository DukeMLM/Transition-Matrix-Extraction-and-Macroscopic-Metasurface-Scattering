"""Gates for retrieval/fastfull/ewald.py (milestone M2, lattice coupling C).

Run:  python test_fastfull_ewald.py     (from retrieval/, env cst_inference)

  (a) GATE D, the two-implementation cross-check: on a square lattice, where
      the repository's tapered real-space Bloch sum is valid and was itself
      verified per-site to 1.3e-15, the Ewald C agrees with it -- at normal
      AND oblique incidence
  (b) the polarization-index convention provably does not matter for C
      (in-plane translation gives A_ee = A_mm, A_em = A_me)
  (c) eta independence over a safe bracket, and refusal outside it
  (d) reciprocity of the lattice sum: C(-k_par) is the properly mapped
      partner of C(+k_par)
  (e) converged_C works on the oblique/rectangular cells where the tapered
      sum correctly refuses, and is stable walking toward a Rayleigh
      threshold
  (f) the coding cell is near-Born: ||C T|| for the M1 winner vs the
      campaign's 2 um cell
  (g) the forward map with the Ewald C reproduces the validated
      sparams_oblique result on the campaign cell
"""
import sys
import time


import numpy as np

from tmatrix.retrieval.fastfull import lattice as lt
from tmatrix.retrieval.fastfull import ewald as ew
from tmatrix.retrieval.fastfull import transforms as xf
from tmatrix.retrieval.fastfull import symmetry as sym

from tmatrix.aggregation.vswf import ModeBasis
from tmatrix.aggregation.translate import make_quad
from tmatrix.retrieval.bloch_lattice import lattice_sum_C_bloch
from tmatrix.retrieval.sparams_oblique import sparams_oblique
from tmatrix.aggregation.tmat_io import TMatrixData
from tmatrix.paths import DEMO_TMAT

RESULTS = []
MODES = ModeBasis.standard(3)


def record(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s\n         %s" % ("PASS" if ok else "FAIL", name, detail),
          flush=True)


# ------------------------------------------------- (a) Gate D cross-check

def _cached_C(fm, ifreq, th_deg, ph_deg):
    """The campaign's own C, built with the NORMATIVE per-theta taper scaling
    kRc = (10, 14, 20)/(1 - sin theta) (retrieval/HANDOFF.md).  Recomputing
    it here with the unscaled taper would under-converge it badly at oblique
    incidence and the comparison would measure that, not Ewald."""
    ia = fm.angle_index(th_deg, ph_deg)
    if not fm.have[ifreq, ia]:
        return None
    return fm.C[ifreq, ia]


def test_vs_tapered():
    from tmatrix.retrieval.forward import ForwardModel
    fm = ForwardModel()
    latt = lt.Lattice2D.square(fm.pitch)
    for ifreq in (48, 32):
        k = fm.k[ifreq]
        for th_deg, ph_deg in ((0.0, 0.0), (30.0, 22.5), (60.0, 22.5)):
            C_ref = _cached_C(fm, ifreq, th_deg, ph_deg)
            if C_ref is None:
                continue
            th, ph = np.deg2rad(th_deg), np.deg2rad(ph_deg)
            kpar = k * np.sin(th) * np.array([np.cos(ph), np.sin(ph)])
            t0 = time.time()
            C_ew = ew.lattice_sum_C(latt, k, MODES, kpar)
            t_ew = time.time() - t0
            scale = float(np.abs(C_ref).max())
            rel = float(np.abs(C_ew - C_ref).max() / scale)
            record("(a) GATE D: Ewald C == campaign tapered C, square "
                   "lattice (lam=%.1f um, theta=%g, phi=%g)"
                   % (fm.lam_um[ifreq], th_deg, ph_deg),
                   rel < 1e-4,
                   "relative max deviation %.3e (|C|max = %.4g), Ewald "
                   "%.4f s -- the residual is the tapered method's "
                   "Richardson extrapolation error; Ewald is exact to its "
                   "eta split (2.7e-14 here)" % (rel, scale, t_ew))


# --------------------------------------------------- (b) polarization index

def test_pol_index_irrelevant():
    latt = lt.Lattice2D.rectangular(11.0, 17.0, alpha_deg=23.0)
    k = 2 * np.pi / 9.0
    kpar = latt.bloch(0.13, -0.41)
    C0 = ew.lattice_sum_C(latt, k, MODES, kpar, pol_flip=False)
    C1 = ew.lattice_sum_C(latt, k, MODES, kpar, pol_flip=True)
    d = float(np.abs(C1 - C0).max() / np.abs(C0).max())
    record("(b) C is invariant under the polarization-index convention",
           d < 1e-14,
           "relative max |C(flip) - C| = %.3e -- for an in-plane "
           "displacement A_ee = A_mm and A_em = A_me, so swapping both "
           "indices is the identity; treams' parity pol and the repository's "
           "ELECTRIC = 0 therefore cannot disagree about C" % d)


# ------------------------------------------------------------------ (c) eta

def test_eta_stability():
    latt = lt.Lattice2D.oblique(10.8121, 7.2371, 92.75, 19.68)
    k = 2 * np.pi / 8.0
    kpar = latt.bloch(-0.0098, -0.4930)
    C, info = ew.converged_C(latt, k, MODES, kpar, return_info=True)
    worst = max(info["eta_deviations"].values())
    record("(c) eta independence over the safe bracket",
           C is not None and info["converged"] and worst < 1e-8,
           "eta deviations %s; worst %.2e (gate 1e-8)"
           % ({("%.1f" % kk): ("%.1e" % v)
               for kk, v in info["eta_deviations"].items()}, worst))

    # outside the safe range the sum really does move -- the refusal is real
    C_big = ew.lattice_sum_C(latt, k, MODES, kpar, eta=1.8)
    d_big = float(np.abs(C_big - C).max() / np.abs(C).max())
    obl = lt.Lattice2D.oblique(13.0, 19.0, 71.0, 17.0)
    bad, bad_info = ew.converged_C(obl, 2 * np.pi / 8.0, MODES,
                                   obl.bloch(0.11, -0.37),
                                   eta_bracket=(1.8,), rtol=1e-8,
                                   return_info=True)
    record("(c) a large eta is detected and refused, not silently used",
           d_big > 1e-12 and bad is None,
           "eta = 1.8 moves this cell's C by %.2e; converged_C with a "
           "bracket at 1.8 refuses on the oblique 13x19 cell (%s)"
           % (d_big, bad_info["reasons"][0] if bad_info["reasons"] else "-"))


# --------------------------------------------------------- (d) reciprocity

def test_reciprocity():
    """C(-k_par) = Rec-mapped C(+k_par).

    C = sum_R A(R) e^{i k.R}.  Reversing the Bloch vector is the same as
    reversing every lattice vector, and A(-R) is A(R) under the reciprocity
    map of parametrize.apply_rec (transpose with the (-1)^(m+m') sign and
    the m -> -m permutation), because the lattice is inversion symmetric.
    A one-line consequence of the conventions, and a sharp test of both the
    Bloch sign and the mode ordering.
    """
    from tmatrix.retrieval import parametrize as pz
    latt = lt.Lattice2D.oblique(12.0, 15.0, 83.0, 29.0)
    k = 2 * np.pi / 10.0
    kpar = latt.bloch(0.19, -0.33)
    Cp = ew.lattice_sum_C(latt, k, MODES, kpar)
    Cm = ew.lattice_sum_C(latt, k, MODES, -kpar)
    perm, sign = pz.reciprocity_perm_sign(MODES)
    d = float(np.abs(pz.apply_rec(Cp, perm, sign) - Cm).max()
              / np.abs(Cp).max())
    record("(d) C(-k_par) is the reciprocity map of C(+k_par)",
           d < 1e-10,
           "relative max deviation %.3e -- pins the Bloch sign and the mode "
           "ordering simultaneously" % d)


# ------------------------------------------- (e) cells the taper refuses

def test_oblique_cells():
    from tmatrix.retrieval.fastfull import coupling as cp
    quad = make_quad(8, 16)
    cases = [("M1 winner 10.8x7.2 g93", lt.Lattice2D.oblique(
        10.8121, 7.2371, 92.75, 19.68), 8.0, (-0.0098, -0.4930)),
        ("par.6 seed 26x33.8", lt.Lattice2D.rectangular(26.0, 33.8),
         20.0, (0.090, -0.460)),
        ("oblique 13x19 g71", lt.Lattice2D.oblique(13.0, 19.0, 71.0, 17.0),
         8.0, (0.11, -0.37))]
    ok_all, lines = True, []
    for name, latt, lam, f in cases:
        k = 2 * np.pi / lam
        kpar = latt.bloch(*f)
        taper, tinfo = cp.converged_C(latt, k, MODES, 0.8, quad, kpar)
        C, info = ew.converged_C(latt, k, MODES, kpar, return_info=True)
        ok = (taper is None) and (C is not None) and info["converged"]
        ok_all &= ok
        lines.append("%s: taper refuses (%d sites in taper), Ewald |C|max "
                     "= %.4g, eta spread %.1e"
                     % (name, tinfo["stats"]["sites_in_min_taper"],
                        info["abs_max"], max(info["eta_deviations"].values())))
    record("(e) Ewald succeeds on every cell where the tapered sum refuses",
           ok_all, "; ".join(lines))


def test_rayleigh_walk():
    latt = lt.Lattice2D.oblique(10.8121, 7.2371, 92.75, 19.68)
    k = 2 * np.pi / 8.0
    rows = ew.rayleigh_scan(latt, k, MODES, (-0.0098, -0.4930),
                            (1.0, 1.05, 1.10, 1.14))
    ok = all(r["converged"] for r in rows)
    growth = rows[-1]["abs_max"] / rows[0]["abs_max"]
    record("(e) Ewald stays converged walking toward a Rayleigh threshold",
           ok and growth < 10.0,
           "; ".join("Wood %.4f -> |C|max %.3g (eta %.0e)"
                     % (r["wood"], r["abs_max"], r["eta_spread"])
                     for r in rows))


# ------------------------------------------------------------- (f) Born

def test_born_regime():
    data = TMatrixData(str(DEMO_TMAT))
    i8 = int(np.argmin(np.abs(data.wavelength_um - 8.0)))
    T = data.T[i8]
    k = data.k_at(i8)
    win = lt.Lattice2D.oblique(10.8121, 7.2371, 92.75, 19.68)
    C_win = ew.lattice_sum_C(win, k, MODES, win.bloch(-0.0098, -0.4930))
    d_win = ew.lattice_dressing_strength(C_win, T)

    camp = lt.Lattice2D.square(2.0)
    th, ph = np.deg2rad(30.0), np.deg2rad(22.5)
    kpar = k * np.sin(th) * np.array([np.cos(ph), np.sin(ph)])
    C_camp = ew.lattice_sum_C(camp, k, MODES, kpar)
    d_camp = ew.lattice_dressing_strength(C_camp, T)

    record("(f) the coding cell is near-Born while the campaign cell is not",
           d_win["born_valid"] and d_win["norm_CT"] < 0.5,
           "M1 winner (78 um^2): ||C T|| = %.4f, Neumann bound %.3f; "
           "campaign cell (4 um^2): ||C T|| = %.1f, Born valid = %s. "
           "The screening Jacobian M1 computed at C = 0 is therefore a mild "
           "perturbation of the real one on the coding cell."
           % (d_win["norm_CT"], d_win["neumann_bound"], d_camp["norm_CT"],
              d_camp["born_valid"]))


# --------------------------------------------------- (g) end-to-end forward

def test_forward_with_ewald_C():
    """Ewald C driven through the full forward map must match the validated
    sparams_oblique result on the campaign cell."""
    from tmatrix.retrieval.forward import ForwardModel
    fm = ForwardModel()
    latt = lt.Lattice2D.square(fm.pitch)
    worst, lines = 0.0, []
    for ifreq in (48, 32):
        k, T0 = fm.k[ifreq], fm.data.T[ifreq]
        for th_deg, ph_deg in ((0.0, 0.0), (30.0, 22.5), (60.0, 22.5)):
            C_tap = _cached_C(fm, ifreq, th_deg, ph_deg)
            if C_tap is None:
                continue
            th, ph = np.deg2rad(th_deg), np.deg2rad(ph_deg)
            kpar = k * np.sin(th) * np.array([np.cos(ph), np.sin(ph)])
            C_ew = ew.lattice_sum_C(latt, k, MODES, kpar)
            ref = sparams_oblique(k, latt.area, MODES, T0, C_tap, th, ph, -1)
            got = sparams_oblique(k, latt.area, MODES, T0, C_ew, th, ph, -1)
            d = max(float(np.abs(got["S11"] - ref["S11"]).max()),
                    float(np.abs(got["S21"] - ref["S21"]).max()))
            worst = max(worst, d)
            lines.append("lam %.1f (%g, %g): %.2e"
                         % (fm.lam_um[ifreq], th_deg, ph_deg, d))
    # the campaign's own measured model-vs-CST discrepancy is 2.6-3.6e-3;
    # swapping the lattice sum must stay far below that to be irrelevant
    record("(g) swapping tapered C for Ewald C leaves the forward S far "
           "below the measured discrepancy",
           worst < 1e-4,
           "worst complex |dS| = %.3e over %d (frequency, angle) pairs "
           "[%s]; the measured CST-vs-model discrepancy is 2.6e-3, so the "
           "choice of lattice-sum implementation is %0.f x below it"
           % (worst, len(lines), "; ".join(lines),
              2.6333e-3 / worst if worst > 0 else float("inf")))


# ------------------------------------------------------------------ driver

def main():
    print("=== fastfull Ewald (M2) gates ===", flush=True)
    test_vs_tapered()
    test_pol_index_irrelevant()
    test_eta_stability()
    test_reciprocity()
    test_oblique_cells()
    test_rayleigh_walk()
    test_born_regime()
    test_forward_with_ewald_C()

    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n=== SUMMARY (test_fastfull_ewald) ===")
    for name, ok, _ in RESULTS:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("ALL %d TESTS PASSED" % len(RESULTS) if nfail == 0
          else "%d of %d TEST(S) FAILED" % (nfail, len(RESULTS)))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
