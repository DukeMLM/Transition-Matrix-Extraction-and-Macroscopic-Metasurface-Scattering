"""PASS/FAIL test suite for retrieval/parametrize.py.

Checklist item 4 of INVERSE_TMATRIX_FROM_FLOQUET.md: C4v (x) reciprocity
constrained subspace -- numerical sigma_v derivation, group validation,
projector construction, dimension counts (228 / 114 / 68), bright-subspace
reduction (25 entries -> 11 orbits), pack/unpack helpers, and the
reference-T validation battery (projector invariance, explicit sigma_v
selection rule, passivity).

Run:
    conda activate cst_inference
    cd "D:/Claude/T matrix/retrieval"
    python test_parametrize.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "aggregation"))
sys.path.insert(0, HERE)

from tmat_io import TMatrixData                       # noqa: E402
import parametrize as par                             # noqa: E402

REF = os.path.join(HERE, "..", "test", "single",
                   "saw_gold_wl15p0025um.tmat.h5")

RESULTS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print("[%s] %s%s" % (tag, name, ("  --  " + detail) if detail else ""))
    RESULTS.append((name, bool(ok)))
    return ok


def info(name, detail):
    print("[info] %s  --  %s" % (name, detail))


def main():
    t_start = time.time()
    print("=" * 76)
    print("retrieval/parametrize.py test suite  (checklist item 4)")
    print("=" * 76)

    # ---------------------------------------------------------------- [0]
    print("\n--- [0] reference file ---")
    data = TMatrixData(REF)
    modes = data.modes
    nf = len(data.freq)
    lam = data.wavelength_um
    print("file: %s" % os.path.abspath(REF))
    print("nf=%d  nmodes=%d  lmax=%d  lambda %.2f-%.2f um"
          % (nf, modes.n, modes.lmax, lam.min(), lam.max()))
    check("reference geometry (49 freqs, 30 modes, lmax=3)",
          nf == 49 and modes.n == 30 and modes.lmax == 3,
          "nf=%d n=%d lmax=%d" % (nf, modes.n, modes.lmax))

    # ---------------------------------------------------------------- [1]
    print("\n--- [1] sigma_v: numerical derivation (E-field lstsq, "
          "M = diag(1,-1,1)) ---")
    D_num, ndiag = par.derive_sigma_v_numeric(modes, k=1.0, r_sample=1.3)
    D_exact, sperm, ssigns = par.sigma_v_exact(modes)
    err_match = float(np.abs(D_num - D_exact).max())
    err_sq = float(np.abs(D_num @ D_num - np.eye(modes.n)).max())
    print("lstsq design-matrix condition number: %.3e" % ndiag["cond"])
    check("E-field lstsq relative residual <= 1e-9",
          ndiag["resid_E"] <= 1e-9, "resid_E = %.3e" % ndiag["resid_E"])
    check("H-field (pseudovector, Ht' = -M.Ht(M.r)) validation of the SAME "
          "D <= 1e-9",
          ndiag["resid_H"] <= 1e-9, "resid_H = %.3e" % ndiag["resid_H"])
    check("numerical D(sigma_v) matches analytic signed permutation "
          "<= 1e-10",
          err_match <= 1e-10, "max|D_num - D_exact| = %.3e" % err_match)
    info("numerical D(sigma_v)^2 = I residual", "%.3e" % err_sq)

    # write out the rule found, straight from the numerical matrix
    print("sigma_v action found numerically (row = image mode, snapped):")
    ok_rule = True
    pol_name = {0: "E", 1: "M"}
    for nu in range(modes.n):
        col = D_num[:, nu]
        mu = int(np.abs(col).argmax())
        val = col[mu]
        snapped = round(float(val.real))
        clean = (abs(val - snapped) <= 1e-10
                 and np.abs(np.delete(col, mu)).max() <= 1e-10)
        l, m, p = int(modes.l[nu]), int(modes.m[nu]), int(modes.pol[nu])
        lm, mm, pm = int(modes.l[mu]), int(modes.m[mu]), int(modes.pol[mu])
        expect_sign = (1 if m % 2 == 0 else -1) * (1 if p == 0 else -1)
        ok_nu = (clean and lm == l and mm == -m and pm == p
                 and snapped == expect_sign)
        ok_rule = ok_rule and ok_nu
        print("  (l=%d, m=%+d, %s) -> %+d * (l=%d, m=%+d, %s)%s"
              % (l, m, pol_name[p], snapped, lm, mm, pol_name[pm],
                 "" if ok_nu else "   <-- UNEXPECTED"))
    check("rule = (l,m,E) -> +(-1)^m (l,-m,E), (l,m,M) -> -(-1)^m (l,-m,M) "
          "for all 30 modes", ok_rule)

    # anti-trap structure checks
    m_nonzero = modes.m != 0
    diag_on_mflip = float(np.abs(np.diag(D_exact)[m_nonzero]).max())
    check("anti-trap: D(sigma_v) has ZERO diagonal entry for every m != 0 "
          "mode (signed permutation m -> -m, NOT diagonal)",
          diag_on_mflip == 0.0, "max diag over m!=0 modes = %.1e"
          % diag_on_mflip)
    from mirror import mirror_parity_signs
    s_h = mirror_parity_signs(modes)
    D_h = np.diag(s_h)
    check("anti-trap: D(sigma_v) differs from mirror.py's horizontal "
          "z-mirror diag (C4h element)",
          float(np.abs(D_exact - D_h).max()) >= 1.0,
          "max|D_sigma_v - diag(s_h)| = %.1f (s_h is diagonal; sigma_v is "
          "not)" % float(np.abs(D_exact - D_h).max()))

    # ---------------------------------------------------------------- [2]
    print("\n--- [2] group validation (exact matrices) ---")
    B, meta = par.build_c4v_reciprocity_basis(modes, verify_numeric=True)
    check("D(sigma_v)^2 = I exactly",
          meta["sigma_sq_resid"] == 0.0,
          "residual = %.1e" % meta["sigma_sq_resid"])
    check("all 8 elements unitary (D D^dag = I) <= 1e-12",
          meta["unitarity_resid"] <= 1e-12,
          "worst = %.3e" % meta["unitarity_resid"])
    check("group closure: all 64 pairwise products match an element "
          "<= 1e-12", meta["closure_resid"] <= 1e-12,
          "worst pair residual = %.3e" % meta["closure_resid"])
    check("conjugation identity D(C4) D(sigma_v) D(C4)^-1 = "
          "D(C4^2 sigma_v) (yz mirror) <= 1e-12",
          meta["conjugation_resid"] <= 1e-12,
          "residual = %.3e" % meta["conjugation_resid"])
    info("builder re-derived sigma_v numerically",
         "match = %.3e (guard threshold 1e-10)" % meta["sigma_numeric_err"])

    # ---------------------------------------------------------------- [3]
    print("\n--- [3] projectors on vec(T) (900 x 900) ---")
    check("group projectors real (imag residue) <= 1e-15",
          meta["imag_resid"] <= 1e-15, "%.3e" % meta["imag_resid"])
    check("[P_grp, P_rec] = 0 (matrix commutator) <= 1e-12",
          meta["comm_resid"] <= 1e-12, "max abs = %.3e" % meta["comm_resid"])
    check("commutation on random T (operator forms) <= 1e-12",
          meta["comm_T_resid"] <= 1e-12, "%.3e" % meta["comm_T_resid"])
    check("P_full symmetric <= 1e-12", meta["herm_resid"] <= 1e-12,
          "%.3e" % meta["herm_resid"])
    check("P_full idempotent <= 1e-12", meta["idem_resid"] <= 1e-12,
          "%.3e" % meta["idem_resid"])
    # P_c4 must be the diagonal (m - m') mod 4 mask
    P_c4 = meta["P_c4"]
    off = P_c4 - np.diag(np.diag(P_c4))
    dm = modes.m[:, None] - modes.m[None, :]
    mask_c4 = (np.mod(dm, 4) == 0).astype(float).reshape(-1)
    check("P_c4 is the diagonal (m - m') mod 4 == 0 mask",
          float(np.abs(off).max()) == 0.0
          and float(np.abs(np.diag(P_c4) - mask_c4).max()) == 0.0,
          "offdiag max = %.1e, diag-vs-mask max = %.1e"
          % (float(np.abs(off).max()),
             float(np.abs(np.diag(P_c4) - mask_c4).max())))
    # vec form vs operator form
    check("vec-matrix P_full agrees with operator P on random T <= 1e-12",
          meta["vec_apply_resid"] <= 1e-12, "%.3e" % meta["vec_apply_resid"])
    # reciprocity involution sanity
    rng = np.random.default_rng(7)
    Tr = rng.normal(size=(30, 30)) + 1j * rng.normal(size=(30, 30))
    invol = float(np.abs(par.apply_rec(
        par.apply_rec(Tr, meta["rec_perm"], meta["rec_sign"]),
        meta["rec_perm"], meta["rec_sign"]) - Tr).max())
    check("Rec(Rec(T)) = T (involution, complex-linear) <= 1e-15",
          invol <= 1e-15, "%.1e" % invol)

    # ---------------------------------------------------------------- [4]
    print("\n--- [4] dimension counts (doc section 3: 228 -> 114 -> 68) ---")
    check("rank of C4-only commutant projector == 228",
          meta["rank_c4"] == 228, "measured %d (eig purity %.1e)"
          % (meta["rank_c4"], meta["eig_purity_c4"]))
    check("rank of full C4v commutant projector == 114",
          meta["rank_c4v"] == 114, "measured %d (eig purity %.1e)"
          % (meta["rank_c4v"], meta["eig_purity_c4v"]))
    check("rank of C4v intersect reciprocity projector == 68",
          meta["rank_full"] == 68, "measured %d (eig purity %.1e)"
          % (meta["rank_full"], meta["eigenvalue_purity"]))
    info("reciprocity-only projector rank (context)",
         "%d (= (900 + 30)/2)" % meta["rank_rec"])

    # ---------------------------------------------------------------- [5]
    print("\n--- [5] subspace basis B (shape %s) ---" % (B.shape,))
    check("basis shape == (68, 30, 30)", B.shape == (68, 30, 30),
          str(B.shape))
    check("basis is real (complex span supplies the imaginary DOF)",
          np.isrealobj(B))
    check("Frobenius orthonormality <= 1e-12",
          meta["orthonormality_resid"] <= 1e-12,
          "max|Gram - I| = %.3e" % meta["orthonormality_resid"])
    check("each B_k invariant under operator P <= 1e-12",
          meta["basis_invariance_resid"] <= 1e-12,
          "worst = %.3e" % meta["basis_invariance_resid"])
    check("projection via B equals operator P on random T <= 1e-12",
          meta["basis_projection_resid"] <= 1e-12,
          "%.3e" % meta["basis_projection_resid"])

    # ---------------------------------------------------------------- [6]
    print("\n--- [6] pack / unpack helpers ---")
    nb = B.shape[0]
    check("n_params(B) == 2 * 68 == 136", par.n_params(B) == 136,
          "n_params = %d" % par.n_params(B))
    rng = np.random.default_rng(42)
    t0 = rng.normal(size=2 * nb)
    T0 = par.unpack(t0, B)
    rt = float(np.abs(par.pack(T0, B) - t0).max())
    check("roundtrip pack(unpack(t)) = t <= 1e-12", rt <= 1e-12,
          "max err = %.3e" % rt)
    Tr = rng.normal(size=(30, 30)) + 1j * rng.normal(size=(30, 30))
    PT = par.apply_full_projector(Tr, meta["group"], meta["rec_perm"],
                                  meta["rec_sign"])
    rt2 = float(np.abs(par.unpack(par.pack(Tr, B), B) - PT).max())
    check("unpack(pack(T)) equals the full projector on arbitrary T "
          "<= 1e-12", rt2 <= 1e-12, "max err = %.3e" % rt2)
    inband = float(np.abs(par.unpack(par.pack(PT, B), B) - PT).max())
    check("subspace elements reproduced exactly <= 1e-12",
          inband <= 1e-12, "max err = %.3e" % inband)

    # ---------------------------------------------------------------- [7]
    print("\n--- [7] bright-subspace reduction (threshold 1e-3 of global "
          "|T|max) ---")
    T_stack = data.T
    mask = par.bright_mask(T_stack, threshold=1e-3)
    check("bright entry count == 25 (doc section 3 exact count)",
          int(mask.sum()) == 25, "measured %d" % int(mask.sum()))
    Bb, binfo = par.bright_orbit_basis(B, T_stack, threshold=1e-3,
                                       modes=modes)
    check("bright entries collapse into 11 orbits under "
          "{id, sigma, transpose, reciprocity}",
          binfo.get("n_orbits") == 11,
          "measured %d orbits, mask closed under action: %s"
          % (binfo.get("n_orbits", -1),
             binfo.get("mask_closed_under_group")))
    orb_sizes = sorted(len(o) for o in binfo["orbits"])
    info("orbit sizes", "%s (sum = %d; orbits may extend to sub-threshold "
         "mates)" % (orb_sizes, sum(orb_sizes)))
    n_conf = binfo["n_orbits_c4_conforming"]
    info("C4 conformance per orbit",
         "%d of %d orbits obey (m - m') mod 4 == 0; the violating orbit(s) "
         "contain the doc's 2 noise-level bright entries (doc: '23 of the "
         "25 bright entries obey it') and are annihilated by the C4 average"
         % (n_conf, binfo["n_orbits"]))
    viol = [o for o, c in zip(binfo["orbits"], binfo["orbit_c4_conforming"])
            if not c]
    for o in viol:
        stat = ", ".join("%s%s peak=%.2e"
                         % (p, "" if binfo["mask"][p] else " (subthr)",
                            float(np.abs(T_stack[:, p[0], p[1]]).max()))
                         for p in o)
        info("C4-violating orbit", stat)
    check("bright basis dimension == number of C4-conforming bright orbits",
          binfo["rank"] == n_conf,
          "measured rank %d == %d conforming orbits (singular values %s)"
          % (binfo["rank"], n_conf,
             np.array2string(binfo["singular_values"][:binfo["rank"] + 3],
                             precision=3)))
    info("comparison to doc's 11-15 window",
         "measured %d basis vectors: the doc's estimate counts all 11 "
         "position orbits, but 1 orbit is pure C4-violating noise (its "
         "transpose/reciprocity mates even fall below the bright threshold)"
         ", so the physical bright unknown count is %d"
         % (binfo["rank"], binfo["rank"]))
    # support of the bright basis must be confined to the union of the
    # C4-conforming orbits it was built from
    supp = np.zeros((modes.n, modes.n), dtype=bool)
    for o, c in zip(binfo["orbits"], binfo["orbit_c4_conforming"]):
        if c:
            for p in o:
                supp[p] = True
    off_supp = float(np.abs(Bb[:, ~supp]).max()) if (~supp).any() else 0.0
    check("bright basis supported only on the C4-conforming bright orbits "
          "<= 1e-12", off_supp <= 1e-12,
          "max |B_bright| off the %d-position orbit union = %.3e"
          % (int(supp.sum()), off_supp))
    Gb = Bb.reshape(len(Bb), -1) @ Bb.reshape(len(Bb), -1).T
    check("bright basis Frobenius-orthonormal <= 1e-12",
          float(np.abs(Gb - np.eye(len(Bb))).max()) <= 1e-12,
          "%.3e" % float(np.abs(Gb - np.eye(len(Bb))).max()))
    memb = max(float(np.abs(par.apply_full_projector(
        Bbk, meta["group"], meta["rec_perm"], meta["rec_sign"])
        - Bbk).max()) for Bbk in Bb)
    check("bright basis lies inside the 68-dim subspace <= 1e-12",
          memb <= 1e-12, "worst = %.3e" % memb)
    print("bright entries (i, j) -> (l,m,pol)x(l',m',pol'):")
    pol_name = {0: "E", 1: "M"}
    for (i, j) in np.argwhere(mask):
        print("  T[%2d,%2d] = (%d,%+d,%s | %d,%+d,%s)  peak|T| = %.3e"
              % (i, j, modes.l[i], modes.m[i], pol_name[int(modes.pol[i])],
                 modes.l[j], modes.m[j], pol_name[int(modes.pol[j])],
                 float(np.abs(T_stack[:, i, j]).max())))

    # ---------------------------------------------------------------- [8]
    print("\n--- [8] reference-T validation (doc section 4) ---")
    rep = par.invariance_report(T_stack, meta["group"], meta["rec_perm"],
                                meta["rec_sign"])
    iw = int(rep[:, 1].argmax())
    print("projector invariance ||P(T)-T|| per frequency:")
    print("  abs Frobenius: median %.3e, band-worst %.3e (lam=%.2f um)"
          % (float(np.median(rep[:, 0])), float(rep[:, 0].max()),
             lam[int(rep[:, 0].argmax())]))
    print("  rel Frobenius: median %.3e, band-worst %.3e (lam=%.2f um)"
          % (float(np.median(rep[:, 1])), float(rep[:, 1].max()), lam[iw]))
    print("  max-entry abs: median %.3e, band-worst %.3e (lam=%.2f um)"
          % (float(np.median(rep[:, 2])), float(rep[:, 2].max()),
             lam[int(rep[:, 2].argmax())]))
    check("projector invariance: band-worst relative Frobenius <= 2e-2 "
          "(expected noise floor: few 1e-3)",
          float(rep[:, 1].max()) <= 2e-2, "worst = %.3e" % rep[:, 1].max())

    sel = par.sigma_v_selection_residual(T_stack, meta["sigma_perm"],
                                         meta["sigma_signs"], mask=mask)
    gmax = float(np.abs(T_stack).max())
    sw = int(sel.argmax())
    print("explicit sigma_v selection rule "
          "T[i,j] = s_i s_j T[flip(i),flip(j)] on bright entries:")
    print("  worst abs residual over band: %.3e (lam=%.2f um); "
          "global |T|max = %.3e; relative = %.3e"
          % (float(sel.max()), lam[sw], gmax, float(sel.max()) / gmax))
    check("sigma_v selection rule on bright entries: worst residual "
          "<= 5e-3 x global |T|max (anti-trap test)",
          float(sel.max()) <= 5e-3 * gmax,
          "relative worst = %.3e" % (float(sel.max()) / gmax))
    sel_all = par.sigma_v_selection_residual(T_stack, meta["sigma_perm"],
                                             meta["sigma_signs"])
    info("sigma_v selection rule over ALL 900 entries",
         "band-worst abs = %.3e (relative %.3e)"
         % (float(sel_all.max()), float(sel_all.max()) / gmax))

    sv = par.passivity_max_sv(T_stack)
    check("passivity: max SV(I + 2T) over band <= 1 + 1e-3",
          sv <= 1.0 + 1e-3, "max SV = %.6f" % sv)

    # ---------------------------------------------------------------- end
    print("\n" + "=" * 76)
    npass = sum(1 for _, ok in RESULTS if ok)
    nfail = len(RESULTS) - npass
    for name, ok in RESULTS:
        if not ok:
            print("FAILED: %s" % name)
    print("%d / %d checks passed, %d failed  (%.1f s)"
          % (npass, len(RESULTS), nfail, time.time() - t_start))
    print("=" * 76)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
