"""Fast deterministic identity tests for retrieval/fit.py (the
pure-optimization gate, independent of the physics target).

Gates (smoke frequencies 32 and 48 by default; frequencies whose campaign
angles are not yet cached are skipped gracefully):

  G1  AnalyticJacobian construction identity: the affine assembly
      S = const + L f reproduces fm.predict at the reference T to <= 1e-9
      (measured: ~1e-16).
  G2  Analytic Jacobian == INDEPENDENT central-finite-difference Jacobian
      (observability.jacobian; no shared derivative code), column-wise
      <= 1e-6 relative, at the healthy-scale linearization
      t_lin = pack(T_ref), FD step_scale 1e-8.  (At the default FD step
      1e-6 the FD reference ITSELF is truncation-limited to ~5e-4 along
      high-leverage directions -- third derivatives scale with
      ||C||^2 ~ 1e12; measured 5.2e-8 / 7.2e-9 at step 1e-8.)
  G3  Bright-basis identity: seeded random target in the bright-10 span
      RESTRICTED to its Born-observable directions (12 of 20 real; the 8
      Born-dark even-m directions enter S only quadratically and their
      true value is 0 here), scaled to spectral radius
      max_angle rho(T0 C) = 0.25 (the BINDING constraint of the task; see
      note below), noise-free S on campaign 13 at direction -1, fit in
      the FULL bright basis with tikhonov = 1e-10 (pins the signal-free
      directions at the seed 0 = truth), Born seed with multi-start(3):
      recover ||t_hat - t_true|| / ||t_true|| <= 1e-6.
  G4  Observable-span identity ("the FULL 68-basis restricted to" its
      observable directions): B_obs = fit.observable_subbasis at the Born
      point, cutoff 1e-3 s_max (the spectrum there is machine-split:
      58 real directions at s >= ~0.1, the rest at ~1e-13, so the cutoff
      is parameter-insensitive); random target IN that span, rho = 0.25;
      'lm' from the Born seed: recover <= 1e-6.  Includes the complex-
      basis pack/unpack round trip.
  G5  'lm' row guard and the tikhonov/'trf' branch (normal-only angle set,
      16 rows < 20 bright parameters).

  D6  DIAGNOSTIC (reported, not gated): a GENERIC random 68-basis target
      cannot be identified from campaign-13 specular data -- the fit
      matches the data to ~1e-9 while the parameter error stays O(1),
      split here into observable / dark components via the exact-J SVD.
      This is the measured rank deficiency documented in fit.py (Born
      rank 58/136), NOT an optimizer failure.
  D7  DIAGNOSTIC (reported, not gated): a GENERIC bright-10 target at the
      rho-capped amplitude exhibits the doc-par.-9 local-minima trap for
      its Born-dark (even-m) components -- at ifreq 32 the quadratic
      signal of those components is ~(6e-4)^2-weak and Born multi-start
      lands in shallow false basins (measured: 13 starts, best objective
      5.8e-10 vs true 0, rel error 3.7); at ifreq 48 (9x larger amplitude
      at the same rho) the same protocol recovers to ~1e-13.  Reported
      with the best-of-3 multistart numbers per frequency.

Scaling note (measured deviation from the task's suggested
"max|T0| ~ 1e-2"): a generic random in-subspace direction at max|T0| =
1e-2 has rho(T0 C) = 421 (ifreq 32) / 15.8 (ifreq 48) -- ||C||_2 ~ 1e6
(quasi-static l=3 coupling) and a random mix lacks the physical T's
near-sparse structure that keeps rho(T_ref C) at 0.17-0.32 with
max|T_ref| ~ 2e-2.  rho is linear in the scale, so all targets here are
scaled to rho = 0.25 exactly; the resulting max|T0| is printed.

Run:
    conda activate cst_inference
    cd "D:/Claude/T matrix/retrieval"
    python test_fit_smoke.py [--freqs 32,48]
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "aggregation"))
sys.path.insert(0, HERE)

import numpy as np

from forward import ForwardModel                      # noqa: E402
import parametrize as par                             # noqa: E402
import fit as fitmod                                  # noqa: E402
import observability as obs                           # noqa: E402

RESULTS = []
RHO_TARGET = 0.25


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print("[%s] %s%s" % (tag, name, ("  --  " + detail) if detail else ""),
          flush=True)
    RESULTS.append((name, bool(ok)))
    return ok


def rho_scaled_target(fm, ifreq, Bx, angles, seed_key):
    """Seeded random coefficient vector in span{Bx}, scaled so that
    max over angles of rho(T0 C) = RHO_TARGET (rho is linear in scale)."""
    rng = np.random.default_rng(seed_key)
    npar = 2 * np.asarray(Bx).shape[0]
    t_raw = rng.standard_normal(npar)
    T_raw = par.unpack(t_raw, Bx)
    rho_raw = max(np.abs(np.linalg.eigvals(T_raw @ fm.C[ifreq, a])).max()
                  for a in angles)
    t_true = (RHO_TARGET / rho_raw) * t_raw
    return t_true, par.unpack(t_true, Bx)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", default="32,48")
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args(argv)
    freqs = [int(x) for x in args.freqs.split(",")]

    t_all = time.time()
    print("=" * 76)
    print("retrieval/fit.py smoke identity (pure-optimization gate)")
    print("=" * 76)
    fm = ForwardModel()
    print("cache: %s; available freqs: %s"
          % (fm.cache_path, fm.available_freqs))
    B, meta = par.build_c4v_reciprocity_basis(fm.modes, verify_numeric=False)
    nb = B.shape[0]
    check("full basis has 68 complex directions", nb == 68, "nb=%d" % nb)
    B_b, binfo = par.bright_orbit_basis(B, fm.data.T, threshold=1e-3,
                                        modes=fm.modes)
    check("bright basis has 10 complex directions", B_b.shape[0] == 10,
          "rank=%d, n_bright=%d, orbits=%d"
          % (binfo["rank"], binfo["n_bright"], binfo.get("n_orbits", -1)))
    campaign = fitmod.ANGLE_SETS["campaign"]

    usable = []
    for ifreq in freqs:
        if fm.have[ifreq, campaign].all():
            usable.append(ifreq)
        else:
            print("[skip] ifreq %d: campaign angles not all cached yet"
                  % ifreq, flush=True)
    check("at least one smoke frequency has the campaign set cached",
          len(usable) > 0, "usable=%s" % usable)

    for ifreq in usable:
        lam = fm.lam_um[ifreq]
        print("\n--- ifreq %d (lam = %.2f um) ---" % (ifreq, lam))
        eng = fitmod.AnalyticJacobian(fm, ifreq, B, campaign, direction=-1)

        # G1: engine construction identity
        check("G1 ifreq %d: engine affine model == fm.predict at T_ref"
              % ifreq, eng.validation_resid <= 1e-9,
              "max|dS| = %.2e" % eng.validation_resid)

        # G2: analytic vs central-FD Jacobian at the reference t
        t_lin = par.pack(fm.data.T[ifreq], B)
        tj = time.time()
        Ja = eng.jac_packed(t_lin)
        t_ana = time.time() - tj
        tj = time.time()
        Jf = obs.jacobian(fm, ifreq, B, campaign, t_lin=t_lin,
                          direction=-1, step_scale=1e-8)
        t_fd = time.time() - tj
        rel = float((np.linalg.norm(Ja - Jf, axis=0)
                     / np.maximum(np.linalg.norm(Jf, axis=0), 1.0)).max())
        check("G2 ifreq %d: analytic J == central-FD J (col rel <= 1e-6, "
              "FD step 1e-8)" % ifreq, rel <= 1e-6,
              "max col rel = %.2e; timing analytic %.0f ms vs central-FD "
              "%.1f s" % (rel, t_ana * 1e3, t_fd))

        # G3: bright-basis identity, target in the Born-observable part
        engb = fitmod.AnalyticJacobian(fm, ifreq, B_b, campaign,
                                       direction=-1)
        Bb_obs, sb_born, _ = fitmod.observable_subbasis(engb, t_lin=None,
                                                        s_cutoff_rel=1e-3)
        ub_true, Tb = rho_scaled_target(fm, ifreq, Bb_obs, campaign,
                                        [args.seed, ifreq, 9])
        tb_true = par.pack(Tb, B_b)          # exact: Tb in span(B_b)
        Sb = fm.predict(Tb, ifreq, campaign, direction=-1)
        rb = fitmod.fit_frequency_multistart(
            fm, ifreq, B_b, Sb, campaign, n_starts=3,
            seed=args.seed, direction=-1, engine=engb, tikhonov=1e-10,
            xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=600)
        relb = (np.linalg.norm(rb["t_hat"] - tb_true)
                / np.linalg.norm(tb_true))
        check("G3 ifreq %d: bright Born-observable identity rel <= 1e-6"
              % ifreq, relb <= 1e-6,
              "rel = %.3e; %d/10 complex dirs Born-observable; max|T0| = "
              "%.2e; obj = %.3e; starts objs %s; nfev(best) = %d; "
              "wall(best) = %.1f s"
              % (relb, Bb_obs.shape[0], np.abs(Tb).max(), rb["objective"],
                 ["%.1e" % o for o in rb["multistart_objectives"]],
                 rb["nfev"], rb["wall_s"]))

        # D7: generic bright target (branch-trap diagnostic, not gated)
        tg_true, Tg = rho_scaled_target(fm, ifreq, B_b, campaign,
                                        [args.seed, ifreq, 5])
        Sg = fm.predict(Tg, ifreq, campaign, direction=-1)
        rg = fitmod.fit_frequency_multistart(
            fm, ifreq, B_b, Sg, campaign, n_starts=3, seed=args.seed,
            direction=-1, engine=engb, xtol=1e-14, ftol=1e-14,
            gtol=1e-14, max_nfev=600)
        relg = (np.linalg.norm(rg["t_hat"] - tg_true)
                / np.linalg.norm(tg_true))
        print("[info] D7 ifreq %d: GENERIC bright target (not gated): "
              "rel = %.3e at obj = %.3e (multistart objs %s; true "
              "objective 0; Born-dark branch trap when the quadratic "
              "even-m signal is weak -- doc par. 9)"
              % (ifreq, relg, rg["objective"],
                 ["%.1e" % o for o in rg["multistart_objectives"]]),
              flush=True)

        # G4: observable-span identity (full 68 restricted, Born cutoff)
        B_obs, s_born, V = fitmod.observable_subbasis(eng, t_lin=None,
                                                      s_cutoff_rel=1e-3)
        r_obs = B_obs.shape[0]
        # complex-basis pack/unpack round trip
        rng = np.random.default_rng([args.seed, ifreq, 3])
        u_rt = rng.standard_normal(2 * r_obs)
        rt = np.abs(par.pack(par.unpack(u_rt, B_obs), B_obs) - u_rt).max()
        check("G4a ifreq %d: complex sub-basis pack/unpack round trip"
              % ifreq, rt <= 1e-12,
              "r_obs = %d complex (Born spectrum s[0]=%.2e, s[r-1]=%.2e, "
              "s[r]=%.2e); max|d| = %.1e"
              % (r_obs, s_born[0], s_born[r_obs - 1],
                 s_born[r_obs] if r_obs < len(s_born) else np.nan, rt))
        tu_true, Tu = rho_scaled_target(fm, ifreq, B_obs, campaign,
                                        [args.seed, ifreq, 7])
        Su = fm.predict(Tu, ifreq, campaign, direction=-1)
        engo = fitmod.AnalyticJacobian(fm, ifreq, B_obs, campaign,
                                       direction=-1)
        ro = fitmod.fit_frequency(fm, ifreq, B_obs, Su, campaign,
                                  direction=-1, engine=engo,
                                  xtol=1e-14, ftol=1e-14, gtol=1e-14,
                                  max_nfev=600)
        relo = (np.linalg.norm(ro["t_hat"] - tu_true)
                / np.linalg.norm(tu_true))
        check("G4b ifreq %d: observable-span identity rel <= 1e-6"
              % ifreq, relo <= 1e-6,
              "rel = %.3e; max|T0| = %.2e; obj = %.3e; nfev = %d; "
              "wall = %.1f s"
              % (relo, np.abs(Tu).max(), ro["objective"], ro["nfev"],
                 ro["wall_s"]))

        # D6: generic-target diagnostic (measured rank deficiency)
        tg_true, Tg = rho_scaled_target(fm, ifreq, B, campaign,
                                        [args.seed, ifreq])
        Sg = fm.predict(Tg, ifreq, campaign, direction=-1)
        rg = fitmod.fit_frequency(fm, ifreq, B, Sg, campaign,
                                  direction=-1, engine=eng,
                                  tikhonov=1e-10, xtol=1e-12, ftol=1e-12,
                                  gtol=1e-12, max_nfev=300)
        d = rg["t_hat"] - tg_true
        Jt = eng.jac_packed(tg_true)
        U2, s2, Vt2 = np.linalg.svd(Jt, full_matrices=False)
        sel = s2 >= 1e-3 * s2[0]
        e_obs = np.linalg.norm(Vt2[sel] @ d)
        e_all = np.linalg.norm(d)
        print("[info] D6 ifreq %d: GENERIC 68-basis target (not gated): "
              "max|dS| = %.2e, full rel = %.3e, of which observable-span "
              "component %.3e / dark %.3e (norm t_true %.3e; Born rank "
              "%d/136) -- measured non-identifiability, see fit.py "
              "docstring"
              % (ifreq, np.abs(rg["dS"]).max(),
                 e_all / np.linalg.norm(tg_true),
                 e_obs / np.linalg.norm(tg_true),
                 np.sqrt(max(e_all ** 2 - e_obs ** 2, 0.0))
                 / np.linalg.norm(tg_true),
                 np.linalg.norm(tg_true), int((s_born >= 1e-3
                                               * s_born[0]).sum()) * 2),
              flush=True)

    # G5: 'lm' row guard + tikhonov branch (first usable freq)
    ifreq = usable[0]
    T_proj = par.unpack(par.pack(fm.data.T[ifreq], B), B)
    S_n = fm.predict(T_proj, ifreq, [0], direction=-1)
    try:
        fitmod.fit_frequency(fm, ifreq, B_b, S_n, [0], direction=-1)
        lm_guard = False
    except ValueError:
        lm_guard = True
    check("G5a: 'lm' with fewer rows than params raises the guard",
          lm_guard)
    r_tik = fitmod.fit_frequency(fm, ifreq, B_b, S_n, [0], direction=-1,
                                 tikhonov=1e-10)
    check("G5b: tikhonov>0 path (trf, augmented rows) converges on "
          "normal-only", r_tik["success"] and r_tik["method"] == "trf",
          "obj = %.3e, tik term = %.3e, nfev = %d"
          % (r_tik["objective"], r_tik["objective_tik"], r_tik["nfev"]))

    nfail = sum(1 for _, ok in RESULTS if not ok)
    print("\n=== SUMMARY (test_fit_smoke) ===")
    for name, ok in RESULTS:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("total wall time: %.1f s" % (time.time() - t_all))
    print("ALL TESTS PASSED" if nfail == 0
          else "%d TEST(S) FAILED" % nfail)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
