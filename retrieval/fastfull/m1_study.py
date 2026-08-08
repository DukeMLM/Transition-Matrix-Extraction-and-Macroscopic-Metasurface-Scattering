"""Milestone M1: the physical rank / SNR / cost study (proposal par. 14 M1).

  * enumerate rectangular/oblique Floquet channels
  * build flux-normalized A, W, and the D4h coefficient basis
  * build the wheel Jacobian H and a CST mesh/RHS cost proxy
  * optimize cell, Bloch shift, rotation and channel set first at 8 um, then
    test the difficult 20 um point
  * compare the wheel and generic branches using noise-whitened singular
    spectra, expected signal, and predicted cost

Configurations compared
-----------------------
  seed        the par. 6 preliminary cell (26.0 x 33.8 um, kB = 0.090 b1
              - 0.460 b2), as the published baseline
  small       wheel track, 2-6 retained orders (the "8-12 channel" fast
              branch of par. 8.5)
  medium      wheel track, 6-20 retained orders
  generic     generic algebraic track, >= 8 orders so that 32+ channels can
              span all 30 VSWFs
  pool-2/3    two and three small encodings measured jointly (par. 1 item 4)

Every configuration is designed WITHOUT the reference wheel T and only then
benchmarked against it, per par. 7.3.

Run:  python -m fastfull.m1_study [--quick] [--samples N] [--out DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "aggregation"))
sys.path.insert(0, os.path.dirname(HERE))

from vswf import ModeBasis                                    # noqa: E402
from tmat_io import TMatrixData                               # noqa: E402

from . import design as dz                                    # noqa: E402
from . import symmetry as sym                                 # noqa: E402
from . import cost as ct                                      # noqa: E402

REF_TMAT = os.path.join(HERE, "..", "..", "test", "single",
                        "saw_gold_wl15p0025um.tmat.h5")
OUT_DIR = os.path.join(os.path.dirname(HERE), "results", "fastfull")

LAM_DESIGN = 8.0        # par. 8: design at the strong response first
LAM_HARD = 20.0         # par. 8: the hard low-SNR test


def _ref_T(lam_um):
    d = TMatrixData(REF_TMAT)
    i = int(np.argmin(np.abs(d.wavelength_um - lam_um)))
    return d.T[i], float(d.wavelength_um[i]), float(d.k_at(i)), i


def _snr_line(rc):
    """One-line invariant Gate E summary (systematic / iid bracket)."""
    if not rc or not rc.get("full_rank"):
        return "n/a"
    txt = ("global %.2f%% sys / %.2f%% iid, worst dominant block %.2f%% sys "
           "/ %.2f%% iid  [%s]"
           % (100 * rc["fro_err_sys"], 100 * rc["fro_err_iid"],
              100 * rc["block_err_sys"], 100 * rc["block_err_iid"],
              "PASS" if (rc["pass_global"] and rc["pass_block"]) else "FAIL"))
    if rc.get("sigma_for_global"):
        txt += ("; needs sigma <= %.2e (global) / %.2e (block) vs measured "
                "%.2e" % (rc["sigma_for_global"], rc["sigma_for_block"],
                          rc["sigma_used"]))
    return txt


def _box_for(lam_um, cons_kw=None):
    """Design box scaled with the wavelength AND the requested order count.

    Everything kinematic scales with lambda: a cell opening n orders has
    area ~ n lambda^2 / pi, i.e. primitive lengths ~ lambda sqrt(n/pi).  A
    box fixed in um silently forbids the 20 um designs; a box fixed in
    lambda but not in n wastes almost every sample -- measured at 5 feasible
    draws in 400 for the 2-6 order configuration before this was fixed,
    because a box reaching 5.5 lambda almost always opens dozens of orders.
    """
    box = dict(dz.DEFAULT_BOX)
    n_lo = 2 if cons_kw is None else max(1, cons_kw["n_orders_min"])
    n_hi = 24 if cons_kw is None else cons_kw["n_orders_max"]
    p_lo = 0.55 * lam_um * np.sqrt(n_lo / np.pi)
    p_hi = 1.8 * lam_um * np.sqrt(n_hi / np.pi)
    box["p1"] = (p_lo, p_hi)
    box["p2"] = (p_lo, p_hi)
    return box


def _area_max(lam_um, cons_kw):
    """Area bound scaled with lambda^2 and the requested order count.

    Roughly n_orders orders need area ~ n lambda^2 / pi; 2.5x that leaves
    the search room without letting it wander into cells it could never
    afford.
    """
    return 2.5 * cons_kw["n_orders_max"] * lam_um ** 2 / np.pi


def verify_with_coupling(design, k, modes, B, T_ref, n_ensemble=6,
                         seed=7, constraints=None):
    """Re-measure a design with the REAL Ewald lattice coupling (M2).

    M1's screening Jacobian sets C = 0, which is a necessary caveat but not a
    result.  Now that `fastfull.ewald` supplies a converged C for these
    cells, the caveat can be discharged directly: this recomputes the wheel
    metrics over a passive D4h ensemble WITH C, and against the reference T,
    and reports how far each number moved.

    Returns None when the Ewald sum does not converge for the design (it then
    says so rather than reporting an unconverged number).
    """
    from . import ewald as ew
    from . import lattice as ltc
    from . import transforms as xf
    from . import jacobian as jac

    cons = constraints or dz.Constraints()
    lat = design.lattice()
    orders = ltc.enumerate_orders(lat, k, f_bloch=(design.f1, design.f2),
                                  kz_min_frac=cons.kz_min_frac,
                                  wood_margin=cons.wood_margin)
    if not orders.n_retained:
        return None
    ch = ltc.ChannelSet(orders)
    A = xf.build_A(k, ch, modes)
    W = xf.build_W(k, ch, modes)
    kpar = lat.bloch(design.f1, design.f2)
    C, info = ew.converged_C(lat, k, modes, kpar, return_info=True)
    if C is None:
        return dict(converged=False, reasons=info["reasons"])

    sig = jac.sigma_uniform(ch.n ** 2, dz.measured_sigma(2 * np.pi / k))
    rng = np.random.default_rng(seed)
    T_ens, _ = sym.random_passive_d4h(B, rng, n_draw=int(n_ensemble),
                                      target_fro=float(np.linalg.norm(T_ref)))
    m0 = jac.wheel_track_metrics(W, A, B, sigma=sig)
    ens = jac.ensemble_metrics(W, A, B, T_ens, C=C, sigma=sig)
    rec0 = jac.reference_recovery(W, A, B, modes, T_ref, sig)
    recC = jac.reference_recovery(W, A, B, modes, T_ref, sig, C=C)
    dress = ew.lattice_dressing_strength(C, T_ref)
    Teff = xf.t_effective(T_ref, C)
    _, dediag = xf.deembed_lattice(Teff, C, return_diag=True)
    return dict(
        converged=True, abs_max_C=info["abs_max"],
        eta_spread=max(info["eta_deviations"].values()),
        dressing=dress, deembed=dediag,
        sigma40_C0=m0["sigma_min"],
        sigma40_ensemble_worst=ens["worst_sigma_min"],
        sigma40_ratio=(ens["worst_sigma_min"] / m0["sigma_min"]
                       if m0["sigma_min"] > 0 else np.inf),
        rank_C0=m0["rank"], rank_ensemble_worst=ens["worst_rank"],
        recovery_C0=rec0, recovery_C=recC,
        fro_ratio=(recC["fro_err_sys"] / rec0["fro_err_sys"]
                   if rec0["fro_err_sys"] > 0 else np.inf))


def fmt_coupling(vc, indent=16):
    """Report the M2 with-C verification of a design."""
    pad = " " * indent
    if vc is None:
        return pad + "with C: no retained orders"
    if not vc.get("converged"):
        return pad + "with C: Ewald did not converge (%s)" % \
            "; ".join(vc.get("reasons", []))
    d = vc["dressing"]
    a, b = vc["recovery_C0"], vc["recovery_C"]
    return "\n".join([
        pad + "with the real Ewald C: |C|max %.4g, ||C T|| = %.4f "
        "(Born-valid %s, Neumann bound %.3f)"
        % (vc["abs_max_C"], d["norm_CT"], d["born_valid"],
           d["neumann_bound"]),
        pad + "  de-embedding sigma_min(I + Teff C) = %.5f, cond %.3f "
        "(Gate D error amplification)"
        % (vc["deembed"]["sigma_min"], vc["deembed"]["cond"]),
        pad + "  sigma_40 %.4g -> %.4g over a passive D4h ensemble "
        "(ratio %.4f), rank %d -> %d"
        % (vc["sigma40_C0"], vc["sigma40_ensemble_worst"],
           vc["sigma40_ratio"], vc["rank_C0"], vc["rank_ensemble_worst"]),
        pad + "  global dT %.4f%% -> %.4f%% systematic, %.4f%% -> %.4f%% "
        "iid  (ratio %.4f)"
        % (100 * a["fro_err_sys"], 100 * b["fro_err_sys"],
           100 * a["fro_err_iid"], 100 * b["fro_err_iid"],
           vc["fro_ratio"])])


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, complex):
        return dict(re=o.real, im=o.imag)
    return float(o)


def run(samples=800, polish=3, quick=False, out_dir=OUT_DIR, seed=20260807):
    if quick:
        samples, polish = 120, 1
    os.makedirs(out_dir, exist_ok=True)
    modes = ModeBasis.standard(3)
    print("building the D4h + reciprocity basis ...", flush=True)
    B, meta = sym.build_d4h_reciprocity_basis(modes)
    print("  rank %d (character prediction %d), sigma_h numeric err %.2e"
          % (meta["rank_full"], meta["rank_full_predicted"],
             meta["sigma_h_numeric_err"]), flush=True)

    T8, lam8, k8, i8 = _ref_T(LAM_DESIGN)
    T20, lam20, k20, i20 = _ref_T(LAM_HARD)
    print("  reference wheel: lambda %.3f um max|T| = %.5f;  lambda %.3f um "
          "max|T| = %.5f" % (lam8, np.abs(T8).max(), lam20, np.abs(T20).max()),
          flush=True)
    ks = np.array([k8])
    ks_both = np.array([k8, k20])

    out = dict(generated_by="fastfull.m1_study",
               lam_design_um=lam8, lam_hard_um=lam20,
               sigma_used=dict(lam8=dz.measured_sigma(lam8),
                               lam20=dz.measured_sigma(lam20),
                               source="results/fit_sigma_from_closure.npz, "
                                      "interpolated at the design wavelength"),
               basis=dict(rank=meta["rank_full"],
                          rank_predicted=meta["rank_full_predicted"],
                          sectors=meta["sector_multiplicities"]),
               reference=dict(path=os.path.abspath(REF_TMAT),
                              lam_um=[lam8, lam20],
                              freq_index=[i8, i20],
                              max_abs_T=[float(np.abs(T8).max()),
                                         float(np.abs(T20).max())]),
               configs={})

    configs = [
        ("small", dict(kz_min_frac=0.2, wood_margin=0.05, n_orders_min=2,
                       n_orders_max=6), "wheel"),
        ("medium", dict(kz_min_frac=0.2, wood_margin=0.05, n_orders_min=6,
                        n_orders_max=20), "wheel"),
        ("generic", dict(kz_min_frac=0.2, wood_margin=0.05, n_orders_min=8,
                         n_orders_max=24), "generic"),
    ]

    # ---- baseline: the proposal's own par. 6 seed cell
    seed_design = dz.Design(26.0, 33.8, 90.0, 0.0, 0.090, -0.460)
    loose = dz.Constraints(kz_min_frac=0.0, wood_margin=0.0, n_orders_min=1,
                           n_orders_max=80, area_max_um2=5000.0)
    print("\n[baseline] par. 6 seed cell", flush=True)
    rep = dz.benchmark_reference(seed_design, ks_both, modes, B,
                                 constraints=loose)
    out["configs"]["seed"] = rep
    print(dz._fmt_report(rep, "wheel"), flush=True)

    # ---- single-cell searches, INDEPENDENTLY at each design wavelength
    #      (par. 11: one fixed cell is unlikely to serve 8 and 20 um; the
    #      cell must be rescaled per subband).  The 8 um winner is also
    #      evaluated at 20 um so the fixed-cell claim is measured, not
    #      assumed.
    for name, cons_kw, track in configs:
        for lam, kk, T_ref in ((lam8, k8, T8), (lam20, k20, T20)):
            tag = "%s@%.0f" % (name, lam)
            cons = dz.Constraints(area_max_um2=_area_max(lam, cons_kw),
                                  **cons_kw)
            box = _box_for(lam, cons_kw)
            print("\n[%s] searching (%d samples, track=%s, area <= %.0f "
                  "um^2) ..." % (tag, samples, track, cons.area_max_um2),
                  flush=True)
            t0 = time.time()
            best, polished = dz.search(np.array([kk]), modes, B=B,
                                       track=track, n_samples=samples,
                                       n_polish=polish, constraints=cons,
                                       box=box, seed=seed, verbose=False)
            dt = time.time() - t0
            if best is None:
                print("  no feasible design (%.1f s)" % dt, flush=True)
                out["configs"][tag] = dict(feasible=False)
                continue
            rep = dz.benchmark_reference(best, np.array([kk]), modes, B,
                                         constraints=cons)
            rep["search"] = dict(seconds=dt, n_samples=samples,
                                 n_feasible_top=len(polished), track=track,
                                 lam_um=lam, constraints=dict(cons_kw))
            out["configs"][tag] = rep
            print("  %.1f s" % dt, flush=True)
            print(dz._fmt_report(rep, track), flush=True)
            # M2: discharge the C = 0 caveat with the real Ewald coupling
            vc = verify_with_coupling(best, kk, modes, B, T_ref,
                                      constraints=cons)
            rep["with_coupling"] = vc
            print(fmt_coupling(vc), flush=True)
            if lam == lam8:
                # fixed-cell broadband check (par. 11)
                fixed = dz.evaluate(best, np.array([k20]), modes, B=B,
                                    constraints=cons)
                f = fixed["per_freq"][0] if fixed["per_freq"] else None
                out["configs"][tag]["fixed_cell_at_hard"] = fixed
                print("  fixed-cell check at %.1f um: %s"
                      % (lam20, "no retained orders" if f is None else
                         "%d orders, wheel rank %d/40"
                         % (f["n_orders"], f.get("wheel", {}).get("rank", 0))),
                      flush=True)

    # ---- pooled small encodings (the proposal's fast hypothesis)
    for lam, kk, T_ref in ((lam8, k8, T8), (lam20, k20, T20)):
        cons_small = dz.Constraints(area_max_um2=_area_max(lam, configs[0][1]),
                                    **configs[0][1])
        box = _box_for(lam, configs[0][1])
        for n_enc in (2, 3):
            tag = "pool-%d@%.0f" % (n_enc, lam)
            print("\n[%s] greedy pooled search ..." % tag, flush=True)
            t0 = time.time()
            chosen = dz.search_pool(n_enc, np.array([kk]), modes, B,
                                    constraints=cons_small, box=box,
                                    n_samples=max(80, samples // 3),
                                    n_polish=1, seed=seed + n_enc,
                                    verbose=True)
            dt = time.time() - t0
            if len(chosen) < n_enc:
                out["configs"][tag] = dict(feasible=False)
                print("  only %d of %d encodings feasible" % (len(chosen),
                                                              n_enc))
                continue
            rp = dz.pooled_evaluate(chosen, [kk], modes, B,
                                    constraints=cons_small, T_ref=T_ref)
            out["configs"][tag] = dict(designs=[d.to_dict() for d in chosen],
                                       at_design=rp, lam_um=lam,
                                       search_seconds=dt)
            f = rp["per_freq"][0]
            print("  %.1f um: %d/%d encodings usable, channels %s  rank "
                  "%d/40  sigma_40 %.4g\n           SNR %s\n"
                  "           cost %.1f min, %d RHS total"
                  % (lam, f["n_encodings_used"], len(chosen),
                     f["n_channels"], f["rank"], f["sigma_min"],
                     _snr_line(f.get("recovery")),
                     rp["cost"]["t_campaign_min"],
                     rp["cost"]["n_rhs_total"]), flush=True)

    # ---- the benchmark the fast route must beat
    out["conventional_baseline"] = ct.baseline_conventional()

    js = os.path.join(out_dir, "m1_study.json")
    with open(js, "w") as fh:
        json.dump(out, fh, indent=1, default=_json_default)
    print("\nwrote %s" % js, flush=True)
    md = write_markdown(out, os.path.join(out_dir, "M1_DESIGN_STUDY.md"))
    print("wrote %s" % md, flush=True)
    return out


def _CONFIG_ORDER(out):
    """Configuration keys in report order: baseline, then per wavelength."""
    keys = list(out["configs"].keys())
    head = [k for k in keys if k == "seed"]
    rest = [k for k in keys if k != "seed"]

    def sort_key(k):
        base, _, lam = k.partition("@")
        fam = {"small": 0, "medium": 1, "generic": 2}.get(base, 3)
        return (float(lam) if lam else 0.0, fam, base)

    return head + sorted(rest, key=sort_key)


def write_markdown(out, path):
    L = ["# M1 design study - rank, SNR and cost of a coded Floquet cell", "",
         "Generated by `retrieval/fastfull/m1_study.py`.  Milestone M1 of "
         "`FAST_FULL_TMATRIX_WHEEL_PROPOSAL.md`: no CST, no lattice coupling "
         "(C = 0 screening Jacobian - see `jacobian.py`).", "",
         "Error level: the frequency-matched complex-S discrepancy MEASURED "
         "by the specular campaign's normal-incidence closure "
         "(`results/fit_sigma_from_closure.npz`): %.4e at %.1f um, %.4e at "
         "%.1f um.  That residual is dominated by MODEL error, not CST "
         "numerical noise, so it is **not** an iid floor that averages down "
         "over modal entries; every error figure below is therefore bracketed "
         "systematic / iid.  A multimode cell's real covariance is an M3 "
         "deliverable."
         % (out["sigma_used"]["lam8"], out["lam_design_um"],
            out["sigma_used"]["lam20"], out["lam_hard_um"]), "",
         "Reference wheel (benchmark only, never used to design): "
         "max|T| = %.5f at %.3f um and %.5f at %.3f um."
         % (out["reference"]["max_abs_T"][0], out["reference"]["lam_um"][0],
            out["reference"]["max_abs_T"][1], out["reference"]["lam_um"][1]),
         "",
         "## Summary", "",
         "`dT glob` and `dT block` are the PREDICTED relative T-recovery "
         "errors at the frequency-matched measured sigma, each given as "
         "**systematic / iid**.  The whitening level is a MODEL-vs-CST "
         "discrepancy, not iid noise (`results/REAL_RETRIEVAL.md` par. 4.3), "
         "so the iid figure -- which averages the discrepancy down over all "
         "modal entries -- is a lower bound only, while the systematic figure "
         "assumes the discrepancy aligns with the worst-conditioned direction "
         "of H+.  The truth depends on the discrepancy's STRUCTURE, which is "
         "an M3 measurement: **no Gate E verdict is claimed from either "
         "end**.  Blocks are invariant multipole blocks "
         "(l, pol) x (l', pol'), not coordinates of the symmetry basis.  "
         "`sigma need` is the accuracy required to reach the 5% global "
         "target under the systematic model.", "",
         "| config | cell | orders | channels | wheel rank | "
         "sigma_40 | dT glob % | dT block % | sigma need | signal/sigma | "
         "cost proxy |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def _snr_cells(rc):
        if not rc or not rc.get("full_rank"):
            return "n/a", "n/a", "n/a"
        return ("%.1f / %.1f" % (100 * rc["fro_err_sys"],
                                 100 * rc["fro_err_iid"]),
                "%.1f / %.1f" % (100 * rc["block_err_sys"],
                                 100 * rc["block_err_iid"]),
                "%.1e" % rc.get("sigma_for_global", 0.0))

    def row(name, rep):
        if not rep or rep.get("feasible") is False:
            return ("| %s | - | - | - | - | - | - | - | - | - | infeasible |"
                    % name)
        if "per_freq" in rep and rep["per_freq"]:
            f = rep["per_freq"][0]
            d = rep["design"]
            cell = "%.1fx%.1f g%.0f a%.0f" % (d["p1_um"], d["p2_um"],
                                              d["gamma_deg"], d["alpha_deg"])
            wr = f.get("wheel", {})
            sr, sg, sn = _snr_cells(f.get("recovery"))
            rs = f.get("reference_signal") or {}
            return ("| %s | %s | %d | %d | %s | %.4g | %s | %s | %s | %.2f | "
                    "%.0f min |"
                    % (name, cell, f["n_orders"], f["n_channels"],
                       "%d/40" % wr.get("rank", 0), wr.get("sigma_min", 0.0),
                       sr, sg, sn, rs.get("snr_max", float("nan")),
                       rep["cost"]["t_campaign_min"]))
        f = rep["at_design"]["per_freq"][0]
        sr, sg, sn = _snr_cells(f.get("recovery"))
        return ("| %s | %d cells | %s | %s | %d/40 | %.4g | %s | %s | %s | - "
                "| %.0f min |"
                % (name, rep["at_design"]["n_encodings"],
                   "+".join(str(x) for x in f["n_orders"]),
                   "+".join(str(x) for x in f["n_channels"]),
                   f["rank"], f["sigma_min"], sr, sg, sn,
                   rep["at_design"]["cost"]["t_campaign_min"]))

    for name in _CONFIG_ORDER(out):
        if name in out["configs"]:
            L.append(row(name, out["configs"][name]))
    cb = out["conventional_baseline"]
    L += ["", "Conventional isolated-particle benchmark to beat: %d runs, "
          "%.1f min (cost proxy, matched-accuracy comparison is an M5 "
          "deliverable)." % (cb["n_runs"], cb["t_total_min"]), "",
          "## Per-configuration detail", ""]
    for name in _CONFIG_ORDER(out):
        rep = out["configs"].get(name)
        if not rep:
            continue
        L.append("### %s" % name)
        L.append("")
        L.append("```")
        if rep.get("feasible") is False:
            L.append(" no feasible design in the sampled box")
        elif "per_freq" in rep:
            L.append(dz._fmt_report(rep, "wheel"))
            if "with_coupling" in rep:
                L.append(fmt_coupling(rep["with_coupling"]))
        else:
            L.append(" designs: " + "; ".join(
                "%.2fx%.2f gamma %.1f alpha %.1f f (%.3f, %.3f)"
                % (d["p1_um"], d["p2_um"], d["gamma_deg"], d["alpha_deg"],
                   d["f1"], d["f2"]) for d in rep["designs"]))
            f = rep["at_design"]["per_freq"][0]
            if not f.get("feasible"):
                L.append(" infeasible (every encoding dropped)")
            else:
                L.append(" lam %.3f um: %d/%d encodings usable, channels %s, "
                         "rank %d/40, sigma_40 %.4g"
                         % (f["lam_um"], f["n_encodings_used"],
                            rep["at_design"]["n_encodings"], f["n_channels"],
                            f["rank"], f["sigma_min"]))
                L.append(" reference coefficient SNR: %s"
                         % _snr_line(f.get("recovery")))
            L.append(" total cost proxy %.1f min, %d RHS"
                     % (rep["at_design"]["cost"]["t_campaign_min"],
                        rep["at_design"]["cost"]["n_rhs_total"]))
        L.append("```")
        L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--samples", type=int, default=800)
    p.add_argument("--polish", type=int, default=3)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--out", default=OUT_DIR)
    a = p.parse_args(argv)
    run(samples=a.samples, polish=a.polish, quick=a.quick, out_dir=a.out,
        seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
