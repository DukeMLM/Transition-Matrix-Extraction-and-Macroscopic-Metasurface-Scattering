"""Gates for retrieval/fastfull: jacobian, cost, design.

Run:  python test_fastfull_design.py        (from retrieval/, env cst_inference)

  (a) the analytic Jacobian (J1) matches a central finite difference, at
      C = 0 and with the real cached lattice coupling, along BOTH the real
      and the imaginary coefficient directions (holomorphy)
  (b) the Kronecker identity: sv(W kron A^T) = {sv_i(W) sv_j(A)}, so the
      generic-track objective sigma_30(W) sigma_30(A) really is the weakest
      measured T_eff direction
  (c) inequality (D1): sigma_40(H_wheel) >= sigma_30(W) sigma_30(A) -- the
      quantitative form of "the generic gate is the stronger one"
  (d) noise-free generic algebraic loop: S -> T_eff = W+ S A+ -> T0 by the
      stable solve, recovering an injected T0 exactly
  (e) noise-free wheel algebraic loop at C = 0: the 40 coefficients come back
      from a linear solve against H
  (f) whitening and posterior-covariance semantics
  (g) cost proxy: reproduces its measured anchor, and is monotone in area
      and RHS count
  (h) design.evaluate on the proposal par. 6 seed design; constraint
      rejection actually rejects
  (i) the search driver runs end to end and returns a feasible design
"""
import sys


import numpy as np

from tmatrix.retrieval.fastfull import lattice as lt
from tmatrix.retrieval.fastfull import transforms as xf
from tmatrix.retrieval.fastfull import symmetry as sym
from tmatrix.retrieval.fastfull import jacobian as jac
from tmatrix.retrieval.fastfull import cost as ct
from tmatrix.retrieval.fastfull import design as dz

from tmatrix.aggregation.vswf import ModeBasis
from tmatrix.paths import DEMO_TMAT

RESULTS = []


def record(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s\n         %s" % ("PASS" if ok else "FAIL", name, detail),
          flush=True)


# --------------------------------------------------------------- fixtures

def seed_design_pieces(lam_um=20.0, modes=None):
    """A, W, channels of the proposal par. 6 seed cell at `lam_um`."""
    modes = modes or ModeBasis.standard(3)
    k = 2 * np.pi / lam_um
    lat = lt.Lattice2D.rectangular(26.0, 33.8)
    o = lt.enumerate_orders(lat, k, f_bloch=(0.090, -0.460), kz_min_frac=0.0,
                            wood_margin=0.0)
    ch = lt.ChannelSet(o)
    return k, ch, xf.build_A(k, ch, modes), xf.build_W(k, ch, modes), modes


def single_order_pieces(ifreq=48, th_deg=60.0, ph_deg=22.5):
    """A, W and the REAL cached lattice coupling of the campaign cell."""
    from tmatrix.retrieval.forward import ForwardModel
    fm = ForwardModel()
    k = fm.k[ifreq]
    ia = fm.angle_index(th_deg, ph_deg)
    C = fm.C[ifreq, ia] if fm.have[ifreq, ia] else None
    th, ph = np.deg2rad(th_deg), np.deg2rad(ph_deg)
    kB = k * np.sin(th) * np.array([np.cos(ph), np.sin(ph)])
    lat = lt.Lattice2D.square(fm.pitch)
    o = lt.enumerate_orders(lat, k, k_bloch=kB, kz_min_frac=0.0,
                            wood_margin=0.0)
    ch = lt.ChannelSet(o)
    return (k, ch, xf.build_A(k, ch, fm.modes), xf.build_W(k, ch, fm.modes),
            fm.modes, C, fm.data.T[ifreq])


# ------------------------------------------------------------ (a) Jacobian

def test_jacobian_fd(B):
    modes = ModeBasis.standard(3)
    rng = np.random.default_rng(3)
    T_ens, _ = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.15)
    T0 = T_ens[0]

    cases = []
    k, ch, A, W, _ = seed_design_pieces(20.0, modes)
    cases.append(("seed cell, C = 0", W, A, None))
    k2, ch2, A2, W2, _, C2, _ = single_order_pieces()
    if C2 is not None:
        cases.append(("campaign cell, real cached C", W2, A2, C2))

    for name, Wm, Am, C in cases:
        H = jac.jacobian(Wm, Am, B, T0=T0, C=C)
        Hfd = jac.jacobian_fd(Wm, Am, B, T0, C=C, h=1e-6)
        rel = float(np.abs(H - Hfd).max() / np.abs(H).max())
        # holomorphy: perturbing along i * B must give i * H
        h = 1e-6
        Cz = np.zeros_like(T0) if C is None else C
        col = []
        for a in range(B.shape[0]):
            Sp = xf.modal_S(Wm, xf.t_effective(T0 + 1j * h * B[a], Cz), Am)
            Sm = xf.modal_S(Wm, xf.t_effective(T0 - 1j * h * B[a], Cz), Am)
            col.append(((Sp - Sm) / (2.0 * h)).ravel())
        Himag = np.stack(col, axis=1)
        rel_i = float(np.abs(Himag - 1j * H).max() / np.abs(H).max())
        record("(a) analytic Jacobian == central FD [%s]" % name,
               rel < 1e-7 and rel_i < 1e-7,
               "real direction %.3e, imaginary direction %.3e (relative, "
               "gate 1e-7); H is %dx%d" % (rel, rel_i, H.shape[0],
                                           H.shape[1]))


# ------------------------------------------------------- (b, c) structure

def test_kronecker_identity():
    k, ch, A, W, modes = seed_design_pieces(20.0)
    K = np.kron(W, A.T)                       # row-major vec(W T A)
    sv = np.linalg.svd(K, compute_uv=False)
    svW = np.linalg.svd(W, compute_uv=False)
    svA = np.linalg.svd(A, compute_uv=False)
    want = np.sort(np.outer(svW, svA).ravel())[::-1]
    d = float(np.abs(np.sort(sv)[::-1][:len(want)] - want).max() / want[0])
    # and the vec identity itself
    rng = np.random.default_rng(4)
    T = rng.normal(size=(30, 30)) + 1j * rng.normal(size=(30, 30))
    dv = float(np.abs((W @ T @ A).ravel() - K @ T.ravel()).max()
               / np.abs(W @ T @ A).max())
    record("(b) vec(W T A) = (W kron A^T) vec(T) and sv(K) = sv(W) x sv(A)",
           d < 1e-10 and dv < 1e-12,
           "singular-value set agreement %.3e, vec identity %.3e; "
           "sigma_min(K) = %.4e = sigma_30(W) sigma_30(A) = %.4e"
           % (d, dv, sv.min(), svW[-1] * svA[-1]))


def test_generic_bounds_wheel(B):
    k, ch, A, W, modes = seed_design_pieces(20.0)
    mw = jac.wheel_track_metrics(W, A, B)
    gm = xf.generic_track_metrics(A, W)
    lower = gm["sigma_min_A"] * gm["sigma_min_W"]
    ok = mw["sigma_min"] >= lower * (1 - 1e-9)
    record("(c) sigma_40(H_wheel) >= sigma_30(W) sigma_30(A)  [ineq. (D1)]",
           ok and mw["rank"] == 40,
           "sigma_40(H) = %.4e, sigma_30(W) sigma_30(A) = %.4e, ratio %.3f; "
           "wheel rank %d/40, rank(A) = %d, rank(W) = %d"
           % (mw["sigma_min"], lower, mw["sigma_min"] / lower, mw["rank"],
              gm["rank_A"], gm["rank_W"]))


# ------------------------------------------------------- (d, e) inversions

def test_generic_algebraic_loop(B):
    """Noise-free S -> T_eff -> T0, the proposal's par. 9.3 branch G."""
    modes = ModeBasis.standard(3)
    k, ch, A, W, _ = seed_design_pieces(20.0, modes)
    gm = xf.generic_track_metrics(A, W)
    if not gm["full_rank"]:
        record("(d) generic algebraic loop recovers T0", False,
               "transforms are rank deficient: rank(A) = %d, rank(W) = %d"
               % (gm["rank_A"], gm["rank_W"]))
        return
    rng = np.random.default_rng(7)
    T0 = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.15)[0][0]
    # a synthetic, non-symmetric coupling: lattice dressing must NOT be
    # assumed D4h (proposal par. 5.1)
    C = (rng.normal(size=(30, 30)) + 1j * rng.normal(size=(30, 30))) * 0.02
    Teff = xf.t_effective(T0, C)
    S = xf.scattered_S(W, T0, C, A)
    Teff_hat = np.linalg.pinv(W) @ S @ np.linalg.pinv(A)
    T0_hat, diag = xf.deembed_lattice(Teff_hat, C, return_diag=True)
    e_eff = float(np.abs(Teff_hat - Teff).max() / np.abs(Teff).max())
    e_0 = float(np.abs(T0_hat - T0).max() / np.abs(T0).max())
    record("(d) generic algebraic loop W+ S A+ then (I + Teff C) T0 = Teff",
           e_eff < 1e-9 and e_0 < 1e-9,
           "max rel err: T_eff %.3e, T0 %.3e; sigma_min(I + Teff C) = %.4f, "
           "cond %.2f" % (e_eff, e_0, diag["sigma_min"], diag["cond"]))


def test_wheel_linear_loop(B):
    """Noise-free C = 0 recovery of the 40 coefficients from H."""
    modes = ModeBasis.standard(3)
    k, ch, A, W, _ = seed_design_pieces(20.0, modes)
    rng = np.random.default_rng(8)
    T0 = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.15)[0][0]
    c_true = (B.reshape(len(B), -1).conj() @ T0.reshape(-1))
    S = xf.scattered_S(W, T0, np.zeros((30, 30), dtype=complex), A)
    H = jac.jacobian(W, A, B)
    c_hat, *_ = np.linalg.lstsq(H, S.ravel(), rcond=None)
    e_c = float(np.abs(c_hat - c_true).max() / np.abs(c_true).max())
    T_hat = np.tensordot(c_hat, B, axes=(0, 0))
    e_T = float(np.abs(T_hat - T0).max() / np.abs(T0).max())
    record("(e) wheel linear loop recovers all 40 coefficients at C = 0",
           e_c < 1e-9 and e_T < 1e-9,
           "max rel err: coefficients %.3e, T0 %.3e (40 complex unknowns "
           "from %d complex observables)" % (e_c, e_T, H.shape[0]))


# ----------------------------------------------------------- (f) whitening

def test_whitening_semantics(B):
    modes = ModeBasis.standard(3)
    k, ch, A, W, _ = seed_design_pieces(20.0, modes)
    n_obs = ch.n ** 2
    sig = 3e-3
    m1 = jac.wheel_track_metrics(W, A, B, sigma=jac.sigma_uniform(n_obs, 1.0))
    m2 = jac.wheel_track_metrics(W, A, B, sigma=jac.sigma_uniform(n_obs, sig))
    scale_ok = abs(m2["sigma_min"] * sig - m1["sigma_min"]) \
        <= 1e-9 * m1["sigma_min"]
    # posterior covariance: a Monte-Carlo linear-model check
    Hw = m2["H"]
    rng = np.random.default_rng(9)
    ntr = 400
    errs = np.zeros((ntr, B.shape[0]), dtype=complex)
    Hp = np.linalg.pinv(Hw)
    for i in range(ntr):
        e = (rng.normal(size=Hw.shape[0])
             + 1j * rng.normal(size=Hw.shape[0])) / np.sqrt(2.0)
        errs[i] = Hp @ e
    meas = np.sqrt((np.abs(errs) ** 2).mean(axis=0))
    pred = m2["post_std"]
    ratio = float(np.abs(meas / pred - 1.0).max())
    record("(f) whitening scales as 1/sigma and post_std is the true "
           "posterior spread", scale_ok and ratio < 0.15,
           "sigma_40 x sigma is invariant to %.1e; measured/predicted "
           "coefficient std within %.1f%% over %d noise trials "
           "(worst of 40)" % (abs(m2["sigma_min"] * sig - m1["sigma_min"]),
                              100 * ratio, ntr))


# --------------------------------------------------------------- (g) cost

def test_cost_model():
    cm = ct.CostModel()
    anchor = cm.project(ct.A_REF_UM2, n_orders=1, n_evanescent=0)
    # at the anchor: dof ratio 1, 4 RHS, t = t_fac_ref + 4 t_rhs_ref = 64 s
    ok_anchor = (abs(anchor["dof_ratio"] - 1.0) < 1e-12
                 and anchor["n_rhs"] == 4
                 and abs(anchor["t_total_s"] - 64.0) < 1e-9)
    big = cm.project(878.8, n_orders=8, n_evanescent=2)
    mono = (big["t_total_s"] > anchor["t_total_s"]
            and big["n_rhs"] == 36 and big["mem_gb"] > anchor["mem_gb"])
    pen = ct.cost_penalty(big["t_total_s"], anchor["t_total_s"], 0.5)
    record("(g) cost proxy reproduces its measured anchor and is monotone",
           ok_anchor and mono and 0 < pen < 1,
           "anchor: 4 um^2 / 4 RHS -> %.1f s (campaign measured 50-78 s); "
           "seed cell: 878.8 um^2 / %d RHS -> %.0f s (%.1f min), %.1f GB, "
           "penalty %.4f" % (anchor["t_total_s"], big["n_rhs"],
                             big["t_total_s"], big["t_total_s"] / 60,
                             big["mem_gb"], pen))
    base = ct.baseline_conventional()
    record("(g) conventional benchmark is stated explicitly", True,
           "%d isolated-particle runs x %.0f s = %.1f min (the number the "
           "fast route must beat at matched accuracy)"
           % (base["n_runs"], 64.0, base["t_total_min"]))


# ------------------------------------------------------------- (h, i) design

def test_evaluate_seed(B):
    modes = ModeBasis.standard(3)
    d = dz.Design(26.0, 33.8, 90.0, 0.0, 0.090, -0.460)
    ks = np.array([2 * np.pi / 20.0])
    cons = dz.Constraints(kz_min_frac=0.0, wood_margin=0.0, n_orders_min=1,
                          n_orders_max=40, area_max_um2=5000.0)
    rep = dz.evaluate(d, ks, modes, B=B, constraints=cons)
    f = rep["per_freq"][0]
    ok = (f["n_orders"] == 8 and f["n_channels"] == 32
          and f["generic"]["rank_A"] == 30 and f["generic"]["rank_W"] == 30
          and rep["ok"])
    record("(h) design.evaluate reproduces the par. 6 seed design",
           ok, "orders %d, channels %d, rank(A)/rank(W) = %d/%d, "
               "kappa %.3g/%.3g, wheel rank %d/40, objective %.4g"
               % (f["n_orders"], f["n_channels"], f["generic"]["rank_A"],
                  f["generic"]["rank_W"], f["generic"]["kappa_A"],
                  f["generic"]["kappa_W"], f["wheel"]["rank"],
                  rep["objective_wheel"]))

    # unit-field transforms must reproduce the proposal's quoted kappa ~ 4.5
    k, ch, _, _, _ = seed_design_pieces(20.0, modes)
    Au, Wu = xf.unit_field_transforms(k, ch, modes)
    gm = xf.generic_track_metrics(Au, Wu)
    close = abs(gm["kappa_A"] - 4.5) < 1.0 and abs(gm["kappa_W"] - 4.5) < 1.0
    record("(h) unit-field angular conditioning reproduces the par. 6 result "
           "kappa ~ 4.5", close and gm["full_rank"],
           "unit-field kappa(A) = %.3f, kappa(W) = %.3f, rank %d/%d "
           "(proposal quotes ~4.5); flux-normalized kappa is %.3g/%.3g -- "
           "the physical operator is far worse conditioned"
           % (gm["kappa_A"], gm["kappa_W"], gm["rank_A"], gm["rank_W"],
              f["generic"]["kappa_A"], f["generic"]["kappa_W"]))


def test_constraints_reject():
    modes = ModeBasis.standard(3)
    cons = dz.Constraints(kz_min_frac=0.2, wood_margin=0.05, n_orders_min=8,
                          n_orders_max=24, area_max_um2=500.0)
    # the seed cell is 878.8 um^2: must be rejected by the area bound
    d = dz.Design(26.0, 33.8, 90.0, 0.0, 0.090, -0.460)
    rep = dz.evaluate(d, [2 * np.pi / 20.0], modes, constraints=cons)
    # a sub-wavelength cell has no diffraction orders at all
    d2 = dz.Design(2.0, 2.0, 90.0, 0.0, 0.1, 0.1)
    rep2 = dz.evaluate(d2, [2 * np.pi / 20.0], modes, constraints=cons)
    record("(h) hard constraints actually reject",
           (not rep["ok"]) and rep["objective_generic"] == 0.0
           and (not rep2["ok"]) and rep2["objective_generic"] == 0.0,
           "878.8 um^2 vs 500 bound -> %r; 2x2 um cell -> %r"
           % (rep["reasons"][:1], rep2["reasons"][:1]))


def test_schur_positive_prior():
    """The Schur complement must be right for Q_eta > 0, not only at 0.

    The shortcut M = (I - Je (Je^T Je + Q)^-1 Je^T) Jc, svd(M) is valid only
    at Q = 0, where the bracket is an orthogonal projector.  For Q > 0 it is
    not idempotent and M^T M is not the Schur complement -- on `small@8` it
    gave 3.53 against the true 8.27 at q_eta = 1, and 191 against 407 at
    q_eta = 1e4.  This gates the direct computation against explicit block
    Fisher algebra for zero, isotropic, diagonal and correlated priors.
    """
    from tmatrix.retrieval.fastfull import nuisance as nz
    rng = np.random.default_rng(17)
    n_obs, n_c, n_e = 240, 14, 6
    Jc = rng.normal(size=(n_obs, n_c))
    Je = rng.normal(size=(n_obs, n_e))
    R = rng.normal(size=(n_e, n_e))
    priors = [("zero", 0.0), ("isotropic", 3.0),
              ("diagonal", np.linspace(0.5, 5.0, n_e)),
              ("correlated", R @ R.T + 0.1 * np.eye(n_e))]
    worst, rows = 0.0, []
    for name, Q in priors:
        F = nz.schur_complement(Jc, Je, Q_eta=Q)
        # explicit block inverse of the joint Fisher information
        Qm = (np.zeros((n_e, n_e)) if np.ndim(Q) == 0 and Q == 0
              else (Q * np.eye(n_e) if np.ndim(Q) == 0
                    else (np.diag(Q) if np.ndim(Q) == 1 else Q)))
        Jf = np.block([[Jc.T @ Jc, Jc.T @ Je],
                       [Je.T @ Jc, Je.T @ Je + Qm]])
        F_blk = np.linalg.inv(np.linalg.inv(Jf)[:n_c, :n_c])
        d = float(np.abs(F - F_blk).max() / np.abs(F_blk).max())
        worst = max(worst, d)
        rows.append("%s %.1e" % (name, d))
        sym_err = float(np.abs(F - F.T).max())
        worst = max(worst, sym_err / np.abs(F).max())
    record("(l) the Schur complement is exact for zero and positive priors",
           worst < 1e-9,
           "vs explicit block-Fisher inversion: %s (all relative, gate 1e-9); "
           "matrices are symmetric to roundoff" % ", ".join(rows))


def _small8_pieces():
    """The PUBLISHED small@8 cell with its real Ewald C, at 8 um."""
    from tmatrix.retrieval.fastfull import ewald as ew
    modes = ModeBasis.standard(3)
    d = dz.Design(10.8121, 7.2371, 92.75, 19.68, -0.0098, -0.4930)
    k = 2 * np.pi / 8.0
    lat = d.lattice()
    o = lt.enumerate_orders(lat, k, f_bloch=(d.f1, d.f2), kz_min_frac=0.2,
                            wood_margin=0.05)
    ch = lt.ChannelSet(o)
    return (k, ch, xf.build_A(k, ch, modes), xf.build_W(k, ch, modes), modes,
            ew.lattice_sum_C(lat, k, modes, lat.bloch(d.f1, d.f2)))


def test_worst_nuisance_direction(B):
    """The per-class audit must report the WORST member, not the leading one.

    Run on the PUBLISHED `small@8` cell with its real Ewald C, so the gate
    reproduces the numbers the documents quote (an earlier version of this
    test called `seed_design_pieces`, i.e. the proposal's 26 x 33.8 um seed
    at C = 0, and therefore gated different numbers than it claimed to).

    Also covers the rank-deficiency counterexample: using every returned left
    singular vector of Jc as col(Jc) reports perfect collinearity for a
    rank-deficient T Jacobian.
    """
    from tmatrix.retrieval.fastfull import nuisance as nz
    from tmatrix.aggregation.tmat_io import TMatrixData
    k, ch, A, W, modes, C = _small8_pieces()
    data = TMatrixData(str(DEMO_TMAT))
    T = data.T[int(np.argmin(np.abs(data.wavelength_um - 8.0)))]
    info = nz.marginalized_information(W, A, B, T, C, 2.8417e-3, ch)
    ph = info["per_class"]["phase_rx"]
    rp = info["per_class"]["ref_plane"]
    ok = (abs(100 * ph["apparent_T_error"] - 24.37) < 0.5
          and abs(100 * ph["projection_into_colH"] - 99.989) < 0.01
          and abs(ph["min_principal_angle_deg"] - 0.863) < 0.05
          and abs(100 * rp["apparent_T_error"] - 23.92) < 0.5
          and ph["apparent_T_error"] > 3 * ph["apparent_T_error_leading"])
    record("(l) worst-direction audit reproduces the published small@8 "
           "numbers", ok,
           "phase_rx worst dT %.2f%% (published 24.37; leading-vector "
           "%.2f%%), max projection %.4f%% (published 99.989), principal "
           "angle %.3f deg (published 0.863); ref_plane worst dT %.2f%% "
           "(published 23.92)"
           % (100 * ph["apparent_T_error"],
              100 * ph["apparent_T_error_leading"],
              100 * ph["projection_into_colH"],
              ph["min_principal_angle_deg"],
              100 * rp["apparent_T_error"]))

    # rank-deficient col(Jc) must not read as perfect collinearity
    Jc = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    Jg = np.array([[0.0], [1.0], [0.0], [0.0]])
    Bt = np.zeros((1, 2, 2))
    Bt[0, 0, 0] = 1.0
    d = nz.worst_direction(Jc, Jg, 4, Bt, 0.1 * np.eye(2))
    record("(l) a rank-deficient T Jacobian does not read as collinear",
           d["projection_into_colH"] < 1e-10
           and d["min_principal_angle_deg"] > 89.9,
           "projection %.2e, principal angle %.2f deg (truth 0 and 90); "
           "without rank truncation this reported 1.0 and 0 deg"
           % (d["projection_into_colH"], d["min_principal_angle_deg"]))


def test_generalized_loss():
    """The loss metric must equal the generalized eigenvalue EXACTLY.

    An earlier version only checked that sampled directions stayed below the
    reported value, so monkeypatching the result to inf would have passed.
    This asserts the closed-form value, covers the singular and near-singular
    branches, and keeps the sampling check as a secondary consistency test.
    """
    from tmatrix.retrieval.fastfull import nuisance as nz
    from scipy.linalg import eigh
    rng = np.random.default_rng(29)
    Jc = rng.normal(size=(150, 10))
    Je = rng.normal(size=(150, 4))
    F_free = Jc.T @ Jc
    F = nz.schur_complement(Jc, Je)
    gl = nz.generalized_loss(F_free, F)
    exact = float(np.sqrt(eigh(F_free, F, eigvals_only=True).max()))
    d_exact = abs(gl - exact) / exact

    Fi, Ffi = np.linalg.inv(F), np.linalg.inv(F_free)
    best = 0.0
    for _ in range(4000):
        v = rng.normal(size=10)
        best = max(best, (v @ Fi @ v) / (v @ Ffi @ v))
    sv_ratio = (np.linalg.svd(Jc, compute_uv=False)[-1]
                / np.sqrt(np.clip(np.linalg.eigvalsh(F), 0, None)).min())

    # SINGULAR branch: any nuisance whose column space lies inside col(Jc)
    # absorbs that direction exactly, whatever its amplitude or tilt WITHIN
    # col(Jc) -- tilting inside the space does not make it near-singular.
    Jc_s = np.eye(2)
    gl_sing = nz.generalized_loss(
        Jc_s.T @ Jc_s, nz.schur_complement(Jc_s, np.array([[0.02], [1.0]])))
    # NEAR-singular branch: the nuisance must lean partly OUT of col(Jc), so
    # a small component of the absorbed direction survives.
    Jc_n = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    Je_n = np.array([[0.0], [1.0], [0.05]])
    gl_near = nz.generalized_loss(Jc_n.T @ Jc_n,
                                  nz.schur_complement(Jc_n, Je_n))
    record("(l) generalized loss equals the exact generalized eigenvalue",
           d_exact < 1e-10 and np.sqrt(best) <= gl * (1 + 1e-9)
           and gl >= sv_ratio and not np.isfinite(gl_sing)
           and np.isfinite(gl_near) and gl_near > 10.0,
           "closed form %.6f matches eigh(F_free, F_marg) to %.1e; 4000 "
           "random directions reach %.4f (never above); the singular-value "
           "ratio %.4f is strictly weaker; a fully absorbed direction gives "
           "inf and a near-absorbed one %.3e (finite)"
           % (gl, d_exact, np.sqrt(best), sv_ratio, gl_near))


def test_prior_interfaces():
    """Objective and estimator must accept and reject the same prior."""
    from tmatrix.retrieval.fastfull import nuisance as nz
    from tmatrix.retrieval.fastfull import synthetic as sy
    bad = [("negative scalar", -2.0), ("negative vector", np.array([1.0, -1.0])),
           ("nonsymmetric", np.array([[1.0, 2.0], [0.0, 1.0]])),
           ("indefinite", np.array([[1.0, 0.0], [0.0, -1.0]]))]
    rows, ok = [], True
    for name, Q in bad:
        n_e = 2 if np.ndim(Q) else 1
        r1 = r2 = "accepted"
        try:
            nz.validate_prior_precision(Q, n_e)
        except ValueError:
            r1 = "rejected"
        try:
            sy._prior_whitener(Q, n_e)
        except ValueError:
            r2 = "rejected"
        ok &= (r1 == "rejected" and r2 == "rejected")
        rows.append("%s: objective %s / estimator %s" % (name, r1, r2))
    # and the whitener really reproduces the precision
    R = np.random.default_rng(5).normal(size=(4, 4))
    Q = R @ R.T + 0.5 * np.eye(4)
    L = sy._prior_whitener(Q, 4)
    d = float(np.abs(L.T @ L - Q).max() / np.abs(Q).max())
    ok &= d < 1e-12
    record("(l) prior precision is validated identically on both paths", ok,
           "%s; the whitener satisfies L^T L = Q to %.1e (a negative scalar "
           "previously passed silently on the estimator side and produced "
           "F_marg > F_free on the objective side)" % ("; ".join(rows), d))


def test_schur_unit_invariance():
    """The Schur complement must not depend on nuisance UNITS, and must not
    delete valid modes of an anisotropic prior.

    `pinv(Je^T Je + Q, rcond)` failed both: Je = diag(1, 1e-6) and Je = I
    span the same nuisance space but returned F_marg = 1 and 0, and with
    Jc = Je = I and Q = diag(1e12, 1) the relative cutoff dropped the weaker
    positive-prior mode, giving F = I where the exact answer is diag(1, 0.5).
    """
    from tmatrix.retrieval.fastfull import nuisance as nz
    Jc = np.eye(2)
    f_I = np.linalg.eigvalsh(nz.schur_complement(Jc, np.eye(2))).min()
    f_s = np.linalg.eigvalsh(
        nz.schur_complement(Jc, np.diag([1.0, 1e-6]))).min()
    F = nz.schur_complement(np.eye(2), np.eye(2),
                            Q_eta=np.diag([1e12, 1.0]))
    exact = np.diag([1e12 / (1e12 + 1.0), 0.5])
    d_exact = float(np.abs(F - exact).max())
    # a random rescaling of the nuisance coordinates must not move anything
    rng = np.random.default_rng(11)
    Jc3 = rng.normal(size=(60, 6))
    Je3 = rng.normal(size=(60, 3))
    D = np.diag([1e-4, 1.0, 1e5])
    Q3 = np.diag([2.0, 3.0, 5.0])
    # eta -> D^-1 eta sends Je -> Je D and the PRECISION Q -> D Q D (the
    # covariance would map the other way; getting this backwards is the same
    # covariance/precision slip the prior interface had)
    F_a = nz.schur_complement(Jc3, Je3, Q_eta=Q3)
    F_b = nz.schur_complement(Jc3, Je3 @ D, Q_eta=D @ Q3 @ D)
    d_scale = float(np.abs(F_a - F_b).max() / np.abs(F_a).max())

    # FULL-MATRIX equality, free nuisance: rank revelation must depend only
    # on the column SPACE.  Reading the rank from the unscaled Je made
    # diag(1, 1e-12) come out as rank 1 and return diag(0, 1) where I
    # returns 0, even though the two span the same space.
    F_I = nz.schur_complement(np.eye(2), np.eye(2))
    F_w = nz.schur_complement(np.eye(2), np.diag([1.0, 1e-12]))
    d_free = float(np.abs(F_I - F_w).max())
    # and every representation of a zero prior must take the SAME branch
    zeros = [None, 0.0, np.zeros(2), np.zeros((2, 2))]
    Fz = [nz.schur_complement(np.eye(2), np.diag([1.0, 1e-12]), Q_eta=q)
          for q in zeros]
    d_zero = max(float(np.abs(F - Fz[0]).max()) for F in Fz[1:])
    record("(l) the Schur complement is unit invariant and keeps valid "
           "prior modes",
           abs(f_I) < 1e-10 and abs(f_s) < 1e-10 and d_exact < 1e-6
           and d_scale < 1e-8 and d_free < 1e-12 and d_zero < 1e-12,
           "Je = I and Je = diag(1, 1e-6) both give F_marg_min = %.1e / %.1e; "
           "Q = diag(1e12, 1) gives the exact diag(1, 0.5) to %.1e; a "
           "1e-4..1e5 rescaling moves F by %.1e; full-matrix equality for "
           "diag(1, 1e-12) vs I is %.1e; all four zero-prior representations "
           "(None, 0.0, zero vector, zero matrix) agree to %.1e"
           % (f_I, f_s, d_exact, d_scale, d_free, d_zero))


def test_prior_projection_consistency():
    """A borderline-indefinite prior must be projected ONCE and shared.

    Previously the validator returned a slightly negative eigenvalue
    unchanged: the Schur path produced NEGATIVE information while the
    whitener clipped the same eigenvalue to zero, so objective and estimator
    were using different priors.  Non-finite input passed both.
    """
    from tmatrix.retrieval.fastfull import nuisance as nz
    from tmatrix.retrieval.fastfull import synthetic as sy
    rows, ok = [], True
    for name, Q in (("NaN", float("nan")), ("Inf", float("inf")),
                    ("-Inf vector", np.array([1.0, -np.inf]))):
        n_e = 2 if np.ndim(Q) else 1
        try:
            nz.validate_prior_precision(Q, n_e)
            rows.append("%s ACCEPTED" % name)
            ok = False
        except ValueError:
            rows.append("%s rejected" % name)
    worst_gap, worst_eig = 0.0, 0.0
    for Q in (np.diag([1.0, -5e-11]), np.diag([1e9, -0.05])):
        Qv = nz.validate_prior_precision(Q, 2)
        L = sy._prior_whitener(Q, 2)
        worst_gap = max(worst_gap,
                        float(np.abs(L.T @ L - Qv).max()
                              / max(np.abs(Qv).max(), 1.0)))
        worst_eig = min(worst_eig,
                        float(np.linalg.eigvalsh(np.linalg.eigvalsh(Qv)
                                                 * np.eye(2)).min()))
        F = nz.schur_complement(np.eye(2), np.eye(2), Q_eta=Q)
        ok &= np.linalg.eigvalsh(F).min() >= -1e-12
    ok &= worst_gap < 1e-10 and worst_eig >= 0.0
    record("(l) a borderline prior is projected once and shared exactly",
           ok, "%s; after projection the validated Q is PSD, the whitener "
               "satisfies L^T L = Q_validated to %.1e, and the Schur "
               "complement is non-negative -- objective and estimator now "
               "consume the identical matrix" % ("; ".join(rows), worst_gap))


def test_basis_rotation_invariance(B):
    """Every REPORTED recovery number must survive a rotation of the basis.

    The 40 basis matrices come from eigenvectors of a degenerate projector,
    so any real orthogonal O gives an equally valid basis O B spanning the
    same D4h + reciprocity space and describing the same physics.  The 2026-
    08-07 review demonstrated that the previous per-coordinate metrics moved
    by ~4x under exactly this rotation (a "1.10% error on 2 dominant
    coefficients" became 4.4-5.3% on 26-37 of them).  This gate locks the
    replacement metrics down: global Frobenius error, per-multipole-block
    error and the sigma requirements are computed through Q_b Cov Q_b^H and
    are invariant by construction -- so any future regression that
    reintroduces a coordinate-dependent statistic fails here.
    """
    modes = ModeBasis.standard(3)
    k, ch, A, W, _ = seed_design_pieces(20.0, modes)
    from tmatrix.aggregation.tmat_io import TMatrixData
    data = TMatrixData(str(DEMO_TMAT))
    i = int(np.argmin(np.abs(data.wavelength_um - 8.0)))
    T_ref = data.T[i]
    sig = jac.sigma_uniform(ch.n ** 2, 2.8417e-3)

    base = jac.reference_recovery(W, A, B, modes, T_ref, sig)
    rng = np.random.default_rng(31)
    worst = dict(fro=0.0, block=0.0, names=0)
    for _ in range(12):
        O, _r = np.linalg.qr(rng.normal(size=(len(B), len(B))))
        B2 = np.tensordot(O, B, axes=(1, 0))
        g = np.abs(B2.reshape(len(B2), -1) @ B2.reshape(len(B2), -1).T
                   - np.eye(len(B2))).max()
        assert g < 1e-10, "rotated basis lost orthonormality"
        rot = jac.reference_recovery(W, A, B2, modes, T_ref, sig)
        worst["fro"] = max(worst["fro"],
                           abs(rot["fro_err_sys"] - base["fro_err_sys"])
                           / base["fro_err_sys"],
                           abs(rot["fro_err_iid"] - base["fro_err_iid"])
                           / base["fro_err_iid"])
        worst["block"] = max(worst["block"],
                             abs(rot["block_err_sys"] - base["block_err_sys"])
                             / base["block_err_sys"])
        worst["names"] = max(worst["names"],
                             0 if rot["dominant_names"]
                             == base["dominant_names"] else 1)
    record("(k) reported recovery metrics are invariant under a rotation of "
           "the 40-dim symmetry basis",
           worst["fro"] < 1e-9 and worst["block"] < 1e-9
           and worst["names"] == 0,
           "over 12 random orthogonal rotations: global error changes by "
           "%.2e, worst-block error by %.2e, dominant-block naming "
           "unchanged = %s (base: %.4f%% global sys, %.4f%% block sys, "
           "blocks %s)"
           % (worst["fro"], worst["block"], worst["names"] == 0,
              100 * base["fro_err_sys"], 100 * base["block_err_sys"],
              base["dominant_names"]))


def test_systematic_bracket(B):
    """The systematic bound must be the exact worst case, above the iid one.

    Checks (i) fro_err_sys / fro_err_iid = sqrt(n_obs lambda_max / trace),
    the averaging gain an iid model claims and a systematic discrepancy does
    not deliver; (ii) a Monte-Carlo search over unit-RMS discrepancy vectors
    never exceeds the bound and approaches it along the worst direction.
    """
    modes = ModeBasis.standard(3)
    k, ch, A, W, _ = seed_design_pieces(20.0, modes)
    rng = np.random.default_rng(41)
    T_ref = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.15)[0][0]
    n_obs = ch.n ** 2
    sig = jac.sigma_uniform(n_obs, 3e-3)
    m = jac.wheel_track_metrics(W, A, B, sigma=sig)
    Cov = jac.coefficient_covariance(m["H"])
    rc = jac.recovery_errors(B, modes, Cov, n_obs, T_ref, sigma_used=3e-3)

    ev = np.linalg.eigvalsh(Cov).real
    ratio = np.sqrt(n_obs * ev.max() / ev.sum())
    ok_ratio = abs(rc["averaging_gain"] - ratio) <= 1e-9 * ratio

    # empirical worst case over deterministic dS with ||dS_w||_2 = sqrt(n_obs)
    Hp = np.linalg.pinv(m["H"])
    nrm = np.linalg.norm(T_ref)
    worst = 0.0
    for _ in range(300):
        d = rng.normal(size=n_obs) + 1j * rng.normal(size=n_obs)
        d *= np.sqrt(n_obs) / np.linalg.norm(d)
        worst = max(worst, np.linalg.norm(Hp @ d) / nrm)
    U, S, Vh = np.linalg.svd(m["H"], full_matrices=False)
    d_worst = U[:, -1] * np.sqrt(n_obs)
    attained = np.linalg.norm(Hp @ d_worst) / nrm
    record("(k) the systematic bound is the exact worst case and brackets "
           "the iid figure",
           ok_ratio and worst <= rc["fro_err_sys"] * (1 + 1e-9)
           and abs(attained - rc["fro_err_sys"]) <= 1e-9 * rc["fro_err_sys"],
           "sys/iid = %.4f (= sqrt(n_obs lam_max/trace) = %.4f); 300 random "
           "unit-RMS discrepancies reach %.4f%% <= bound %.4f%%, the worst "
           "singular direction attains it exactly"
           % (rc["averaging_gain"], ratio, 100 * worst,
              100 * rc["fro_err_sys"]))


def test_scale_invariance(B):
    """Scaling cell and wavelength together must leave W T A untouched.

    S-parameters are dimensionless, so for a FIXED T the whole measurement
    operator is invariant under (lambda, a1, a2) -> alpha (lambda, a1, a2).
    Checking it catches any dimensional slip in the flux factors: A carries
    nu ~ 1/alpha, W carries alpha, and only their product is physical.  The
    M1 study leans on this -- it is why the 20 um designs come out as exact
    rescalings of the 8 um ones and why the ONLY thing that changes between
    the two wavelengths is the wheel's own T and the CST cost.
    """
    modes = ModeBasis.standard(3)
    alpha = 2.5
    rng = np.random.default_rng(21)
    T0 = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.15)[0][0]
    outs = []
    for a in (1.0, alpha):
        k = 2 * np.pi / (8.0 * a)
        lat = lt.Lattice2D.oblique(10.8121 * a, 7.2371 * a, 92.75, 19.68)
        o = lt.enumerate_orders(lat, k, f_bloch=(-0.0098, -0.4930),
                                kz_min_frac=0.2, wood_margin=0.05)
        ch = lt.ChannelSet(o)
        A = xf.build_A(k, ch, modes)
        W = xf.build_W(k, ch, modes)
        outs.append((ch.n, xf.scattered_S(W, T0, np.zeros((30, 30),
                                                          dtype=complex), A),
                     jac.wheel_track_metrics(W, A, B)["sigma_min"]))
    same_n = outs[0][0] == outs[1][0]
    dS = (float(np.abs(outs[0][1] - outs[1][1]).max()
                / np.abs(outs[0][1]).max()) if same_n else np.inf)
    dsv = abs(outs[0][2] - outs[1][2]) / outs[0][2] if same_n else np.inf
    record("(j) lambda-scaling invariance of the flux-normalized operator",
           same_n and dS < 1e-12 and dsv < 1e-12,
           "channels %d vs %d; max rel |dS| = %.3e; sigma_40 relative "
           "change %.3e under a %.1fx rescaling"
           % (outs[0][0], outs[1][0], dS, dsv, alpha))


def test_search_runs(B):
    modes = ModeBasis.standard(3)
    cons = dz.Constraints(kz_min_frac=0.2, wood_margin=0.04, n_orders_min=6,
                          n_orders_max=20, area_max_um2=1200.0)
    best, polished = dz.search([2 * np.pi / 8.0], modes, B=B, track="wheel",
                               n_samples=120, n_polish=2, constraints=cons,
                               seed=101, verbose=False)
    ok = best is not None and polished and polished[0][0] > 0
    if ok:
        rep = dz.evaluate(best, [2 * np.pi / 8.0], modes, B=B,
                          constraints=cons)
        ok = rep["ok"] and rep["per_freq"][0]["wheel"]["rank"] == 40
        detail = ("best %r; objective %.4g (screening %.4g); orders %d, "
                  "wheel rank %d/40, area %.1f um^2"
                  % (best, polished[0][0], polished[0][2],
                     rep["per_freq"][0]["n_orders"],
                     rep["per_freq"][0]["wheel"]["rank"], rep["area_um2"]))
    else:
        detail = "search returned no feasible design"
    record("(i) design.search runs end to end and returns a feasible cell",
           ok, detail)


# ------------------------------------------------------------------ driver

def main():
    print("=== fastfull design gates ===", flush=True)
    modes = ModeBasis.standard(3)
    B, meta = sym.build_d4h_reciprocity_basis(modes)
    assert meta["rank_full"] == 40
    test_jacobian_fd(B)
    test_kronecker_identity()
    test_generic_bounds_wheel(B)
    test_generic_algebraic_loop(B)
    test_wheel_linear_loop(B)
    test_whitening_semantics(B)
    test_cost_model()
    test_evaluate_seed(B)
    test_constraints_reject()
    test_schur_positive_prior()
    test_worst_nuisance_direction(B)
    test_generalized_loss()
    test_prior_interfaces()
    test_schur_unit_invariance()
    test_prior_projection_consistency()
    test_basis_rotation_invariance(B)
    test_systematic_bracket(B)
    test_scale_invariance(B)
    test_search_runs(B)

    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n=== SUMMARY (test_fastfull_design) ===")
    for name, ok, _ in RESULTS:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("ALL %d TESTS PASSED" % len(RESULTS) if nfail == 0
          else "%d of %d TEST(S) FAILED" % (nfail, len(RESULTS)))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
