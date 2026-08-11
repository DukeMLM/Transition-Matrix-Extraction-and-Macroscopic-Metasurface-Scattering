"""End-to-end identifiability: the noise-whitened Jacobian and its spectrum.

The proposal is emphatic (par. 3, 6, 10 Gate A) that two different gates must
not be confused:

  * WHEEL track   -- write T0 = sum_{alpha=1..40} c_alpha B_alpha^{D4h} and
                     require the noise-whitened complex Jacobian of ALL
                     measured modal S entries with respect to c to have
                     complex rank 40, with usable singular-direction SNR.
  * GENERIC track -- require rank(A) = rank(W) = 30 after flux normalization
                     (transforms.generic_track_metrics), which is a stronger
                     and object-independent condition.

It is also emphatic that angular rank is not evidence of usable information:
the par. 6 seed design has kappa(A) = kappa(W) = 4.5 and is still judged
inconclusive because the cell is 200x larger, the sheet amplitude falls as
1/A_cell, and the reference wheel is weak at that wavelength.  Everything
here therefore reports SIGNAL alongside rank.

Analytic Jacobian
-----------------
With  S = S_empty + W T_eff A,  T_eff = T0 (I - C T0)^{-1},  T0 = sum c B,
a directional derivative in delta T0 = B_alpha is

    dT_eff = (I + T_eff C) B_alpha (I - C T0)^{-1}

(differentiate T_eff (I - C T0) = T0), hence, with the two constant flanks

    L = W (I + T_eff C),        R = (I - C T0)^{-1} A,

    H[:, alpha] = vec( L B_alpha R ).                                 (J1)

S is holomorphic in c, so (J1) is the full complex Jacobian: no separate
Re/Im treatment is needed, and rank(H) = 40 is a statement about complex
rank.  At C = 0 the map is complex-LINEAR, L = W, R = A, and (J1) reduces to
vec(W B_alpha A).

Scope note for milestone M1
---------------------------
A converged rectangular/oblique lattice coupling C is M2 work (Ewald).  With
C = 0 this module computes the BARE Jacobian, which screens designs on the
structural question "do the flux-normalized A and W spans see all 40 D4h
directions at all".  That is a screening statistic, not the final gate:
lattice dressing generically mixes sectors and can only add reachable
directions, but it also degrades conditioning near collective resonances.
A design that fails at C = 0 is not yet disqualified, and one that passes is
not yet qualified.  `coupling.py` supplies a converged C for cells small
enough for the tapered real-space sum; anything larger waits for M2.

Whitening and scaling
---------------------
The gate is on  Sigma_S^{-1/2} H D_c, where Sigma_S is the complex-S error
covariance and D_c carries declared physical coefficient scales (par. 7.3).
With noise normalized so that E|delta S_i / sigma_i|^2 = 1, the posterior
standard deviation along whitened singular direction i is 1 / sigma_i, so

    SNR_i = |c|_i * sigma_i(Sigma_S^{-1/2} H D_c)

and with D_c already carrying the coefficient scale the singular value IS
the SNR of that direction.  The proposal's Gate A threshold is SNR > 10 on
every retained direction.

D_c defaults to the identity: the basis is Frobenius-orthonormal, so an
isotropic prior on the 40 coefficients is the target-INDEPENDENT choice the
design search is required to use (par. 7.3, "must not use the reference
wheel T to choose bright entries or priors").
"""
import numpy as np

from .transforms import t_effective, modal_S, generic_track_metrics


# ------------------------------------------------------------ noise model

def sigma_uniform(n_obs, sigma):
    """Flat complex-S error model: E|dS_i|^2 = sigma^2 for every observable.

    The campaign MEASURED sigma = 2.6333e-3 on the 2x2 specular blocks
    (retrieval/HANDOFF.md, retrieval/results/fit_sigma_from_closure.npz).
    There is no measurement yet for a multimode diffractive cell, so a flat
    model at that level is the honest default: it must be replaced by a
    per-channel covariance from the M3 empty-cell repeatability study before
    any Gate A verdict is final.
    """
    return np.full(int(n_obs), float(sigma))


def whiten(H, sigma):
    """Sigma_S^{-1/2} H for a diagonal covariance given as per-entry sigma."""
    sigma = np.asarray(sigma, dtype=float).ravel()
    if sigma.shape[0] != H.shape[0]:
        raise ValueError("sigma must have one entry per observable row")
    if np.any(sigma <= 0):
        raise ValueError("sigma entries must be positive")
    return H / sigma[:, None]


# ------------------------------------------------------------ the Jacobian

def jacobian(W, A, B, T0=None, C=None):
    """Complex Jacobian H (n_out*n_in, n_basis) of vec(S) w.r.t. c, eq. (J1).

    Parameters
    ----------
    W : (M_out, n) outgoing transform
    A : (n, M_in) incoming transform
    B : (n_basis, n, n) real symmetry basis
    T0 : (n, n), optional
        Expansion point.  Required when C is given; ignored when C is None
        (the map is then linear and the Jacobian is constant).
    C : (n, n), optional
        Lattice coupling.  None (default) means the bare C = 0 Jacobian --
        see the module docstring's scope note.

    vec() is row-major (numpy C order) on the (M_out, M_in) S block, matching
    `modal_S(...).ravel()`.
    """
    B = np.asarray(B)
    if C is None:
        L, R = W, A
    else:
        if T0 is None:
            raise ValueError("T0 is required when C is given")
        n = T0.shape[0]
        Teff = t_effective(T0, C)
        L = W @ (np.eye(n, dtype=complex) + Teff @ C)
        R = np.linalg.solve(np.eye(n, dtype=complex) - C @ T0, A)
    H = np.empty((W.shape[0] * A.shape[1], B.shape[0]), dtype=complex)
    for a in range(B.shape[0]):
        H[:, a] = (L @ B[a] @ R).ravel()
    return H


def jacobian_fd(W, A, B, T0, C=None, h=1e-6):
    """Central finite-difference Jacobian, for gating `jacobian`.

    Perturbs each complex coefficient along the real axis; because S is
    holomorphic in c that single direction determines the whole complex
    derivative, and the imaginary direction is checked separately by the
    caller (tests/retrieval/test_fastfull_design.py does both).
    """
    B = np.asarray(B)
    Cz = np.zeros_like(T0) if C is None else C
    out = np.empty((W.shape[0] * A.shape[1], B.shape[0]), dtype=complex)
    for a in range(B.shape[0]):
        Sp = modal_S(W, t_effective(T0 + h * B[a], Cz), A).ravel()
        Sm = modal_S(W, t_effective(T0 - h * B[a], Cz), A).ravel()
        out[:, a] = (Sp - Sm) / (2.0 * h)
    return out


# ------------------------------------------------------------ the metrics

def wheel_track_metrics(W, A, B, T0=None, C=None, sigma=None,
                        coeff_scale=None, rank_tol=1e-10, snr_floor=10.0):
    """Noise-whitened singular spectrum of the 40-coefficient Jacobian.

    Returns a dict with
      sv               whitened singular values, descending (length n_basis)
      rank             count above rank_tol * sv[0]
      sigma_min/max    sv[-1], sv[0]
      kappa            sv[0] / sv[-1]
      snr_min          sv[-1]         (= SNR of the weakest direction when
                                       coeff_scale carries physical scales)
      n_above_floor    directions with sv >= snr_floor
      full_rank        rank == n_basis
      post_std         per-coefficient posterior std sqrt(diag((H^H H)^-1)),
                       in coeff_scale units -- the observability heatmap
      H                the whitened, scaled Jacobian itself
    """
    H = jacobian(W, A, B, T0=T0, C=C)
    n_obs, n_basis = H.shape
    if sigma is None:
        sigma = sigma_uniform(n_obs, 1.0)
    Hw = whiten(H, sigma)
    if coeff_scale is not None:
        Hw = Hw * np.asarray(coeff_scale, dtype=float).ravel()[None, :]
    sv = np.linalg.svd(Hw, compute_uv=False)
    tol = rank_tol * sv[0] if sv.size and sv[0] > 0 else 0.0
    rank = int((sv > tol).sum())
    out = dict(sv=sv, rank=rank, n_basis=n_basis, n_obs=n_obs,
               sigma_min=float(sv[-1]), sigma_max=float(sv[0]),
               kappa=float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf,
               snr_min=float(sv[-1]),
               n_above_floor=int((sv >= snr_floor).sum()),
               full_rank=bool(rank == n_basis), H=Hw)
    if rank == n_basis:
        G = Hw.conj().T @ Hw
        out["post_std"] = np.sqrt(np.abs(np.diag(np.linalg.inv(G))))
    else:
        out["post_std"] = np.full(n_basis, np.inf)
    return out


def signal_metrics(W, A, T0, C=None, sigma=None, S_empty=None):
    """Flux-normalized scattered-signal level, and its ratio to the noise.

    The proposal's central warning about the generic track is that a cell big
    enough to open eight orders suppresses the sheet amplitude roughly as
    1 / A_cell.  Rank without this number is meaningless, so every design
    report carries both.
    """
    n = T0.shape[0]
    Cz = np.zeros((n, n), dtype=complex) if C is None else C
    S_sca = modal_S(W, t_effective(T0, Cz), A)
    a = np.abs(S_sca)
    out = dict(max_abs=float(a.max()), rms=float(np.sqrt((a ** 2).mean())),
               fro=float(np.linalg.norm(S_sca)))
    if sigma is not None:
        s = np.asarray(sigma, dtype=float).ravel()
        s = s.reshape(S_sca.shape) if s.size == S_sca.size else \
            np.full(S_sca.shape, float(s.mean()))
        out["snr_max"] = float((a / s).max())
        out["snr_rms"] = float(np.sqrt(((a / s) ** 2).mean()))
    if S_empty is not None:
        out["max_abs_total"] = float(np.abs(S_empty + S_sca).max())
    return out


def ensemble_metrics(W, A, B, T_ensemble, C=None, sigma=None,
                     coeff_scale=None, snr_floor=10.0):
    """Worst-case wheel-track metrics over a passive D4h ensemble (par. 7.3).

    The design objective is a MINIMAX over the ensemble, not a value at one
    target: a cell tuned to a single T is exactly the failure the proposal
    forbids.  With C = None the Jacobian does not depend on the expansion
    point at all and the ensemble collapses to one evaluation (reported as
    `ensemble_trivial`), which is worth knowing when reading an M1 report.
    """
    T_ensemble = np.asarray(T_ensemble)
    if T_ensemble.ndim == 2:
        T_ensemble = T_ensemble[None]
    per = []
    for T0 in T_ensemble:
        m = wheel_track_metrics(W, A, B, T0=T0, C=C, sigma=sigma,
                                coeff_scale=coeff_scale, snr_floor=snr_floor)
        m.pop("H", None)
        s = signal_metrics(W, A, T0, C=C, sigma=sigma)
        m["signal"] = s
        per.append(m)
    svmins = [m["sigma_min"] for m in per]
    return dict(per_draw=per,
                ensemble_trivial=bool(C is None),
                worst_sigma_min=float(np.min(svmins)),
                median_sigma_min=float(np.median(svmins)),
                worst_rank=int(min(m["rank"] for m in per)),
                full_rank_all=bool(all(m["full_rank"] for m in per)),
                worst_signal_max=float(min(m["signal"]["max_abs"]
                                           for m in per)))


def multipole_blocks(modes):
    """Invariant (l, pol) x (l', pol') blocks of the T-matrix.

    These are the "per-multipole-block" units Gate E asks for.  Unlike a
    coordinate of the symmetry basis, a block is a property of T itself: it
    is fixed by the mode labels in the tmat.h5 file and is untouched by any
    change of basis inside the 40-dimensional D4h subspace.

    Returns an ordered dict name -> boolean (n, n) mask, name like "E1<-E1"
    (receive electric l = 1, from incident electric l = 1).
    """
    pol = {0: "E", 1: "M"}
    ls = sorted(set(int(x) for x in modes.l))
    out = {}
    for l in ls:
        for p in (0, 1):
            row = (modes.l == l) & (modes.pol == p)
            for lp in ls:
                for pp in (0, 1):
                    col = (modes.l == lp) & (modes.pol == pp)
                    out["%s%d<-%s%d" % (pol[p], l, pol[pp], lp)] = \
                        np.outer(row, col)
    return out


def coefficient_covariance(H_whitened):
    """Cov(c) = (H^H H)^{-1} for the whitened complex-linear model."""
    G = H_whitened.conj().T @ H_whitened
    return np.linalg.inv(G)


def _block_stats(Q_block, Cov):
    """(trace, lambda_max) of Q_b Cov Q_b^H, computed in the small basis.

    Q_b Cov Q_b^H and Cov^(1/2) (Q_b^H Q_b) Cov^(1/2) share their non-zero
    spectrum, so both statistics come from an n_basis x n_basis eigenproblem
    no matter how many T entries the block holds.
    """
    if Q_block.shape[0] == 0:
        return 0.0, 0.0
    G = Q_block.conj().T @ Q_block                 # (n_basis, n_basis)
    w, V = np.linalg.eigh(Cov)
    Vs = V * np.sqrt(np.clip(w.real, 0.0, None))[None, :]
    R = Vs.conj().T @ G @ Vs
    ev = np.linalg.eigvalsh(0.5 * (R + R.conj().T)).real
    ev = np.clip(ev, 0.0, None)
    return float(ev.sum()), float(ev.max())


def recovery_errors(B, modes, Cov, n_obs, T_ref, blocks=None,
                    target_global=0.05, target_block=0.02,
                    dominant_frac=0.05, sigma_used=None):
    """Predicted T-recovery error, invariant, under TWO error models.

    WHY TWO MODELS.  The whitening level sigma is the campaign's measured
    normal-incidence CLOSURE residual, and `retrieval/results/REAL_RETRIEVAL.md`
    par. 4.3 establishes that this residual is dominated by MODEL error, not by
    CST numerical noise: "because the dominant error is systematic rather
    than i.i.d. Gaussian, every chi2 significance quoted ... is indicative".
    A systematic discrepancy does NOT average down over the n_obs modal
    entries, so the usual iid posterior is a LOWER BOUND on the error, not
    an estimate of it.  Both ends of the bracket are therefore reported:

      iid         E||dT||_F^2 = trace(Cov)                    [optimistic]
      systematic  ||dT||_F <= sqrt(n_obs lambda_max(Cov))     [conservative]

    The second is the exact worst case over all deterministic discrepancy
    vectors whose per-entry RMS is sigma: ||dS_whitened||_2 = sqrt(n_obs),
    and ||dc|| <= ||dS_w|| / sigma_min(H_w) = sqrt(n_obs lambda_max(Cov)).
    Its ratio to the iid figure is exactly the averaging gain that a
    systematic error does not deliver.

    NEITHER END IS THE ANSWER.  The iid figure assumes an averaging gain the
    measured discrepancy does not provide; the systematic figure assumes the
    discrepancy aligns with the single worst-conditioned direction of H^+,
    which nothing suggests it does.  The true error depends on the STRUCTURE
    of the discrepancy, which is unmeasured until M3's per-channel
    covariance study.  These numbers therefore bracket a sensitivity, and no
    Gate E verdict should be read off either one alone; `pass_*` flags exist
    so a caller can state which model a claim rests on, not to certify one.

    INVARIANCE.  Everything is computed from Cov through Q_b Cov Q_b^H with
    Q_b = vec(B) restricted to a block of T ENTRIES.  A real orthogonal
    rotation O of the symmetry basis sends B -> O B, c -> O c,
    Cov -> O Cov O^T and Q -> Q O^T, so Q_b Cov Q_b^H is unchanged.  This
    replaces the earlier per-coordinate "dominant coefficient" statistics,
    which lived in the arbitrary eigenbasis of a degenerate projector and
    moved by a factor of ~4 under exactly such a rotation.
    """
    B = np.asarray(B)
    nb = B.shape[0]
    Q = B.reshape(nb, -1).T                       # (n^2, n_basis), real
    T_ref = np.asarray(T_ref)
    if blocks is None:
        blocks = multipole_blocks(modes)

    tr_all, lam_all = _block_stats(Q, Cov)
    nrm = float(np.linalg.norm(T_ref))
    e_iid = np.sqrt(tr_all) / nrm if nrm > 0 else np.inf
    e_sys = np.sqrt(n_obs * lam_all) / nrm if nrm > 0 else np.inf

    rows = []
    for name, mask in blocks.items():
        idx = np.flatnonzero(mask.ravel())
        nb_ref = float(np.linalg.norm(T_ref[mask]))
        tr, lam = _block_stats(Q[idx], Cov)
        e_b_iid = float(np.sqrt(tr))
        e_b_sys = float(np.sqrt(n_obs * lam))
        rows.append(dict(
            name=name, n_entries=int(len(idx)),
            ref_fro=nb_ref, ref_share=(nb_ref / nrm if nrm > 0 else 0.0),
            err_iid=e_b_iid, err_sys=e_b_sys,
            rel_iid=(e_b_iid / nb_ref if nb_ref > 0 else np.inf),
            rel_sys=(e_b_sys / nb_ref if nb_ref > 0 else np.inf),
            rel_global_sys=(e_b_sys / nrm if nrm > 0 else np.inf)))
    rows.sort(key=lambda r: -r["ref_fro"])
    dom = [r for r in rows if r["ref_share"] >= dominant_frac]
    worst_dom = max((r["rel_sys"] for r in dom), default=np.inf)
    worst_dom_iid = max((r["rel_iid"] for r in dom), default=np.inf)

    out = dict(
        full_rank=True, n_obs=int(n_obs),
        blocks=rows, n_dominant_blocks=len(dom),
        dominant_names=[r["name"] for r in dom],
        dominant_frac=float(dominant_frac),
        fro_err_iid=float(e_iid), fro_err_sys=float(e_sys),
        averaging_gain=(float(e_sys / e_iid) if e_iid > 0 else np.inf),
        block_err_sys=float(worst_dom), block_err_iid=float(worst_dom_iid),
        fro_target=float(target_global), block_target=float(target_block),
        pass_global=bool(e_sys <= target_global),
        pass_block=bool(worst_dom <= target_block),
        pass_global_iid=bool(e_iid <= target_global),
        pass_block_iid=bool(worst_dom_iid <= target_block))
    if sigma_used:
        out["sigma_used"] = float(sigma_used)
        out["sigma_for_global"] = (float(sigma_used) * target_global / e_sys
                                   if np.isfinite(e_sys) and e_sys > 0
                                   else 0.0)
        out["sigma_for_block"] = (float(sigma_used) * target_block / worst_dom
                                  if np.isfinite(worst_dom) and worst_dom > 0
                                  else 0.0)
    return out


def reference_recovery(W, A, B, modes, T_ref, sigma, C=None, **kw):
    """Invariant predicted recovery error for a SPECIFIC target (reporting).

    The design objective must stay target independent (par. 7.3), so this is
    never called by the search.  It answers Gate E's question directly:
    global Frobenius error and per-multipole-block error, under both the
    optimistic iid and the conservative systematic model.
    """
    B = np.asarray(B)
    m = wheel_track_metrics(W, A, B, T0=T_ref, C=C, sigma=sigma)
    if not m["full_rank"]:
        return dict(rank=m["rank"], full_rank=False, n_obs=m["n_obs"],
                    fro_err_iid=np.inf, fro_err_sys=np.inf,
                    block_err_sys=np.inf, block_err_iid=np.inf,
                    pass_global=False, pass_block=False, blocks=[],
                    n_dominant_blocks=0, dominant_names=[])
    Cov = coefficient_covariance(m["H"])
    sig_scalar = float(np.mean(np.asarray(sigma, dtype=float)))
    out = recovery_errors(B, modes, Cov, m["n_obs"], T_ref,
                          sigma_used=sig_scalar, **kw)
    out.update(rank=m["rank"], full_rank=True, sigma_min=m["sigma_min"],
               n_obs=m["n_obs"])
    return out


def track_report(k, channels, A, W, B, T_ensemble=None, C=None, sigma=None,
                 snr_floor=10.0):
    """Both tracks' Gate-A numbers for one (design, frequency) pair."""
    rep = dict(k=float(k), lam_um=float(2 * np.pi / k),
               n_orders=channels.n // 4, n_channels=channels.n,
               n_obs=channels.n ** 2,
               area=float(channels.orders.lattice.area),
               wood_margin=channels.orders.wood_margin_actual(),
               grazing_margin=channels.orders.grazing_margin_actual())
    rep["generic"] = generic_track_metrics(A, W)
    if T_ensemble is not None:
        if sigma is None:
            sigma = sigma_uniform(channels.n ** 2, 1.0)
        rep["wheel"] = ensemble_metrics(W, A, B, T_ensemble, C=C,
                                        sigma=sigma, snr_floor=snr_floor)
    return rep


def format_report(rep, snr_floor=10.0):
    g = rep["generic"]
    lines = [
        "  lambda = %.3f um   cell = %.1f um^2   orders = %d   channels = %d"
        % (rep["lam_um"], rep["area"], rep["n_orders"], rep["n_channels"]),
        "  margins: Wood %.4f   grazing kz/k %.4f"
        % (rep["wood_margin"], rep["grazing_margin"]),
        "  generic track: rank(A) = %d, rank(W) = %d  (need 30)"
        % (g["rank_A"], g["rank_W"]),
        "                 kappa(A) = %.3g, kappa(W) = %.3g  (prefer <= 10)"
        % (g["kappa_A"], g["kappa_W"]),
    ]
    if "wheel" in rep:
        w = rep["wheel"]
        lines += [
            "  wheel track:   worst rank = %d / 40%s"
            % (w["worst_rank"], "" if w["full_rank_all"] else "  <-- FAIL"),
            "                 worst sigma_40 = %.4g   (SNR floor %.3g)"
            % (w["worst_sigma_min"], snr_floor),
            "                 worst |S_sca|max = %.4g"
            % (w["worst_signal_max"]),
        ]
        if w["ensemble_trivial"]:
            lines.append("                 [C = 0: bare screening Jacobian, "
                         "ensemble is degenerate]")
    return "\n".join(lines)
