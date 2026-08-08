"""Blind noisy synthetic recovery — the M2 Gate A experiment.

Answers the question the M1 error bracket left open. M1 could only say that
the predicted T error lies somewhere between an iid posterior and a
norm-bounded adversarial bound (a factor of ~7 apart for the winning cell).
Neither end is a physical claim. This module measures where *structured*
discrepancies — the kind a CST comparison actually produces — land inside
that bracket, by synthesizing S from a known T0, perturbing it, and running a
blind recovery that never sees the answer.

It implements the reviewer's recommendations 2 and 3 (`review.md`,
2026-08-07): a robustness ENVELOPE over several error models rather than one
flat sigma, and a falsifiable CANDIDATE SET rather than a single refined
winner, with a geometrically distinct encoding held out.

Error models (all scaled to the same per-entry RMS)
--------------------------------------------------
  iid              circular complex Gaussian, independent per modal entry.
                   The optimistic reference; the one the measured closure
                   residual is known NOT to be.
  reference_plane  a port-plane offset dL, giving S -> D S D with
                   D = diag(exp(i k_z dL / 2)).  One scalar parameter, so the
                   induced dS is highly structured and follows the channels'
                   k_z.  Modelled on the DE-EMBEDDED block, i.e. after the
                   common-mode part that the matched empty cell cancels.
  mode_mixing      a small TE/TM rotation within each order, S -> D S D with
                   D block-diagonal 2x2 rotations of angle ~eps.  This is the
                   realistic form of the proposal's "label/field-overlap
                   perturbation": a full sign flip is not small, a few-degree
                   mixing is.
  angular_smooth   a smooth multiplicative error, low-order polynomial in
                   (k_z/k, cos phi, sin phi) applied to each channel, standing
                   in for mesh/discretization error that varies slowly with
                   observation direction.
  adversarial      the exact worst case: the left singular vector of the
                   whitened Jacobian belonging to its smallest singular
                   value.  Attains the conservative bound by construction.

`truncation` is deliberately absent: a genuine lmax = 5 tail for this wheel
does not exist in the repository, and faking one would measure the fake.
That is Gate F's job (proposal par. 10 Gate F).

Recovery (proposal par. 9.2 branch W, par. 9.3 branch G)
--------------------------------------------------------
Wheel branch: blind linear seed from the C = 0 model, then continuation
C -> eta C with eta from 0 to 1, re-minimizing the whitened residual over all
40 complex coefficients at each step, then a final Levenberg-Marquardt
refinement with the analytic Jacobian.  Multistart from perturbed seeds
checks that the basin is unique.  No oracle, no bright mask, no reference T.

Generic branch: T_eff = W^+ S A^+ then the stable solve (I + T_eff C) T0 =
T_eff.  Needs rank(A) = rank(W) = 30, so it only applies to candidates with
at least 30 independent channels.
"""
import numpy as np

from . import lattice as lt
from . import transforms as xf
from . import jacobian as jac

ERROR_MODELS = ("iid", "reference_plane", "mode_mixing", "angular_smooth",
                "adversarial")

# Stable seed tree.  Python's hash() is process-randomized, so deriving a
# trial seed from hash(model_name) makes a run non-reproducible across
# invocations -- it did, and two nominally identical probes moved the iid
# error from 3.014% to 2.774%.  Enumerating the models fixes the offsets.
_MODEL_INDEX = {m: i for i, m in enumerate(ERROR_MODELS)}


def trial_seed(base_seed, model, trial):
    """Deterministic, collision-free seed for (model, trial)."""
    idx = _MODEL_INDEX.get(model, len(_MODEL_INDEX))
    return int(np.random.SeedSequence(
        [int(base_seed), int(idx), int(trial)]).generate_state(1)[0])


# ------------------------------------------------------------ perturbations

def _scale_to_rms(dS, sigma):
    r = float(np.sqrt(np.mean(np.abs(dS) ** 2)))
    if r == 0:
        return dS
    return dS * (float(sigma) / r)


def _congruence(S, d):
    """S -> D S D for a diagonal D (the form every gauge/plane error takes)."""
    return d[:, None] * S * d[None, :]


def _calibrate(make_dS, sigma, t_hi=8.0, iters=60):
    """Bisect a model's PHYSICAL parameter until the RMS discrepancy is sigma.

    Rescaling a structured dS to hit the RMS would destroy the structure --
    lambda (D S D - S) is an interpolation between S and a congruence, not a
    congruence, and its entrywise ratio to S stops being rank 1.  Calibrating
    the parameter itself (a port-plane offset, a mixing angle, a mesh-error
    amplitude) keeps the model exactly the one-parameter family it claims to
    be.  Gated in test_fastfull_synthetic.py (b).
    """
    lo, hi = 0.0, float(t_hi)
    f = lambda t: float(np.sqrt(np.mean(np.abs(make_dS(t)) ** 2)))
    # grow the bracket if the target is not reached at t_hi
    for _ in range(20):
        if f(hi) >= sigma:
            break
        hi *= 2.0
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if f(mid) < sigma:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return make_dS(t), t


def make_perturbation(model, channels, S_clean, rng, sigma, H_whitened=None):
    """A discrepancy of per-entry RMS `sigma`, of the requested structure.

    Returns (dS, info).  `info["rank"]` is the numerical rank of dS, which is
    the quantitative sense in which a model is "structured": iid noise fills
    the whole matrix, a reference-plane error does not.
    """
    n = channels.n
    kz = np.asarray(channels.kz, dtype=float)
    k = float(channels.k)

    param = None
    if model == "iid":
        dS = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) \
            / np.sqrt(2.0)
        dS = _scale_to_rms(dS, sigma)
    elif model == "reference_plane":
        # the physical parameter is the port-plane offset dL, in um
        dS, param = _calibrate(
            lambda t: _congruence(S_clean, np.exp(0.5j * t * kz)) - S_clean,
            sigma)
    elif model == "mode_mixing":
        # the physical parameter is the TE/TM mixing angle, in rad
        eps = rng.normal(size=n // 2)

        def _mix(t):
            M = np.eye(n, dtype=complex)
            for j in range(n // 2):
                a, b = 2 * j, 2 * j + 1     # the TE/TM pair of one port mode
                c, s = np.cos(t * eps[j]), np.sin(t * eps[j])
                M[a, a], M[a, b] = c, -s
                M[b, a], M[b, b] = s, c
            return M @ S_clean @ M.T - S_clean

        dS, param = _calibrate(_mix, sigma)
    elif model == "angular_smooth":
        # the physical parameter is the amplitude of a smooth angular error
        u = kz / k
        ph = np.asarray(channels.phi, dtype=float)
        basis = np.stack([np.ones_like(u), u, u ** 2,
                          np.cos(ph), np.sin(ph),
                          u * np.cos(ph), u * np.sin(ph)])
        w = rng.normal(size=basis.shape[0])
        w /= np.linalg.norm(w)
        eps = (w @ basis).astype(complex)
        dS, param = _calibrate(
            lambda t: _congruence(S_clean, 1.0 + t * eps) - S_clean, sigma)
    elif model == "adversarial":
        if H_whitened is None:
            raise ValueError("adversarial needs the whitened Jacobian")
        U, s, _ = np.linalg.svd(H_whitened, full_matrices=False)
        dS = _scale_to_rms(U[:, -1].reshape(n, n), sigma)
    else:
        raise ValueError("unknown error model %r" % model)

    sv = np.linalg.svd(dS, compute_uv=False)
    return dS, dict(model=model, parameter=param,
                    rank=int((sv > 1e-10 * sv[0]).sum()) if sv[0] > 0 else 0,
                    sv_ratio=float(sv[0] / max(sv.sum(), 1e-300)),
                    rms=float(np.sqrt(np.mean(np.abs(dS) ** 2))),
                    max_abs=float(np.abs(dS).max()))


# --------------------------------------------------------------- recovery

def _real_jacobian(H):
    """Real 2n_obs x 2n_basis Jacobian of [Re r; Im r] w.r.t. [Re c; Im c]."""
    return np.block([[H.real, -H.imag], [H.imag, H.real]])


def _residual(c_real, blocks, B, eta, sig):
    nb = B.shape[0]
    c = c_real[:nb] + 1j * c_real[nb:]
    T0 = np.tensordot(c, B, axes=(0, 0))
    parts = []
    for W, A, C, S_meas in blocks:
        S = xf.scattered_S(W, T0, eta * C, A)
        parts.append(((S - S_meas) / sig).ravel())
    r = np.concatenate(parts)
    return np.concatenate([r.real, r.imag])


def _jac_real(c_real, blocks, B, eta, sig):
    nb = B.shape[0]
    c = c_real[:nb] + 1j * c_real[nb:]
    T0 = np.tensordot(c, B, axes=(0, 0))
    H = np.vstack([jac.jacobian(W, A, B, T0=T0, C=eta * C) / sig
                   for W, A, C, _ in blocks])
    return _real_jacobian(H)


def recover_wheel(blocks, B, sigma, n_eta=5, seed=None, n_multistart=0,
                  rng=None, ftol=1e-12):
    """Blind wheel-branch recovery: linear seed, C-continuation, LM refine.

    `blocks` is a list of (W, A, C, S_measured) tuples -- one per coded cell,
    so a pooled encoding set is just a longer list and the residual is
    stacked.  S_measured is the de-embedded SCATTERED block.

    Nothing here sees the true T0: the seed comes from the C = 0 linear
    model, and continuation walks the lattice coupling in from eta = 0 to 1.
    Returns (T0_hat, info).
    """
    from scipy.optimize import least_squares
    nb = B.shape[0]
    sig = float(sigma)

    # blind linear seed (exact when C = 0)
    H0 = np.vstack([jac.jacobian(W, A, B) for W, A, _, _ in blocks])
    y0 = np.concatenate([S.ravel() for _, _, _, S in blocks])
    c0, *_ = np.linalg.lstsq(H0, y0, rcond=None)
    if seed is not None:
        c0 = np.asarray(seed, dtype=complex)

    etas = np.linspace(0.0, 1.0, int(n_eta)) if n_eta > 1 else np.array([1.0])
    x = np.concatenate([c0.real, c0.imag])
    path = []
    for eta in etas:
        res = least_squares(_residual, x, jac=_jac_real, method="lm",
                            xtol=ftol, ftol=ftol, gtol=ftol,
                            args=(blocks, B, float(eta), sig))
        x = res.x
        path.append(dict(eta=float(eta), cost=float(res.cost),
                         nfev=int(res.nfev)))
    c = x[:nb] + 1j * x[nb:]
    T0 = np.tensordot(c, B, axes=(0, 0))
    info = dict(path=path, final_cost=path[-1]["cost"],
                seed_norm=float(np.linalg.norm(c0)))

    if n_multistart:
        rng = rng or np.random.default_rng(0)
        spread = []
        for _ in range(int(n_multistart)):
            pert = c0 * (1.0 + 0.5 * (rng.normal(size=nb)
                                      + 1j * rng.normal(size=nb)))
            T_alt, _ = recover_wheel(blocks, B, sigma, n_eta=n_eta,
                                     seed=pert, n_multistart=0)
            spread.append(float(np.linalg.norm(T_alt - T0)
                                / max(np.linalg.norm(T0), 1e-300)))
        info["multistart_spread"] = float(max(spread))
        info["multistart_unique"] = bool(max(spread) < 1e-6)
    return T0, info


def _prior_whitener(prior_precision, n_e):
    """Cholesky factor L with L^T L = Q, for a prior PRECISION Q.

    Q is a PRECISION (inverse covariance), not a covariance.  Passing a
    covariance reverses the regularization -- a more uncertain nuisance would
    be constrained more strongly -- so the argument is named
    `prior_precision` and validated here.  Accepts a scalar, a per-parameter
    vector, or a full symmetric PSD matrix, matching what
    `nuisance.schur_complement` takes, so the objective and the estimator can
    be given the identical operator.
    """
    from . import nuisance as nz
    Q = nz.validate_prior_precision(prior_precision, n_e)
    if Q is None:
        return None
    w, V = np.linalg.eigh(Q)
    return (V * np.sqrt(np.clip(w, 0.0, None))[None, :]).T


def recover_joint(blocks, B, sigma, calibs, param_map=None, n_eta=None,
                  prior_precision=None, prior_mean=None, seed_eta=None,
                  ftol=1e-12):
    """Blind recovery of T AND the calibration parameters, jointly.

    `calibs` is one nuisance.Calibration per block.  `param_map` is an
    explicit list of index arrays, one per block, selecting that block's
    parameters out of ONE global eta vector; omit it for the default, in
    which every block gets its own disjoint slice.

    SHARING IS EXPLICIT.  An earlier version of this docstring claimed that
    passing the same Calibration object to two blocks tied their parameters.
    It did not -- the slices were allocated disjointly regardless -- so a
    port-plane offset physically common to two coded cells was silently
    fitted twice.  Use `param_map` to tie them, e.g. for two blocks sharing
    one 1-parameter offset:  param_map=[[0], [0]].

    This is the estimator the marginalized design objective assumes.  A
    T-only fit absorbs a port-plane offset into T -- that is the 24 % error
    Gate A measured -- while this one spends the calibration freedom the cell
    was designed to expose.

    `prior_precision` is the SAME object `nuisance.schur_complement` takes as
    `Q_eta`: a prior PRECISION (inverse covariance), scalar, per-parameter or
    full matrix, so objective and estimator can be handed one calibrated
    operator.  `prior_mean` is the calibrated nuisance mean, defaulting to
    zero.  None/0 precision leaves the parameters free -- conservative, and
    the setting under which the misspecification blow-up was measured.

    Returns (T0_hat, eta_hat, info).
    """
    from scipy.optimize import least_squares
    nb = B.shape[0]
    sig = float(sigma)
    if len(calibs) != len(blocks):
        raise ValueError("need one Calibration per block: %d calibrations "
                         "for %d blocks (zip would silently discard the "
                         "extra)" % (len(calibs), len(blocks)))
    if param_map is None:
        param_map, off = [], 0
        for c in calibs:
            param_map.append(np.arange(off, off + c.n_params))
            off += c.n_params
        n_e = off
    else:
        checked = []
        for ix in param_map:
            arr = np.asarray(ix)
            # validate integrality BEFORE casting: [[0.9], [0.1]] would
            # otherwise floor to a shared index 0 and silently change the
            # inverse problem
            if arr.size and not np.all(arr == np.round(arr)):
                raise ValueError("param_map indices must be integers, got %r"
                                 % (ix,))
            checked.append(arr.astype(int))
        param_map = checked
        if len(param_map) != len(calibs):
            raise ValueError("param_map needs one index array per block")
        for ix, c in zip(param_map, calibs):
            if len(ix) != c.n_params:
                raise ValueError("param_map block has %d indices but the "
                                 "Calibration needs %d" % (len(ix),
                                                           c.n_params))
            if ix.size and int(ix.min()) < 0:
                raise ValueError("param_map indices must be non-negative")
        used = set()
        for ix in param_map:
            used.update(int(v) for v in ix)
        n_e = (int(max(used)) + 1) if used else 0
        # A hole is an unidentifiable free parameter that no block drives, so
        # it is rejected rather than silently carried into the solve.
        if used != set(range(n_e)):
            missing = sorted(set(range(n_e)) - used)
            raise ValueError("param_map leaves parameter(s) %s unused; every "
                             "index in 0..%d must be driven by some block"
                             % (missing[:5], n_e - 1))
    if n_eta is not None:
        if (not np.isfinite(n_eta) or n_eta < 0
                or float(n_eta) != int(n_eta)):
            raise ValueError("n_eta must be a non-negative integer, got %r"
                             % (n_eta,))
        if int(n_eta) != n_e:
            raise ValueError("n_eta = %d does not match the %d parameters "
                             "the param_map drives; a larger n_eta leaves "
                             "flat unidentifiable directions"
                             % (int(n_eta), n_e))
        n_e = int(n_eta)
    L = _prior_whitener(prior_precision, n_e)
    mu = (np.zeros(n_e) if prior_mean is None
          else np.asarray(prior_mean, dtype=float).reshape(n_e))

    def unpack(x):
        c = x[:nb] + 1j * x[nb:2 * nb]
        return c, x[2 * nb:]

    def resid(x):
        c, eta = unpack(x)
        T0 = np.tensordot(c, B, axes=(0, 0))
        parts = []
        for (W, A, C, S_meas), cal, ix in zip(blocks, calibs, param_map):
            S = cal.apply(xf.scattered_S(W, T0, C, A), eta[ix])
            parts.append(((S - S_meas) / sig).ravel())
        r = np.concatenate(parts)
        out = np.concatenate([r.real, r.imag])
        if L is not None:
            out = np.concatenate([out, L @ (eta - mu)])
        return out

    # seed: T from the blind nuisance-free recovery, calibration at identity
    T_seed, _ = recover_wheel(blocks, B, sigma)
    c_seed = B.reshape(nb, -1).conj() @ T_seed.reshape(-1)
    if seed_eta is not None:
        seed_eta = np.asarray(seed_eta, dtype=float).ravel()
        if seed_eta.shape[0] != n_e:
            raise ValueError("seed_eta has %d entries but there are %d "
                             "nuisance parameters"
                             % (seed_eta.shape[0], n_e))
    x0 = np.concatenate([c_seed.real, c_seed.imag,
                         np.zeros(n_e) if seed_eta is None else seed_eta])
    res = least_squares(resid, x0, method="lm", xtol=ftol, ftol=ftol,
                        gtol=ftol)
    c, eta = unpack(res.x)
    T0 = np.tensordot(c, B, axes=(0, 0))
    return T0, eta, dict(cost=float(res.cost), nfev=int(res.nfev),
                         n_eta=n_e)


def recover_generic(S_meas, W, A, C, rcond=1e-12):
    """Branch G: T_eff = W^+ S A^+, then (I + T_eff C) T0 = T_eff."""
    Teff = np.linalg.pinv(W, rcond=rcond) @ S_meas @ np.linalg.pinv(A,
                                                                    rcond=rcond)
    T0, diag = xf.deembed_lattice(Teff, C, return_diag=True)
    return T0, dict(deembed=diag)


# --------------------------------------------------------------- experiment

def candidate_pieces(design, k, modes, constraints):
    """(channels, A, W, C) for one candidate, or None if C does not converge."""
    from . import ewald as ew
    lat = design.lattice()
    orders = lt.enumerate_orders(lat, k, f_bloch=(design.f1, design.f2),
                                 kz_min_frac=constraints.kz_min_frac,
                                 wood_margin=constraints.wood_margin)
    if not orders.n_retained:
        return None
    ch = lt.ChannelSet(orders)
    A = xf.build_A(k, ch, modes)
    W = xf.build_W(k, ch, modes)
    C, info = ew.converged_C(lat, k, modes, lat.bloch(design.f1, design.f2),
                             return_info=True)
    if C is None:
        return None
    return ch, A, W, C, info


def run_candidate(designs, k, modes, B, T_true, sigma, constraints,
                  models=ERROR_MODELS, n_trials=3, seed=12345,
                  n_multistart=2, do_generic=True, label=None):
    """Blind recovery of `T_true` from one candidate ENCODING SET.

    `designs` is a single Design or a list of them; a list is a pooled
    encoding, whose residual is stacked (reviewer recommendation 3: does
    pooling buy robust nonlinear information, or merely more rows?).
    """
    if not isinstance(designs, (list, tuple)):
        designs = [designs]
    got = [candidate_pieces(d, k, modes, constraints) for d in designs]
    got = [g for g in got if g is not None]
    if not got:
        return dict(feasible=False, label=label)

    blocks_clean, chans, n_obs, absC = [], [], 0, 0.0
    for ch, A, W, C, cinfo in got:
        S = xf.scattered_S(W, T_true, C, A)
        blocks_clean.append((W, A, C, S))
        chans.append(ch)
        n_obs += ch.n ** 2
        absC = max(absC, cinfo["abs_max"])

    sig_vec = jac.sigma_uniform(n_obs, sigma)
    H = np.vstack([jac.jacobian(W, A, B, T0=T_true, C=C)
                   for W, A, C, _ in blocks_clean]) / sigma
    sv = np.linalg.svd(H, compute_uv=False)
    rank = int((sv > 1e-10 * sv[0]).sum())
    pred = None
    if rank == B.shape[0]:
        Cov = jac.coefficient_covariance(H)
        pred = jac.recovery_errors(B, modes, Cov, n_obs, T_true,
                                   sigma_used=sigma)

    T_clean, info_clean = recover_wheel(blocks_clean, B, sigma,
                                        n_multistart=n_multistart,
                                        rng=np.random.default_rng(seed))
    e_clean = float(np.linalg.norm(T_clean - T_true)
                    / np.linalg.norm(T_true))

    out = dict(feasible=True, label=label,
               designs=[d.to_dict() for d in designs],
               n_channels=[int(c.n) for c in chans], n_obs=int(n_obs),
               rank=rank, sigma=float(sigma), abs_max_C=float(absC),
               sigma_min=float(sv[-1]),
               predicted=(dict(fro_iid=pred["fro_err_iid"],
                               fro_sys=pred["fro_err_sys"],
                               block_iid=pred["block_err_iid"],
                               block_sys=pred["block_err_sys"],
                               dominant=pred["dominant_names"])
                          if pred else None),
               noise_free=dict(err=e_clean,
                               multistart_spread=info_clean.get(
                                   "multistart_spread"),
                               unique=info_clean.get("multistart_unique")),
               models={})
    if pred is None:
        return out

    dom_masks = {name: mask for name, mask in
                 jac.multipole_blocks(modes).items()
                 if float(np.linalg.norm(T_true[mask]))
                 >= 0.05 * float(np.linalg.norm(T_true))}

    for model in models:
        errs, blks, ranks, uniq = [], [], [], []
        n_rep = 1 if model == "adversarial" else int(n_trials)
        for t in range(n_rep):
            rng = np.random.default_rng(trial_seed(seed, model, t))
            noisy, ds_rank = [], []
            if model == "adversarial":
                # The worst case belongs to the STACKED system, not to any
                # one cell: building it per block would understate a pooled
                # candidate's exposure (it did, by 9x, before this fix).
                U, _, _ = np.linalg.svd(H, full_matrices=False)
                v = _scale_to_rms(U[:, -1], sigma)
                off = 0
                for (W, A, C, S) in blocks_clean:
                    dS = v[off:off + S.size].reshape(S.shape)
                    off += S.size
                    noisy.append((W, A, C, S + dS))
                    sv = np.linalg.svd(dS, compute_uv=False)
                    ds_rank.append(int((sv > 1e-10 * sv[0]).sum())
                                   if sv[0] > 0 else 0)
            else:
                for (W, A, C, S) in blocks_clean:
                    ch = chans[len(noisy)]
                    dS, dinfo = make_perturbation(model, ch, S, rng, sigma,
                                                  H_whitened=None)
                    noisy.append((W, A, C, S + dS))
                    ds_rank.append(dinfo["rank"])
            T_hat, rinfo = recover_wheel(noisy, B, sigma,
                                         n_multistart=n_multistart, rng=rng)
            dT = T_hat - T_true
            errs.append(float(np.linalg.norm(dT) / np.linalg.norm(T_true)))
            blks.append({nm: float(np.linalg.norm(dT[msk])
                                   / np.linalg.norm(T_true[msk]))
                         for nm, msk in dom_masks.items()})
            ranks.append(int(np.median(ds_rank)))
            uniq.append(bool(rinfo.get("multistart_unique", True)))
        worst_block = max((max(b.values()) if b else np.inf) for b in blks)
        out["models"][model] = dict(
            fro_err=float(np.median(errs)), fro_err_worst=float(max(errs)),
            block_err_worst=float(worst_block), blocks=blks[0],
            dS_rank=int(np.median(ranks)),
            multistart_unique=bool(all(uniq)),
            position_in_bracket=float(
                (np.median(errs) - pred["fro_err_iid"])
                / max(pred["fro_err_sys"] - pred["fro_err_iid"], 1e-300)))

    if do_generic and len(blocks_clean) == 1 and chans[0].n >= 30:
        W, A, C, S = blocks_clean[0]
        if xf.generic_track_metrics(A, W)["full_rank"]:
            rng = np.random.default_rng(seed + 77)
            dS, _ = make_perturbation("reference_plane", chans[0], S, rng,
                                      sigma)
            T_g, ginfo = recover_generic(S + dS, W, A, C)
            T_g0, _ = recover_generic(S, W, A, C)
            out["generic"] = dict(
                err_noise_free=float(np.linalg.norm(T_g0 - T_true)
                                     / np.linalg.norm(T_true)),
                err_perturbed=float(np.linalg.norm(T_g - T_true)
                                    / np.linalg.norm(T_true)),
                sigma_min_deembed=ginfo["deembed"]["sigma_min"])
    out["_T_hat_reference_plane"] = None
    return out


def gate_a_study(candidates, k, modes, B, T_true, sigma, constraints,
                 holdout=None, n_trials=3, seed=12345, n_multistart=2,
                 verbose=True):
    """Run the whole candidate set through the blind recovery (rec. 3).

    `candidates` is a list of (label, design-or-list-of-designs).  `holdout`
    is a geometrically distinct encoding never used in any recovery; each
    recovered T0 is asked to predict it.
    """
    out = dict(k=float(k), lam_um=float(2 * np.pi / k), sigma=float(sigma),
               candidates={})
    for label, des in candidates:
        if verbose:
            print("  [%s] ..." % label, flush=True)
        r = run_candidate(des, k, modes, B, T_true, sigma, constraints,
                          n_trials=n_trials, seed=seed,
                          n_multistart=n_multistart, label=label)
        if holdout is not None and r.get("feasible") and r.get("predicted"):
            dl = des if isinstance(des, (list, tuple)) else [des]
            got = [candidate_pieces(d, k, modes, constraints) for d in dl]
            got = [g for g in got if g is not None]
            rng = np.random.default_rng(seed + 5)
            noisy = []
            for ch, A, W, C, _ in got:
                S = xf.scattered_S(W, T_true, C, A)
                dS, _ = make_perturbation("reference_plane", ch, S, rng,
                                          sigma)
                noisy.append((W, A, C, S + dS))
            T_hat, _ = recover_wheel(noisy, B, sigma)
            r["holdout"] = cross_cell_check(T_hat, holdout, k, modes, T_true,
                                            constraints, sigma)
        out["candidates"][label] = r
    return out


def format_gate_a(out):
    L = ["Blind synthetic recovery at lambda = %.3f um, sigma = %.4e"
         % (out["lam_um"], out["sigma"]), ""]
    for label, r in out["candidates"].items():
        if not r.get("feasible"):
            L.append("%-12s infeasible" % label)
            continue
        L.append("%-12s channels %s, %d observables, rank %d/40, "
                 "|C|max %.3g" % (label, r["n_channels"], r["n_obs"],
                                  r["rank"], r["abs_max_C"]))
        nf = r["noise_free"]
        L.append("             noise-free err %.3e, multistart spread %.1e "
                 "(basin unique: %s)"
                 % (nf["err"], nf["multistart_spread"] or 0.0, nf["unique"]))
        p = r.get("predicted")
        if p:
            L.append("             predicted bracket: %.3f%% iid ... "
                     "%.3f%% systematic" % (100 * p["fro_iid"],
                                            100 * p["fro_sys"]))
        for nm, m in r.get("models", {}).items():
            L.append("               %-16s dT %7.3f%%  worst block %7.3f%%  "
                     "dS rank %2d  bracket position %+.2f  unique %s"
                     % (nm, 100 * m["fro_err"], 100 * m["block_err_worst"],
                        m["dS_rank"], m["position_in_bracket"],
                        m["multistart_unique"]))
        if r.get("generic"):
            g = r["generic"]
            L.append("               generic branch: noise-free %.3e, "
                     "perturbed %.3f%%, sigma_min(I+TeffC) %.4f"
                     % (g["err_noise_free"], 100 * g["err_perturbed"],
                        g["sigma_min_deembed"]))
        if r.get("holdout"):
            h = r["holdout"]
            L.append("               holdout cell (%d channels): max |dS| "
                     "%.3e = %.2f sigma, on |S| ~ %.3e"
                     % (h["n_channels"], h["max_abs"], h["rel_to_sigma"],
                        h["signal_max"]))
        L.append("")
    return "\n".join(L)


def cross_cell_check(T_hat, design_holdout, k, modes, T_true, constraints,
                     sigma):
    """Predict a materially different encoding from a recovered T0.

    Gate E's cell-independence test in its cheapest synthetic form: take the
    T0 recovered from one cell and predict the held-out cell's S, then
    compare against the truth's prediction.  A T0 that merely fits its own
    cell fails here.
    """
    pieces = candidate_pieces(design_holdout, k, modes, constraints)
    if pieces is None:
        return None
    ch, A, W, C, _ = pieces
    S_true = xf.scattered_S(W, T_true, C, A)
    S_pred = xf.scattered_S(W, T_hat, C, A)
    d = np.abs(S_pred - S_true)
    return dict(n_channels=ch.n, max_abs=float(d.max()),
                rms=float(np.sqrt((d ** 2).mean())),
                rel_to_sigma=float(d.max() / sigma),
                signal_max=float(np.abs(S_true).max()))
