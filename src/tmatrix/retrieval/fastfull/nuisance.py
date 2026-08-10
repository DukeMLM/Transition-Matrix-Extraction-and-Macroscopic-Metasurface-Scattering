"""Nuisance-marginalized identifiability: can this cell tell T from calibration?

Gate A found that the coded-cell inverse is not algebraically
underdetermined — `small@8` is rank 40/40 with an exact noise-free recovery
and a unique basin — but becomes practically non-identifiable once physical
calibration directions are admitted, a port-plane offset alone producing 24 %
apparent T error.  The reviewer's tangent-space audit (`review.md`,
2026-08-07 15:45) made the mechanism explicit: 99.98 % of the reference-plane
tangent lies inside col(H), so only ~2 % of that perturbation is
distinguishable from a change in T.

The consequence for design is that maximizing sigma_40(H) optimizes the wrong
thing.  What matters is the information about T that SURVIVES marginalizing
the nuisance parameters — the Schur complement of the joint Fisher
information (reviewer recommendation 6):

    F_T = J_c^T J_c - J_c^T J_eta (J_eta^T J_eta + Q_eta)^-1 J_eta^T J_c

With Q_eta = 0 (nuisance parameters completely free, the conservative and
likelihood-only choice) this is exactly

    F_T = J_c^T (I - P_eta) J_c,        P_eta = projector onto col(J_eta),

so sqrt(lambda_min(F_T)) is the smallest singular value of the part of the
T-Jacobian ORTHOGONAL to every calibration direction.  That is the quantity a
coded cell should maximize, and it is what this module computes.

Real parametrization
--------------------
The nuisance parameters are physical and REAL (a port-plane offset in um, a
mixing angle in rad, a mesh-error amplitude), while c is complex.  The whole
calculation is therefore done in real coordinates: the whitened residual
contributes [Re; Im] rows, c contributes [Re c; Im c] columns through
[[Re H, -Im H], [Im H, Re H]], and each nuisance parameter contributes one
column [Re dS; Im dS].  Mixing a complex and a real parametrization here
would silently give the nuisance space twice its true dimension.

Nuisance classes
----------------
Declared explicitly, and deliberately wider than the Gate A error models,
because the reviewer correctly noted that the Gate A `mode_mixing` family
could not represent the campaign's actual failure — a receive-only TM-row
sign fault (`retrieval/RESULTS.md`).  A congruence S -> M S M^T can never
produce a receive-only defect, so INDEPENDENT transmit and receive maps are
carried here:

  ref_plane        one port-plane offset, S -> D S D, D = exp(i k_z dL/2)
  ref_plane_split  independent transmit/receive offsets (2 parameters)
  phase_rx/tx      independent per-channel receive / transmit phases
  tm_row           a receive-only TM-row scale -- the campaign's own fault,
                   expressed as a tangent
  mix_rx/tx        independent per-order TE/TM rotations on receive/transmit
  angular          a smooth angular (mesh) error, low-order polynomial in
                   (k_z/k, cos phi, sin phi), separately on rx and tx

Every tangent is evaluated at a given T (the passive D4h ensemble for design,
never the reference wheel), because a multiplicative calibration error acts
on the signal and therefore depends on it.
"""
import numpy as np

from . import transforms as xf
from . import jacobian as jac

NUISANCE_CLASSES = ("ref_plane", "ref_plane_split", "phase_rx", "phase_tx",
                    "tm_row", "mix_rx", "mix_tx", "angular_rx", "angular_tx")

DEFAULT_CLASSES = ("ref_plane", "phase_rx", "phase_tx", "tm_row",
                   "mix_rx", "mix_tx", "angular_rx", "angular_tx")


def _angular_basis(channels):
    u = np.asarray(channels.kz, dtype=float) / float(channels.k)
    ph = np.asarray(channels.phi, dtype=float)
    return np.stack([np.ones_like(u), u, u ** 2, np.cos(ph), np.sin(ph),
                     u * np.cos(ph), u * np.sin(ph)])


def nuisance_tangents(channels, S, classes=DEFAULT_CLASSES):
    """Per-parameter dS/d(nuisance), as a list of (name, (n, n) complex).

    Every entry is a first-order tangent at the identity calibration, which
    is the right object for a Fisher/Schur calculation regardless of how
    large the actual calibration error turns out to be.

    Receive acts on the ROW index and transmit on the COLUMN index; a class
    that touches only one of them cannot be written as a congruence, which is
    exactly why the campaign's TM-row fault needed its own hypothesis family.
    """
    n = channels.n
    kz = np.asarray(channels.kz, dtype=float)
    pol = np.asarray(channels.pol, dtype=int)
    out = []

    def row_tangent(v):        # receive-side diagonal generator
        return v[:, None] * S

    def col_tangent(v):        # transmit-side diagonal generator
        return S * v[None, :]

    for cls in classes:
        if cls == "ref_plane":
            out.append(("ref_plane", 0.5j * (kz[:, None] + kz[None, :]) * S))
        elif cls == "ref_plane_split":
            out.append(("ref_plane_rx", row_tangent(0.5j * kz)))
            out.append(("ref_plane_tx", col_tangent(0.5j * kz)))
        elif cls == "phase_rx":
            for c in range(n):
                v = np.zeros(n, dtype=complex)
                v[c] = 1j
                out.append(("phase_rx_%d" % c, row_tangent(v)))
        elif cls == "phase_tx":
            for c in range(n):
                v = np.zeros(n, dtype=complex)
                v[c] = 1j
                out.append(("phase_tx_%d" % c, col_tangent(v)))
        elif cls == "tm_row":
            v = np.where(pol == 1, 1.0, 0.0).astype(complex)
            out.append(("tm_row", row_tangent(v)))
        elif cls in ("mix_rx", "mix_tx"):
            for j in range(n // 2):
                a, b = 2 * j, 2 * j + 1
                G = np.zeros((n, n), dtype=complex)
                G[a, b], G[b, a] = -1.0, 1.0        # so(2) generator
                dS = G @ S if cls == "mix_rx" else S @ G.T
                out.append(("%s_%d" % (cls, j), dS))
        elif cls in ("angular_rx", "angular_tx"):
            for i, row in enumerate(_angular_basis(channels)):
                v = row.astype(complex)
                dS = row_tangent(v) if cls == "angular_rx" else col_tangent(v)
                out.append(("%s_%d" % (cls, i), dS))
        else:
            raise ValueError("unknown nuisance class %r" % cls)
    return out


def _real_stack(mats, sigma):
    """[(n,n) complex] -> real (2*n_obs, n_param) whitened design matrix."""
    if not mats:
        return np.zeros((0, 0))
    cols = []
    for M in mats:
        v = (M / sigma).ravel()
        cols.append(np.concatenate([v.real, v.imag]))
    return np.stack(cols, axis=1)


def validate_prior_precision(Q, n_e):
    """One validator for the prior PRECISION, used by objective and estimator.

    Q is an inverse covariance, never a covariance: passing a covariance
    reverses the regularization, so a more uncertain nuisance would be
    constrained more strongly.  Accepts scalar, per-parameter vector, or full
    matrix; requires symmetry and positive semidefiniteness.  The two code
    paths previously disagreed -- the Schur side accepted a negative or
    nonsymmetric precision (Jc = Je = [1], Q = -2 gave the impossible
    F_marg = 2 > F_free = 1) while the estimator silently ignored a negative
    scalar.  Returns a full (n_e, n_e) matrix, or None for "no prior".
    """
    if Q is None:
        return None
    Q = np.asarray(Q, dtype=float)
    if not np.all(np.isfinite(Q)):
        raise ValueError("prior precision must be finite (got NaN or Inf)")
    if Q.ndim == 0:
        if float(Q) == 0.0:
            return None
        if float(Q) < 0.0:
            raise ValueError("prior precision must be non-negative, got %g"
                             % float(Q))
        Q = float(Q) * np.eye(n_e)
    elif Q.ndim == 1:
        if Q.shape[0] != n_e:
            raise ValueError("prior precision vector must have %d entries"
                             % n_e)
        if np.any(Q < 0):
            raise ValueError("prior precision entries must be non-negative")
        Q = np.diag(Q)
    elif Q.shape != (n_e, n_e):
        raise ValueError("prior precision must be scalar, (%d,) or (%d, %d)"
                         % (n_e, n_e, n_e))
    scale = max(np.abs(Q).max(), 1.0)
    if np.abs(Q - Q.T).max() > 1e-10 * scale:
        raise ValueError("prior precision must be symmetric")
    Q = 0.5 * (Q + Q.T)
    # PROJECT ONCE, and return the projected matrix to BOTH consumers.  The
    # previous version accepted a slightly negative eigenvalue under a
    # relative tolerance and returned it unchanged: the Schur path then
    # produced negative information (F_marg = -0.143 on a two-observation
    # fixture) while the whitener silently clipped the same eigenvalue to
    # zero, so objective and estimator were using different priors.
    w, V = np.linalg.eigh(Q)
    if w.min() < -1e-10 * scale:
        raise ValueError("prior precision must be positive semidefinite "
                         "(smallest eigenvalue %.3e)" % w.min())
    w = np.clip(w, 0.0, None)
    if not np.any(w > 0.0):
        # a zero precision is "no prior", whichever representation it
        # arrived in: scalar 0, a zero vector and a zero matrix previously
        # took different branches and gave different answers.
        return None
    return (V * w[None, :]) @ V.T


def schur_complement(Jc, Je, Q_eta=None, rank_tol=1e-10):
    """F_T = Jc^T Jc - Jc^T Je (Je^T Je + Q_eta)^-1 Je^T Jc, computed directly.

    The obvious shortcut -- form M = (I - Je (Je^T Je + Q)^-1 Je^T) Jc and
    take svd(M) -- is only valid at Q = 0, where the bracket is an orthogonal
    projector.  For Q > 0 the bracket is NOT idempotent, so M^T M is not the
    Schur complement.  The error is invisible on well-separated problems and
    large here precisely because the nuisance space is nearly collinear with
    col(Jc): on `small@8` at q_eta = 1 the shortcut gave a weakest value of
    3.53 against the true 8.27, and 191 against 407 at q_eta = 1e4.

    Q_eta may be a scalar (isotropic ridge), a 1-D vector of per-parameter
    precisions, or a full matrix.  None or 0 means completely free nuisance
    parameters, which is the conservative likelihood-only choice.
    """
    G = Jc.T @ Jc
    if Je is None or Je.size == 0:
        return 0.5 * (G + G.T)
    n_e = Je.shape[1]
    Q = validate_prior_precision(Q_eta, n_e)

    if Q is None:
        # FREE nuisance: eliminate the column SPACE of Je, which is what the
        # Schur complement reduces to at Q = 0.  Using pinv(Je^T Je) is not
        # invariant to nuisance units -- Je = diag(1, 1e-6) and Je = I span
        # the same space but gave F_marg = 1 and 0.  Revealing the rank from
        # the UNSCALED Je is not invariant either: diag(1, 1e-12) then reads
        # as rank 1 and returns diag(0, 1) where diag(1, 1e-12) and I have
        # the identical column space.  Normalizing the non-zero columns
        # first makes the elimination depend only on that space.
        #
        # SCOPE OF THE INVARIANCE CLAIM: this buys invariance under DIAGONAL
        # rescaling of the nuisance coordinates, which is what physical unit
        # changes are, and that is what the gate checks.  It is NOT invariant
        # under an arbitrary invertible reparameterization: D = [[1, 1],
        # [0, 1e-10]] leaves the mathematical column space alone but makes
        # one direction fall below rank_tol in this metric.  Since the real
        # nuisance families are already nearly collinear, the supported
        # coordinates and their scales must be declared physically -- the
        # numerical metric must not be left to decide what "free calibration"
        # removes.  Declaring that metric from a measured covariance is M3
        # work and is not done.
        col = np.linalg.norm(Je, axis=0)
        keep = col > 0.0
        if not np.any(keep):
            return 0.5 * (G + G.T)
        Jen = Je[:, keep] / col[keep][None, :]
        U, sv, _ = np.linalg.svd(Jen, full_matrices=False)
        r = int((sv > rank_tol * sv[0]).sum()) if sv.size and sv[0] > 0 else 0
        if r == 0:
            return 0.5 * (G + G.T)
        Uq = U[:, :r]
        M = Jc - Uq @ (Uq.T @ Jc)
        F = M.T @ M
        return 0.5 * (F + F.T)

    # POSITIVE prior: nondimensionalize the nuisance coordinates before the
    # solve.  The Schur complement is invariant under eta -> D eta with
    # Q -> D^-1 Q D^-1, but a relative pseudoinverse cutoff is not: with
    # Jc = Je = I and Q = diag(1e12, 1) the cutoff deleted the weaker -- and
    # perfectly valid -- prior mode, returning F = I where the exact answer
    # is diag(1, 0.5).  A symmetric solve on the scaled system is exact.
    col = np.linalg.norm(Je, axis=0)
    d = np.where(col > 0, 1.0 / np.maximum(col, 1e-300), 1.0)
    Jed = Je * d[None, :]
    Qd = (d[:, None] * Q) * d[None, :]
    M = Jed.T @ Jed + Qd
    M = 0.5 * (M + M.T)
    rhs = Jed.T @ Jc
    try:
        X = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(M, rhs, rcond=None)[0]
    F = G - (Jc.T @ Jed) @ X
    return 0.5 * (F + F.T)


def generalized_loss(F_free, F_marg, rcond=1e-12):
    """Worst directional inflation of the posterior standard deviation.

    `sigma_free / sigma_marg` -- the ratio of the two smallest singular
    values -- is NOT this quantity: the weakest direction before and after
    marginalization need not coincide, so the ratio understates the worst
    case.  On the deterministic fixture used by the loss gate it reads 2.7343
    where the true worst inflation is 5.6822; on the reference wheel with all
    nuisance classes, 149.37 against 240.48.

    The variance ratio along a direction v is
    (v^T F_marg^-1 v) / (v^T F_free^-1 v), whose maximum over v is the
    largest eigenvalue of the pencil (F_free, F_marg).  The standard-
    deviation inflation is its square root.  Returns inf when F_marg is
    singular (a direction that marginalization destroys entirely).
    """
    from scipy.linalg import eigh
    F_marg = 0.5 * (F_marg + F_marg.T)
    F_free = 0.5 * (F_free + F_free.T)
    ev_m = np.linalg.eigvalsh(F_marg)
    if ev_m.min() <= rcond * max(ev_m.max(), 1e-300):
        return np.inf
    try:
        w = eigh(F_free, F_marg, eigvals_only=True)
    except Exception:
        return np.inf
    return float(np.sqrt(max(w.max(), 0.0)))


def worst_direction(Jc, Jg, n_obs, B, T0, rank_tol=1e-10):
    """The genuinely worst direction WITHIN a nuisance family.

    Taking the leading left singular vector of Jg (the previous behaviour)
    maximizes output norm per unit parameter -- it says nothing about
    collinearity with T, and it understates the damage badly: for `small@8`
    the phase-rx family reported 6.65 % apparent T error and a 41.3 %
    projection, while its true worst member gives ~24 % and a 99.99 %
    principal projection.

    At fixed output RMS the family spans the unit sphere of col(Jg), so with
    Qg an orthonormal basis of col(Jg):

      max principal projection into col(Jc)  = sigma_max(Uc^T Qg)
          (= cos of the SMALLEST principal angle between the two subspaces)
      worst apparent T error                = sigma_max(Jc^+ Qg) * sqrt(n_obs)

    Returns a dict; `apparent_T_error` is now the worst case, and the old
    leading-singular-vector figure is kept as `apparent_T_error_leading` so
    the change is auditable.
    """
    def _neutral(n_params):
        # A family whose tangents all vanish has no effect on T.  Physically
        # reachable: every tangent here is proportional to the signal, so a
        # channel set with no signal produces an all-zero family, which
        # previously raised IndexError.
        return dict(n_params=int(n_params), projection_into_colH=0.0,
                    distinguishable_fraction=1.0, apparent_T_error=0.0,
                    apparent_T_error_leading=0.0,
                    min_principal_angle_deg=90.0, rank=0)

    if Jg.size == 0:
        return _neutral(0)
    # RANK-TRUNCATE col(Jc).  Using every returned left singular vector
    # treats the null directions of Jc as part of its column space, which
    # reports perfect collinearity for a rank-deficient T Jacobian: for
    # Jc = [[1,0],[0,0],[0,0],[0,0]] and a nuisance along e2 the untruncated
    # form gave projection 1 / angle 0 deg where the truth is 0 / 90 deg.
    Uc, sc, _ = np.linalg.svd(Jc, full_matrices=False)
    rc = int((sc > rank_tol * sc[0]).sum()) if sc.size and sc[0] > 0 else 0
    if rc == 0:
        return _neutral(Jg.shape[1])
    Uc = Uc[:, :rc]
    Qg, sg, _ = np.linalg.svd(Jg, full_matrices=False)
    r = int((sg > rank_tol * sg[0]).sum()) if sg.size and sg[0] > 0 else 0
    if r == 0:
        return _neutral(Jg.shape[1])
    Qg = Qg[:, :r]
    cos_pa = np.linalg.svd(Uc.T @ Qg, compute_uv=False)
    proj_max = float(np.clip(cos_pa.max(), 0.0, 1.0)) if cos_pa.size else 0.0

    Jc_pinv = np.linalg.pinv(Jc, rcond=rank_tol)
    nrmT = float(np.linalg.norm(T0))
    nb = B.shape[0]

    def _apparent(u):
        dc = Jc_pinv @ (u * np.sqrt(n_obs))
        dT = np.tensordot(dc[:nb] + 1j * dc[nb:], B, axes=(0, 0))
        return float(np.linalg.norm(dT) / nrmT)

    Uw, sw, _ = np.linalg.svd(Jc_pinv @ Qg, full_matrices=False)
    _, _, Vw = np.linalg.svd(Jc_pinv @ Qg, full_matrices=True)
    worst_u = Qg @ Vw[0].conj()
    worst_u = worst_u / np.linalg.norm(worst_u)
    lead_u = Qg[:, 0] / np.linalg.norm(Qg[:, 0])
    return dict(n_params=int(Jg.shape[1]),
                projection_into_colH=proj_max,
                distinguishable_fraction=float(
                    np.sqrt(max(1.0 - proj_max ** 2, 0.0))),
                min_principal_angle_deg=float(
                    np.rad2deg(np.arccos(np.clip(proj_max, -1.0, 1.0)))),
                rank=int(r),
                apparent_T_error=_apparent(worst_u),
                apparent_T_error_leading=_apparent(lead_u))


def marginalized_information(W, A, B, T0, C, sigma, channels,
                             classes=DEFAULT_CLASSES, q_eta=0.0,
                             rank_tol=1e-10):
    """Schur complement F_T and its spectrum, with the collinearity audit.

    Returns a dict with
      sv_free       singular values of the T-Jacobian alone  (= sqrt eig of
                    J_c^T J_c); sv_free[-1] is the M1 objective sigma_40
      sv_marg       singular values of the nuisance-marginalized problem,
                    sqrt(eig(F_T)); sv_marg[-1] is the new objective
      loss          sv_free[-1] / sv_marg[-1], the factor by which admitting
                    calibration freedom degrades the worst-determined
                    physical direction
      per_class     for each nuisance class: the fraction of its tangent norm
                    lying inside col(J_c) (the reviewer's audit number), and
                    the apparent T error a unit-RMS perturbation of that
                    class would induce
    """
    sig = float(sigma)
    S = xf.scattered_S(W, T0, C, A)
    H = jac.jacobian(W, A, B, T0=T0, C=C) / sig
    Jc = np.block([[H.real, -H.imag], [H.imag, H.real]])

    tangents = nuisance_tangents(channels, S, classes)
    Je = _real_stack([t for _, t in tangents], sig)

    sv_free = np.linalg.svd(Jc, compute_uv=False)
    F_free = Jc.T @ Jc
    F = schur_complement(Jc, Je, Q_eta=q_eta, rank_tol=rank_tol)
    ev = np.clip(np.linalg.eigvalsh(F).real, 0.0, None)
    sv_marg = np.sqrt(ev)[::-1]            # descending, like sv_free
    gen_loss = generalized_loss(F_free, F)

    n_obs = S.size
    per_class = {}
    for cls in classes:
        sub = nuisance_tangents(channels, S, (cls,))
        Jg = _real_stack([t for _, t in sub], sig)
        if Jg.size == 0:
            continue
        per_class[cls] = worst_direction(Jc, Jg, n_obs, B, T0,
                                         rank_tol=rank_tol)

    return dict(sv_free=sv_free, sv_marg=sv_marg,
                sigma_free=float(sv_free[-1]), sigma_marg=float(sv_marg[-1]),
                sigma_marg_per_obs=float(sv_marg[-1] / np.sqrt(n_obs)),
                n_nuisance=int(Je.shape[1]) if Je.size else 0,
                q_eta=q_eta,
                # ratio of the two weakest directions: an information
                # diagnostic ONLY.  The worst directional variance inflation
                # is `generalized_loss` (the two weakest directions differ).
                sigma_ratio=(float(sv_free[-1] / sv_marg[-1])
                             if sv_marg[-1] > 0 else np.inf),
                generalized_loss=gen_loss, loss=gen_loss,
                identifiable=bool(sv_marg[-1] > rank_tol * sv_free[0]),
                per_class=per_class, n_obs=int(n_obs))


def ensemble_marginalized(W, A, B, T_ensemble, C, sigma, channels,
                          classes=DEFAULT_CLASSES, q_eta=0.0):
    """Worst-case marginalized objective over a passive D4h ensemble.

    The tangents depend on the signal, hence on T, so the design objective
    has to be a minimax over the ensemble exactly as the unmarginalized one
    is (proposal par. 7.3).
    """
    T_ensemble = np.atleast_3d(np.asarray(T_ensemble))
    if T_ensemble.ndim == 2:
        T_ensemble = T_ensemble[None]
    rows = [marginalized_information(W, A, B, T0, C, sigma, channels,
                                     classes=classes, q_eta=q_eta)
            for T0 in T_ensemble]
    return dict(per_draw=rows,
                worst_sigma_marg=float(min(r["sigma_marg"] for r in rows)),
                # BOTH normalizations are emitted: whether the post-projection
                # remainder averages like iid noise is an open assumption, so
                # neither raw nor per-observable is privileged in the artifact.
                worst_sigma_marg_per_obs=float(
                    min(r["sigma_marg_per_obs"] for r in rows)),
                worst_sigma_free=float(min(r["sigma_free"] for r in rows)),
                worst_loss=float(max(r["loss"] for r in rows)),
                worst_sigma_ratio=float(max(r["sigma_ratio"] for r in rows)),
                all_identifiable=bool(all(r["identifiable"] for r in rows)),
                n_obs=rows[0]["n_obs"] if rows else 0,
                n_nuisance=rows[0]["n_nuisance"] if rows else 0)


class Calibration:
    """A parametrized calibration map S -> D_rx S D_tx, for JOINT recovery.

    The marginalized objective (above) measures the information about T that
    survives calibration freedom.  That number only becomes the right design
    criterion if the ESTIMATOR actually spends that freedom -- a recovery
    that fits T alone absorbs a port-plane offset into T no matter how well
    the cell could have separated them.  Optimizing the marginalized
    objective while running a T-only fit improved the objective by 32 % and
    the recovered T error by nothing, which is exactly that mismatch.

    SCOPE, stated because the objective and the estimator must not silently
    differ: only `ref_plane`, `tm_row` and the angular families have a finite
    form here (16 parameters for a 24-channel cell), while
    `marginalized_information`'s DEFAULT_CLASSES removes 88 columns including
    72 per-channel phase and per-order mixing directions.  Optimizing against
    a class the estimator cannot fit is not meaningful; either extend this
    map or restrict the objective's `classes` to what is implemented.  Note
    that the corrected worst-direction audit puts the phase families at the
    SAME ~24 % apparent-T level as the port plane, so this gap matters.

    Sharing a parameter between blocks is done by the caller through
    `synthetic.recover_joint`'s `param_map`; passing the same Calibration
    object to two blocks does NOT tie them.

    Classes are the physical ones of `nuisance_tangents`, in their finite
    (not tangent) form:
      ref_plane   dL          S -> D S D,  D = exp(i k_z dL / 2)
      tm_row      g           receive-only gain (1 + g) on TM rows -- the
                              campaign's own fault, which no congruence can
                              express
      angular_rx  w[0..6]     receive gain 1 + sum_j w_j b_j(k_z, phi)
      angular_tx  w[0..6]     transmit gain, same basis
    """

    def __init__(self, channels, classes=("ref_plane",)):
        self.channels = channels
        self.classes = tuple(classes)
        self.kz = np.asarray(channels.kz, dtype=float)
        self.pol = np.asarray(channels.pol, dtype=int)
        self.basis = _angular_basis(channels)
        self._sizes = []
        for c in self.classes:
            if c == "ref_plane":
                self._sizes.append(1)
            elif c == "tm_row":
                self._sizes.append(1)
            elif c in ("angular_rx", "angular_tx"):
                self._sizes.append(self.basis.shape[0])
            else:
                raise ValueError("class %r has no finite form here" % c)
        self.n_params = int(sum(self._sizes))

    def split(self, eta):
        out, off = [], 0
        for s in self._sizes:
            out.append(np.asarray(eta, dtype=float)[off:off + s])
            off += s
        return out

    def apply(self, S, eta):
        d_rx = np.ones(self.channels.n, dtype=complex)
        d_tx = np.ones(self.channels.n, dtype=complex)
        for cls, p in zip(self.classes, self.split(eta)):
            if cls == "ref_plane":
                ph = np.exp(0.5j * float(p[0]) * self.kz)
                d_rx = d_rx * ph
                d_tx = d_tx * ph
            elif cls == "tm_row":
                d_rx = d_rx * (1.0 + float(p[0]) * (self.pol == 1))
            elif cls == "angular_rx":
                d_rx = d_rx * (1.0 + p @ self.basis)
            elif cls == "angular_tx":
                d_tx = d_tx * (1.0 + p @ self.basis)
        return d_rx[:, None] * S * d_tx[None, :]


def format_audit(info, indent=2):
    pad = " " * indent
    L = [pad + "sigma_40 free %.4g -> marginalized %.4g over %d nuisance "
         "parameters (loss %.1fx, identifiable %s)"
         % (info["sigma_free"], info["sigma_marg"], info["n_nuisance"],
            info["loss"], info["identifiable"]),
         pad + "%-16s %6s %12s %9s %13s %11s"
         % ("class", "params", "max proj", "angle", "worst dT",
            "(leading)")]
    for cls, d in sorted(info["per_class"].items(),
                         key=lambda kv: -kv[1]["apparent_T_error"]):
        L.append(pad + "%-16s %6d %11.5f%% %8.3f%s %12.2f%% %10.2f%%"
                 % (cls, d["n_params"], 100 * d["projection_into_colH"],
                    d["min_principal_angle_deg"], "°",
                    100 * d["apparent_T_error"],
                    100 * d["apparent_T_error_leading"]))
    L.append(pad + "(max proj = cos of the smallest principal angle between "
             "the family and col(J_c); 'worst dT' is the constrained worst "
             "member at fixed output RMS, 'leading' the old "
             "leading-singular-vector figure)")
    return "\n".join(L)
