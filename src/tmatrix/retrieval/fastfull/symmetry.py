"""D4h + reciprocity parametrization of the free-standing wheel's T-matrix.

The validated specular pipeline parametrized the cell with C4v x reciprocity
(68 complex coefficients at lmax = 3) because its CST cell had a substrate-
free but z-ASYMMETRIC treatment inherited from the campaign geometry.  The
fast-full proposal targets the free-standing wheel in a homogeneous
background with uniform material through the thickness, whose ideal group is
D4h (proposal par. 2.1), and states the resulting count: 40 independent
complex coefficients per frequency (par. 3).  This module builds exactly that
basis and PROVES the count rather than assuming it.

D4h = C4v x {E, sigma_h}
------------------------
sigma_h is the horizontal mirror (x, y, z) -> (x, y, -z).  Its action on the
tmat.h5 VSWF basis is DIAGONAL (m is preserved) and is derived here in closed
form, then confirmed numerically from the field transformation:

    E'(r) = M . E(M r),   M = diag(1, 1, -1)        (E is a true vector)
    Ht'(r) = -M . Ht(M r)                           (H is a pseudovector)

Using Y_lm(pi - th, ph) = (-1)^(l+m) Y_lm(th, ph), th_hat(pi - th) =
-M th_hat, ph_hat(pi - th) = M ph_hat, one gets M X_lm(M r_hat) =
-(-1)^(l+m) X_lm and M Z_lm(M r_hat) = +(-1)^(l+m) Z_lm, hence

    (l, m, electric) -> +(-1)^(l+m) (l, m, electric)
    (l, m, magnetic) -> -(-1)^(l+m) (l, m, magnetic)                   (S1)

Spot checks that (S1) is the physical horizontal mirror and not a sign slip:
E_{1,0} (a z electric dipole) gets -1, E_{1,+-1} (in-plane electric dipole)
gets +1, M_{1,0} (z magnetic dipole, a pseudovector component) gets +1,
M_{1,+-1} gets -1.  These are exactly the horizontal-parity assignments the
proposal quotes in par. 3.1.

MIRROR TRAP, restated.  tmatrix.aggregation.mirror's `mirror_parity_signs`
looks like this operator but is NOT: it computes D . V(sigma rho) with
D = diag(-1, -1, +1) = -M, the PEC IMAGE construction, so its signs are the
NEGATIVE of (S1).  tmatrix.retrieval.parametrize warns against using it as the
C4v vertical mirror; it is equally wrong here up to an overall sign, and the
numeric gate below is what settles the question.

Counts this module verifies (none of them assumed)
--------------------------------------------------
    horizontal-parity split of the 30 modes        15 even / 15 odd
    C4v sector multiplicities per parity           proposal par. 3.1 table
    dim commutant(C4v)                             114   (parametrize)
    dim commutant(D4h)                              58   = sum_s n_s^2
    dim commutant(D4h) & reciprocity                40   = sum_s n_s(n_s+1)/2

The last number is the proposal's headline: the stored artifact is still a
full 30x30 matrix, but only 40 complex numbers per frequency are independent.
"""
from tmatrix.numerics import maxabs
import numpy as np

from tmatrix.aggregation.vswf import ELECTRIC, MAGNETIC, sphere_quadrature, vswf_fields

from tmatrix.retrieval import parametrize as pz

MIRROR_XY = np.array([1.0, 1.0, -1.0])       # sigma_h: (x, y, z) -> (x, y, -z)

# C4v character table, columns ordered (E, 2C4, C2, 2 sigma_v, 2 sigma_d)
C4V_IRREPS = ("A1", "A2", "B1", "B2", "E")
C4V_CHARACTERS = np.array([
    [1.0,  1.0,  1.0,  1.0,  1.0],       # A1
    [1.0,  1.0,  1.0, -1.0, -1.0],       # A2
    [1.0, -1.0,  1.0,  1.0, -1.0],       # B1
    [1.0, -1.0,  1.0, -1.0,  1.0],       # B2
    [2.0,  0.0, -2.0,  0.0,  0.0],       # E
])
# parametrize.c4v_group order: 0=E, 1=C4, 2=C2, 3=C4^3, then C4^k sigma_v.
# C4^k sigma_v is the mirror at azimuth k*pi/4, so k in {0, 2} are the two
# sigma_v (through the spokes / their perpendicular) and k in {1, 3} the two
# diagonal sigma_d.
C4V_CLASS_MEMBERS = ([0], [1, 3], [2], [4, 6], [5, 7])
C4V_CLASS_SIZES = np.array([1.0, 2.0, 1.0, 2.0, 2.0])



# ------------------------------------------------------------- sigma_h

def sigma_h_exact(modes):
    """Diagonal D(sigma_h) of eq. (S1). Returns (D, signs)."""
    par = np.where(np.mod(modes.l + modes.m, 2) == 0, 1.0, -1.0)
    pol_sign = np.where(modes.pol == ELECTRIC, 1.0, -1.0)
    signs = par * pol_sign
    return np.diag(signs), signs


def derive_sigma_h_numeric(modes, k=1.0, r_sample=1.3, quad=None):
    """Numerically derive D(sigma_h) from the field transformation.

    Same technique as parametrize.derive_sigma_v_numeric: least-squares fit
    of E'_nu(r) = M . E_nu(M r) on the E channel only, with the pseudovector
    identity Ht'_nu(r) = -M . Ht_nu(M r) held back as an independent check.
    Returns (D_num, diag).
    """
    if quad is None:
        quad = sphere_quadrature(16, 32)
    TH, PH, _ = quad
    pts = r_sample * np.stack(
        [np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
        axis=1)
    E, Ht = vswf_fields(k, modes, pts, "outgoing")
    Em, Htm = vswf_fields(k, modes, pts * MIRROR_XY, "outgoing")
    Ep = Em * MIRROR_XY
    Htp = -Htm * MIRROR_XY

    A = E.reshape(modes.n, -1).T
    Bm = Ep.reshape(modes.n, -1).T
    D, _, _, sv = np.linalg.lstsq(A, Bm, rcond=None)
    resid_E = np.linalg.norm(A @ D - Bm) / np.linalg.norm(Bm)
    AH = Ht.reshape(modes.n, -1).T
    BH = Htp.reshape(modes.n, -1).T
    resid_H = np.linalg.norm(AH @ D - BH) / np.linalg.norm(BH)
    return D, dict(resid_E=float(resid_E), resid_H=float(resid_H),
                   cond=float(sv[0] / sv[-1]))


def horizontal_parity(modes):
    """+1 / -1 per mode under sigma_h (the diagonal of eq. (S1))."""
    return sigma_h_exact(modes)[1]


def d4h_group(modes, D_sigma_v=None, D_sigma_h=None):
    """The 16 matrices of D4h = C4v x {E, sigma_h}. Returns (Ds, labels)."""
    Ds_c4v, labels_c4v = pz.c4v_group(modes, D_sigma_v)
    if D_sigma_h is None:
        D_sigma_h, _ = sigma_h_exact(modes)
    D_sigma_h = D_sigma_h.astype(complex)
    Ds = [D.astype(complex) for D in Ds_c4v] + \
         [D.astype(complex) @ D_sigma_h for D in Ds_c4v]
    labels = list(labels_c4v) + [s + ".sigma_h" for s in labels_c4v]
    return Ds, labels


# --------------------------------------------------------- sector counting

def c4v_sector_multiplicities(modes, Ds_c4v=None, parity=None):
    """Multiplicities of the five C4v irreps in the 30-mode representation.

    Computed by the character formula n_s = (1/8) sum_g conj(chi_s(g)) chi(g)
    with chi(g) the trace of D(g) restricted to a horizontal-parity subspace
    (parity = +1 / -1) or to everything (parity = None).

    Returns (mult, resid) with mult a dict irrep -> multiplicity (rounded to
    int) and resid the worst distance of a raw multiplicity from an integer,
    which is the actual gate that the character bookkeeping is right.
    """
    if Ds_c4v is None:
        Ds_c4v, _ = pz.c4v_group(modes)
    if parity is None:
        sel = np.ones(modes.n, dtype=bool)
    else:
        sel = horizontal_parity(modes) == parity
    chi = np.array([np.trace(D[np.ix_(sel, sel)]) for D in Ds_c4v])
    chi_class = np.array([chi[m[0]] for m in C4V_CLASS_MEMBERS])
    raw = (C4V_CHARACTERS.conj() * C4V_CLASS_SIZES[None, :]
           * chi_class[None, :]).sum(axis=1) / 8.0
    resid = float(np.abs(raw - np.round(raw.real)).max())
    return ({name: int(round(float(v.real)))
             for name, v in zip(C4V_IRREPS, raw)}, resid)


def independent_coefficient_count(modes):
    """(n_commutant, n_reciprocal, per-sector detail) from the character
    multiplicities alone -- an algebraic prediction to check the numerically
    built basis against.

    A sector appearing with multiplicity n contributes n^2 complex freedoms
    to the commutant and n(n+1)/2 after reciprocity (which acts as a
    transpose inside each sector block).
    """
    detail = {}
    n_comm = 0
    n_rec = 0
    for parity, tag in ((+1, "even"), (-1, "odd")):
        mult, _ = c4v_sector_multiplicities(modes, parity=parity)
        detail[tag] = mult
        for n in mult.values():
            n_comm += n * n
            n_rec += n * (n + 1) // 2
    return n_comm, n_rec, detail


# --------------------------------------------------------------- the basis

def build_d4h_reciprocity_basis(modes, verify_numeric=True, k=1.0,
                                r_sample=1.3, atol_numeric=1e-10,
                                rng_seed=20260807):
    """Frobenius-orthonormal real basis {B_alpha} of

        commutant(D4h)  intersect  reciprocity-symmetric.

    Returns (B, meta) with B (40, 30, 30) real at lmax = 3.  The physical
    subspace is the COMPLEX span; use parametrize.pack / unpack (re-exported
    below) for the 80 real fit parameters.

    Every structural claim is measured into `meta`, not assumed:
      * sigma_h numeric-vs-exact residual (the mirror-trap guard),
      * group closure / unitarity / commutation with reciprocity,
      * commutant ranks with and without sigma_h and reciprocity,
      * the character-formula prediction of the same ranks,
      * horizontal-parity block-diagonality of every basis element (a
        commutant element cannot connect opposite sigma_h parities).
    """
    n = modes.n
    D_sv, sperm, ssigns = pz.sigma_v_exact(modes)
    D_sh, hsigns = sigma_h_exact(modes)

    numeric = dict(resid_E=None, resid_H=None, cond=None)
    D_sh_num, err_num = None, None
    if verify_numeric:
        D_sh_num, numeric = derive_sigma_h_numeric(modes, k=k,
                                                   r_sample=r_sample)
        err_num = maxabs(D_sh_num - D_sh)
        if err_num > atol_numeric:
            raise RuntimeError(
                "numerically derived D(sigma_h) does not match eq. (S1) "
                "(max err %.3e > %.1e) -- wrong mirror or wrong parity?"
                % (err_num, atol_numeric))

    Ds, labels = d4h_group(modes, D_sv, D_sh)
    I = np.eye(n)
    sh_sq_resid = maxabs(D_sh @ D_sh - I)
    commute_resid = maxabs(D_sh @ D_sv - D_sv @ D_sh)
    unitarity_resid = max(maxabs(D @ D.conj().T - I) for D in Ds)
    closure_resid = 0.0
    for A in Ds:
        for Bm in Ds:
            prod = A @ Bm
            closure_resid = max(closure_resid,
                                min(maxabs(prod - C) for C in Ds))

    # vec-space projectors
    P_grp = sum(pz.conjugation_matrix_vec(D) for D in Ds) / float(len(Ds))
    imag_resid = maxabs(P_grp.imag)
    P_grp = P_grp.real
    perm, sign = pz.reciprocity_perm_sign(modes)
    P_rec = 0.5 * (np.eye(n * n) + pz.rec_matrix_vec(perm, sign))
    comm_resid = maxabs(P_grp @ P_rec - P_rec @ P_grp)
    P_full = P_grp @ P_rec
    herm_resid = maxabs(P_full - P_full.T)
    P_full = 0.5 * (P_full + P_full.T)
    idem_resid = maxabs(P_full @ P_full - P_full)

    w_grp = np.linalg.eigvalsh(P_grp)
    rank_d4h = int((w_grp > 0.5).sum())
    w, V = np.linalg.eigh(P_full)
    purity = float(np.abs(w - np.round(w)).max())
    sel = w > 0.5
    rank_full = int(sel.sum())
    B = V[:, sel].T.reshape(rank_full, n, n).copy()

    Bf = B.reshape(rank_full, -1)
    ortho_resid = maxabs(Bf @ Bf.T - np.eye(rank_full))
    inv_resid = max(maxabs(apply_d4h_projector(Bk, Ds, perm, sign) - Bk)
                    for Bk in B)

    # horizontal-parity block structure: a D4h commutant element is block
    # diagonal in sigma_h parity, so both off-parity blocks must vanish.
    even = hsigns > 0
    odd = ~even
    parity_leak = max(maxabs(B[:, even][:, :, odd]),
                      maxabs(B[:, odd][:, :, even]))

    n_comm_pred, n_rec_pred, sector_detail = \
        independent_coefficient_count(modes)
    _, char_resid = c4v_sector_multiplicities(modes)

    rng = np.random.default_rng(rng_seed)
    Trand = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    PT_apply = apply_d4h_projector(Trand, Ds, perm, sign)
    PT_vec = (P_full @ Trand.reshape(-1)).reshape(n, n)
    vec_apply_resid = maxabs(PT_apply - PT_vec)
    basis_proj_resid = maxabs(pz.unpack(pz.pack(PT_apply, B), B) - PT_apply)

    meta = dict(
        modes=modes, group=Ds, group_labels=labels,
        D_sigma_v=D_sv, D_sigma_h=D_sh,
        sigma_h_signs=hsigns, sigma_v_signs=ssigns, sigma_v_perm=sperm,
        D_sigma_h_numeric=D_sh_num, sigma_h_numeric_err=err_num,
        sigma_h_lstsq_resid_E=numeric["resid_E"],
        sigma_h_resid_H=numeric["resid_H"],
        sigma_h_lstsq_cond=numeric["cond"],
        sigma_h_sq_resid=sh_sq_resid,
        sigma_hv_commute_resid=commute_resid,
        unitarity_resid=unitarity_resid, closure_resid=closure_resid,
        imag_resid=imag_resid, comm_resid=comm_resid,
        herm_resid=herm_resid, idem_resid=idem_resid,
        rec_perm=perm, rec_sign=sign,
        P_grp=P_grp, P_rec=P_rec, P_full=P_full,
        rank_d4h=rank_d4h, rank_full=rank_full,
        rank_d4h_predicted=n_comm_pred, rank_full_predicted=n_rec_pred,
        sector_multiplicities=sector_detail,
        character_integrality_resid=char_resid,
        n_even=int(even.sum()), n_odd=int(odd.sum()),
        eigenvalue_purity=purity, orthonormality_resid=ortho_resid,
        basis_invariance_resid=inv_resid, parity_leak=parity_leak,
        vec_apply_resid=vec_apply_resid,
        basis_projection_resid=basis_proj_resid,
    )
    return B, meta


def apply_d4h_projector(T, Ds, perm, sign):
    """P(T) = P_grp(P_rec(T)) in operator form (the two commute)."""
    return pz.apply_group_average(pz.apply_rec_projector(T, perm, sign), Ds)


# ------------------------------------------------------- pack / unpack / util

pack = pz.pack
unpack = pz.unpack
n_params = pz.n_params
passivity_max_sv = pz.passivity_max_sv


def project(T, B):
    """Orthogonal projection of an arbitrary T onto the complex span of B."""
    return unpack(pack(T, B), B)


def symmetry_residual(T_stack, B):
    """Per-frequency ||T - P_B(T)||_F / ||T||_F.

    Applied to the reference wheel tmat.h5 this measures how D4h-consistent
    the independently supplied file actually is; the campaign already
    established that its C4v violation is ~3e-3 noise, so anything at that
    level here is the file's noise floor, not a basis error.
    """
    T_stack = np.atleast_3d(np.asarray(T_stack))
    out = []
    for T in T_stack:
        d = T - project(T, B)
        out.append(np.linalg.norm(d) / np.linalg.norm(T))
    return np.array(out)


# ------------------------------------------------------------- test ensemble

def ensemble_diversity(B, T_stack):
    """How much of the 40-dim coefficient space an ensemble actually covers.

    Reports the alignment of every draw with the IDENTITY direction, the
    pairwise alignments, and the participation-ratio effective rank of the
    coefficient matrix.  A passive T must carry some negative identity
    component -- passivity forces it -- but if that component DOMINATES, the
    ensemble is a one-parameter family wearing 40 coordinates and any design
    optimized on it will not transfer.  The reference wheel's identity cosine
    is 0.21; the original `convex` generator produced 0.88-0.91 with an
    effective rank of 1.44, which is why it is no longer the default.
    """
    B = np.asarray(B)
    nb = B.shape[0]
    Bf = B.reshape(nb, -1)
    T_stack = np.asarray(T_stack)
    if T_stack.ndim == 2:
        T_stack = T_stack[None]

    def _c(T):
        c = Bf.conj() @ np.asarray(T).reshape(-1)
        n = np.linalg.norm(c)
        return c / n if n > 0 else c

    cI = _c(np.eye(B.shape[1], dtype=complex))
    cs = np.stack([_c(T) for T in T_stack])
    ident = np.abs(cs @ cI.conj())
    pair = np.abs(cs @ cs.conj().T)
    iu = np.triu_indices(len(cs), 1)
    sv = np.linalg.svd(cs, compute_uv=False)
    eff = float((sv ** 2).sum() ** 2 / (sv ** 4).sum()) if sv.size else 0.0
    return dict(identity_cosine=ident,
                identity_cosine_max=float(ident.max()),
                pairwise_cosine_max=(float(pair[iu].max()) if len(iu[0])
                                     else 0.0),
                pairwise_cosine_min=(float(pair[iu].min()) if len(iu[0])
                                     else 0.0),
                effective_rank=eff, n_basis=int(nb))


def latent_draws(B, rng, n_draw):
    """Latent coefficient vectors for the Cayley generator.

    Drawing these ONCE and reusing them across loss factors is what makes a
    loss ablation paired: the reactive/sector direction is then identical and
    only the Hermitian multiplier differs.  Re-seeding a fresh RNG per loss
    stratum does NOT achieve this -- it reproduces the same block three times
    instead.
    """
    nb = B.shape[0]
    return [(rng.normal(size=nb) + 1j * rng.normal(size=nb)) / np.sqrt(2.0)
            for _ in range(int(n_draw))]


def random_passive_d4h_cayley(B, rng, n_draw=1, target_fro=0.2,
                              loss_factor=0.15, passivity_tol=1e-9,
                              c_draws=None):
    """Passive D4h draws with genuine sector diversity (the default generator).

    The `convex` construction below is exactly passive but collapses: with
    S = (1-t) I + t Y_hat, every draw carries the same -I direction, and at
    the small t needed for a weak scatterer that term dominates.  Measured on
    six draws at ||T||_F = 0.25: identity cosine 0.884-0.906, pairwise
    0.743-0.830, participation-ratio effective rank **1.44 of 40**.  A design
    optimized against that is optimized against one direction.

    This generator uses the bounded-real (Cayley) correspondence instead:

        S = (I - K)(I + K)^-1,     T = (S - I)/2 = -K (I + K)^-1

    which satisfies ||S||_2 <= 1 exactly whenever Herm(K) = (K + K^H)/2 is
    positive semidefinite.  Both stay inside the symmetry subspace: V is a
    subspace closed under conjugate transpose (verified in the builder) and
    under powers of a SINGLE element -- Rec(T^k) = Rec(T)^k -- so any analytic
    function of one element of V is again in V, and I in V.  (V is not closed
    under products of two DIFFERENT elements, which is why the construction
    is built from one K.)

    The scale of K is root-solved so that ||T||_F hits `target_fro`; T is
    never rescaled after the map, because that would leave the bounded-real
    manifold and inject absorption that the caller did not ask for.

    `loss_factor` scales the Hermitian (absorptive) part of K relative to its
    anti-Hermitian (reactive) part.  Small values give a low-loss, resonant
    draw -- the regime the wheel is actually in -- and therefore a small
    identity component; 1.0 gives a strongly absorbing, grey draw.  It is the
    knob a predeclared loss grid varies.

    Returns (T_stack, c_stack).
    """
    B = np.asarray(B)
    nb, n, _ = B.shape
    if not np.isfinite(target_fro) or float(target_fro) <= 0.0:
        raise ValueError("target_fro must be finite and positive, got %r"
                         % (target_fro,))
    if not np.isfinite(loss_factor) or float(loss_factor) < 0.0:
        raise ValueError("loss_factor must be finite and non-negative, got "
                         "%r" % (loss_factor,))
    I = np.eye(n, dtype=complex)
    Bf = B.reshape(nb, -1)
    Ts, cs = [], []
    draws = (latent_draws(B, rng, n_draw) if c_draws is None
             else [np.asarray(c) for c in c_draws])
    for c in draws:
        K0 = np.tensordot(c, B, axes=(0, 0))
        K0 = K0 / max(np.linalg.norm(K0), 1e-300)
        herm = 0.5 * (K0 + K0.conj().T)
        anti = 0.5 * (K0 - K0.conj().T)
        lam = float(np.linalg.eigvalsh(herm).min())
        herm_psd = herm + max(0.0, -lam) * I        # PSD, still in V
        gen = anti + float(loss_factor) * herm_psd

        # Root-solve the SCALE OF K so that ||T||_F hits the target, and
        # never touch T afterwards.  Rescaling T after the map leaves the
        # Cayley manifold and silently injects absorption: at loss_factor 0
        # the map should give a UNITARY S, but post-scaling produced
        # ||S^H S - I||_2 = 0.043 and mean apparent absorption 0.0075 against
        # the reference wheel's 0.00055.  ||T(t)||_F is monotone in t, so a
        # bisection is exact and cheap.
        def _T(t):
            K = t * gen
            return -K @ np.linalg.inv(I + K)

        lo, hi = 0.0, 1.0
        for _ in range(200):
            if np.linalg.norm(_T(hi)) >= target_fro:
                break
            hi *= 2.0
        else:
            raise RuntimeError("cannot reach the requested Frobenius norm")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if np.linalg.norm(_T(mid)) < target_fro:
                lo = mid
            else:
                hi = mid
        T = _T(0.5 * (lo + hi))
        if passivity_max_sv(T[None]) > 1.0 + passivity_tol:
            raise RuntimeError("exact Cayley draw is not passive -- the "
                               "bounded-real construction is violated")
        Ts.append(T)
        cs.append(Bf.conj() @ T.reshape(-1))
    return np.array(Ts), np.array(cs)


def absorption_spectrum(T_stack):
    """Singular values of S = I + 2T and the implied absorption per draw.

    `max SV(S) <= 1` alone cannot see attenuation in the other singular
    directions, which is exactly how post-map rescaling hid artificial loss.
    Reports the full spectrum, the unitarity residual ||S^H S - I||_2 (zero
    for a lossless scatterer) and the mean absorbed power fraction
    1 - mean(sv^2).
    """
    T_stack = np.asarray(T_stack)
    if T_stack.ndim == 2:
        T_stack = T_stack[None]
    n = T_stack.shape[-1]
    I = np.eye(n, dtype=complex)
    out = []
    for T in T_stack:
        S = I + 2.0 * T
        sv = np.linalg.svd(S, compute_uv=False)
        # the documented quantity is the OPERATOR norm; the max element
        # magnitude understates it (0.184 against 0.211 on a loss-0.5 draw)
        out.append(dict(sv_max=float(sv.max()), sv_min=float(sv.min()),
                        unitarity_resid=float(
                            np.linalg.norm(S.conj().T @ S - I, 2)),
                        mean_absorption=float(1.0 - (sv ** 2).mean())))
    return out


def random_passive_d4h(B, rng, n_draw=1, target_fro=0.2, t_max=1.0,
                       method="cayley", loss_factor=0.15, c_draws=None):
    """Passive D4h ensemble.  `method="cayley"` (default) is the diverse
    generator above; `method="convex"` is the original construction, kept
    only to reproduce earlier runs -- see `random_passive_d4h_cayley` for the
    measured collapse that demoted it."""
    if method == "cayley":
        return random_passive_d4h_cayley(B, rng, n_draw=n_draw,
                                         target_fro=target_fro,
                                         loss_factor=loss_factor,
                                         c_draws=c_draws)
    if method != "convex":
        raise ValueError("method must be 'cayley' or 'convex'")
    return _random_passive_d4h_convex(B, rng, n_draw=n_draw,
                                      target_fro=target_fro, t_max=t_max)


def _random_passive_d4h_convex(B, rng, n_draw=1, target_fro=0.2, t_max=1.0):
    """Draw exactly-passive T-matrices from the D4h + reciprocity space.

    This is the proposal's "target-independent passive D4h ensemble" (par.
    7.3): the design objective must hold over an ENSEMBLE, not only at the
    reference wheel, or the cell is tuned to one answer.

    Passivity CANNOT be obtained by shrinking a random T.  With S = I + 2T,
    max SV(S) ~ 1 + 2 s lambda_max(Herm(T)) + O(s^2) for T -> sT, so any
    draw whose Hermitian part has a positive eigenvalue is super-unitary at
    every positive scale.  Passivity has to be built into the S-matrix
    instead.  The construction used here works on S directly:

        Y  = a random element of the subspace,     Y_hat = Y / ||Y||_2
        S  = (1 - t) I + t Y_hat,        t in [0, 1]
        T  = (S - I) / 2 = (t / 2) (Y_hat - I)

    S is a convex combination of two matrices of spectral norm <= 1, hence
    ||S||_2 <= 1 exactly, i.e. max SV(I + 2T) <= 1 by the triangle
    inequality and not by luck.  S stays in the subspace because the
    subspace is linear and contains the identity (I commutes with every
    group element and Rec(I) = I; asserted below).  t is then chosen to hit
    `target_fro` in Frobenius norm, so the ensemble is SCALE MATCHED to the
    weakly scattering wheel (||T||_F ~ 0.1-0.2) instead of sitting at the
    black-body-like T ~ -I/2 that an unconstrained sub-unitary draw gives.

    It is NOT claimed to be a uniform measure on the passive set; it is a
    scale-matched, symmetry-exact, reference-independent ensemble, which is
    what a target-independent design objective needs.

    Returns (T_stack (n_draw, n, n), c_stack (n_draw, n_basis) complex).
    """
    B = np.asarray(B)
    nb, n, _ = B.shape
    Bf = B.reshape(nb, -1)
    I = np.eye(n, dtype=complex)
    c_I = Bf.conj() @ I.reshape(-1)
    if np.abs(np.tensordot(c_I, B, axes=(0, 0)) - I).max() > 1e-10:
        raise RuntimeError("the identity is not inside the symmetry subspace "
                           "-- the passive construction is invalid")
    Ts, cs = [], []
    for _ in range(int(n_draw)):
        c = (rng.normal(size=nb) + 1j * rng.normal(size=nb)) / np.sqrt(2.0)
        Y = np.tensordot(c, B, axes=(0, 0))
        s2 = np.linalg.norm(Y, 2)
        if s2 == 0:
            continue
        Yh = Y / s2
        D = Yh - I
        nD = np.linalg.norm(D)
        t = float(np.clip(2.0 * float(target_fro) / nD, 0.0, float(t_max)))
        Ts.append(0.5 * t * D)
        cs.append(0.5 * t * (c / s2 - c_I))
    return np.array(Ts), np.array(cs)
