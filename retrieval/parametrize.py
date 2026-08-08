"""C4v (x) reciprocity constrained parametrization of the isolated-cell T-matrix.

Checklist item 4 of INVERSE_TMATRIX_FROM_FLOQUET.md (spec: doc sections 3, 4,
10.4).  Builds the symmetry-constrained subspace S in which the inverse
retrieval searches:

    T0(t) = sum_k (t_k + i t_{k+nb}) B_k,        t real, length 2*nb,

with {B_k} a Frobenius-orthonormal REAL basis of the commutant of the C4v
representation intersected with the reciprocity-symmetric subspace.  At
lmax = 3 (30 modes) the measured dimensions are: C4-only commutant 228,
full C4v commutant 114, intersected with reciprocity 68 (doc section 3).
The physical subspace is the COMPLEX span of the 68 real matrices.

Group representation on the VSWF basis (tmat.h5 convention, mode order from
/modes; every constraint here is complex-linear):

  * C4 rotations about z act by conjugation with the DIAGONAL matrix
        D_phi = diag(e^{i m_nu phi}),   phi in {0, pi/2, pi, 3pi/2},
    the same azimuthal phase family translate.rotate_inplane applies.
    Entries are built from an exact i^m lookup table (no exp roundoff).

  * sigma_v is the VERTICAL mirror through the xz-plane,
    (x, y, z) -> (x, -y, z), i.e. M = diag(1, -1, 1).  It is derived
    NUMERICALLY here (derive_sigma_v_numeric) by the field-transformation
    technique: sample the outgoing VSWF E-field of every mode on a sphere,
    form the mirrored field E'(r) = M.E(M.r) (E is a true vector), and solve
    the least-squares system E'_nu = sum_mu D[mu, nu] E_mu over all sample
    points and cartesian components.  The magnetic field is a PSEUDOVECTOR,
    Ht'(r) = -M.Ht(M.r); it is deliberately left out of the fit and used as
    an independent validation of the same D.  The result is the signed
    permutation (verified to <= 1e-10, then used in EXACT form):

        (l, m, electric) -> +(-1)^m (l, -m, electric)
        (l, m, magnetic) -> -(-1)^m (l, -m, magnetic)

  MIRROR TRAP (doc section 4): aggregation/mirror.py's mirror_parity_signs
  encodes the HORIZONTAL z-mirror diag(1, 1, -1) -- an element of C4h, NOT a
  C4v mirror.  It is diagonal in (l, m); a vertical mirror maps m -> -m (a
  signed permutation, block-antidiagonal in m).  Using sigma_h builds the
  wrong group, and the reference T (a D4h-symmetric wheel) is invariant under
  that wrong projector too, so projector-invariance of the reference does NOT
  validate the mirror choice.  The numerical derivation from the explicit
  M = diag(1,-1,1) field transformation, plus the entrywise selection-rule
  check (sigma_v_selection_residual), do.

  The 8 group elements are {D(C4^k), D(C4^k) D(sigma_v)}, k = 0..3.  The
  four vertical/diagonal mirrors all appear in the second coset
  (R_phi sigma_v = reflection through the plane at azimuth phi/2), and
  conjugation satisfies D(g sigma_v g^-1) = D(g) D(sigma_v) D(g)^-1
  (concretely C4 sigma_v C4^-1 = C4^2 sigma_v, the yz mirror).

Reciprocity (the repo's validated convention, run_demo.py:45-58 --
complex-linear: transpose, NO conjugation): with perm the index map
(l, m, p) -> (l, -m, p) and sign = (-1)^{m+m'},

    Rec(T)  = sign * T[np.ix_(perm, perm)].T          (an involution)
    P_rec(T) = (T + Rec(T)) / 2.

Full projector: P = P_grp o P_rec,  P_grp(T) = (1/8) sum_g D(g) T D(g)^-1.
P_grp and P_rec commute (validated numerically), so P is itself an
orthogonal (here: real-symmetric) projector on vec(T).

Public API for the fit driver:
    build_c4v_reciprocity_basis(modes) -> (B, meta)   B: (68, 30, 30) real
    bright_orbit_basis(B, T_ref_stack, threshold=1e-3) -> (B_bright, info)
    pack(T, B) -> t          real, length n_params(B) = 2 * len(B)
    unpack(t, B) -> T0       complex (n, n)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "aggregation"))

from vswf import ELECTRIC, sphere_quadrature, vswf_fields  # noqa: E402

# exact values of i^k, k = 0..3 (avoids exp() roundoff in the group matrices)
_I4 = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j])

# sigma_v(xz): (x, y, z) -> (x, -y, z)
MIRROR_XZ = np.array([1.0, -1.0, 1.0])


def _maxabs(x):
    return float(np.abs(x).max())


# --------------------------------------------------------------- group action

def c4_rotation_diag(modes, k_quarter):
    """Exact D(C4^k) = diag(e^{i m_nu k pi/2}) = diag(i^{m_nu k})."""
    ph = _I4[np.mod(modes.m * int(k_quarter), 4)]
    return np.diag(ph)


def mflip_perm(modes):
    """Index permutation nu -> (l, -m, p) (an involution)."""
    perm = np.empty(modes.n, dtype=int)
    for i in range(modes.n):
        j = modes.index(int(modes.l[i]), -int(modes.m[i]), int(modes.pol[i]))
        if j is None:
            raise ValueError("mode basis is not closed under m -> -m")
        perm[i] = j
    return perm


def sigma_v_exact(modes):
    """Exact D(sigma_v) for the xz-plane mirror (signed permutation).

    (l,m,E) -> +(-1)^m (l,-m,E),  (l,m,M) -> -(-1)^m (l,-m,M).
    Returns (D, perm, signs) with D[perm[nu], nu] = signs[nu].
    """
    perm = mflip_perm(modes)
    m_sign = np.where(np.mod(modes.m, 2) == 0, 1.0, -1.0)
    pol_sign = np.where(modes.pol == ELECTRIC, 1.0, -1.0)
    signs = m_sign * pol_sign
    D = np.zeros((modes.n, modes.n))
    D[perm, np.arange(modes.n)] = signs
    return D, perm, signs


def derive_sigma_v_numeric(modes, k=1.0, r_sample=1.3, quad=None):
    """Numerically derive D(sigma_v) from the field transformation.

    Samples the outgoing E-field of every VSWF mode on the sphere
    |r| = r_sample (kr ~ O(1)) on a Gauss-Legendre x uniform-phi product
    grid, forms the mirrored fields E'_nu(r) = M.E_nu(M.r) with
    M = diag(1,-1,1) (E is a true vector), and least-squares solves

        E'_nu = sum_mu D[mu, nu] E_mu

    on the stacked (npts*3, nmodes) system.  H is a pseudovector and is NOT
    used in the fit; instead the same D must reproduce
    Ht'_nu(r) = -M.Ht_nu(M.r), reported as the independent resid_H.

    Returns (D_num, diag) with diag = dict(resid_E, resid_H, cond).
    """
    if quad is None:
        quad = sphere_quadrature(16, 32)
    TH, PH, _ = quad
    pts = r_sample * np.stack(
        [np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
        axis=1)
    E, Ht = vswf_fields(k, modes, pts, "outgoing")
    Em, Htm = vswf_fields(k, modes, pts * MIRROR_XZ, "outgoing")
    Ep = Em * MIRROR_XZ            # E'(r)  = +M . E(M r)   (true vector)
    Htp = -Htm * MIRROR_XZ         # Ht'(r) = -M . Ht(M r)  (pseudovector)

    A = E.reshape(modes.n, -1).T       # (npts*3, n), column mu = E_mu
    Bm = Ep.reshape(modes.n, -1).T     # column nu = E'_nu
    D, _, _, sv = np.linalg.lstsq(A, Bm, rcond=None)
    resid_E = np.linalg.norm(A @ D - Bm) / np.linalg.norm(Bm)
    AH = Ht.reshape(modes.n, -1).T
    BH = Htp.reshape(modes.n, -1).T
    resid_H = np.linalg.norm(AH @ D - BH) / np.linalg.norm(BH)
    return D, dict(resid_E=float(resid_E), resid_H=float(resid_H),
                   cond=float(sv[0] / sv[-1]))


def c4v_group(modes, D_sigma=None):
    """The 8 exact representation matrices {D(C4^k), D(C4^k).D(sigma_v)}.

    Returns (Ds, labels); rotations first (Ds[:4]), mirror coset after.
    """
    if D_sigma is None:
        D_sigma, _, _ = sigma_v_exact(modes)
    rots = [c4_rotation_diag(modes, kq) for kq in range(4)]
    Ds = [r.astype(complex) for r in rots] + \
         [r.astype(complex) @ D_sigma for r in rots]
    labels = ["C4^%d" % kq for kq in range(4)] + \
             ["C4^%d.sigma_v" % kq for kq in range(4)]
    return Ds, labels


# ---------------------------------------------------------------- reciprocity

def reciprocity_perm_sign(modes):
    """(perm, sign) of the validated reciprocity convention
    (run_demo.py:45-58): Rec(T) = sign * T[np.ix_(perm, perm)].T with
    perm: (l,m,p) -> (l,-m,p) and sign = (-1)^{m+m'}.  Complex-linear
    (transpose, no conjugation)."""
    perm = mflip_perm(modes)
    msum = modes.m[:, None] + modes.m[None, :]
    sign = np.where(np.mod(msum, 2) == 0, 1.0, -1.0)
    return perm, sign


def apply_rec(T, perm, sign):
    """The reciprocity involution Rec(T) = sign * T[ix(perm, perm)].T."""
    return sign * T[np.ix_(perm, perm)].T


def apply_rec_projector(T, perm, sign):
    return 0.5 * (T + apply_rec(T, perm, sign))


def apply_group_average(T, Ds):
    """P_grp(T) = (1/|G|) sum_g D(g) T D(g)^-1  (D unitary: D^-1 = D^dag)."""
    acc = np.zeros(T.shape, dtype=complex)
    for D in Ds:
        acc += D @ T @ D.conj().T
    return acc / len(Ds)


def apply_full_projector(T, Ds, perm, sign):
    """P(T) = P_grp(P_rec(T)); the two commute (validated in the builder)."""
    return apply_group_average(apply_rec_projector(T, perm, sign), Ds)


# ----------------------------------------------------- vec-space (n^2 x n^2)

def conjugation_matrix_vec(D):
    """Matrix of T -> D T D^-1 on row-major vec(T), for unitary D.

    Row-major identity: vec(A X B) = kron(A, B.T) vec(X); here B = D^dag so
    kron(D, conj(D))."""
    return np.kron(D, D.conj())


def rec_matrix_vec(perm, sign):
    """Matrix of the reciprocity involution on row-major vec(T)."""
    n = len(perm)
    M = np.zeros((n * n, n * n))
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rows = (ii * n + jj).ravel()
    cols = (perm[jj] * n + perm[ii]).ravel()
    M[rows, cols] = sign[ii, jj].ravel()
    return M


def projector_matrices(modes, Ds=None):
    """Vec-space projectors P_c4 (rotations only), P_grp (full C4v),
    P_rec, and P_full = P_grp @ P_rec, with self-check residuals.

    All four are real; the returned matrices are the real parts after
    asserting the imaginary residue (imag_resid) is at roundoff.
    """
    if Ds is None:
        Ds, _ = c4v_group(modes)
    n = modes.n
    P_c4 = sum(conjugation_matrix_vec(D) for D in Ds[:4]) / 4.0
    P_grp = sum(conjugation_matrix_vec(D) for D in Ds) / 8.0
    imag_resid = max(_maxabs(P_c4.imag), _maxabs(P_grp.imag))
    P_c4 = P_c4.real
    P_grp = P_grp.real
    perm, sign = reciprocity_perm_sign(modes)
    P_rec = 0.5 * (np.eye(n * n) + rec_matrix_vec(perm, sign))
    comm_resid = _maxabs(P_grp @ P_rec - P_rec @ P_grp)
    P_full = P_grp @ P_rec
    herm_resid = _maxabs(P_full - P_full.T)
    P_full = 0.5 * (P_full + P_full.T)
    idem_resid = _maxabs(P_full @ P_full - P_full)
    return dict(P_c4=P_c4, P_grp=P_grp, P_rec=P_rec, P_full=P_full,
                imag_resid=imag_resid, comm_resid=comm_resid,
                herm_resid=herm_resid, idem_resid=idem_resid)


def _projector_rank(P, tol=0.5):
    w = np.linalg.eigvalsh(P)
    return int((w > tol).sum()), float(np.abs(w - np.round(w)).max())


# -------------------------------------------------------------- basis builder

def build_c4v_reciprocity_basis(modes, verify_numeric=True, k=1.0,
                                r_sample=1.3, atol_numeric=1e-10,
                                rng_seed=12345):
    """Frobenius-orthonormal basis {B_k} of the C4v-commutant intersected
    with the reciprocity-symmetric subspace.

    Returns (B, meta):
      B    : real array (n_basis, n, n); n_basis = 68 at lmax = 3.  The
             physical subspace is the COMPLEX span (see pack/unpack;
             n_params = 2 * n_basis real parameters).
      meta : dict with the group matrices, the derived sigma_v action, the
             projector matrices, subspace ranks and every validation
             residual (see keys below).

    If verify_numeric (default), sigma_v is first derived numerically from
    the M = diag(1,-1,1) field transformation and MUST match the exact
    signed permutation to atol_numeric, else RuntimeError -- this is the
    guard against the C4h horizontal-mirror trap.  The exact matrix (not the
    lstsq one) is what enters the projector.
    """
    n = modes.n
    D_sigma, sperm, ssigns = sigma_v_exact(modes)

    # -- numerical derivation + trap guard
    numeric = dict(resid_E=None, resid_H=None, cond=None)
    D_num = None
    err_num = None
    sq_err_num = None
    if verify_numeric:
        D_num, numeric = derive_sigma_v_numeric(modes, k=k,
                                                r_sample=r_sample)
        err_num = _maxabs(D_num - D_sigma)
        sq_err_num = _maxabs(D_num @ D_num - np.eye(n))
        if err_num > atol_numeric:
            raise RuntimeError(
                "numerically derived D(sigma_v) does not match the exact "
                "signed permutation (max err %.3e > %.1e) -- wrong mirror?"
                % (err_num, atol_numeric))

    # -- exact group and its validation
    Ds, labels = c4v_group(modes, D_sigma)
    I = np.eye(n)
    sigma_sq_resid = _maxabs(D_sigma @ D_sigma - I)
    unitarity_resid = max(_maxabs(D @ D.conj().T - I) for D in Ds)
    closure_resid = 0.0
    for A in Ds:
        for Bm in Ds:
            prod = A @ Bm
            best = min(_maxabs(prod - C) for C in Ds)
            closure_resid = max(closure_resid, best)
    # conjugation identity: C4 sigma_v C4^-1 = C4^2 sigma_v (the yz mirror)
    conj_resid = _maxabs(Ds[1] @ D_sigma @ Ds[1].conj().T - Ds[6])

    # -- vec-space projectors, ranks, eigenbasis
    mats = projector_matrices(modes, Ds)
    rank_c4, pur_c4 = _projector_rank(mats["P_c4"])
    rank_c4v, pur_c4v = _projector_rank(mats["P_grp"])
    rank_rec, pur_rec = _projector_rank(mats["P_rec"])
    w, V = np.linalg.eigh(mats["P_full"])
    purity = float(np.abs(w - np.round(w)).max())
    sel = w > 0.5
    rank_full = int(sel.sum())
    B = V[:, sel].T.reshape(rank_full, n, n).copy()

    G = B.reshape(rank_full, -1) @ B.reshape(rank_full, -1).T
    ortho_resid = _maxabs(G - np.eye(rank_full))

    # -- consistency: eigh basis vs the operator form of P
    perm, sign = reciprocity_perm_sign(modes)
    inv_resid = max(
        _maxabs(apply_full_projector(Bk, Ds, perm, sign) - Bk) for Bk in B)
    rng = np.random.default_rng(rng_seed)
    Trand = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    PT_apply = apply_full_projector(Trand, Ds, perm, sign)
    PT_vec = (mats["P_full"] @ Trand.reshape(-1)).reshape(n, n)
    vec_apply_resid = _maxabs(PT_apply - PT_vec)
    PT_basis = unpack(pack(PT_apply, B), B)   # B-projection of P(T) = P(T)
    basis_proj_resid = _maxabs(PT_basis - PT_apply)
    comm_T_resid = _maxabs(
        apply_group_average(apply_rec_projector(Trand, perm, sign), Ds)
        - apply_rec_projector(apply_group_average(Trand, Ds), perm, sign))

    meta = dict(
        modes=modes,
        D_sigma_v=D_sigma, sigma_perm=sperm, sigma_signs=ssigns,
        D_sigma_v_numeric=D_num, sigma_numeric_err=err_num,
        sigma_numeric_sq_err=sq_err_num,
        sigma_lstsq_resid_E=numeric["resid_E"],
        sigma_resid_H=numeric["resid_H"], sigma_lstsq_cond=numeric["cond"],
        group=Ds, group_labels=labels,
        sigma_sq_resid=sigma_sq_resid, unitarity_resid=unitarity_resid,
        closure_resid=closure_resid, conjugation_resid=conj_resid,
        rec_perm=perm, rec_sign=sign,
        P_c4=mats["P_c4"], P_grp=mats["P_grp"], P_rec=mats["P_rec"],
        P_full=mats["P_full"],
        imag_resid=mats["imag_resid"], comm_resid=mats["comm_resid"],
        herm_resid=mats["herm_resid"], idem_resid=mats["idem_resid"],
        comm_T_resid=comm_T_resid,
        rank_c4=rank_c4, rank_c4v=rank_c4v, rank_rec=rank_rec,
        rank_full=rank_full,
        eig_purity_c4=pur_c4, eig_purity_c4v=pur_c4v, eig_purity_rec=pur_rec,
        eigenvalue_purity=purity,
        orthonormality_resid=ortho_resid,
        basis_invariance_resid=inv_resid,
        vec_apply_resid=vec_apply_resid,
        basis_projection_resid=basis_proj_resid,
    )
    return B, meta


# ---------------------------------------------------------- bright subspace

def bright_mask(T_stack, threshold=1e-3):
    """Entries whose |T|, maximized over frequency, exceeds threshold times
    the GLOBAL |T| max (doc section 3: 25 of 900 at 1e-3)."""
    absT = np.abs(np.asarray(T_stack))
    if absT.ndim == 2:
        absT = absT[None]
    return absT.max(axis=0) > threshold * absT.max()


def bright_orbits(mask, perm):
    """Orbits of the bright entry POSITIONS under the group + reciprocity
    action on entries.  C4 rotations act diagonally (they fix positions);
    the position-moving generators are
        s: (i, j) -> (perm i, perm j)     [sigma_v conjugation]
        r: (i, j) -> (perm j, perm i)     [reciprocity]
        t = s o r: (i, j) -> (j, i)
    i.e. the Klein four-group {id, s, r, t}.

    Returns (orbits, closed): orbits = list of sorted position tuples
    containing at least one bright entry; closed = True iff the mask is a
    union of whole orbits.
    """
    entries = [tuple(int(x) for x in e) for e in np.argwhere(mask)]
    entry_set = set(entries)
    seen = set()
    orbits = []
    closed = True
    for e in entries:
        if e in seen:
            continue
        i, j = e
        orb = {(i, j), (int(perm[i]), int(perm[j])),
               (j, i), (int(perm[j]), int(perm[i]))}
        orbits.append(sorted(orb))
        for o in orb:
            if o in entry_set:
                seen.add(o)
            else:
                closed = False
    return orbits, closed


def bright_orbit_basis(B, T_ref_stack, threshold=1e-3, sv_tol=1e-8,
                       perm=None, modes=None):
    """Orthonormal sub-basis of span{B_k} with support on the bright entries.

    Constructed as an orthonormal basis of span{ P(E_ij) : (i,j) bright },
    where E_ij is the canonical basis matrix and P the full projector.  In
    coefficient space P(E_ij) = sum_k B_k[i,j] B_k, so the span is that of
    the rows of G[e, k] = B_k[i_e, j_e]; an SVD of G gives the rank and the
    re-orthonormalized combinations.

    Expected rank = the number of bright symmetry orbits whose entries obey
    the C4 selection rule (m - m') mod 4 == 0: an orbit that violates it is
    annihilated by the C4 average, and every conforming orbit here carries
    exactly one invariant direction.  On the reference file the doc's 25
    bright entries form 11 position orbits (doc section 3, estimate "11-15
    unknowns"), but one orbit consists solely of the two C4-VIOLATING
    noise-level entries (the doc's "23 of the 25 obey it"), so the physical
    bright basis has rank 10.

    Returns (B_bright, info): B_bright real (rank, n, n),
    Frobenius-orthonormal, each element inside span{B_k}; info carries the
    mask, bright count, singular values, the coefficient matrix
    (B_bright = coeff @ B), and orbit statistics when perm (or modes, from
    which perm and the per-orbit C4 conformance are derived) is given.
    """
    B = np.asarray(B)
    nb = B.shape[0]
    mask = bright_mask(T_ref_stack, threshold)
    flat = np.flatnonzero(mask.ravel())
    G = B.reshape(nb, -1)[:, flat].T          # (n_bright, nb), real
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    rank = int((s > sv_tol * (s[0] if s.size else 1.0)).sum())
    coeff = Vt[:rank]
    B_bright = np.tensordot(coeff, B, axes=(1, 0))
    info = dict(mask=mask, n_bright=int(mask.sum()), rank=rank,
                singular_values=s, coeff=coeff)
    if modes is not None and perm is None:
        perm = mflip_perm(modes)
    if perm is not None:
        orbits, closed = bright_orbits(mask, perm)
        info["n_orbits"] = len(orbits)
        info["orbits"] = orbits
        info["mask_closed_under_group"] = closed
        if modes is not None:
            conf = [bool((int(modes.m[o[0][0]]) - int(modes.m[o[0][1]]))
                         % 4 == 0) for o in orbits]
            info["orbit_c4_conforming"] = conf
            info["n_orbits_c4_conforming"] = int(sum(conf))
    return B_bright, info


# ------------------------------------------------------------- pack / unpack

def n_params(B):
    """Number of real fit parameters spanning the complex span of B."""
    return 2 * np.asarray(B).shape[0]


def pack(T, B):
    """T (n, n) -> real t, length 2*nb: t = [Re z; Im z], z_k = <B_k, T>_F.

    For T inside the subspace, unpack(pack(T, B), B) == T; for arbitrary T
    the roundtrip is the orthogonal projection onto the complex span of B.
    """
    B = np.asarray(B)
    z = np.tensordot(B.conj(), T, axes=([1, 2], [0, 1]))
    return np.concatenate([z.real, z.imag])


def unpack(t, B):
    """Real t (2*nb) -> T0 = sum_k (t_k + i t_{nb+k}) B_k, complex (n, n)."""
    B = np.asarray(B)
    nb = B.shape[0]
    t = np.asarray(t, dtype=float)
    if t.shape[-1] != 2 * nb:
        raise ValueError("t must have length 2 * n_basis = %d" % (2 * nb))
    z = t[..., :nb] + 1j * t[..., nb:]
    return np.tensordot(z, B, axes=(-1, 0))


# ------------------------------------------------- reference-T validation

def invariance_report(T_stack, Ds, perm, sign):
    """Per-frequency ||P(T) - T||: columns (abs Frobenius, rel Frobenius,
    max-entry abs).  Doc section 4 expectation: relative at the few-1e-3
    noise-floor level, absolute ~3.6e-5."""
    rows = []
    for T in T_stack:
        d = apply_full_projector(T, Ds, perm, sign) - T
        nT = np.linalg.norm(T)
        rows.append((np.linalg.norm(d), np.linalg.norm(d) / nT,
                     np.abs(d).max()))
    return np.array(rows)


def sigma_v_selection_residual(T_stack, sperm, ssigns, mask=None):
    """Entrywise sigma_v selection rule (the anti-trap test, doc section 4):

        T[i, j] = s_i s_j T[sperm[i], sperm[j]],
        s_nu = +(-1)^m (electric) / -(-1)^m (magnetic),

    i.e. s_i s_j = (-1)^{m+m'} x (pol-dependent sign: +1 same-pol, -1 EM
    cross blocks).  Returns per-frequency worst |residual| over mask
    (default: all entries)."""
    smat = np.outer(ssigns, ssigns)
    out = []
    for T in T_stack:
        R = np.abs(T - smat * T[np.ix_(sperm, sperm)])
        out.append(R[mask].max() if mask is not None else R.max())
    return np.array(out)


def passivity_max_sv(T_stack):
    """max over frequency of max SV(I + 2T) (<= 1 + eps for passive cells)."""
    n = T_stack.shape[-1]
    I = np.eye(n)
    return float(max(np.linalg.svd(I + 2.0 * T, compute_uv=False).max()
                     for T in T_stack))
