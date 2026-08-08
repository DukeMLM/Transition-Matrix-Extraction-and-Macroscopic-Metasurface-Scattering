"""Heterogeneous periodic supercell: pair-resolved block-Bloch aggregation.

Implements the extension of the one-atom periodic solve (`translate.lattice_sum_C`
+ `aggregate.solve_periodic`) to a cell that contains M *different* meta-atoms,
following the operational manual (expanded Aug-2026 edition) sections 6.5.2-6.5.5
and 7.5.

Physics
-------
The supercell lattice is L = {R = n1 a1 + n2 a2}; the reference cell holds M
atoms at basis positions rho_1 ... rho_M, each with its own isolated T_t.  Bloch's
theorem (a_{t,R} = e^{i k_par . R} a_t) reduces the infinite problem to the M
atoms of one cell, coupled by the *pair-resolved* lattice sums

    W_st(k_par) = sum'_R  A(R + rho_t - rho_s) e^{i k_par . R}          (Eq. 48)

where the prime removes ONLY the (s == t, R == 0) self term: for s != t the
R = 0 term is the intracell coupling and must be kept.  A(d) is the same
outgoing->regular translation operator used everywhere else in this package
(`translate.translation_matrix`, d = r_source - r_target), so the whole block
matrix inherits the package's convention lock -- no second set of Wigner/Gaunt
coefficients is programmed.

The self-consistent system and the reusable aggregate are

    (I - W T0) a_uc = a_inc,uc ,   f_uc = T0 a_uc ,                     (Eq. 50)
    T_B^loc(k_par) = [I - T0 W]^{-1} T0 .                               (Eq. 52)

Regularization
--------------
Each W_st is conditionally convergent and is regularized exactly like the
one-atom sum: a Gaussian taper on the *full displacement*

    W_st(Rc) = sum'_R e^{-|R + rho_t - rho_s|^2 / Rc^2} e^{i k_par . R} A(...)

followed by Richardson extrapolation in 1/Rc^2 over the taper set (Eq. 55/56).
Tapering the displacement rather than the lattice vector is what makes the
sub-lattice decomposition exact: for identical atoms

    sum_t W_st(Rc) == C_p(Rc)  (the one-atom lattice sum, same taper)

*termwise*, because the union of the M displacement sets is the full atom
lattice with the self term removed.  That identity is the 6.5.5 test 2
equivalence check, and it holds before extrapolation, so the extrapolated
matrices satisfy it too.

Cost note: every pair block reuses ONE set of shell matrices A(r x_hat).  The
in-plane rotation identity A(Rot_phi d) = e^{i(m_nu - m_mu) phi} A(d) turns each
displacement into a phase on a shared radius matrix, so the M^2 blocks cost
essentially one lattice sum plus BLAS.

Output channels
---------------
A supercell of side P has Rayleigh onsets at lambda = P / sqrt(n1^2 + n2^2), so
higher diffraction orders generally open inside a band where the primitive cell
had none.  `floquet_smatrix` therefore returns every propagating order, with the
coherent basis-position phase of manual Eq. (64) in the output channel map:

    E^{+-}_G = (2 pi i / (A_uc k_z,G)) sum_s e^{-i k^{+-}_G . rho_s} F_s(k_G^{+-})

power-normalized by sqrt(k_z,G / k_z,inc).  Eqs. (65)/(66) are the G = 0,
normal-incidence special case and are reproduced exactly.
"""
import numpy as np

from vswf import (ModeBasis, RegularProjector, far_field_amplitude,
                  plane_wave_coeffs, sphere_quadrature, vswf_fields)
from translate import sphere_points, translation_shells


# --------------------------------------------------------------- geometry ---
def reciprocal_vectors(a1, a2):
    """In-plane reciprocal basis with b_i . a_j = 2 pi delta_ij."""
    A = np.array([[a1[0], a2[0]], [a1[1], a2[1]]], dtype=float)
    B = 2 * np.pi * np.linalg.inv(A).T          # columns b1, b2
    return B[:, 0], B[:, 1]


def lattice_points(a1, a2, r_max, offset=(0.0, 0.0), drop_zero_offset=False):
    """Displacements d = R + offset with |d| <= r_max, R over the lattice.

    `drop_zero_offset` removes the R = 0 term (only meaningful when offset is
    the zero vector, i.e. the s == t self term).  Returns (d, R), both (nd, 2).
    """
    a1 = np.asarray(a1, float)[:2]
    a2 = np.asarray(a2, float)[:2]
    off = np.asarray(offset, float)[:2]
    area = abs(a1[0] * a2[1] - a1[1] * a2[0])
    if area <= 0:
        raise ValueError("degenerate lattice vectors")
    reach = r_max + np.linalg.norm(off)
    n1max = int(np.ceil(reach * np.linalg.norm(a2) / area)) + 1
    n2max = int(np.ceil(reach * np.linalg.norm(a1) / area)) + 1
    n1, n2 = np.meshgrid(np.arange(-n1max, n1max + 1),
                         np.arange(-n2max, n2max + 1), indexing="ij")
    n1, n2 = n1.ravel(), n2.ravel()
    R = n1[:, None] * a1[None, :] + n2[:, None] * a2[None, :]
    d = R + off[None, :]
    r = np.linalg.norm(d, axis=1)
    keep = (r <= r_max) & (r > 1e-12)
    if drop_zero_offset:
        keep &= ~((n1 == 0) & (n2 == 0))
    return d[keep], R[keep]


# ------------------------------------------------------ shell bookkeeping ---
class ShellSet:
    """Translation matrices A(r x_hat) for a set of in-plane radii.

    One instance serves every pair block at one frequency: each displacement d
    is (radius, angle), and A(d) = rotate_inplane(A[radius], angle).
    """

    def __init__(self, k, radii, modes, r0, quad, projector=None, chunk=48):
        self.radii = np.asarray(radii, float)
        self.key = np.round(self.radii, 9)
        self.lookup = {float(r): i for i, r in enumerate(self.key)}
        self.A = translation_shells(k, self.radii, modes, r0, quad,
                                    projector=projector, chunk=chunk)
        self.modes = modes

    def index(self, radii):
        return np.array([self.lookup[float(r)] for r in np.round(radii, 9)],
                        dtype=int)


def _richardson(values, Rcs):
    """Extrapolate a sequence of tapered sums to Rc -> infinity in 1/Rc^2."""
    x = 1.0 / np.asarray(Rcs, float) ** 2
    out = np.zeros_like(values[0])
    for i in range(len(Rcs)):
        L = 1.0
        for j in range(len(Rcs)):
            if j != i:
                L *= x[j] / (x[j] - x[i])
        out = out + L * values[i]
    return out


def _assemble(shells, sh_idx, radii, angles, bloch, modes, Rcs):
    """Tapered + Richardson-extrapolated sum over one displacement list.

    sh_idx: shell index of each displacement; radii/angles: its polar form;
    bloch: e^{i k_par . R} per displacement.
    """
    dm = modes.m[None, :] - modes.m[:, None]        # (mu, nu): m_nu - m_mu
    dms = np.unique(dm)
    dm_idx = np.searchsorted(dms, dm)
    nshell = len(shells.radii)
    terms = []
    for Rc in Rcs:
        w = np.exp(-(radii / Rc) ** 2) * bloch
        live = np.abs(w) > 1e-16
        P = np.zeros((nshell, len(dms)), dtype=complex)
        if live.any():
            ph = np.exp(1j * np.outer(angles[live], dms)) * w[live, None]
            np.add.at(P, sh_idx[live], ph)
        terms.append(np.einsum("smn,smn->mn", shells.A, P[:, dm_idx]))
    return _richardson(terms, Rcs)


# ------------------------------------------- pair-resolved Bloch coupling ---
def block_lattice_sums(k, a1, a2, rho, modes, r0, quad, kpar=(0.0, 0.0),
                       kRc=(10.0, 14.0, 20.0), r_max_factor=3.5,
                       projector=None, return_shells=False):
    """W[s, t] of manual Eq. (48), Gaussian-tapered + Richardson-extrapolated.

    a1, a2  : in-plane supercell lattice vectors (length units)
    rho     : (M, 2) or (M, 3) basis positions inside the cell (coplanar, z = 0)
    kpar    : in-plane Bloch vector (rad / length unit)
    Returns W with shape (M, M, n, n).
    """
    rho = np.atleast_2d(np.asarray(rho, float))[:, :2]
    M = len(rho)
    kpar = np.asarray(kpar, float)[:2]
    Rcs = np.asarray(kRc, float) / k
    r_max = r_max_factor * Rcs.max()

    # one enumeration per (s, t); collect every distinct radius first
    disp = {}
    all_r = []
    for s in range(M):
        for t in range(M):
            off = rho[t] - rho[s]
            d, R = lattice_points(a1, a2, r_max, off,
                                  drop_zero_offset=(s == t))
            r = np.round(np.linalg.norm(d, axis=1), 9)
            disp[(s, t)] = (d, R, r)
            all_r.append(r)
    radii = np.unique(np.concatenate(all_r)) if all_r else np.zeros(0)

    shells = ShellSet(k, radii, modes, r0, quad, projector=projector)

    W = np.empty((M, M, modes.n, modes.n), dtype=complex)
    for (s, t), (d, R, r) in disp.items():
        ang = np.arctan2(d[:, 1], d[:, 0])
        bloch = np.exp(1j * (R @ kpar))
        W[s, t] = _assemble(shells, shells.index(r), r, ang, bloch, modes, Rcs)
    return (W, shells) if return_shells else W


# ------------------------------------------------------------- the solve ---
def build_block_system(W, T_list):
    """M = I - W T0 of manual Eq. (50), as one dense (M n) x (M n) matrix."""
    Ms, n = W.shape[0], W.shape[2]
    big = np.eye(Ms * n, dtype=complex)
    for s in range(Ms):
        for t in range(Ms):
            big[s * n:(s + 1) * n, t * n:(t + 1) * n] -= W[s, t] @ T_list[t]
    return big


def solve_supercell(W, T_list, a_inc_blocks):
    """(I - W T0) a = a_inc ;  f = T0 a.  Returns a, f, each (M, n)."""
    Ms, n = W.shape[0], W.shape[2]
    big = build_block_system(W, T_list)
    x = np.linalg.solve(big, np.asarray(a_inc_blocks).reshape(Ms * n))
    a = x.reshape(Ms, n)
    f = np.stack([T_list[s] @ a[s] for s in range(Ms)])
    return a, f


def block_bloch_T(W, T_list):
    """T_B^loc = [I - T0 W]^{-1} T0 of manual Eq. (52), as an (M n) x (M n)
    matrix mapping stacked a_inc to stacked f.  Diagnostic / reuse object;
    `solve_supercell` never forms it."""
    Ms, n = W.shape[0], W.shape[2]
    T0 = np.zeros((Ms * n, Ms * n), dtype=complex)
    Wb = np.zeros_like(T0)
    for s in range(Ms):
        T0[s * n:(s + 1) * n, s * n:(s + 1) * n] = T_list[s]
        for t in range(Ms):
            Wb[s * n:(s + 1) * n, t * n:(t + 1) * n] = W[s, t]
    return np.linalg.solve(np.eye(Ms * n) - T0 @ Wb, T0)


def incident_blocks(k_hat, e_hat, modes, rho, k):
    """a_inc,s = e^{i k_inc . rho_s} a_pw of manual Eq. (54)."""
    rho = np.atleast_2d(np.asarray(rho, float))
    if rho.shape[1] == 2:
        rho = np.column_stack([rho, np.zeros(len(rho))])
    a_pw = plane_wave_coeffs(k_hat, e_hat, modes)
    k_vec = k * np.asarray(k_hat, float) / np.linalg.norm(k_hat)
    return np.exp(1j * (rho @ k_vec))[:, None] * a_pw[None, :]


# --------------------------------------------------------- Floquet output ---
def open_orders(k, a1, a2, kpar=(0.0, 0.0), tol=1e-9):
    """Propagating diffraction orders: list of dicts (n1, n2, kpar_G, kz)."""
    b1, b2 = reciprocal_vectors(a1, a2)
    kpar = np.asarray(kpar, float)[:2]
    nb = max(np.linalg.norm(b1), np.linalg.norm(b2))
    nmax = int(np.ceil((k + np.linalg.norm(kpar)) / min(
        np.linalg.norm(b1), np.linalg.norm(b2)))) + 1
    out = []
    for n1 in range(-nmax, nmax + 1):
        for n2 in range(-nmax, nmax + 1):
            G = n1 * b1 + n2 * b2
            kp = kpar + G
            kz2 = k * k - kp @ kp
            if kz2 > tol * k * k:
                out.append(dict(n=(n1, n2), kpar=kp, kz=float(np.sqrt(kz2)),
                                G=G))
    out.sort(key=lambda o: (-o["kz"], o["n"]))
    del nb
    return out


def order_pol_vectors(kpar_G, kz, k, sign):
    """(e_TE, e_TM) for the plane wave (kpar_G, sign*kz).

    Degenerate at normal incidence -- the fixed (y_hat, x_hat) pair is used
    there so that the G = 0 co-polarized entry is x_hat for both sides, which is
    what `sparams.sparams_normal` assumes.
    """
    khat = np.array([kpar_G[0], kpar_G[1], sign * kz]) / k
    if np.linalg.norm(kpar_G) < 1e-12 * k:
        e_te = np.array([0.0, 1.0, 0.0])
        e_tm = np.array([1.0, 0.0, 0.0])
    else:
        e_te = np.array([-kpar_G[1], kpar_G[0], 0.0])
        e_te = e_te / np.linalg.norm(e_te)
        e_tm = np.cross(e_te, khat)
    return e_te, e_tm, khat


def floquet_amplitudes(k, A_uc, modes, f_blocks, rho, orders, sign):
    """Vector plane-wave amplitude of every order on side `sign` (+1: z > 0).

    E_G = (2 pi i / (A_uc k_z,G)) sum_s e^{-i k_G . rho_s} F_s(k_hat_G)
    (manual Eq. 64 folded into the output channel map).  Returns (n_orders, 3).
    """
    rho = np.atleast_2d(np.asarray(rho, float))
    if rho.shape[1] == 2:
        rho = np.column_stack([rho, np.zeros(len(rho))])
    f_blocks = np.atleast_2d(f_blocks)
    E = np.zeros((len(orders), 3), dtype=complex)
    for i, o in enumerate(orders):
        kvec = np.array([o["kpar"][0], o["kpar"][1], sign * o["kz"]])
        f_sum = (np.exp(-1j * (rho @ kvec))[:, None] * f_blocks).sum(axis=0)
        F = far_field_amplitude(k, f_sum, modes, (kvec / k)[None, :])[0]
        E[i] = 2j * np.pi / (A_uc * o["kz"]) * F
    return E


def floquet_smatrix(k, a1, a2, rho, modes, f_blocks, e_inc=(1, 0, 0),
                    kpar=(0.0, 0.0), k_hat_inc=(0, 0, 1)):
    """Complete power-normalized Floquet S-matrix column for one incident wave.

    Returns a dict with, per side, a list of orders carrying the complex vector
    amplitude, its TE/TM projections and the power fraction.  The incident wave
    (unit amplitude, order 0 from z < 0) is added to the transmitted field.
    """
    A_uc = abs(a1[0] * a2[1] - a1[1] * a2[0])
    orders = open_orders(k, a1, a2, kpar)
    kz_in = float(np.abs(k * np.asarray(k_hat_inc, float)[2]))
    e_inc = np.asarray(e_inc, dtype=complex)

    out = {"A_uc": A_uc, "orders": orders, "kz_in": kz_in}
    for side, name in ((+1, "trans"), (-1, "refl")):
        E = floquet_amplitudes(k, A_uc, modes, f_blocks, rho, orders, side)
        if side == +1:
            for i, o in enumerate(orders):
                if o["n"] == (0, 0):
                    E[i] = E[i] + e_inc          # direct background path
        rows = []
        for i, o in enumerate(orders):
            e_te, e_tm, _ = order_pol_vectors(o["kpar"], o["kz"], k, side)
            nrm = np.sqrt(o["kz"] / kz_in)
            rows.append(dict(n=o["n"], kz=o["kz"], E=E[i],
                             S_te=nrm * np.vdot(e_te, E[i]),
                             S_tm=nrm * np.vdot(e_tm, E[i]),
                             power=float(o["kz"] / kz_in *
                                         np.vdot(E[i], E[i]).real)))
        out[name] = rows
    out["T"] = sum(r["power"] for r in out["trans"])
    out["R"] = sum(r["power"] for r in out["refl"])
    out["A"] = 1.0 - out["T"] - out["R"]
    return out


def zeroth_order(res, e_co=(1, 0, 0), e_cross=(0, 1, 0)):
    """Scalar S11/S21 of the (0, 0) order in a chosen polarization pair."""
    e_co = np.asarray(e_co, dtype=complex)
    e_cross = np.asarray(e_cross, dtype=complex)
    g = {name: next(r for r in res[name] if r["n"] == (0, 0))
         for name in ("trans", "refl")}
    return {
        "S21_co": np.vdot(e_co, g["trans"]["E"]),
        "S21_cross": np.vdot(e_cross, g["trans"]["E"]),
        "S11_co": np.vdot(e_co, g["refl"]["E"]),
        "S11_cross": np.vdot(e_cross, g["refl"]["E"]),
    }


# ------------------------------------- finite cluster: single-origin T (6.4) --
def _sphere_projection(k, d, modes_src, modes_tgt, r0, quad, kind,
                       projector=None):
    """Matrix re-expanding `kind` source waves at offset d into target waves.

    d = r_source - r_target.  `kind` 'outgoing' gives U^{31} (the manual's
    translation block, needs r0 < |d|); 'regular' gives the regular addition
    matrix Rg(d), which is entire, so any r0 works -- and Rg is *also* the
    outgoing -> outgoing operator, valid outside |r - r_target| > |d|.
    """
    pts = sphere_points(r0, quad)
    E, Ht = vswf_fields(k, modes_src, pts, kind, origin=np.asarray(d, float))
    proj = projector if projector is not None else RegularProjector(modes_tgt,
                                                                    quad)
    return proj.project(k, r0, E, Ht).T          # (n_tgt, n_src)


def regular_translation(k, d, modes_src, modes_tgt, r0, quad, projector=None):
    """Rg(d): regular->regular (and outgoing->outgoing) addition matrix."""
    return _sphere_projection(k, d, modes_src, modes_tgt, r0, quad, "regular",
                              projector=projector)


def outgoing_translation(k, d, modes_src, modes_tgt, r0, quad, projector=None):
    """U^{31}(d): outgoing at source -> regular at target.  Needs r0 < |d|."""
    return _sphere_projection(k, d, modes_src, modes_tgt, r0, quad, "outgoing",
                              projector=projector)


def finite_interaction(k, positions, modes, r0, quad, projector=None):
    """U of manual Eq. (34) for a finite cluster (zero diagonal blocks)."""
    positions = np.atleast_2d(np.asarray(positions, float))
    if positions.shape[1] == 2:
        positions = np.column_stack([positions, np.zeros(len(positions))])
    N, n = len(positions), modes.n
    proj = projector if projector is not None else RegularProjector(modes, quad)
    U = np.zeros((N, N, n, n), dtype=complex)
    cache = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = positions[j] - positions[i]     # source j -> target i
            key = tuple(np.round(d, 9))
            if key not in cache:
                cache[key] = outgoing_translation(k, d, modes, modes, r0, quad,
                                                  projector=proj)
            U[i, j] = cache[key]
    return U


def cluster_projection_grid(k, lmax_c, reach):
    """(r0, quad) for the P/Q regular translations of a cluster of order lmax_c.

    Unlike U, these are projections onto a HIGH-order basis, and the regular
    radial functions collapse below the sampling sphere: j_l(k r0) ~
    (k r0)^l / (2l+1)!!, so at k r0 = 1.4 and l = 14 both the projected
    coefficient and the j_l^2 + xi_l^2 the projector divides by are ~1e-26 and
    the ratio is pure round-off.  Regular waves are entire, so r0 is free: put
    the sphere where the target order is on the propagating side of its turning
    point, k r0 > lmax_c, and then size the quadrature for the angular content
    the sphere actually sees, l ~ k (r0 + reach), so the projection is not
    aliased.  `reach` is the largest source offset (max |rho_s - O|).
    """
    r0 = (lmax_c + 2.0) / k
    l_eff = int(np.ceil(k * (r0 + reach))) + 2
    return r0, sphere_quadrature(2 * l_eff + 6, 4 * l_eff + 12)


def cluster_T(k, positions, T_list, modes, lmax_c, r0, quad, origin=(0, 0, 0),
              U=None, projector=None, r0_c=None, quad_c=None):
    """Single-origin finite-array T-matrix T^O = Q (I - T0 U)^{-1} T0 P (Eq. 40).

    Valid outside a sphere about `origin` enclosing the whole cluster.  `r0` and
    `quad` are used for the inter-atom operator U (they must satisfy
    r0 < min |r_i - r_j|); `r0_c` / `quad_c` are used for the global P and Q and
    default to `cluster_projection_grid`.  Returns (T^O, cluster ModeBasis).
    """
    positions = np.atleast_2d(np.asarray(positions, float))
    if positions.shape[1] == 2:
        positions = np.column_stack([positions, np.zeros(len(positions))])
    origin = np.asarray(origin, float)
    N, n = len(positions), modes.n
    mb_c = ModeBasis.standard(lmax_c)
    if U is None:
        U = finite_interaction(k, positions, modes, r0, quad,
                               projector=projector)

    reach = float(np.linalg.norm(positions - origin, axis=1).max())
    if r0_c is None or quad_c is None:
        r0_d, quad_d = cluster_projection_grid(k, lmax_c, reach)
        r0_c = r0_d if r0_c is None else r0_c
        quad_c = quad_d if quad_c is None else quad_c
    proj_loc = RegularProjector(modes, quad_c)
    proj_glb = RegularProjector(mb_c, quad_c)

    # P: global regular (about O) -> local regular at rho_s   [source O]
    # Q: local outgoing at rho_s  -> global outgoing (about O) [source rho_s]
    P = np.zeros((N * n, mb_c.n), dtype=complex)
    Q = np.zeros((mb_c.n, N * n), dtype=complex)
    for s in range(N):
        d = origin - positions[s]
        P[s * n:(s + 1) * n] = regular_translation(k, d, mb_c, modes, r0_c,
                                                   quad_c, projector=proj_loc)
        Q[:, s * n:(s + 1) * n] = regular_translation(k, -d, modes, mb_c, r0_c,
                                                      quad_c,
                                                      projector=proj_glb)

    T0 = np.zeros((N * n, N * n), dtype=complex)
    Ub = np.zeros_like(T0)
    for i in range(N):
        T0[i * n:(i + 1) * n, i * n:(i + 1) * n] = T_list[i]
        for j in range(N):
            Ub[i * n:(i + 1) * n, j * n:(j + 1) * n] = U[i, j]
    X = np.linalg.solve(np.eye(N * n) - T0 @ Ub, T0 @ P)
    return Q @ X, mb_c
