"""Vector spherical wave functions (VSWF) and projections.

Conventions (matching "The Physics of the Neural Analytic Modal Method",
with the normalization gap closed as documented below):

- Time dependence e^{+j omega t}; outgoing radial function is the spherical
  Hankel function of the SECOND kind h_n^(2)(kr); regular waves use j_n(kr).
- Orthonormal spherical harmonics with Condon-Shortley phase:
      Y_nm(th, ph) = Ptilde_n^m(cos th) e^{j m ph},
      Ptilde: normalized associated Legendre, int |Y|^2 dOmega = 1.
  (The NAMM paper leaves the normalization unspecified; we fix it here.)
- VSWFs, normalized so the tangential vector harmonics are orthonormal
  (gamma_n = 1/sqrt(n(n+1))):
      M_nm^(c) = gamma_n curl( r  z_n^(c)(kr) Y_nm )
               = gamma_n z_n(kr) [ j pi_nm Y_th^ - tau_nm Y_ph^ ] e^{jm ph}
      N_nm^(c) = (1/k) curl M_nm^(c)
               = gamma_n [ n(n+1) z_n(kr)/(kr) Y_nm r^
                          + ((kr z_n)'/kr) (tau_nm th^ + j pi_nm ph^) e^{jm ph} ]
  with pi_nm = m Ptilde_n^m / sin th, tau_nm = d Ptilde_n^m / d th.
- Fields:  E_inc = sum a_nm M^(1) + b_nm N^(1)
           E_sca = sum f_nm M^(3) + g_nm N^(3)      (c=3 -> h^(2))
           Z0 H  = j [ sum a_nm N + b_nm M ]        (same c as E)
  Stacking: coefficient vectors are [M-block; N-block], n = 1..lmax outer,
  m = -n..n inner (block_index = n^2 - 1 + m + n).  f = T a.

All sign/conjugation choices are locked by tests/test_vswf.py (plane-wave
reconstruction, projection round-trip, Mie synthetic extraction).

Interop: the community convention (treams, tmat.h5) uses e^{-i omega t} with
h^(1); converting our T to it is element-wise complex conjugation (mode
ordering preserved; their default ordering differs — reorder as needed).
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from .quadrature import spherical_basis

# ----------------------------------------------------------------------------
# Mode bookkeeping
# ----------------------------------------------------------------------------

def n_modes(lmax: int) -> int:
    """Total stacked size N = 2 * lmax * (lmax + 2)."""
    return 2 * lmax * (lmax + 2)


def mode_list(lmax: int):
    """Return (pol, n, m) integer arrays of length N in canonical stacking
    order: pol 0 = M/TE block first, pol 1 = N/TM block."""
    ns, ms = [], []
    for n in range(1, lmax + 1):
        for m in range(-n, n + 1):
            ns.append(n)
            ms.append(m)
    half = len(ns)
    pol = np.concatenate([np.zeros(half, int), np.ones(half, int)])
    return pol, np.tile(ns, 2), np.tile(ms, 2)


def block_index(n: int, m: int) -> int:
    """Index within one polarization block: n^2 - 1 + (m + n)."""
    return n * n - 1 + m + n


# ----------------------------------------------------------------------------
# Normalized associated Legendre, pi and tau functions (stable recurrences)
# ----------------------------------------------------------------------------

def _legendre_norm(lmax: int, ct: np.ndarray, st: np.ndarray):
    """Orthonormal associated Legendre Ptilde_n^m(cos th) with Condon-
    Shortley phase, 0 <= m <= n <= lmax.  Array P[n, m, pt]."""
    npts = ct.shape[0]
    P = np.zeros((lmax + 1, lmax + 1, npts))
    P[0, 0] = 1.0 / np.sqrt(4 * np.pi)
    for m in range(1, lmax + 1):
        P[m, m] = -np.sqrt((2 * m + 1) / (2 * m)) * st * P[m - 1, m - 1]
    for m in range(0, lmax):
        P[m + 1, m] = np.sqrt(2 * m + 3) * ct * P[m, m]
    for m in range(0, lmax + 1):
        for n in range(m + 2, lmax + 1):
            a = np.sqrt((4 * n * n - 1) / (n * n - m * m))
            b = np.sqrt(((n - 1) ** 2 - m * m) / (4 * (n - 1) ** 2 - 1))
            P[n, m] = a * (ct * P[n - 1, m] - b * P[n - 2, m])
    return P


def _P_signed(P, n, m, npts):
    """Ptilde_n^m for any m (negative via Ptilde_n^{-m} = (-1)^m Ptilde_n^m)."""
    if abs(m) > n:
        return np.zeros(npts)
    if m >= 0:
        return P[n, m]
    return (-1.0) ** (-m) * P[n, -m]


def pi_tau(lmax: int, theta: np.ndarray):
    """pi[n, m+lmax, pt] = m/sin(th) Ptilde_n^m ;
    tau[n, m+lmax, pt] = d Ptilde_n^m / d th, for all -n <= m <= n.

    tau uses the pole-safe ladder identity (from L+- Y_nm = sqrt((n-+m)(n+-m+1)) Y_n,m+-1):
      d/dth Ptilde_n^m = 1/2 [ sqrt((n-m)(n+m+1)) Ptilde_n^{m+1}
                             - sqrt((n+m)(n-m+1)) Ptilde_n^{m-1} ].
    pi divides by sin(th); quadrature nodes and Fibonacci directions never
    hit the poles, exact poles get the analytic limit (nonzero only |m|=1,
    where lim pi_nm = sign factor * tau_nm).
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    ct, st = np.cos(theta), np.sin(theta)
    P = _legendre_norm(lmax, ct, st)
    npts = theta.shape[0]

    pi = np.zeros((lmax + 1, 2 * lmax + 1, npts))
    tau = np.zeros((lmax + 1, 2 * lmax + 1, npts))
    pole = st < 1e-12
    safe_st = np.where(pole, 1.0, st)
    for n in range(1, lmax + 1):
        for m in range(-n, n + 1):
            tau[n, m + lmax] = 0.5 * (
                np.sqrt((n - m) * (n + m + 1)) * _P_signed(P, n, m + 1, npts)
                - np.sqrt((n + m) * (n - m + 1)) * _P_signed(P, n, m - 1, npts))
            val = m * _P_signed(P, n, m, npts) / safe_st
            if abs(m) == 1:
                lim = np.where(ct > 0, m, -m) * tau[n, m + lmax]
                val = np.where(pole, lim, val)
            else:
                val = np.where(pole, 0.0, val)
            pi[n, m + lmax] = val
    return pi, tau


def _exp_jmphi(lmax: int, phi: np.ndarray):
    """e^{j m phi} for m = -lmax..lmax, array [m + lmax, pt]."""
    m = np.arange(-lmax, lmax + 1)
    return np.exp(1j * np.outer(m, phi))


# ----------------------------------------------------------------------------
# Radial functions
# ----------------------------------------------------------------------------

def radial(kind: str, n: int, kr: np.ndarray):
    """(z_n(kr), [kr z_n(kr)]'/kr) for 'regular' (j_n) or 'outgoing'
    (h_n^(2) = j_n - j y_n, e^{+jwt} convention)."""
    kr = np.asarray(kr, dtype=float)
    jn = spherical_jn(n, kr)
    jnp = spherical_jn(n, kr, derivative=True)
    if kind == "regular":
        z, zp = jn, jnp
    elif kind == "outgoing":
        yn = spherical_yn(n, kr)
        ynp = spherical_yn(n, kr, derivative=True)
        z = jn - 1j * yn
        zp = jnp - 1j * ynp
    else:
        raise ValueError(kind)
    ric = z / kr + zp
    return z, ric


# ----------------------------------------------------------------------------
# Field evaluation
# ----------------------------------------------------------------------------

def _angular_tables(lmax: int, theta, phi):
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pi, tau = pi_tau(lmax, theta)
    ejm = _exp_jmphi(lmax, phi)
    P = _legendre_norm(lmax, np.cos(theta), np.sin(theta))
    return pi, tau, ejm, P


def evaluate_field(coeffs: np.ndarray, lmax: int, kind: str, k: float,
                   points: np.ndarray):
    """E(r) of a stacked coefficient vector at Cartesian points (Npts, 3).
    Returns complex (Npts, 3) Cartesian components."""
    pts = np.asarray(points, dtype=float)
    r = np.linalg.norm(pts, axis=1)
    theta = np.arccos(np.clip(pts[:, 2] / np.where(r == 0, 1, r), -1, 1))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    r_hat, th_hat, ph_hat = spherical_basis(theta, phi)

    pi, tau, ejm, P = _angular_tables(lmax, theta, phi)
    kr = k * r
    npts = pts.shape[0]

    E = np.zeros((npts, 3), dtype=complex)
    half = n_modes(lmax) // 2
    for n in range(1, lmax + 1):
        z, ric = radial(kind, n, kr)
        gam = 1.0 / np.sqrt(n * (n + 1))
        for m in range(-n, n + 1):
            idx = block_index(n, m)
            cM = coeffs[idx]
            cN = coeffs[half + idx]
            if cM == 0 and cN == 0:
                continue
            ang = ejm[m + lmax]
            piv = pi[n, m + lmax] * ang
            tav = tau[n, m + lmax] * ang
            if cM != 0:
                E += (cM * gam) * (z * (1j * piv))[:, None] * th_hat
                E += (cM * gam) * (z * (-tav))[:, None] * ph_hat
            if cN != 0:
                Ynm = _P_signed(P, n, m, npts) * ang
                E += (cN * gam) * (n * (n + 1) * z / kr * Ynm)[:, None] * r_hat
                E += (cN * gam) * (ric * tav)[:, None] * th_hat
                E += (cN * gam) * (ric * (1j * piv))[:, None] * ph_hat
    return E


def evaluate_EH(coeffs: np.ndarray, lmax: int, kind: str, k: float,
                points: np.ndarray):
    """(E, Z0*H) of a stacked coefficient vector.  Z0 H = j * field of the
    block-swapped coefficient vector (curl relations swap M and N roles)."""
    half = n_modes(lmax) // 2
    swapped = np.concatenate([coeffs[half:], coeffs[:half]])
    E = evaluate_field(coeffs, lmax, kind, k, points)
    ZH = 1j * evaluate_field(swapped, lmax, kind, k, points)
    return E, ZH


# ----------------------------------------------------------------------------
# Plane wave
# ----------------------------------------------------------------------------

def plane_wave_field(theta_i, phi_i, pol: str, k: float, points: np.ndarray):
    """Unit plane wave E(r) = e^ exp(-j k k^.r), zero phase at origin,
    propagating along k^(theta_i, phi_i); pol in {'theta','phi'} picks e^ as
    the spherical unit vector at the propagation direction.
    Returns (E, Z0*H), each (Npts, 3) Cartesian;  Z0 H = k^ x E."""
    r_hat, th_hat, ph_hat = spherical_basis(np.atleast_1d(float(theta_i)),
                                            np.atleast_1d(float(phi_i)))
    k_hat, e_th, e_ph = r_hat[0], th_hat[0], ph_hat[0]
    e_hat = e_th if pol == "theta" else e_ph
    h_hat = np.cross(k_hat, e_hat)
    phase = np.exp(-1j * k * np.asarray(points) @ k_hat)
    return np.outer(phase, e_hat), np.outer(phase, h_hat)


def plane_wave_coefficients(theta_i, phi_i, pol: str, lmax: int):
    """Stacked VSWF coefficient vector a of the unit plane wave defined by
    plane_wave_field, in this module's conventions:

        a^M_nm = 4 pi (-j)^n     e_hat . conj( C_nm(k_hat) )
        a^N_nm = 4 pi (-j)^{n-1} e_hat . conj( B_nm(k_hat) )

    with C_nm = gamma_n (j pi_nm th^ - tau_nm ph^) e^{jm ph_i} and
    B_nm = gamma_n (tau_nm th^ + j pi_nm ph^) e^{jm ph_i} evaluated at the
    propagation direction.  Derived by numerical orthogonality projection
    (m-independent per-n prefactors fit exactly) and locked by
    tests/test_vswf.py — do not change signs without re-running the tests.
    """
    pi, tau = pi_tau(lmax, np.atleast_1d(float(theta_i)))
    N = n_modes(lmax)
    half = N // 2
    a = np.zeros(N, dtype=complex)
    for n in range(1, lmax + 1):
        gam = 1.0 / np.sqrt(n * (n + 1))
        pref = 4 * np.pi * (-1j) ** n
        for m in range(-n, n + 1):
            idx = block_index(n, m)
            piv = pi[n, m + lmax][0]
            tav = tau[n, m + lmax][0]
            emjm = np.exp(-1j * m * float(phi_i))
            if pol == "theta":
                eC = gam * (-1j * piv) * emjm      # e^ . conj(C_nm)
                eB = gam * tav * emjm              # e^ . conj(B_nm)
            elif pol == "phi":
                eC = gam * (-tav) * emjm
                eB = gam * (-1j * piv) * emjm
            else:
                raise ValueError(pol)
            a[idx] = pref * eC
            a[half + idx] = pref * 1j * eB         # (-j)^{n-1} = j (-j)^n
    return a


# ----------------------------------------------------------------------------
# Projection of fields on a sphere onto VSWF coefficients
# ----------------------------------------------------------------------------

def project_surface_field(E: np.ndarray, lmax: int, kind: str, k: float,
                          radius: float, theta, phi, w,
                          H: np.ndarray | None = None):
    """Recover stacked VSWF coefficients of a field from samples on a sphere.

    E : (Npts, 3) complex Cartesian E-field on the sphere of `radius`, at
        quadrature nodes (theta, phi) with weights w (sum w f = solid-angle
        integral).
    kind : 'outgoing' for scattered fields, 'regular' for incident fields.
    H : optional (Npts, 3) Z0*H samples; combining E and H keeps the
        recovery well-conditioned even where j_n(kr) or its Riccati
        derivative vanish (relevant for kind='regular'; h^(2) never
        vanishes so E-only is fine for scattered fields).

    Tangential harmonics used (orthonormal over the sphere):
        C_nm = gamma_n ( j pi th^ - tau ph^ ) e^{jm ph}    [M-type]
        B_nm = gamma_n ( tau th^ + j pi ph^ ) e^{jm ph}    [N-type]
    Tangential field:  E_t = sum cM z_n C_nm + cN ((kr z_n)'/kr) B_nm
                       Z0 H_t = j sum cM ((kr z_n)'/kr) B_nm + cN z_n C_nm
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    _, th_hat, ph_hat = spherical_basis(theta, phi)
    E_th = np.einsum("ij,ij->i", E, th_hat)
    E_ph = np.einsum("ij,ij->i", E, ph_hat)
    if H is not None:
        H_th = np.einsum("ij,ij->i", H, th_hat)
        H_ph = np.einsum("ij,ij->i", H, ph_hat)

    pi, tau = pi_tau(lmax, theta)
    ejm = _exp_jmphi(lmax, phi)
    kr1 = np.array([k * radius])

    N = n_modes(lmax)
    half = N // 2
    c = np.zeros(N, dtype=complex)
    for n in range(1, lmax + 1):
        z, ric = radial(kind, n, kr1)
        z, ric = z[0], ric[0]
        gam = 1.0 / np.sqrt(n * (n + 1))
        for m in range(-n, n + 1):
            idx = block_index(n, m)
            ang = ejm[m + lmax]
            piv = pi[n, m + lmax] * ang
            tav = tau[n, m + lmax] * ang
            pC = gam * np.sum(w * (np.conj(1j * piv) * E_th
                                   + np.conj(-tav) * E_ph))
            pB = gam * np.sum(w * (np.conj(tav) * E_th
                                   + np.conj(1j * piv) * E_ph))
            if H is None:
                c[idx] = pC / z
                c[half + idx] = pB / ric
            else:
                hC = gam * np.sum(w * (np.conj(1j * piv) * H_th
                                       + np.conj(-tav) * H_ph))
                hB = gam * np.sum(w * (np.conj(tav) * H_th
                                       + np.conj(1j * piv) * H_ph))
                A1 = np.array([z, 1j * ric])       # cM seen by (E via C, H via B)
                y1 = np.array([pC, hB])
                c[idx] = (np.conj(A1) @ y1) / (np.conj(A1) @ A1).real
                A2 = np.array([ric, 1j * z])       # cN seen by (E via B, H via C)
                y2 = np.array([pB, hC])
                c[half + idx] = (np.conj(A2) @ y2) / (np.conj(A2) @ A2).real
    return c


def separate_surface_field(E: np.ndarray, H: np.ndarray, lmax: int, k: float,
                           radius: float, theta, phi, w):
    """Split a TOTAL field on a sphere into regular (incident) and outgoing
    (scattered) VSWF coefficients — the self-calibrating primary path for
    CST extraction.

    On one sphere, per mode (n, m), the tangential projections of E and Z0*H
    give four equations in four unknowns (aM, fM) and (aN, gN):
        pC = aM j_n  + fM h_n          hB = j (aM jric + fM hric)
        pB = aN jric + gN hric         hC = j (aN j_n  + gN h_n )
    Each 2x2 system has determinant proportional to the Wronskian
    W{j_n, h_n^(2)} != 0, so the split is always well-conditioned.

    Because both incident and scattered coefficients come from the SAME
    simulation data, any amplitude/phase convention of the source (e.g.
    CST's plane-wave phase reference) cancels in T = F A^+.

    Returns (a_regular, f_outgoing), each a stacked length-N vector.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    _, th_hat, ph_hat = spherical_basis(theta, phi)
    E_th = np.einsum("ij,ij->i", E, th_hat)
    E_ph = np.einsum("ij,ij->i", E, ph_hat)
    H_th = np.einsum("ij,ij->i", H, th_hat)
    H_ph = np.einsum("ij,ij->i", H, ph_hat)

    pi, tau = pi_tau(lmax, theta)
    ejm = _exp_jmphi(lmax, phi)
    kr1 = np.array([k * radius])

    N = n_modes(lmax)
    half = N // 2
    a = np.zeros(N, dtype=complex)
    f = np.zeros(N, dtype=complex)
    for n in range(1, lmax + 1):
        jz, jric = radial("regular", n, kr1)
        hz, hric = radial("outgoing", n, kr1)
        jz, jric, hz, hric = jz[0], jric[0], hz[0], hric[0]
        gam = 1.0 / np.sqrt(n * (n + 1))
        M1 = np.array([[jz, hz], [1j * jric, 1j * hric]])       # (aM,fM)
        M2 = np.array([[jric, hric], [1j * jz, 1j * hz]])       # (aN,gN)
        M1inv = np.linalg.inv(M1)
        M2inv = np.linalg.inv(M2)
        for m in range(-n, n + 1):
            idx = block_index(n, m)
            ang = ejm[m + lmax]
            piv = pi[n, m + lmax] * ang
            tav = tau[n, m + lmax] * ang
            pC = gam * np.sum(w * (np.conj(1j * piv) * E_th
                                   + np.conj(-tav) * E_ph))
            pB = gam * np.sum(w * (np.conj(tav) * E_th
                                   + np.conj(1j * piv) * E_ph))
            hC = gam * np.sum(w * (np.conj(1j * piv) * H_th
                                   + np.conj(-tav) * H_ph))
            hB = gam * np.sum(w * (np.conj(tav) * H_th
                                   + np.conj(1j * piv) * H_ph))
            aM, fM = M1inv @ np.array([pC, hB])
            aN, gN = M2inv @ np.array([pB, hC])
            a[idx], f[idx] = aM, fM
            a[half + idx], f[half + idx] = aN, gN
    return a, f


def to_treams_convention(T: np.ndarray, lmax: int):
    """Convert a T-matrix from the internal convention to the convention of
    the tmat.h5 data format and the treams code (Beutel et al., Comput.
    Phys. Commun. 297, 109076 (2024)).

    Both bases use Jackson-normalized spherical harmonics with the
    Condon-Shortley phase; treams uses the e^{-i omega t} time dependence
    with h_l^(1) outgoing waves, and its basis functions carry a uniform
    prefactor i relative to ours, which cancels in T.  Writing a
    monochromatic physical field in both phasor conventions,
    E(r,t) = Re[E_+ e^{+j w t}] = Re[E_- e^{-i w t}] with E_- = E_+^*,
    and using P_l^{-m} = (-1)^m P_l^m, the expansion coefficients map as
    a^-_{l,m} = (-1)^m (a^+_{l,-m})^*, which gives

        T^-_{(p,l,m),(p',l',m')} = (-1)^{m+m'} [ T^+_{(p,l,-m),(p',l',-m')} ]^*

    (polarizations unmixed).  Note this is NOT plain element-wise
    conjugation; the two agree only for matrices diagonal and even in m
    (e.g. spheres).  The map is verified at the field level in
    tests/test_storage_treams.py.

    Parameters
    ----------
    T : (N, N) or (n_freq, N, N) complex, internal stacking
        ([M block; N block], n = 1..lmax, m = -n..n).

    Returns the converted matrix with the SAME internal stacking; use
    `treams_mode_tables` and `reorder_to_interleaved` for the mode
    bookkeeping written into tmat.h5 files.
    """
    N = n_modes(lmax)
    single = (T.ndim == 2)
    Tin = T[None, ...] if single else T
    if Tin.shape[-2:] != (N, N):
        raise ValueError(f"T shape {T.shape} inconsistent with lmax={lmax}")
    half = N // 2
    pol, ns, ms = mode_list(lmax)
    neg = np.array([(0 if p == 0 else half) + block_index(n, -m)
                    for p, n, m in zip(pol, ns, ms)])
    sign = (-1.0) ** (ms[:, None] + ms[None, :])
    out = sign * np.conj(Tin[:, neg][:, :, neg])
    return out[0] if single else out


def from_treams_convention(T: np.ndarray, lmax: int):
    """Inverse of `to_treams_convention` (the map is an involution)."""
    return to_treams_convention(T, lmax)


def treams_interleaved_order(lmax: int):
    """Permutation and mode tables for the interleaved ordering used in
    tmat.h5 files: l ascending, m = -l..l, and for each (l, m) the two
    polarizations with 'electric' (N-type) first.

    Returns (perm, l_arr, m_arr, pol_labels) such that
    T_interleaved = T_internal[perm][:, perm] and row i of the output
    corresponds to (l_arr[i], m_arr[i], pol_labels[i]).
    """
    N = n_modes(lmax)
    half = N // 2
    perm, ls, ms_, labels = [], [], [], []
    for n in range(1, lmax + 1):
        for m in range(-n, n + 1):
            idx = block_index(n, m)
            perm.extend([half + idx, idx])     # electric (N), then magnetic (M)
            ls.extend([n, n])
            ms_.extend([m, m])
            labels.extend(["electric", "magnetic"])
    return (np.array(perm), np.array(ls, dtype=np.int64),
            np.array(ms_, dtype=np.int64), labels)


def evaluate_farfield(coeffs: np.ndarray, lmax: int, theta, phi):
    """Far-field amplitude F(theta, phi) of a stacked OUTGOING coefficient
    vector -- the exact inverse of project_farfield (same asymptotic
    convention: E_sca(r) -> F(th,ph) e^{-jkr}/(kr) as kr -> inf).

    Returns (F_th, F_ph) complex, one value per (theta, phi) pair.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    pi, tau = pi_tau(lmax, theta)
    ejm = _exp_jmphi(lmax, phi)

    half = n_modes(lmax) // 2
    F_th = np.zeros(theta.shape[0], dtype=complex)
    F_ph = np.zeros(theta.shape[0], dtype=complex)
    for n in range(1, lmax + 1):
        gam = 1.0 / np.sqrt(n * (n + 1))
        aM = (1j) ** (n + 1)
        aN = (1j) ** n
        for m in range(-n, n + 1):
            idx = block_index(n, m)
            cM = coeffs[idx]
            cN = coeffs[half + idx]
            if cM == 0 and cN == 0:
                continue
            ang = ejm[m + lmax]
            piv = pi[n, m + lmax] * ang
            tav = tau[n, m + lmax] * ang
            if cM != 0:
                F_th += (cM * gam * aM) * (1j * piv)
                F_ph += (cM * gam * aM) * (-tav)
            if cN != 0:
                F_th += (cN * gam * aN) * tav
                F_ph += (cN * gam * aN) * (1j * piv)
    return F_th, F_ph


def project_farfield(E_th_far: np.ndarray, E_ph_far: np.ndarray, lmax: int,
                     k: float, theta, phi, w):
    """Recover OUTGOING coefficients from the scattered farfield pattern.

    Farfield convention: E_sca(r) -> F(th, ph) e^{-jkr}/(k r) as kr -> inf,
    i.e. F = lim kr e^{+jkr} E_sca (dimension of E times unity; if the input
    is CST's 1 m reference farfield in V/m, multiply by k*(1 m)*e^{+jk*1m}
    upstream — handled in the CST export layer).

    Using h_n^(2)(kr) ~ j^{n+1} e^{-jkr}/(kr):
        F = sum_nm  j^{n+1} [ f_nm C_nm + j^{-1}... ] — see code.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    pi, tau = pi_tau(lmax, theta)
    ejm = _exp_jmphi(lmax, phi)

    N = n_modes(lmax)
    half = N // 2
    c = np.zeros(N, dtype=complex)
    for n in range(1, lmax + 1):
        gam = 1.0 / np.sqrt(n * (n + 1))
        # asymptotics: z_n -> j^{n+1} e^{-jkr}/kr ; (kr z_n)'/kr -> j^n e^{-jkr}/kr
        aM = (1j) ** (n + 1)
        aN = (1j) ** n
        for m in range(-n, n + 1):
            idx = block_index(n, m)
            ang = ejm[m + lmax]
            piv = pi[n, m + lmax] * ang
            tav = tau[n, m + lmax] * ang
            pC = gam * np.sum(w * (np.conj(1j * piv) * E_th_far
                                   + np.conj(-tav) * E_ph_far))
            pB = gam * np.sum(w * (np.conj(tav) * E_th_far
                                   + np.conj(1j * piv) * E_ph_far))
            c[idx] = pC / aM
            c[half + idx] = pB / aN
    return c
