"""Vector spherical wave functions (VSWFs) in the tmat.h5 convention.

Convention (matches the demo file's root attribute):
  - time dependence exp(-i omega t)
  - outgoing radial functions h_l^(1)
  - Jackson-normalized vector spherical waves with Condon-Shortley phase
  - parity basis: polarization in {electric (N-wave, TM), magnetic (M-wave, TE)}
  - p = T a, modes ordered as listed in /modes of the file

Definitions:
  Y_lm            : orthonormal spherical harmonics with Condon-Shortley phase
                    (scipy.special.sph_harm_y).
  X_lm(th,ph)     = L Y_lm / sqrt(l(l+1)),  L = -i r x grad
                  = [-(m/sin th) Y_lm] th_hat + [-i dY_lm/dth] ph_hat, normalized.
  Z_lm            = r_hat x X_lm.
  M^(c)_lm(r)     = z_l^(c)(kr) X_lm                       (magnetic / TE)
  N^(c)_lm(r)     = i sqrt(l(l+1)) z_l^(c)(kr)/(kr) Y_lm r_hat
                    + xi_l^(c)(kr) Z_lm                    (electric / TM)
  (the i on the radial term follows exactly from N = (1/k) curl M with
   Jackson's X = L Y / sqrt(l(l+1)), L = -i r x grad; verified by FD curl)
  xi_l(x)         = [x z_l(x)]' / x = z_l'(x) + z_l(x)/x
  c=1: z_l = j_l (regular),  c=3: z_l = h_l^(1) (outgoing).

Duality (vacuum, Z0-scaled magnetic field Ht = i*Z0*H):
  E = M  =>  Ht = N     ;     E = N  =>  Ht = M

Plane wave  E = e_hat exp(i k.r):
  a^M_lm = 4 pi i^l     X*_lm(k_hat) . e_hat
  a^E_lm = 4 pi i^(l-1) Z*_lm(k_hat) . e_hat

Far field of outgoing expansion E_sca = sum f^M M^(3) + f^E N^(3):
  E_sca -> (e^{ikr}/r) F(r_hat),
  F(r_hat) = (1/k) sum_lm [ f^M_lm (-i)^{l+1} X_lm + f^E_lm (-i)^l Z_lm ]
"""
import numpy as np
from scipy.special import sph_harm_y, spherical_jn, spherical_yn

ELECTRIC = 0  # N-wave, TM
MAGNETIC = 1  # M-wave, TE

_POLE_EPS = 1e-9


class ModeBasis:
    """Ordered list of (l, m, pol) modes. pol: 0=electric, 1=magnetic."""

    def __init__(self, l, m, pol):
        self.l = np.asarray(l, dtype=int)
        self.m = np.asarray(m, dtype=int)
        self.pol = np.asarray(pol, dtype=int)
        self.n = len(self.l)
        self.lmax = int(self.l.max())

    @classmethod
    def standard(cls, lmax):
        """tmat.h5 default order: l asc, m asc, electric before magnetic."""
        ls, ms, ps = [], [], []
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                for p in (ELECTRIC, MAGNETIC):
                    ls.append(l); ms.append(m); ps.append(p)
        return cls(ls, ms, ps)

    def index(self, l, m, pol):
        hit = np.nonzero((self.l == l) & (self.m == m) & (self.pol == pol))[0]
        return int(hit[0]) if len(hit) else None


def spherical_h1(l, x):
    return spherical_jn(l, x) + 1j * spherical_yn(l, x)


def radial_funcs(kind, l, x):
    """Return z_l(x) and xi_l(x) = z_l'(x) + z_l(x)/x for kind 'regular'/'outgoing'."""
    x = np.asarray(x, dtype=float)
    xs = np.where(x == 0.0, 1e-30, x)
    if kind == "regular":
        z = spherical_jn(l, xs)
        dz = spherical_jn(l, xs, derivative=True)
    elif kind == "outgoing":
        z = spherical_h1(l, xs)
        dz = (spherical_jn(l, xs, derivative=True)
              + 1j * spherical_yn(l, xs, derivative=True))
    else:
        raise ValueError(kind)
    return z, dz + z / xs


def angular_tables(lmax, theta, phi):
    """Y_lm, tau_lm = dY/dtheta, pi_lm = m Y / sin(theta) for 1<=l<=lmax, |m|<=l.

    theta is clipped away from the poles by _POLE_EPS (error O(eps^2)).
    Returns dict (l, m) -> (Y, tau, pi), each shaped like theta.
    """
    theta = np.clip(np.asarray(theta, dtype=float), _POLE_EPS, np.pi - _POLE_EPS)
    phi = np.asarray(phi, dtype=float)
    Y = {}
    for l in range(0, lmax + 1):
        for m in range(-l, l + 1):
            Y[(l, m)] = sph_harm_y(l, m, theta, phi)
    out = {}
    sin_t = np.sin(theta)
    eiph = np.exp(1j * phi)
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            yp = Y[(l, m + 1)] if m + 1 <= l else 0.0
            ym = Y[(l, m - 1)] if m - 1 >= -l else 0.0
            tau = 0.5 * (np.sqrt((l - m) * (l + m + 1)) * yp / eiph
                         - np.sqrt((l + m) * (l - m + 1)) * ym * eiph)
            pi_ = m * Y[(l, m)] / sin_t
            out[(l, m)] = (Y[(l, m)], tau, pi_)
    return out


def _sph_unit_vectors(theta, phi):
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    r_hat = np.stack([st * cp, st * sp, ct], axis=-1)
    t_hat = np.stack([ct * cp, ct * sp, -st], axis=-1)
    p_hat = np.stack([-sp, cp, np.zeros_like(theta)], axis=-1)
    return r_hat, t_hat, p_hat


def xz_vectors(modes, theta, phi):
    """X_lm and Z_lm as cartesian vectors, shape (nmodes_lm_unique, npts, 3).

    Returns dict (l, m) -> (X, Z), each (npts, 3) complex.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    tabs = angular_tables(modes.lmax, theta, phi)
    _, t_hat, p_hat = _sph_unit_vectors(
        np.clip(theta, _POLE_EPS, np.pi - _POLE_EPS), phi)
    out = {}
    for l in range(1, modes.lmax + 1):
        nrm = 1.0 / np.sqrt(l * (l + 1))
        for m in range(-l, l + 1):
            Y, tau, pi_ = tabs[(l, m)]
            X = nrm * (-pi_[:, None] * t_hat - 1j * tau[:, None] * p_hat)
            Z = nrm * (1j * tau[:, None] * t_hat - pi_[:, None] * p_hat)
            out[(l, m)] = (X, Z)
    return out


def vswf_fields(k, modes, points, kind, origin=None):
    """E and Ht (= i Z0 H) of each VSWF mode at cartesian points.

    points: (npts, 3). origin: source position (default 0).
    Returns E, Ht with shape (modes.n, npts, 3), complex.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if origin is not None:
        pts = pts - np.asarray(origin, dtype=float)
    r = np.linalg.norm(pts, axis=1)
    r_safe = np.where(r == 0.0, 1e-30, r)
    theta = np.arccos(np.clip(pts[:, 2] / r_safe, -1.0, 1.0))
    phi = np.arctan2(pts[:, 1], pts[:, 0])

    xz = xz_vectors(modes, theta, phi)
    tabs = angular_tables(modes.lmax, theta, phi)
    r_hat, _, _ = _sph_unit_vectors(theta, phi)
    x = k * r_safe

    zl, xil = {}, {}
    for l in range(1, modes.lmax + 1):
        zl[l], xil[l] = radial_funcs(kind, l, x)

    E = np.empty((modes.n, len(r), 3), dtype=complex)
    Ht = np.empty_like(E)
    for i in range(modes.n):
        l, m, p = modes.l[i], modes.m[i], modes.pol[i]
        X, Z = xz[(l, m)]
        Y = tabs[(l, m)][0]
        Mfld = zl[l][:, None] * X
        Nfld = (1j * np.sqrt(l * (l + 1)) * (zl[l] / x * Y)[:, None] * r_hat
                + xil[l][:, None] * Z)
        if p == MAGNETIC:
            E[i], Ht[i] = Mfld, Nfld
        else:
            E[i], Ht[i] = Nfld, Mfld
    return E, Ht


def plane_wave_coeffs(k_hat, e_hat, modes):
    """Regular VSWF coefficients of E = e_hat exp(i k.r); |k_hat| = 1."""
    k_hat = np.asarray(k_hat, dtype=float)
    k_hat = k_hat / np.linalg.norm(k_hat)
    theta = np.array([np.arccos(np.clip(k_hat[2], -1, 1))])
    phi = np.array([np.arctan2(k_hat[1], k_hat[0])])
    xz = xz_vectors(modes, theta, phi)
    e_hat = np.asarray(e_hat, dtype=complex)
    a = np.empty(modes.n, dtype=complex)
    for i in range(modes.n):
        l, m, p = modes.l[i], modes.m[i], modes.pol[i]
        X, Z = xz[(l, m)]
        if p == MAGNETIC:
            a[i] = 4 * np.pi * (1j ** l) * np.vdot(X[0], e_hat)
        else:
            a[i] = 4 * np.pi * (1j ** (l - 1)) * np.vdot(Z[0], e_hat)
    return a


def far_field_amplitude(k, f, modes, rhat_pts):
    """F(r_hat) with E_sca -> e^{ikr}/r F(r_hat). rhat_pts: (npts, 3) unit vecs.

    F = (1/k) sum_lm [ f^M (-i)^{l+1} X_lm + f^E (-i)^l Z_lm ].
    """
    pts = np.atleast_2d(np.asarray(rhat_pts, dtype=float))
    theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    xz = xz_vectors(modes, theta, phi)
    F = np.zeros((len(pts), 3), dtype=complex)
    for i in range(modes.n):
        l, m, p = modes.l[i], modes.m[i], modes.pol[i]
        X, Z = xz[(l, m)]
        if modes.pol[i] == MAGNETIC:
            F += f[i] * ((-1j) ** (l + 1)) * X
        else:
            F += f[i] * ((-1j) ** l) * Z
    return F / k


def sphere_quadrature(n_theta=24, n_phi=48):
    """Gauss-Legendre x uniform-phi product grid. Returns theta, phi, w (sum w = 4 pi)."""
    x, wx = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(x)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    TH = np.repeat(theta, n_phi)
    PH = np.tile(phi, n_theta)
    W = np.repeat(wx, n_phi) * (2 * np.pi / n_phi)
    return TH, PH, W


class RegularProjector:
    """Projects (E, Ht) sampled on the sphere |r| = r0 onto regular VSWFs.

    Precomputes the weighted conjugate X/Z projection matrices once so that
    each projection is a single GEMM.  Least-squares combination of the E and
    Ht channels:
      <X_lm, E>  = c^M j_l(kr0),   <Z_lm, E>  = c^E xi_l(kr0)
      <X_lm, Ht> = c^E j_l(kr0),   <Z_lm, Ht> = c^M xi_l(kr0)
    """

    def __init__(self, modes, quad):
        self.modes = modes
        self.quad = quad
        TH, PH, W = quad
        xz = xz_vectors(modes, TH, PH)
        npts = len(TH)
        XW = np.empty((modes.n, npts * 3), dtype=complex)
        ZW = np.empty_like(XW)
        for i in range(modes.n):
            X, Z = xz[(modes.l[i], modes.m[i])]
            XW[i] = (np.conj(X) * W[:, None]).reshape(-1)
            ZW[i] = (np.conj(Z) * W[:, None]).reshape(-1)
        self.XW, self.ZW = XW, ZW

    def project(self, k, r0, E, Ht):
        """E, Ht: (..., npts, 3) -> coefficients (..., modes.n)."""
        modes = self.modes
        lead = E.shape[:-2]
        Ef = E.reshape(-1, self.XW.shape[1])
        Hf = Ht.reshape(-1, self.XW.shape[1])
        pX = Ef @ self.XW.T      # (batch, n)
        pZ = Ef @ self.ZW.T
        qX = Hf @ self.XW.T
        qZ = Hf @ self.ZW.T
        x0 = k * r0
        jl = np.empty(modes.n)
        xl = np.empty(modes.n)
        for l in set(modes.l.tolist()):
            z, xi = radial_funcs("regular", l, np.array([float(x0)]))
            sel = modes.l == l
            jl[sel], xl[sel] = z[0], xi[0]
        den = jl ** 2 + xl ** 2
        cM = (jl * pX + xl * qZ) / den
        cE = (jl * qX + xl * pZ) / den
        c = np.where(modes.pol[None, :] == ELECTRIC, cE, cM)
        return c.reshape(lead + (modes.n,))


def project_regular(k, r0, E, Ht, modes, quad):
    """Convenience wrapper around RegularProjector (see class docstring)."""
    return RegularProjector(modes, quad).project(k, r0, E, Ht)
