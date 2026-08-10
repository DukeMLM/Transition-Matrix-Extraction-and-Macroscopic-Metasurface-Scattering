"""Translation-addition operators and periodic lattice sums.

Translation operator A(d): re-expands an OUTGOING VSWF centered at the source
as REGULAR VSWFs about a target origin, with d = r_source - r_target:

    V^(3)_nu(rho - d) = sum_mu A(d)[mu, nu] V^(1)_mu(rho),   |rho| < |d|

Computed by numerical projection: the outgoing fields (E, Ht) are sampled on a
sphere |rho| = r0 < |d| around the target and projected onto the regular basis
(vswf.RegularProjector).  This is convention-locked to vswf.py by construction.

Azimuthal rotation identity (verified in test_translate.py): for a rotation
of d by phi about z,
    A(Rot_phi d)[mu, nu] = e^{i (m_nu - m_mu) phi} A(d)[mu, nu]
so in-plane lattice points can be grouped into shells of equal radius: one
projection per distinct radius, and per-shell azimuthal phase sums
    P_s(Dm) = sum_{R in shell s} w(|R|) e^{i Dm phi_R}.

Periodic lattice sum (in-plane square lattice, Bloch phase k_par = 0):
    C = sum_{R != 0} A(R) w(|R|),   w(R) = exp(-R^2 / Rc^2)
The Gaussian taper regularizes the conditionally convergent free-space sum.
Evanescent (g != 0) channels converge exponentially for a deep-subwavelength
lattice, but the propagating g = 0 channel converges only algebraically: the
taper smooths the k-space spectrum with a kernel of width 2/Rc at distance k
from the light-circle pole, giving corrections that form an even power series
in 1/(k Rc) (Dawson-function asymptotics of the tapered radial integral).
lattice_sum_C therefore Richardson-extrapolates C(Rc) in 1/Rc^2 over several
taper lengths; the shell matrices are shared between tapers, so the
extrapolation is nearly free.
"""
import numpy as np

from tmatrix.aggregation.vswf import vswf_fields, RegularProjector, sphere_quadrature


def sphere_points(r0, quad):
    TH, PH, _ = quad
    return r0 * np.stack(
        [np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=1)


def make_quad(n_theta=16, n_phi=32):
    return sphere_quadrature(n_theta, n_phi)


def translation_matrix(k, d, modes, r0, quad, projector=None):
    """A(d)[mu, nu]: outgoing mode nu at source (offset d from target) ->
    regular modes mu at target.  Requires r0 < |d|."""
    d = np.asarray(d, dtype=float)
    if r0 >= np.linalg.norm(d):
        raise ValueError("projection sphere must not enclose the source")
    pts = sphere_points(r0, quad)
    E, Ht = vswf_fields(k, modes, pts, "outgoing", origin=d)
    proj = projector if projector is not None else RegularProjector(modes, quad)
    C = proj.project(k, r0, E, Ht)      # (n_nu, n_mu)
    return C.T                          # (n_mu, n_nu)


def translation_shells_offset(k, radii, z_off, modes, r0, quad,
                              projector=None, chunk=48):
    """A((r_s, 0, z_off)) for a batch of in-plane radii with a common
    z-offset.  Returns (nshell, n, n).  The azimuthal rotation identity
    applies unchanged (rotation about z leaves z_off fixed)."""
    radii = np.asarray(radii, dtype=float)
    dmin = np.sqrt(radii.min() ** 2 + z_off ** 2) if len(radii) else np.inf
    if r0 >= dmin:
        raise ValueError(
            f"projection sphere r0={r0} must be smaller than the closest "
            f"source distance {dmin}")
    proj = projector if projector is not None else RegularProjector(modes, quad)
    pts = sphere_points(r0, quad)
    npts = len(pts)
    out = np.empty((len(radii), modes.n, modes.n), dtype=complex)
    for i0 in range(0, len(radii), chunk):
        rs = radii[i0:i0 + chunk]
        # observation points relative to each source at (r_s, 0, z_off)
        src = np.stack([rs, np.zeros_like(rs),
                        np.full_like(rs, z_off)], axis=1)
        rel = pts[None, :, :] - src[:, None, :]
        E, Ht = vswf_fields(k, modes, rel.reshape(-1, 3), "outgoing")
        E = E.reshape(modes.n, len(rs), npts, 3).transpose(1, 0, 2, 3)
        Ht = Ht.reshape(modes.n, len(rs), npts, 3).transpose(1, 0, 2, 3)
        c = proj.project(k, r0, E, Ht)          # (nshell_chunk, n_nu, n_mu)
        out[i0:i0 + chunk] = c.transpose(0, 2, 1)
    return out


def translation_shells(k, radii, modes, r0, quad, projector=None, chunk=48):
    """A(r_s * x_hat) for a batch of in-plane radii. Returns (nshell, n, n)."""
    return translation_shells_offset(k, radii, 0.0, modes, r0, quad,
                                     projector=projector, chunk=chunk)


def rotate_inplane(A, phi, modes):
    """A(Rot_phi d) from A(d) for an in-plane rotation about z."""
    dm = modes.m[None, :] - modes.m[:, None]      # m_nu - m_mu
    return A * np.exp(1j * dm * phi)


def square_lattice_shells(pitch, r_max):
    """Group square-lattice vectors (i, j) * pitch, 0 < |R| <= r_max, by radius.

    Returns radii (nshell,) and a dict shell -> array of angles phi_R.
    """
    n = int(np.ceil(r_max / pitch))
    ii, jj = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    ii, jj = ii.ravel(), jj.ravel()
    r2 = ii * ii + jj * jj
    keep = (r2 > 0) & (r2 * pitch ** 2 <= r_max ** 2)
    ii, jj, r2 = ii[keep], jj[keep], r2[keep]
    order = np.argsort(r2, kind="stable")
    ii, jj, r2 = ii[order], jj[order], r2[order]
    uniq, start = np.unique(r2, return_index=True)
    radii = np.sqrt(uniq.astype(float)) * pitch
    angles = []
    bounds = list(start) + [len(r2)]
    for s in range(len(uniq)):
        sl = slice(bounds[s], bounds[s + 1])
        angles.append(np.arctan2(jj[sl], ii[sl]))
    return radii, angles


def assemble_shell_sum(A_s, radii, angles, modes, Rcs):
    """Combine shell matrices with tapered azimuthal phase sums and
    Richardson-extrapolate over the taper lengths Rcs (in length units)."""
    dm = modes.m[None, :] - modes.m[:, None]      # (mu, nu): m_nu - m_mu
    dms = np.unique(dm)
    dm_idx = np.searchsorted(dms, dm)
    Cs = []
    for Rc in Rcs:
        # per-shell azimuthal phase sums, tapered
        P = np.zeros((len(radii), len(dms)), dtype=complex)
        for s, (r, ph) in enumerate(zip(radii, angles)):
            w = np.exp(-(r / Rc) ** 2)
            if w < 1e-16:
                continue
            P[s] = w * np.exp(1j * np.outer(ph, dms)).sum(axis=0)
        Cs.append(np.einsum("smn,smn->mn", A_s, P[:, dm_idx]))
    x = 1.0 / np.asarray(Rcs, dtype=float) ** 2
    C = np.zeros_like(Cs[0])
    for i in range(len(Rcs)):
        L = 1.0
        for j in range(len(Rcs)):
            if j != i:
                L *= x[j] / (x[j] - x[i])
        C += L * Cs[i]
    return C


def lattice_sum_C(k, pitch, modes, r0, quad, kRc=(10.0, 14.0, 20.0),
                  r_max_factor=3.5, projector=None):
    """Richardson-extrapolated lattice sum C = sum_{R != 0} A(R) for a square
    lattice at normal incidence (Bloch phase 1).

    kRc: dimensionless taper lengths k * Rc used for the extrapolation.
    """
    kRc = np.asarray(kRc, dtype=float)
    Rcs = kRc / k
    r_max = r_max_factor * Rcs.max()
    radii, angles = square_lattice_shells(pitch, r_max)
    A_s = translation_shells(k, radii, modes, r0, quad, projector=projector)
    return assemble_shell_sum(A_s, radii, angles, modes, Rcs)
