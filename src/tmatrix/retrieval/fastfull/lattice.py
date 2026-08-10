"""2-D Bravais lattices and Floquet-order enumeration (proposal par. 5, 7.2).

The one-pitch specular campaign used a square lattice so small that only the
g = 0 order propagates.  The fast-full method deliberately does the opposite:
a rectangular or oblique cell large enough to open several diffraction orders,
at a generic Bloch vector away from every mirror line, so that one CST project
delivers many plane-wave illumination and observation directions.

Geometry conventions
--------------------
* Lengths in um, wavevectors in rad/um (the repository convention: k comes
  from tmat_io.TMatrixData.k_at).
* A lattice is given by two in-plane primitive vectors a1, a2 (2-vectors).
  The site set is {n1 a1 + n2 a2 : n integer}; the meta-atom sits at the
  origin, which must remain the VSWF origin (proposal par. 2).
* Reciprocal vectors b1, b2 satisfy a_i . b_j = 2 pi delta_ij.  With
  A = [[a1], [a2]] (primitive vectors as ROWS), B = 2 pi inv(A).T has b1, b2
  as its rows, and A @ B.T = 2 pi I exactly.
* The Bloch vector is stored in FRACTIONAL reciprocal coordinates
  k_B = f1 b1 + f2 b2, because f is what stays meaningful when the cell is
  rescaled during the design search, and f in [-1/2, 1/2)^2 is the first
  Brillouin zone.

Floquet orders
--------------
At fixed frequency (wavenumber k) and Bloch vector, the order g = (g1, g2)
has in-plane wavevector

    q_g = k_B + g1 b1 + g2 b2

and is PROPAGATING iff |q_g| < k, with out-of-plane wavenumber
kz_g = sqrt(k^2 - |q_g|^2) > 0.  Each propagating order supports four
physical port modes: two sides (Zmin, Zmax) x two polarizations (TE, TM).
Each port mode is both an incoming and an outgoing channel, exactly as in a
CST Floquet-port S-matrix, so M_in = M_out = 4 * n_orders.

Sign conventions for sides and directions (used everywhere downstream):

    side = +1  ->  the Zmax port (z -> +inf, above the sheet)
    side = -1  ->  the Zmin port (z -> -inf, below the sheet)
    incoming wave at a port travels INTO the domain: direction = -side
    outgoing wave at a port travels OUT of the domain: direction = +side

`direction` is the sign of k_hat_z and is exactly the `direction` argument of
sparams_oblique.pol_basis / khat_from_angles.  The campaign's illumination
(down-going, from above) is side = +1, direction = -1.

Rayleigh / Wood exclusion (proposal par. 7.2)
---------------------------------------------
An order at |q_g| = k is at a Rayleigh threshold: kz -> 0, the port reference
plane and power normalization become unstable, and the lattice sum loses
convergence.  `enumerate_orders` therefore reports, for EVERY order inside a
scan radius (propagating or not), the dimensionless distance to threshold

    d_g = | |q_g| / k - 1 |,

and `wood_margin` rejects a design in which any order has d_g below the
declared margin.  This is a hard design constraint, not a warning: an order
just above cutoff is as damaging as one just below.
"""
import numpy as np

TWO_PI = 2.0 * np.pi

ZMAX = +1        # side label: port at z -> +infinity
ZMIN = -1        # side label: port at z -> -infinity

TE = 0           # normative Jones index order (retrieval/HANDOFF.md)
TM = 1

_TINY = 1e-14


# ------------------------------------------------------------------- lattice

class Lattice2D:
    """An in-plane Bravais lattice, its reciprocal, and Bloch bookkeeping.

    Parameters
    ----------
    a1, a2 : array-like, length 2
        Primitive vectors in um.
    name : str, optional
        Free-form label carried into reports.

    Attributes
    ----------
    A : (2, 2) primitive vectors as rows.
    B : (2, 2) reciprocal vectors as rows; A @ B.T = 2 pi I.
    area : |a1 x a2| in um^2.
    """

    def __init__(self, a1, a2, name=None):
        a1 = np.asarray(a1, dtype=float).ravel()
        a2 = np.asarray(a2, dtype=float).ravel()
        if a1.size != 2 or a2.size != 2:
            raise ValueError("primitive vectors must be in-plane 2-vectors")
        A = np.stack([a1, a2])
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        scale = max(np.linalg.norm(a1), np.linalg.norm(a2))
        if abs(det) <= _TINY * scale ** 2:
            raise ValueError("degenerate (collinear) primitive vectors")
        self.A = A
        self.a1, self.a2 = A[0], A[1]
        self.area = abs(det)
        self.B = TWO_PI * np.linalg.inv(A).T
        self.b1, self.b2 = self.B[0], self.B[1]
        self.name = name

    # ------------------------------------------------------------ builders
    @classmethod
    def square(cls, p, alpha_deg=0.0, name=None):
        """Square lattice of pitch p, rotated by alpha_deg about z."""
        return cls.oblique(p, p, 90.0, alpha_deg,
                           name=name or "square p=%g" % p)

    @classmethod
    def rectangular(cls, px, py, alpha_deg=0.0, name=None):
        """Rectangular lattice, a1 along alpha_deg and a2 perpendicular."""
        return cls.oblique(px, py, 90.0, alpha_deg,
                           name=name or "rect %gx%g @%g deg"
                           % (px, py, alpha_deg))

    @classmethod
    def oblique(cls, p1, p2, gamma_deg, alpha_deg=0.0, name=None):
        """|a1| = p1 at azimuth alpha_deg, |a2| = p2 at alpha_deg + gamma_deg.

        gamma_deg = 90 gives the rectangular family; alpha_deg is the lattice
        orientation relative to the wheel's spoke axes (the proposal's
        alpha_lat design variable).
        """
        al = np.deg2rad(alpha_deg)
        ga = np.deg2rad(gamma_deg)
        a1 = p1 * np.array([np.cos(al), np.sin(al)])
        a2 = p2 * np.array([np.cos(al + ga), np.sin(al + ga)])
        return cls(a1, a2, name=name or "oblique %gx%g gamma=%g alpha=%g"
                   % (p1, p2, gamma_deg, alpha_deg))

    # -------------------------------------------------------------- Bloch
    def bloch(self, f1, f2):
        """k_B = f1 b1 + f2 b2 (fractional reciprocal coordinates)."""
        return float(f1) * self.b1 + float(f2) * self.b2

    def fractional(self, k_bloch):
        """Inverse of `bloch`: fractional coordinates of an in-plane vector.

        Uses f = k_B @ inv(B) = k_B @ A.T / (2 pi), exact to roundoff.
        """
        k_bloch = np.asarray(k_bloch, dtype=float).ravel()[:2]
        return self.A @ k_bloch / TWO_PI

    @staticmethod
    def wrap_fractional(f):
        """Fold fractional coordinates into the first BZ [-1/2, 1/2)."""
        f = np.asarray(f, dtype=float)
        return (f + 0.5) % 1.0 - 0.5

    # ------------------------------------------------------------ mirrors
    def mirror_azimuths_deg(self, tol=1e-9):
        """Azimuths (deg, mod 180) of the lattice's own mirror lines.

        A generic oblique lattice has mirror lines only along a1 +/- a2
        directions when |a1| = |a2|; a rectangular lattice has them along a1
        and a2.  The proposal requires the Bloch vector to avoid every mirror
        line (par. 1 item 3), so this is reported by the design search as a
        diagnostic rather than enforced here.
        """
        cand = []
        n1, n2 = np.linalg.norm(self.a1), np.linalg.norm(self.a2)
        if abs(self.a1 @ self.a2) <= tol * n1 * n2:          # rectangular
            cand += [self.a1, self.a2]
        if abs(n1 - n2) <= tol * max(n1, n2):                # rhombic
            cand += [self.a1 + self.a2, self.a1 - self.a2]
        out = []
        for v in cand:
            ang = np.rad2deg(np.arctan2(v[1], v[0])) % 180.0
            if not any(abs(ang - a) < 1e-7 or abs(abs(ang - a) - 180) < 1e-7
                       for a in out):
                out.append(float(ang))
        return sorted(out)

    # -------------------------------------------------------------- shells
    def shells(self, r_max, rtol=1e-9):
        """Group lattice sites 0 < |R| <= r_max by radius.

        Returns (radii, angles) in exactly the format of
        translate.square_lattice_shells -- radii ascending, angles a list of
        per-shell azimuth arrays -- so that the validated Bloch assembly in
        bloch_lattice.assemble_shell_sum_bloch applies unchanged.  That
        assembly is lattice-agnostic: it only ever consumes radii, per-site
        azimuths and the rotation identity A(Rot_phi d) = A(d) e^{i dm phi},
        which holds for any in-plane displacement.

        For a square lattice this reproduces square_lattice_shells exactly
        (gated in tests/retrieval/test_fastfull_core.py).  For a generic
        oblique lattice most shells hold only the exact +/-R pair, so the
        number of distinct radii
        -- and hence the number of translation-operator projections -- grows
        like half the site count.  That is the cost driver that makes the
        tapered real-space sum impractical for a large diffractive cell and
        is why the proposal calls for Ewald at M2.
        """
        n1 = int(np.ceil(r_max / self._line_spacing(0))) + 1
        n2 = int(np.ceil(r_max / self._line_spacing(1))) + 1
        I1, I2 = np.meshgrid(np.arange(-n1, n1 + 1), np.arange(-n2, n2 + 1),
                             indexing="ij")
        idx = np.stack([I1.ravel(), I2.ravel()], axis=1)
        R = idx @ self.A
        r = np.linalg.norm(R, axis=1)
        keep = (r > _TINY * max(1.0, r_max)) & (r <= r_max)
        R, r = R[keep], r[keep]
        order = np.argsort(r, kind="stable")
        R, r = R[order], r[order]
        # group by radius within a relative tolerance (exact ties -- the
        # +/-R pairs and any accidental degeneracy -- land in one shell)
        cut = np.nonzero(np.diff(r) > rtol * np.maximum(r[1:], 1.0))[0] + 1
        bounds = [0] + cut.tolist() + [len(r)]
        radii, angles = [], []
        for s in range(len(bounds) - 1):
            sl = slice(bounds[s], bounds[s + 1])
            radii.append(float(r[sl].mean()))
            angles.append(np.arctan2(R[sl, 1], R[sl, 0]))
        return np.array(radii), angles

    def _line_spacing(self, i):
        """Perpendicular distance between adjacent lattice lines n_i = const.

        Equals area / |a_j| (j the other index); bounding the site search by
        r_max / spacing is exact, unlike bounding by |a_i| on a skewed cell.
        """
        other = self.a2 if i == 0 else self.a1
        return self.area / np.linalg.norm(other)

    def __repr__(self):
        return ("Lattice2D(a1=[%.4f, %.4f], a2=[%.4f, %.4f], area=%.4f um^2%s)"
                % (self.a1[0], self.a1[1], self.a2[0], self.a2[1], self.area,
                   ", name=%r" % self.name if self.name else ""))


# -------------------------------------------------------------- order sets

class OrderSet:
    """Floquet orders of one (lattice, k, k_B), with retention bookkeeping.

    Attributes (all aligned, length n_scan = number of scanned orders)
    ------------------------------------------------------------------
    g        : (n_scan, 2) int      reciprocal indices (g1, g2)
    q        : (n_scan, 2) float    in-plane wavevector q_g, rad/um
    qabs     : (n_scan,)   float    |q_g|
    prop     : (n_scan,)   bool     |q_g| < k  (propagating)
    kz       : (n_scan,)   float    sqrt(k^2 - |q|^2) where propagating,
                                    NaN otherwise (never a signed value:
                                    the sign lives in `direction`)
    kz_frac  : (n_scan,)   float    kz / k in (0, 1]; NaN where evanescent
    theta    : (n_scan,)   float    arcsin(|q|/k), rad; NaN where evanescent
    phi      : (n_scan,)   float    atan2(q_y, q_x), rad
    wood     : (n_scan,)   float    | |q|/k - 1 |, distance to threshold
    retained : (n_scan,)   bool     propagating AND kz_frac >= kz_min_frac
                                    AND inside an explicit `keep` selection

    `idx_retained` gives the indices of retained orders in scan order.
    """

    def __init__(self, lattice, k, k_bloch, g, q, prop, kz, theta, phi, wood,
                 retained, kz_min_frac, wood_margin, q_scan):
        self.lattice = lattice
        self.k = float(k)
        self.k_bloch = np.asarray(k_bloch, dtype=float).ravel()[:2]
        self.g = g
        self.q = q
        self.qabs = np.linalg.norm(q, axis=1)
        self.prop = prop
        self.kz = kz
        self.kz_frac = kz / self.k
        self.theta = theta
        self.phi = phi
        self.wood = wood
        self.retained = retained
        self.kz_min_frac = float(kz_min_frac)
        self.wood_margin = float(wood_margin)
        self.q_scan = float(q_scan)

    # ------------------------------------------------------------ inventory
    @property
    def n_scan(self):
        return len(self.g)

    @property
    def n_prop(self):
        return int(self.prop.sum())

    @property
    def n_retained(self):
        return int(self.retained.sum())

    @property
    def idx_retained(self):
        return np.flatnonzero(self.retained)

    @property
    def n_channels(self):
        """4 per retained order: 2 sides x 2 polarizations."""
        return 4 * self.n_retained

    # ---------------------------------------------------------- diagnostics
    def wood_margin_actual(self):
        """Smallest | |q_g|/k - 1 | over all SCANNED orders.

        Includes evanescent orders: an order just ABOVE cutoff is as
        dangerous as one just below (it opens under a small frequency or
        angle change and destabilizes the port model).
        """
        return float(self.wood.min()) if self.n_scan else np.inf

    def grazing_margin_actual(self):
        """Smallest kz/k over RETAINED orders (1.0 if none retained)."""
        idx = self.idx_retained
        return float(self.kz_frac[idx].min()) if len(idx) else 1.0

    def passes_constraints(self):
        """(ok, reasons) for the proposal par. 7.2 hard constraints."""
        reasons = []
        if self.wood_margin_actual() < self.wood_margin:
            reasons.append("wood margin %.4f < %.4f"
                           % (self.wood_margin_actual(), self.wood_margin))
        if self.grazing_margin_actual() < self.kz_min_frac:
            reasons.append("grazing margin %.4f < %.4f"
                           % (self.grazing_margin_actual(), self.kz_min_frac))
        if self.n_retained == 0:
            reasons.append("no retained orders")
        return (len(reasons) == 0), reasons

    def table(self):
        """Human-readable per-order table (list of dicts, scan order)."""
        rows = []
        for i in range(self.n_scan):
            rows.append(dict(
                g1=int(self.g[i, 0]), g2=int(self.g[i, 1]),
                qx=float(self.q[i, 0]), qy=float(self.q[i, 1]),
                q_over_k=float(self.qabs[i] / self.k),
                kz=float(self.kz[i]), kz_frac=float(self.kz_frac[i]),
                theta_deg=float(np.rad2deg(self.theta[i])),
                phi_deg=float(np.rad2deg(self.phi[i])),
                wood=float(self.wood[i]),
                propagating=bool(self.prop[i]),
                retained=bool(self.retained[i])))
        return rows

    def format_table(self, only_scanned=True):
        lines = ["  g1  g2   |q|/k   kz/k  theta   phi    wood  prop keep",
                 "  " + "-" * 56]
        for r in self.table():
            if only_scanned and not (r["propagating"] or r["wood"] < 0.5):
                continue
            lines.append("%4d%4d  %6.3f %6.3f %6.1f %6.1f  %6.3f   %s   %s"
                         % (r["g1"], r["g2"], r["q_over_k"], r["kz_frac"],
                            r["theta_deg"], r["phi_deg"], r["wood"],
                            "y" if r["propagating"] else "n",
                            "y" if r["retained"] else "n"))
        return "\n".join(lines)

    def __repr__(self):
        return ("OrderSet(k=%.4f, n_prop=%d, n_retained=%d, n_channels=%d, "
                "wood=%.3f, grazing=%.3f)"
                % (self.k, self.n_prop, self.n_retained, self.n_channels,
                   self.wood_margin_actual(), self.grazing_margin_actual()))


def enumerate_orders(lattice, k, k_bloch=None, f_bloch=None,
                     kz_min_frac=0.2, wood_margin=0.05, q_scan=1.6,
                     keep=None):
    """Enumerate Floquet orders of `lattice` at wavenumber k and Bloch vector.

    Parameters
    ----------
    lattice : Lattice2D
    k : float
        Embedding wavenumber, rad/um.
    k_bloch : array-like, length 2, optional
        In-plane Bloch vector (rad/um).  Exactly one of k_bloch / f_bloch.
    f_bloch : array-like, length 2, optional
        Bloch vector in fractional reciprocal coordinates (preferred: it is
        scale-invariant under cell rescaling).
    kz_min_frac : float
        Grazing exclusion; retained orders need kz/k >= kz_min_frac.  The
        proposal's initial value is 0.2 (par. 7.2), to be tightened by the
        SNR study.
    wood_margin : float
        Required distance | |q_g|/k - 1 | for EVERY scanned order.  Reported
        by OrderSet.passes_constraints; not silently enforced here so that a
        failing design can still be inspected.
    q_scan : float
        Scan radius in units of k.  Orders with |q_g| <= q_scan * k are
        enumerated so that evanescent orders near cutoff enter the Wood
        diagnostics.  1.6 covers everything that could open under a 60%
        frequency increase.
    keep : callable or sequence of (g1, g2), optional
        Explicit retention selection (the proposal's design variable
        `mathcal G`).  A callable receives the per-order dict of
        OrderSet.table() and returns a bool.  Default: keep every
        propagating order that clears the grazing cut.

    Returns
    -------
    OrderSet
    """
    if (k_bloch is None) == (f_bloch is None):
        raise ValueError("give exactly one of k_bloch / f_bloch")
    if f_bloch is not None:
        f_bloch = np.asarray(f_bloch, dtype=float).ravel()
        if f_bloch.size != 2:
            raise ValueError("f_bloch must have 2 components")
        k_bloch = lattice.bloch(f_bloch[0], f_bloch[1])
    k_bloch = np.asarray(k_bloch, dtype=float).ravel()[:2]
    k = float(k)
    if k <= 0:
        raise ValueError("k must be positive")

    # Integer search box.  q - k_B = g @ B, so |g| <= |q - k_B| / sigma_min(B)
    # and |q - k_B| <= q_scan k + |k_B|; +1 guards the ceil.
    smin = np.linalg.svd(lattice.B, compute_uv=False).min()
    n_max = int(np.ceil((q_scan * k + np.linalg.norm(k_bloch)) / smin)) + 1
    gg = np.arange(-n_max, n_max + 1)
    G1, G2 = np.meshgrid(gg, gg, indexing="ij")
    g = np.stack([G1.ravel(), G2.ravel()], axis=1).astype(int)
    q = k_bloch[None, :] + g @ lattice.B
    qabs = np.linalg.norm(q, axis=1)

    inside = qabs <= q_scan * k
    g, q, qabs = g[inside], q[inside], qabs[inside]
    order = np.lexsort((g[:, 1], g[:, 0], np.round(qabs, 12)))
    g, q, qabs = g[order], q[order], qabs[order]

    prop = qabs < k
    kz = np.full(len(g), np.nan)
    theta = np.full(len(g), np.nan)
    kz[prop] = np.sqrt(np.maximum(k ** 2 - qabs[prop] ** 2, 0.0))
    theta[prop] = np.arcsin(np.clip(qabs[prop] / k, 0.0, 1.0))
    phi = np.arctan2(q[:, 1], q[:, 0])
    wood = np.abs(qabs / k - 1.0)

    retained = prop.copy()
    with np.errstate(invalid="ignore"):
        retained &= (kz / k) >= kz_min_frac

    os_ = OrderSet(lattice, k, k_bloch, g, q, prop, kz, theta, phi, wood,
                   retained, kz_min_frac, wood_margin, q_scan)
    if keep is not None:
        if callable(keep):
            sel = np.array([bool(keep(r)) for r in os_.table()])
        else:
            want = {(int(a), int(b)) for a, b in keep}
            sel = np.array([(int(g[i, 0]), int(g[i, 1])) in want
                            for i in range(len(g))])
        os_.retained = os_.retained & sel
    return os_


# ------------------------------------------------------------ channel table

class ChannelSet:
    """Port modes of an OrderSet: one entry per (side, order, polarization).

    A channel is BOTH an incoming and an outgoing S-matrix index, exactly as
    a CST Floquet port mode is.  The propagation direction differs between
    the two uses and is derived, never stored ambiguously:

        direction_in  = -side      (into the domain)
        direction_out = +side      (out of the domain)

    Ordering is deterministic and is the normative channel order for every A,
    W and S produced by this package: side descending (Zmax first, matching
    the campaign's illumination side), then retained-order index ascending
    (which is |q| ascending, ties broken by g), then polarization TE, TM.
    """

    def __init__(self, orders):
        idx = orders.idx_retained
        n = len(idx)
        self.orders = orders
        self.order_index = np.repeat(idx, 4)
        self.side = np.tile(np.repeat([ZMAX, ZMIN], 2), n)
        self.pol = np.tile([TE, TM], 2 * n)
        self.g = orders.g[self.order_index]
        self.q = orders.q[self.order_index]
        self.kz = orders.kz[self.order_index]
        self.theta = orders.theta[self.order_index]
        self.phi = orders.phi[self.order_index]
        self.k = orders.k
        self.n = 4 * n

    @property
    def direction_in(self):
        return -self.side

    @property
    def direction_out(self):
        return +self.side

    def labels(self):
        """('Zmax', g1, g2, 'TE') style tuples, one per channel."""
        return [("Zmax" if s == ZMAX else "Zmin", int(g1), int(g2),
                 "TE" if p == TE else "TM")
                for s, (g1, g2), p in zip(self.side, self.g, self.pol)]

    def format_table(self):
        lines = ["  ch  side  g1  g2  pol   theta    phi    kz/k",
                 "  " + "-" * 44]
        for i, (sd, g1, g2, pl) in enumerate(self.labels()):
            lines.append("%4d  %4s%4d%4d  %3s  %6.1f %6.1f  %6.3f"
                         % (i, sd, g1, g2, pl,
                            np.rad2deg(self.theta[i]),
                            np.rad2deg(self.phi[i]),
                            self.kz[i] / self.k))
        return "\n".join(lines)

    def __repr__(self):
        return "ChannelSet(n=%d, orders=%d)" % (self.n, self.n // 4)
