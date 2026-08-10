"""Rank-optimized coding-cell design (proposal par. 7, milestone M1).

"The cell should be chosen by an explicit numerical design problem, not by
visual intuition" (par. 7).  This module states that problem and solves it.

Design variables (par. 7.1)
---------------------------
    x = (p1, p2, gamma_lat, alpha_lat, f1, f2)

with p1, p2 the primitive lengths in um, gamma_lat the angle between them
(90 deg = the rectangular family), alpha_lat the lattice orientation relative
to the wheel's spoke axes, and (f1, f2) the Bloch vector in FRACTIONAL
reciprocal coordinates.  The retained reciprocal-vector set `mathcal G` is
handled by the grazing cut plus an optional explicit selection.

Objectives
----------
Both tracks are reduced to one dimensionally consistent SNR-like scalar, so
that cost and signal enter on the same footing.

  WHEEL track.  sigma_40 of the noise-whitened Jacobian
  Sigma_S^{-1/2} H D_c (jacobian.py) DIVIDED BY sqrt(n_obs).  With D_c = I
  and a flat sigma, sqrt(n_obs) / sigma_40 is the exact worst-case
  ||dT||_F over all deterministic discrepancy vectors of per-entry RMS
  sigma, so its reciprocal is the figure of merit.

  GENERIC track.  vec(W T A) = (W kron A^T) vec(T) in row-major order, so the
  singular values of the end-to-end operator are the products
  sigma_i(W) sigma_j(A) and the weakest T_eff direction is bounded by

      sigma_30(W) sigma_30(A) / (sigma_noise sqrt(n_obs)).

  This sharpens par. 7.3's minimax form into a worst-case SNR while keeping
  its ingredients: kappa(A) and kappa(W) are reported alongside, because the
  proposal's Gate A states its preference in those terms.

  MARGINALIZED track (the one to prefer once C is available).  sigma_40 of
  the T-Jacobian after projecting out every declared calibration tangent --
  the Schur complement of the joint Fisher information (nuisance.py).  Gate A
  showed the coded-cell inverse is not rank deficient but is nearly
  degenerate against a port-plane offset (99.98 % of that tangent lies inside
  col(H)), so this is the quantity that actually limits the method.

  It is NOT divided by sqrt(n_obs), and that is deliberate: the sqrt(n_obs)
  penalty on the other tracks exists because the discrepancy is systematic
  and UNMODELLED.  Once the systematic classes are represented explicitly and
  projected out, what remains is much closer to an iid residual, for which
  averaging is legitimate.  The two normalizations therefore answer different
  questions and are reported side by side, never mixed.

  WHY sqrt(n_obs) IS IN THE DENOMINATOR.  The whitening level is the
  campaign's measured closure residual, which `results/REAL_RETRIEVAL.md`
  par. 4.3 shows is dominated by systematic model error rather than iid
  noise.  An iid objective (plain sigma_40) silently rewards a design for
  having more modal entries, because iid errors average down; a systematic
  discrepancy does not average down, so extra channels only help when they
  raise sigma_40 faster than sqrt(n_obs).  Dividing makes that trade
  explicit, and it materially changes the ranking of pooled encodings.
  `objective_*_iid` keeps the undivided value as the other bracket end.

Because the D4h basis is Frobenius-orthonormal, the wheel Jacobian is the
generic operator restricted to a 40-dimensional subspace, hence

      sigma_40(H)  >=  sigma_30(W) sigma_30(A)                        (D1)

always.  (D1) is gated in test_fastfull_design.py and is the quantitative
form of the proposal's statement that the generic gate is the stronger one.

Penalties (par. 7.2, 7.3)
-------------------------
Hard rejections: Wood/Rayleigh margin, grazing |kz|/k, order count bounds,
cell area bound, and non-overlap of periodic copies.  Soft multiplicative
penalty: the CST cost proxy (cost.py).  Nothing is penalized twice -- the
flux normalization already makes a large cell lose signal, which is the
physically correct penalty; the cost term is about wall time only.

Target independence
-------------------
The search never sees the reference wheel T.  The wheel-track objective at
C = 0 does not depend on T at all; when a converged C is supplied it is
evaluated over a passive D4h ENSEMBLE (symmetry.random_passive_d4h).  The
reference T is used only by `benchmark_reference`, which is reporting, not
design.

Milestone scope
---------------
This is M1: no CST, and by default C = 0 (see jacobian.py's scope note).  The
verdict a run produces is a SCREENING verdict.  Closing Gate A needs M2's
Ewald C, the measured multimode noise covariance, and the blind synthetic
recovery loop.

CLI
---
    python -m retrieval.fastfull.design --lam 8              # design at 8 um
    python -m retrieval.fastfull.design --lam 8,20 --track both
    python -m retrieval.fastfull.design --evaluate 26,33.8,90,0,0.09,-0.46
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

from tmatrix.aggregation.vswf import ModeBasis

from . import lattice as lt
from . import transforms as xf
from . import jacobian as jac
from . import cost as ct
from . import symmetry as sym

from tmatrix.paths import BENCHMARK_SINGLE, FASTFULL_RESULTS, RETRIEVAL_RESULTS

RESULTS = str(FASTFULL_RESULTS)

# Wheel geometry (proposal par. 2); only the circumscribing radius is needed
# here, to forbid overlapping periodic copies.
R_CIRC_UM = 0.7193

SIGMA_CLOSURE_NPZ = os.path.join(RETRIEVAL_RESULTS,
                                 "fit_sigma_from_closure.npz")
SIGMA_FALLBACK = 2.8172e-3     # band RMS of the same file, used only if absent
_SIGMA_CACHE = {}
_MISSING = object()


def _sha_npz(path):
    """Content hash of the closure NPZ, or None when it is absent."""
    import hashlib
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 16), b""):
            h.update(blk)
    return h.hexdigest()


def measured_sigma(lam_um):
    """Frequency-MATCHED complex-S discrepancy from the campaign closure.

    `results/fit_sigma_from_closure.npz` holds the measured normal-incidence
    closure residual per frequency (2.3063e-3 ... 3.6171e-3 across the band;
    2.8417e-3 at 8 um, 3.1751e-3 at 20 um).  Using the band RMS everywhere
    understates the noise at the ends of the band, so the design study
    interpolates the spectrum at the wavelength actually being designed for.

    This is a MODEL-vs-CST discrepancy, not an iid noise floor -- see
    jacobian.recovery_errors for why that distinction changes the verdict.
    """
    # Keyed on the FILE'S CONTENT HASH, not merely on first use.  A cache
    # keyed on presence serves stale values after the NPZ is replaced, while
    # a provenance snapshot taken later hashes the NEW bytes -- so a run
    # would certify an input it did not use.  Rehashing a 3 kB file per call
    # is far cheaper than that failure mode.
    # `_MISSING` rather than None: a missing NPZ hashes to None too, so a
    # None default would make the cold cache compare equal to the
    # absent-file state and fall through to a KeyError on `lam`.
    key = _sha_npz(SIGMA_CLOSURE_NPZ)
    if _SIGMA_CACHE.get("key", _MISSING) != key:
        _SIGMA_CACHE.clear()
        _SIGMA_CACHE["key"] = key
        if not os.path.exists(SIGMA_CLOSURE_NPZ):
            _SIGMA_CACHE["lam"] = None
        else:
            with np.load(SIGMA_CLOSURE_NPZ, allow_pickle=True) as z:
                f_thz = np.asarray(z["f_THz"], dtype=float)
                sig = np.asarray(z["sigma"], dtype=float)
            lam = 299792.458 / f_thz / 1000.0
            o = np.argsort(lam)
            _SIGMA_CACHE["lam"] = lam[o]
            _SIGMA_CACHE["sigma"] = sig[o]
    if _SIGMA_CACHE["lam"] is None:
        return float(SIGMA_FALLBACK)
    return float(np.interp(float(lam_um), _SIGMA_CACHE["lam"],
                           _SIGMA_CACHE["sigma"]))


class Design:
    """One candidate coding cell."""

    __slots__ = ("p1", "p2", "gamma_deg", "alpha_deg", "f1", "f2")

    def __init__(self, p1, p2, gamma_deg=90.0, alpha_deg=0.0, f1=0.0, f2=0.0):
        self.p1 = float(p1)
        self.p2 = float(p2)
        self.gamma_deg = float(gamma_deg)
        self.alpha_deg = float(alpha_deg)
        self.f1 = float(f1)
        self.f2 = float(f2)

    @property
    def vector(self):
        return np.array([self.p1, self.p2, self.gamma_deg, self.alpha_deg,
                         self.f1, self.f2])

    @classmethod
    def from_vector(cls, v):
        return cls(*[float(x) for x in v])

    @classmethod
    def from_dict(cls, d):
        """Inverse of `to_dict` (whose keys carry units, so **d fails)."""
        return cls(d["p1_um"], d["p2_um"], d["gamma_deg"], d["alpha_deg"],
                   d["f1"], d["f2"])

    def lattice(self):
        return lt.Lattice2D.oblique(self.p1, self.p2, self.gamma_deg,
                                    self.alpha_deg)

    def to_dict(self):
        return dict(p1_um=self.p1, p2_um=self.p2, gamma_deg=self.gamma_deg,
                    alpha_deg=self.alpha_deg, f1=self.f1, f2=self.f2)

    def __repr__(self):
        return ("Design(p1=%.4f, p2=%.4f, gamma=%.2f, alpha=%.2f, "
                "f=(%.4f, %.4f))" % (self.p1, self.p2, self.gamma_deg,
                                     self.alpha_deg, self.f1, self.f2))


class Constraints:
    """Proposal par. 7.2 hard constraints, as one inspectable object."""

    def __init__(self, kz_min_frac=0.2, wood_margin=0.05, n_orders_min=8,
                 n_orders_max=24, area_max_um2=2000.0,
                 min_gap_um=2.0 * R_CIRC_UM, mirror_avoid_deg=2.0,
                 dressing_max=0.5, deembed_sigma_min=0.5,
                 signal_min_sigma=3.0):
        # --- physical constraints that only bite once C is available
        # (reviewer recommendation 4).  Without them the marginalized
        # objective is gamed: an unconstrained search found a 3.5 um-pitch
        # cell with ||C T|| = 0.89-0.95 -- sitting on a collective resonance,
        # where the Jacobian is huge for the ensemble draws and collapses by
        # 100x at a different T.  Its Gate A recovery error was 284 % against
        # the incumbent's 5.7 %.
        self.dressing_max = float(dressing_max)          # max ||C T||
        self.deembed_sigma_min = float(deembed_sigma_min)  # sigma_min(I+TeffC)
        self.signal_min_sigma = float(signal_min_sigma)  # max|S_sca| / sigma
        self.kz_min_frac = float(kz_min_frac)
        self.wood_margin = float(wood_margin)
        self.n_orders_min = int(n_orders_min)
        self.n_orders_max = int(n_orders_max)
        self.area_max_um2 = float(area_max_um2)
        self.min_gap_um = float(min_gap_um)
        self.mirror_avoid_deg = float(mirror_avoid_deg)

    def check_geometry(self, design):
        reasons = []
        lat = design.lattice()
        if lat.area > self.area_max_um2:
            reasons.append("area %.1f > %.1f um^2"
                           % (lat.area, self.area_max_um2))
        # no overlap of periodic copies: every non-zero lattice vector must
        # clear two circumscribing radii
        radii, _ = lat.shells(3.0 * max(design.p1, design.p2))
        if len(radii) and radii.min() < self.min_gap_um:
            reasons.append("nearest neighbour %.3f < %.3f um"
                           % (radii.min(), self.min_gap_um))
        return reasons

    def check_bloch_generic(self, design):
        """Warn when the Bloch vector sits on a lattice mirror line.

        Proposal par. 1 item 3 requires a generic k_B away from all mirror
        lines: on a mirror line the encoding stops mixing the sectors it was
        introduced to mix.  Reported as a reason (soft in practice, because
        the rank objective already sees the loss).
        """
        lat = design.lattice()
        kB = lat.bloch(design.f1, design.f2)
        if np.linalg.norm(kB) < 1e-12:
            return ["Bloch vector is zero (Gamma point)"]
        ang = np.rad2deg(np.arctan2(kB[1], kB[0])) % 180.0
        out = []
        for m in lat.mirror_azimuths_deg():
            d = abs(ang - m)
            d = min(d, 180.0 - d)
            if d < self.mirror_avoid_deg:
                out.append("k_B within %.2f deg of the lattice mirror at "
                           "%.2f deg" % (d, m))
        return out


# --------------------------------------------------------------- evaluation

def evaluate(design, k_list, modes, B=None, constraints=None, sigma=None,
             T_ensemble=None, C_list=None, cost_model=None,
             cost_ref_s=None, cost_gamma=0.5, gauge=xf.GAUGE_CST,
             n_evanescent=2, nuisance_classes=None, ewald_C=False):
    """Full per-frequency report plus the two scalar objectives.

    Returns a dict; `ok` is False with `reasons` when a hard constraint
    fails, in which case both objectives are 0.
    """
    constraints = constraints or Constraints()
    cost_model = cost_model or ct.CostModel()
    k_list = np.atleast_1d(np.asarray(k_list, dtype=float))

    out = dict(design=design.to_dict(), ok=True, reasons=[], per_freq=[])
    out["reasons"] += constraints.check_geometry(design)
    out["reasons"] += constraints.check_bloch_generic(design)

    lat = design.lattice()
    out["area_um2"] = float(lat.area)
    out["mirror_azimuths_deg"] = lat.mirror_azimuths_deg()

    obj_wheel, obj_generic, n_orders_max_seen = [], [], 0
    obj_wheel_iid, obj_generic_iid, obj_marg = [], [], []
    for i, k in enumerate(k_list):
        orders = lt.enumerate_orders(
            lat, k, f_bloch=(design.f1, design.f2),
            kz_min_frac=constraints.kz_min_frac,
            wood_margin=constraints.wood_margin)
        ok_c, why = orders.passes_constraints()
        n_ord = orders.n_retained
        n_orders_max_seen = max(n_orders_max_seen, n_ord)
        if not ok_c:
            out["reasons"] += ["lam=%.3f: %s" % (2 * np.pi / k, w)
                               for w in why]
        if not (constraints.n_orders_min <= n_ord
                <= constraints.n_orders_max):
            out["reasons"].append(
                "lam=%.3f: %d retained orders outside [%d, %d]"
                % (2 * np.pi / k, n_ord, constraints.n_orders_min,
                   constraints.n_orders_max))
        if n_ord == 0:
            out["ok"] = False
            continue

        ch = lt.ChannelSet(orders)
        A = xf.build_A(k, ch, modes, gauge=gauge)
        W = xf.build_W(k, ch, modes, gauge=gauge)
        n_obs = ch.n ** 2
        sig = (jac.sigma_uniform(n_obs, measured_sigma(2 * np.pi / k))
               if sigma is None
               else np.broadcast_to(np.asarray(sigma, float).ravel(),
                                    (n_obs,)))
        s_scalar = float(np.asarray(sig).mean())

        # systematic-model normalization: worst-case ||dT|| scales as
        # sqrt(n_obs) / sigma_min, so the merit is sigma_min / sqrt(n_obs)
        sysn = np.sqrt(float(n_obs))

        gm = xf.generic_track_metrics(A, W)
        gen_obj_iid = (gm["sigma_min_A"] * gm["sigma_min_W"] / s_scalar
                       if gm["full_rank"] else 0.0)
        gen_obj = gen_obj_iid / sysn

        C = None if C_list is None else C_list[i]
        if C is None and (ewald_C or nuisance_classes):
            from . import ewald as ew
            C = ew.lattice_sum_C(lat, k, modes,
                                 lat.bloch(design.f1, design.f2))
        wheel = None
        if B is not None:
            if C is None:
                wheel = jac.wheel_track_metrics(W, A, B, sigma=sig)
                wheel.pop("H", None)
                wh_iid = wheel["sigma_min"] if wheel["full_rank"] else 0.0
            else:
                ens = jac.ensemble_metrics(W, A, B, T_ensemble, C=C,
                                           sigma=sig)
                wheel = ens
                wh_iid = (ens["worst_sigma_min"] if ens["full_rank_all"]
                          else 0.0)
            obj_wheel.append(wh_iid / sysn)
            obj_wheel_iid.append(wh_iid)
        obj_generic.append(gen_obj)
        obj_generic_iid.append(gen_obj_iid)

        # physical stability of the lattice-dressed problem, over the same
        # ensemble the objective uses (reviewer recommendation 4)
        if C is not None and T_ensemble is not None:
            from . import ewald as ew
            ens = np.atleast_3d(np.asarray(T_ensemble))
            if ens.ndim == 2:
                ens = ens[None]
            worst_dress, worst_sv, worst_sig = 0.0, np.inf, np.inf
            for T0 in ens:
                worst_dress = max(worst_dress,
                                  ew.lattice_dressing_strength(C, T0)["norm_CT"])
                Teff = xf.t_effective(T0, C)
                _, dd = xf.deembed_lattice(Teff, C, return_diag=True)
                worst_sv = min(worst_sv, dd["sigma_min"])
                worst_sig = min(worst_sig,
                                float(np.abs(xf.scattered_S(W, T0, C,
                                                            A)).max()))
            rec_stab = dict(dressing_max=float(worst_dress),
                            deembed_sigma_min=float(worst_sv),
                            signal_over_sigma=float(worst_sig / s_scalar))
            if worst_dress > constraints.dressing_max:
                out["reasons"].append(
                    "lam=%.3f: ||C T|| = %.3f > %.3f (collective-resonance "
                    "regime)" % (2 * np.pi / k, worst_dress,
                                 constraints.dressing_max))
            if worst_sv < constraints.deembed_sigma_min:
                out["reasons"].append(
                    "lam=%.3f: sigma_min(I + Teff C) = %.3f < %.3f"
                    % (2 * np.pi / k, worst_sv,
                       constraints.deembed_sigma_min))
            if worst_sig / s_scalar < constraints.signal_min_sigma:
                out["reasons"].append(
                    "lam=%.3f: signal/sigma = %.2f < %.2f"
                    % (2 * np.pi / k, worst_sig / s_scalar,
                       constraints.signal_min_sigma))
        else:
            rec_stab = None

        marg = None
        if nuisance_classes and B is not None and T_ensemble is not None:
            from . import nuisance as nz
            cls = (nz.DEFAULT_CLASSES if nuisance_classes is True
                   else tuple(nuisance_classes))
            marg = nz.ensemble_marginalized(W, A, B, T_ensemble, C, s_scalar,
                                            ch, classes=cls)
            obj_marg.append(marg["worst_sigma_marg"]
                            if marg["all_identifiable"] else 0.0)

        rec = dict(lam_um=float(2 * np.pi / k), k=float(k),
                   n_orders=int(n_ord), n_channels=int(ch.n),
                   n_prop=int(orders.n_prop),
                   wood_margin=orders.wood_margin_actual(),
                   grazing_margin=orders.grazing_margin_actual(),
                   generic=dict(rank_A=gm["rank_A"], rank_W=gm["rank_W"],
                                kappa_A=gm["kappa_A"], kappa_W=gm["kappa_W"],
                                sigma_min_A=gm["sigma_min_A"],
                                sigma_min_W=gm["sigma_min_W"],
                                objective=gen_obj),
                   sigma_used=s_scalar)
        if wheel is not None:
            # post_std is a PER-COORDINATE quantity in the arbitrary
            # eigenbasis of a degenerate projector, so it is not reported:
            # a rotation of the symmetry basis changes it while leaving the
            # physics untouched.  Invariant error figures come from
            # jacobian.recovery_errors instead.
            rec["wheel"] = {kk: vv for kk, vv in wheel.items()
                            if kk not in ("H", "per_draw", "post_std")}
        if rec_stab is not None:
            rec["stability"] = rec_stab
        if marg is not None:
            rec["marginalized"] = {kk: vv for kk, vv in marg.items()
                                   if kk != "per_draw"}
            rec["marginalized"]["per_class"] = (
                marg["per_draw"][0]["per_class"] if marg["per_draw"] else {})
            # Per-DRAW scalars, in ensemble order.  The full `per_draw` rows
            # are dropped because they are bulky, but a PAIRED comparison
            # needs draw i of one ensemble against draw i of another: without
            # these, a stress audit can only compare worst-to-worst, which is
            # two different draws whenever the bottlenecks differ.
            for kk in ("sigma_marg", "sigma_free", "loss"):
                rec["marginalized"]["per_draw_" + kk] = [
                    float(r[kk]) for r in marg["per_draw"]]
            # IDENTITY of each draw, so a paired comparison can PROVE that
            # row i is the same T on both sides.  Positional agreement is not
            # evidence: reordered rows would be accepted and mislabelled.
            rec["marginalized"]["per_draw_id"] = [
                hashlib.sha256(np.ascontiguousarray(T0).tobytes()).hexdigest()
                for T0 in np.atleast_3d(np.asarray(T_ensemble))]
        out["per_freq"].append(rec)

    cst = cost_model.campaign(lat.area, n_orders_max_seen,
                              n_evanescent=n_evanescent)
    out["cost"] = cst
    ref = (cost_ref_s if cost_ref_s is not None
           else cost_model.campaign(ct.A_REF_UM2, 1,
                                    n_evanescent=n_evanescent)["t_campaign_s"])
    pen = ct.cost_penalty(cst["t_campaign_s"], ref, cost_gamma)
    out["cost_penalty"] = pen
    out["cost_ref_s"] = float(ref)

    out["ok"] = out["ok"] and not out["reasons"]
    out["objective_generic"] = (float(min(obj_generic)) * pen
                                if obj_generic and out["ok"] else 0.0)
    out["objective_wheel"] = (float(min(obj_wheel)) * pen
                              if obj_wheel and out["ok"] else 0.0)
    out["objective_generic_iid"] = (float(min(obj_generic_iid)) * pen
                                    if obj_generic_iid and out["ok"]
                                    else 0.0)
    out["objective_wheel_iid"] = (float(min(obj_wheel_iid)) * pen
                                  if obj_wheel_iid and out["ok"]
                                  else 0.0)
    out["objective_marginalized"] = (float(min(obj_marg)) * pen
                                     if obj_marg and out["ok"] else 0.0)
    return out


def score(report, track):
    if track == "wheel":
        return report["objective_wheel"]
    if track == "generic":
        return report["objective_generic"]
    if track == "both":
        return min(report["objective_wheel"], report["objective_generic"])
    if track == "marginalized":
        return report.get("objective_marginalized", 0.0)
    raise ValueError("track must be wheel / generic / both / marginalized")


# ------------------------------------------------------------------ search

DEFAULT_BOX = dict(p1=(6.0, 45.0), p2=(6.0, 45.0), gamma_deg=(75.0, 105.0),
                   alpha_deg=(0.0, 90.0), f1=(-0.5, 0.5), f2=(-0.5, 0.5))


def _sample(rng, box):
    return Design(
        p1=rng.uniform(*box["p1"]), p2=rng.uniform(*box["p2"]),
        gamma_deg=rng.uniform(*box["gamma_deg"]),
        alpha_deg=rng.uniform(*box["alpha_deg"]),
        f1=rng.uniform(*box["f1"]), f2=rng.uniform(*box["f2"]))


def bloch_feasible_grid(design, k_list, constraints, n_grid=9, jitter=None):
    """Fractional Bloch points at which `design` satisfies every hard
    constraint, found on a jittered grid over the first Brillouin zone.

    WHY THIS EXISTS.  Sampling (f1, f2) uniformly with the rest of the cell
    almost never lands on a feasible point once the cell is large.  The
    par. 7.2 Wood/Rayleigh rule forbids EVERY order -- propagating or not --
    from coming within `wood_margin` of |q| = k, and the expected number of
    orders inside that forbidden annulus is

        N_ann  ~  2 pi k (margin k) / (4 pi^2 / A_cell)
               =  margin k^2 A_cell / (2 pi),

    so the acceptance probability of a random Bloch vector decays like
    exp(-N_ann).  At lambda = 8 um with margin 0.05 that is ~1 forbidden
    order at A_cell ~ 200 um^2 and ~5 at 1000 um^2, and the measured hit
    rates (45/200, 6/200, 1/200 for the 2-6, 6-20 and 8-24 order
    configurations) match that estimate.  Blind sampling is therefore not a
    search strategy here; the Bloch vector has to be optimized for each cell,
    which is cheap because only the order enumeration is involved.

    Returns a list of (f1, f2) tuples, best Wood margin first.
    """
    out = []
    off = np.zeros(2) if jitter is None else np.asarray(jitter, float)
    g = (np.arange(n_grid) + 0.5) / n_grid - 0.5
    for f1 in g + off[0] / n_grid:
        for f2 in g + off[1] / n_grid:
            d = Design(design.p1, design.p2, design.gamma_deg,
                       design.alpha_deg, float(f1), float(f2))
            if constraints.check_bloch_generic(d):
                continue
            lat = d.lattice()
            worst_wood, ok = np.inf, True
            for k in np.atleast_1d(k_list):
                o = lt.enumerate_orders(lat, k, f_bloch=(d.f1, d.f2),
                                        kz_min_frac=constraints.kz_min_frac,
                                        wood_margin=constraints.wood_margin)
                good, _ = o.passes_constraints()
                if not good or not (constraints.n_orders_min <= o.n_retained
                                    <= constraints.n_orders_max):
                    ok = False
                    break
                worst_wood = min(worst_wood, o.wood_margin_actual())
            if ok:
                out.append((float(f1), float(f2), worst_wood))
    out.sort(key=lambda t: -t[2])
    return [(a, b) for a, b, _ in out]


def best_bloch(design, k_list, modes, evaluate_fn, constraints, n_grid=9,
               n_eval=4, jitter=None):
    """Pick the Bloch vector of `design` by objective, among feasible ones.

    Feasibility is screened on a grid with order enumeration only (cheap),
    then the objective is evaluated on the `n_eval` points with the largest
    Wood margin.  Returns (design, value) or (None, 0.0).
    """
    cands = bloch_feasible_grid(design, k_list, constraints, n_grid, jitter)
    best, best_v = None, 0.0
    for f1, f2 in cands[:int(n_eval)]:
        d = Design(design.p1, design.p2, design.gamma_deg, design.alpha_deg,
                   f1, f2)
        v = evaluate_fn(d)
        if v > best_v:
            best, best_v = d, v
    return best, best_v


def pattern_search(design0, evaluate_fn, box, steps=None, n_rounds=6,
                   shrink=0.5, verbose=False):
    """Coordinate pattern search: robust to the objective's discontinuities.

    The objective jumps whenever an order opens or closes, so gradient and
    simplex methods mislead.  A shrinking coordinate search only ever
    compares evaluated points, which is exactly what a piecewise-constant-
    plus-smooth landscape allows.
    """
    keys = ["p1", "p2", "gamma_deg", "alpha_deg", "f1", "f2"]
    if steps is None:
        steps = {k: 0.08 * (box[k][1] - box[k][0]) for k in keys}
    else:
        steps = dict(steps)
    best = design0
    best_val = evaluate_fn(best)
    for _ in range(int(n_rounds)):
        improved = False
        for kk in keys:
            for sgn in (+1.0, -1.0):
                v = dict(zip(keys, best.vector))
                v[kk] = float(np.clip(v[kk] + sgn * steps[kk],
                                      box[kk][0], box[kk][1]))
                cand = Design(**v)
                val = evaluate_fn(cand)
                if val > best_val:
                    best, best_val, improved = cand, val, True
        if not improved:
            for kk in keys:
                steps[kk] *= shrink
        if verbose:
            print("    pattern round: best = %.6g" % best_val, flush=True)
    return best, best_val


def search(k_list, modes, B=None, track="wheel", n_samples=400, n_polish=5,
           box=None, constraints=None, seed=20260807, verbose=True,
           bloch_grid=9, bloch_eval=4, **kw):
    """Two-stage screening + pattern-search polish over the design box.

    Stage 1 samples the cell GEOMETRY only; the Bloch vector is then chosen
    for that cell by `best_bloch` (see its docstring for why blind sampling
    of (f1, f2) fails).  Stage 2 polishes the winners in all six variables
    with a coordinate pattern search.
    """
    box = dict(DEFAULT_BOX if box is None else box)
    rng = np.random.default_rng(seed)
    constraints = constraints or Constraints()

    def val(d):
        return score(evaluate(d, k_list, modes, B=B,
                              constraints=constraints, **kw), track)

    cands = []
    for i in range(int(n_samples)):
        d0 = _sample(rng, box)
        d, v = best_bloch(d0, k_list, modes, val, constraints,
                          n_grid=bloch_grid, n_eval=bloch_eval,
                          jitter=rng.uniform(-0.5, 0.5, size=2))
        if d is not None and v > 0:
            cands.append((v, d))
        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            print("  screened %d/%d, %d feasible, best %.6g"
                  % (i + 1, n_samples, len(cands),
                     max([c[0] for c in cands], default=0.0)), flush=True)
    if not cands:
        return None, []
    cands.sort(key=lambda t: -t[0])
    polished = []
    for v0, d0 in cands[:int(n_polish)]:
        d, v = pattern_search(d0, val, box, verbose=False)
        polished.append((v, d, v0))
        if verbose:
            print("  polish: %.6g -> %.6g   %r" % (v0, v, d), flush=True)
    polished.sort(key=lambda t: -t[0])
    return polished[0][1], polished


# --------------------------------------------------------- pooled encodings

def _design_pieces(design, k, modes, constraints, enforce_counts=True):
    """(channels, A, W) for one design at one k, or None if infeasible.

    `enforce_counts` applies the SAME order-count window the single-cell
    search uses.  Without it the pooled search would quietly buy its joint
    rank by picking large cells, which is exactly the hypothesis under test
    ("a small channel set pooled over two or three encodings", par. 8.5)
    rather than a way to satisfy it.
    """
    lat = design.lattice()
    orders = lt.enumerate_orders(lat, k, f_bloch=(design.f1, design.f2),
                                 kz_min_frac=constraints.kz_min_frac,
                                 wood_margin=constraints.wood_margin)
    ok, _ = orders.passes_constraints()
    if not orders.n_retained or not ok:
        return None
    if enforce_counts and not (constraints.n_orders_min <= orders.n_retained
                               <= constraints.n_orders_max):
        return None
    ch = lt.ChannelSet(orders)
    return ch, xf.build_A(k, ch, modes), xf.build_W(k, ch, modes)


def pooled_evaluate(designs, k_list, modes, B, constraints=None, sigma=None,
                    cost_model=None, T_ref=None, snr_floor=10.0,
                    n_evanescent=2):
    """Joint identifiability of SEVERAL coded cells measured together.

    The proposal's headline hypothesis for the fast branch (par. 1 item 4,
    par. 8.5) is that a SMALL channel set "pooled over two or three generic
    Bloch/cell encodings" beats one large diffractive cell: each small cell
    keeps the sheet signal strong (amplitudes fall as 1/A_cell) while the
    encodings together supply the angular diversity that one small cell
    lacks.  This function tests exactly that, by stacking the whitened
    Jacobians of all encodings into one system

        H_pool = [ Sigma_1^{-1/2} H_1 ; Sigma_2^{-1/2} H_2 ; ... ]

    and reporting the joint spectrum and summed CST cost.  With T_ref
    supplied it also reports the invariant predicted recovery error the pool
    would achieve on that target (reporting only -- never used to choose
    designs).

    NOTE on stacking.  Pooling multiplies the observable count, so the iid
    posterior improves like 1/sqrt(n_obs).  Because the whitening level is a
    systematic model discrepancy rather than iid noise, that gain is not
    real; `jacobian.recovery_errors` reports the conservative bound
    alongside, and the pooled verdict is driven by the conservative one.
    """
    constraints = constraints or Constraints()
    cost_model = cost_model or ct.CostModel()
    k_list = np.atleast_1d(np.asarray(k_list, dtype=float))

    out = dict(designs=[d.to_dict() for d in designs], per_freq=[],
               n_encodings=len(designs))
    total_cost, total_rhs, worst_area = 0.0, 0, 0.0
    for d in designs:
        area = d.lattice().area
        worst_area = max(worst_area, area)
        n_ord = 0
        for k in k_list:
            p = _design_pieces(d, k, modes, constraints)
            n_ord = max(n_ord, 0 if p is None else p[0].n // 4)
        c = cost_model.campaign(area, n_ord, n_evanescent=n_evanescent)
        total_cost += c["t_campaign_s"]
        total_rhs += c["n_rhs"]
    out["cost"] = dict(t_campaign_s=float(total_cost),
                       t_campaign_min=float(total_cost / 60.0),
                       n_rhs_total=int(total_rhs),
                       max_area_um2=float(worst_area))

    for k in k_list:
        sig_scalar = (measured_sigma(2 * np.pi / k) if sigma is None
                      else float(sigma))
        Hs, n_ch, n_ord, dropped = [], [], [], []
        for j, d in enumerate(designs):
            p = _design_pieces(d, k, modes, constraints)
            if p is None:
                dropped.append(j)
                continue
            ch, A, W = p
            H = jac.jacobian(W, A, B)
            Hs.append(H / sig_scalar)
            n_ch.append(ch.n)
            n_ord.append(ch.n // 4)
        if not Hs:
            out["per_freq"].append(dict(lam_um=float(2 * np.pi / k),
                                        feasible=False,
                                        dropped_encodings=dropped))
            continue
        Hp = np.vstack(Hs)
        sv = np.linalg.svd(Hp, compute_uv=False)
        rank = int((sv > 1e-10 * sv[0]).sum())
        rec = dict(lam_um=float(2 * np.pi / k), feasible=True,
                   n_channels=[int(x) for x in n_ch],
                   n_orders=[int(x) for x in n_ord],
                   n_encodings_used=len(Hs),
                   dropped_encodings=dropped,
                   n_obs=int(Hp.shape[0]), rank=rank,
                   sigma_min=float(sv[-1]), sigma_max=float(sv[0]),
                   kappa=float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf)
        rec["sigma_used"] = float(sig_scalar)
        if rank == Hp.shape[1] and T_ref is not None:
            Cov = jac.coefficient_covariance(Hp)
            rec["recovery"] = jac.recovery_errors(
                B, modes, Cov, Hp.shape[0], T_ref, sigma_used=sig_scalar)
        out["per_freq"].append(rec)
    # systematic-model merit: sigma_min / sqrt(n_obs).  Pooling raises both,
    # so this is where "more encodings" stops being free.
    def _merit(f, iid=False):
        if f.get("rank") != len(B):
            return 0.0
        s = f.get("sigma_min", 0.0)
        return s if iid else s / np.sqrt(max(f.get("n_obs", 1), 1))

    out["objective"] = (float(min(_merit(f) for f in out["per_freq"]))
                        if out["per_freq"] else 0.0)
    out["objective_iid"] = (float(min(_merit(f, True) for f in out["per_freq"]))
                            if out["per_freq"] else 0.0)
    return out


def search_pool(n_encodings, k_list, modes, B, constraints=None, box=None,
                n_samples=300, n_polish=2, seed=20260807, verbose=True,
                **kw):
    """Greedy pooled search: add one encoding at a time, each chosen to
    maximize the JOINT objective given the ones already selected.

    Greedy is the right tool here: the joint objective is a monotone
    function of the stacked row set (adding rows can only raise every
    singular value), so a myopic addition never has to be undone, and the
    alternative -- jointly optimizing 6 * n_encodings variables against a
    discontinuous objective -- is not worth its cost at this stage.
    """
    constraints = constraints or Constraints()
    chosen = []
    box = dict(DEFAULT_BOX if box is None else box)
    rng = np.random.default_rng(seed)
    for j in range(int(n_encodings)):
        def val(d, _chosen=tuple(chosen)):
            rep = pooled_evaluate(list(_chosen) + [d], k_list, modes, B,
                                  constraints=constraints, **kw)
            return rep["objective"]

        best, best_v = None, 0.0
        for i in range(int(n_samples)):
            d0 = _sample(rng, box)
            if constraints.check_geometry(d0):
                continue
            d, v = best_bloch(d0, k_list, modes, val, constraints,
                              jitter=rng.uniform(-0.5, 0.5, size=2))
            if d is not None and v > best_v:
                best, best_v = d, v
        if best is None:
            if verbose:
                print("  encoding %d: no feasible candidate" % (j + 1))
            break
        for _ in range(int(n_polish)):
            best, best_v = pattern_search(best, val, box, n_rounds=4)
        chosen.append(best)
        if verbose:
            print("  encoding %d: joint objective %.6g   %r"
                  % (j + 1, best_v, best), flush=True)
    return chosen


# ----------------------------------------------------------- reporting hook

def benchmark_reference(design, k_list, modes, B, tmat_path=None, **kw):
    """Evaluate a design against the INDEPENDENTLY SUPPLIED wheel T.

    Reporting only.  The proposal reserves the reference T for benchmarking
    (par. 7.3, Gate E) and forbids using it to choose the design, so this is
    a separate entry point that `search` never calls.
    """
    from tmatrix.aggregation.tmat_io import TMatrixData
    if tmat_path is None:
        tmat_path = os.path.join(BENCHMARK_SINGLE, "saw_gold_wl15p0025um.tmat.h5")
    data = TMatrixData(tmat_path)
    lam_target = 2 * np.pi / np.atleast_1d(np.asarray(k_list, float))
    idx = [int(np.argmin(np.abs(data.wavelength_um - l))) for l in lam_target]
    ks = np.array([data.k_at(i) for i in idx])
    rep = evaluate(design, ks, modes, B=B, **kw)
    rep["reference"] = dict(
        tmat_path=os.path.abspath(tmat_path),
        freq_indices=idx,
        lam_um=[float(data.wavelength_um[i]) for i in idx],
        max_abs_T=[float(np.abs(data.T[i]).max()) for i in idx],
        d4h_residual=[float(x) for x in
                      sym.symmetry_residual(data.T[idx], B)])
    cons = kw.get("constraints") or Constraints()
    for j, i in enumerate(idx):
        if j >= len(rep["per_freq"]):
            break
        lat = design.lattice()
        orders = lt.enumerate_orders(lat, ks[j], f_bloch=(design.f1,
                                                          design.f2),
                                     kz_min_frac=cons.kz_min_frac,
                                     wood_margin=cons.wood_margin)
        if not orders.n_retained:
            rep["per_freq"][j]["reference_signal"] = None
            continue
        ch = lt.ChannelSet(orders)
        A = xf.build_A(ks[j], ch, modes)
        W = xf.build_W(ks[j], ch, modes)
        sig = jac.sigma_uniform(ch.n ** 2,
                                measured_sigma(data.wavelength_um[i]))
        rep["per_freq"][j]["reference_signal"] = jac.signal_metrics(
            W, A, data.T[i], sigma=sig)
        rep["per_freq"][j]["recovery"] = jac.reference_recovery(
            W, A, B, modes, data.T[i], sig)
    return rep


# ---------------------------------------------------------------------- CLI

def _fmt_report(rep, track):
    lines = [" design: %s" % Design.from_vector(
        [rep["design"][k] for k in ("p1_um", "p2_um", "gamma_deg",
                                    "alpha_deg", "f1", "f2")]),
        " area = %.2f um^2   cost proxy = %.1f min (%d RHS)   penalty %.3f"
        % (rep["area_um2"], rep["cost"]["t_campaign_min"],
           rep["cost"]["n_rhs"], rep["cost_penalty"])]
    if not rep["ok"]:
        lines.append(" INFEASIBLE: " + "; ".join(rep["reasons"]))
    for f in rep["per_freq"]:
        g = f["generic"]
        lines.append(
            "  lam %6.3f um: orders %2d  channels %3d  Wood %.3f  kz/k %.3f"
            % (f["lam_um"], f["n_orders"], f["n_channels"],
               f["wood_margin"], f["grazing_margin"]))
        lines.append(
            "                generic rank %2d/%2d  kappa %.3g/%.3g  "
            "obj %.4g" % (g["rank_A"], g["rank_W"], g["kappa_A"],
                          g["kappa_W"], g["objective"]))
        if "wheel" in f:
            w = f["wheel"]
            if "rank" in w:
                lines.append("                wheel   rank %2d/40  "
                             "sigma_40 %.4g  kappa %.3g"
                             % (w["rank"], w["sigma_min"], w["kappa"]))
            else:
                lines.append("                wheel   worst rank %2d/40  "
                             "worst sigma_40 %.4g"
                             % (w["worst_rank"], w["worst_sigma_min"]))
        if f.get("reference_signal"):
            rs = f["reference_signal"]
            lines.append("                reference |S_sca|max %.3e  "
                         "S/sigma %.1f" % (rs["max_abs"],
                                           rs.get("snr_max", float("nan"))))
        if f.get("recovery"):
            lines += fmt_recovery(f["recovery"])
    lines.append(" objective(%s) = %.6g" % (track, score(rep, track)))
    return "\n".join(lines)


def fmt_recovery(rc, indent=16):
    """Invariant Gate E lines: predicted T error under both error models."""
    pad = " " * indent
    if not rc.get("full_rank"):
        return [pad + "recovery: rank %d/40 -- T is not determined"
                % rc.get("rank", 0)]
    out = [pad + "predicted T error at sigma = %.4e (%d observables):"
           % (rc.get("sigma_used", float("nan")), rc.get("n_obs", 0))]
    for tag, e_sys, e_iid, target, need in (
            ("global ||dT||_F/||T||_F", rc["fro_err_sys"], rc["fro_err_iid"],
             rc["fro_target"], rc.get("sigma_for_global", 0.0)),
            ("worst dominant block   ", rc["block_err_sys"],
             rc["block_err_iid"], rc["block_target"],
             rc.get("sigma_for_block", 0.0))):
        ok = e_sys <= target
        r = (rc.get("sigma_used", 0.0) / need) if need > 0 else float("inf")
        out.append(pad + "  %s %7.4f%% systematic / %7.4f%% iid  "
                   "(target %.0f%%): %s"
                   % (tag, 100 * e_sys, 100 * e_iid, 100 * target,
                      "PASS" if ok else "FAIL, needs sigma <= %.2e (%.1fx)"
                      % (need, r)))
    out.append(pad + "  dominant blocks (>= %.0f%% of ||T||_F): %s"
               % (100 * rc["dominant_frac"],
                  ", ".join(rc["dominant_names"]) or "none"))
    out.append(pad + "  averaging gain claimed by the iid model: %.1fx "
               "(NOT available for a systematic discrepancy)"
               % rc["averaging_gain"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lam", default="8.0",
                   help="comma-separated design wavelengths in um")
    p.add_argument("--track", default="wheel",
                   choices=["wheel", "generic", "both"])
    p.add_argument("--lmax", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--n-polish", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--kz-min", type=float, default=0.2)
    p.add_argument("--wood-margin", type=float, default=0.05)
    p.add_argument("--orders", default="8,24",
                   help="min,max retained propagating orders")
    p.add_argument("--area-max", type=float, default=2000.0)
    p.add_argument("--cost-gamma", type=float, default=0.5)
    p.add_argument("--evaluate", default=None,
                   help="p1,p2,gamma,alpha,f1,f2 -- evaluate, do not search")
    p.add_argument("--benchmark", action="store_true",
                   help="also report signal against the reference tmat.h5")
    p.add_argument("--out", default=None, help="write the report as JSON")
    a = p.parse_args(argv)

    lams = [float(x) for x in a.lam.split(",")]
    ks = np.array([2 * np.pi / l for l in lams])
    modes = ModeBasis.standard(a.lmax)
    print("building the D4h + reciprocity basis ...", flush=True)
    B, meta = sym.build_d4h_reciprocity_basis(modes)
    print("  rank %d (predicted %d); sigma_h numeric err %.2e"
          % (meta["rank_full"], meta["rank_full_predicted"],
             meta["sigma_h_numeric_err"]), flush=True)

    n_min, n_max = [int(x) for x in a.orders.split(",")]
    cons = Constraints(kz_min_frac=a.kz_min, wood_margin=a.wood_margin,
                       n_orders_min=n_min, n_orders_max=n_max,
                       area_max_um2=a.area_max)
    kw = dict(cost_gamma=a.cost_gamma)

    if a.evaluate:
        d = Design.from_vector([float(x) for x in a.evaluate.split(",")])
        rep = (benchmark_reference(d, ks, modes, B, constraints=cons, **kw)
               if a.benchmark
               else evaluate(d, ks, modes, B=B, constraints=cons, **kw))
        print(_fmt_report(rep, a.track))
        reports = [rep]
    else:
        print("searching (%d samples, track=%s, lambda=%s um) ..."
              % (a.n_samples, a.track, lams), flush=True)
        best, polished = search(ks, modes, B=B, track=a.track,
                                n_samples=a.n_samples, n_polish=a.n_polish,
                                constraints=cons, seed=a.seed, **kw)
        if best is None:
            print("NO FEASIBLE DESIGN in the sampled box -- relax the "
                  "constraints or widen the box")
            return 1
        reports = []
        print("\n=== best %d designs ===" % len(polished))
        for v, d, v0 in polished:
            rep = (benchmark_reference(d, ks, modes, B, constraints=cons,
                                       **kw) if a.benchmark
                   else evaluate(d, ks, modes, B=B, constraints=cons, **kw))
            reports.append(rep)
            print(_fmt_report(rep, a.track))
            print()

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(dict(track=a.track, lam_um=lams, reports=reports), fh,
                      indent=1, default=float)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
