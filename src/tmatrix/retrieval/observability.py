"""Observability map for the inverse T-matrix retrieval (checklist item 5
of INVERSE_TMATRIX_FROM_FLOQUET.md; doc par. 4 first-class deliverable).

Computes the Jacobian J = d packed(S_all-angles) / d t of the forward map
around a linearization point (default: the reference T projected into the
basis), its SVD, a per-parameter resolution measure, and the doc's
entrywise heatmap H[mu, nu] = sum_k res_k |B_k[mu, nu]|^2, plus the
theta=0 visibility verification of the doc's par.-4 claims.

Conventions and definitions (normative for this module):

  * Jacobian: CENTRAL finite differences, step h_k = step_scale * max(1,
    |t_lin,k|) with step_scale = 1e-6 (parameters are O(1e-2), so h = 1e-6
    in practice).  J is (16 * n_angles, 2 * nb) real, rows in
    ForwardModel.pack_S order (all real parts, then all imaginary parts).
    The model is T(t) = T_offset + unpack(t, B) (T_offset = 0 by default;
    a nonzero T_offset supports sub-basis Jacobians linearized at the full
    reference).  Default linearization t_lin = pack(T_ref - T_offset, B),
    i.e. the coefficients of the PROJECTED reference P(T_ref).

  * Resolution: R = V diag(s_j^2 / (s_j^2 + lam^2)) V^T (the Tikhonov
    resolution matrix), res_k = R[k, k] in [0, 1].  The damping is
    lam = (sigma / sqrt(2)) / prior_scale: sigma is the per-COMPLEX-
    observable noise std (E|n|^2 = sigma^2, so each packed real component
    has std sigma/sqrt(2)); prior_scale = 1 encodes an implicit UNIT prior
    on t.  Because |t| ~ 1e-2 physically, the unit prior is deliberately
    loose: res_k ~ 1 for anything with sensitivity well above the noise
    floor and res_k ~ 0 only for structurally dark directions -- res is an
    observability DETECTOR, not a posterior variance.  SIGMA DEFAULT 3e-3
    IS A PLACEHOLDER: it is the magnitude-only CST closure floor on
    record; the true complex floor is unknown until the doc-par.-7
    normal-incidence complex closure measures it, and must replace the
    default then.  J is computed UNWEIGHTED (raw S units) in this default
    workflow; if you pass weights = 1/sigma_i^2 to jacobian() the rows are
    already noise-scaled and svd_resolution should be called with
    sigma = sqrt(2) (i.e. lam = 1).

  * Complex folding (documented choice): the forward map is HOLOMORPHIC in
    T0, so the imaginary-part column is exactly the packed rotation of the
    real-part column, d packed / d t_{k+nb} = packed(i * dS/dt_k); the
    Re/Im columns therefore carry identical singular content and we fold
    res for the complex direction k as the MEAN of the pair,
    res_c[k] = (res[k] + res[k + nb]) / 2 (= min up to FD noise; the
    holomorphy identity is verified numerically and reported).

  * Heatmap: H[mu, nu] = sum_k res_c[k] |B_k[mu, nu]|^2, k over the
    COMPLEX basis index.  With the full 68-basis, sum_k |B_k[mu, nu]|^2 is
    the diagonal of the subspace projector, so H <= diag(P) <= 1
    entrywise: H is the resolution-weighted projector diagonal.

Figures: results/observability_heatmap.png (log color, mode labels
(l,m,P) on both axes, bright entries outlined) and
results/observability_spectrum.png; arrays in
results/observability_{tag}.npz.

theta=0 visibility verification (doc par. 4, measured claims):
  (i)  the (3,-+3)<->(1,+-1) and (3,+-3)<->(3,+-3) content IS visible at
       theta=0 (the lattice sum couples Delta m = 0 mod 4, re-entering the
       m = +-1 specular channel); doc quotes |dS/dT| ~ 70-790 per unit
       entry vs ~7 for dipole entries at 14 um (we measure at the smoke
       frequency, 12 um -- qualitative comparison).
  (ii) even-m content (m, m' in {0, +-2}; e.g. the (2,+-2)(2,+-2)
       quadrupole entries) is STRICTLY dark at theta=0 and lights up when
       oblique angles are added.
Both claims are verified with raw per-ENTRY sensitivities |dS/dT[mu, nu]|
(central differences on the entry, base = P(T_ref) so the mod-4 m-grading
is exact) AND with column norms of sub-basis Jacobians (span of the
projected canonical entries, linearized at P(T_ref)).

CLI:
    python -m tmatrix.retrieval.observability --ifreq 32 --basis full \
        --angles campaign
    python -m tmatrix.retrieval.observability --ifreq 32 --basis bright \
        --angles normal-only
Options: --sigma 3e-3, --direction -1, --no-visibility, --tag NAME.
"""
import argparse
import os
import time


import numpy as np

from tmatrix.plotting import plt                    # noqa: E402
from matplotlib.colors import LogNorm                  # noqa: E402
from matplotlib.patches import Rectangle               # noqa: E402

from tmatrix.aggregation.vswf import ELECTRIC, MAGNETIC                    # noqa: E402
from tmatrix.retrieval import parametrize as par # noqa: E402
from tmatrix.retrieval import fit as fitmod # noqa: E402
from tmatrix.retrieval.fit import (DIRECTION_DEFAULT, resolve_angles,   # noqa: E402
                                   mode_label, entry_label)
from tmatrix.paths import RETRIEVAL_RESULTS             # noqa: E402

RESULTS_DIR = str(RETRIEVAL_RESULTS)
HEATMAP_DEFAULT = os.path.join(RESULTS_DIR, "observability_heatmap.png")
SPECTRUM_DEFAULT = os.path.join(RESULTS_DIR, "observability_spectrum.png")

SIGMA_PLACEHOLDER = 3e-3       # magnitude-only floor; see module docstring


# ---------------------------------------------------------------- Jacobian

def jacobian(fm, ifreq, B, angle_indices=None, t_lin=None, T_offset=None,
             direction=DIRECTION_DEFAULT, weights=None, step_scale=1e-6):
    """Central-difference Jacobian J (16*n_angles, 2*nb) of the packed
    forward map T(t) = T_offset + unpack(t, B) at t_lin.

    t_lin default: pack(T_ref[ifreq] - T_offset, B), i.e. the projected
    reference (with T_offset = 0).  Step: step_scale * max(1, |t_lin,k|)
    per parameter (scale-aware; |t| ~ 1e-2 so h = 1e-6 in practice).
    weights: forwarded to pack_S (leave None for raw S sensitivities --
    the svd_resolution default lam assumes that; see module docstring).
    """
    angles = resolve_angles(angle_indices)
    miss = [a for a in angles if not fm.have[ifreq, a]]
    if miss:
        raise KeyError("C not cached for ifreq=%d, angles %s"
                       % (ifreq, miss))
    B = np.asarray(B)
    n = B.shape[1]
    if T_offset is None:
        T_offset = np.zeros((n, n), dtype=complex)
    if t_lin is None:
        t_lin = par.pack(fm.data.T[ifreq] - T_offset, B)
    t_lin = np.asarray(t_lin, dtype=float)
    npar = t_lin.size

    def packed(t):
        return fm.predict_packed(T_offset + par.unpack(t, B), ifreq,
                                 angles, weights, direction)

    J = np.empty((16 * len(angles), npar))
    for k in range(npar):
        h = step_scale * max(1.0, abs(t_lin[k]))
        tp = t_lin.copy(); tp[k] += h                  # noqa: E702
        tm = t_lin.copy(); tm[k] -= h                  # noqa: E702
        J[:, k] = (packed(tp) - packed(tm)) / (2.0 * h)
    return J


def holomorphy_residual(J):
    """max_k ||J[:, k+nb] - packed(i * col_k)|| / ||col_k||: the exact
    identity d/dIm = i * d/dRe in packed form ([re; im] -> [-im; re]).
    Expect finite-difference noise level (~1e-6 relative)."""
    m2, npar = J.shape
    nb = npar // 2
    m = m2 // 2
    worst = 0.0
    for k in range(nb):
        rot = np.concatenate([-J[m:, k], J[:m, k]])
        nrm = np.linalg.norm(J[:, k])
        if nrm > 0:
            worst = max(worst, np.linalg.norm(J[:, k + nb] - rot) / nrm)
    return float(worst)


# ------------------------------------------------------------- resolution

def svd_resolution(J, sigma=SIGMA_PLACEHOLDER, prior_scale=1.0,
                   null_thresh=0.1):
    """SVD + Tikhonov resolution analysis of J (see module docstring).

    lam = (sigma / sqrt(2)) / prior_scale  (per-real-component noise std of
    complex-observable noise with E|n|^2 = sigma^2, unit prior on t).

    Returns dict: s (singular values, desc), V (npar, r) right vectors,
    filt (r,) = s^2/(s^2+lam^2), res (npar,) = diag of the resolution
    matrix, lam, n_above (count s > lam, equivalently filt > 1/2),
    null_idx (parameter indices with res < null_thresh), sigma.
    """
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    lam = (float(sigma) / np.sqrt(2.0)) / float(prior_scale)
    filt = s ** 2 / (s ** 2 + lam ** 2)
    V = Vt.T
    res = (V ** 2 * filt[None, :]).sum(axis=1)
    return dict(s=s, V=V, filt=filt, res=res, lam=lam,
                n_above=int((s > lam).sum()),
                null_idx=np.nonzero(res < null_thresh)[0],
                null_thresh=float(null_thresh), sigma=float(sigma))


def fold_complex(res):
    """Fold the real-parameter resolution (2*nb,) to complex directions
    (nb,) as the MEAN of each (Re, Im) pair (equal up to FD noise by
    holomorphy; see module docstring)."""
    nb = res.size // 2
    return 0.5 * (res[:nb] + res[nb:])


def entry_heatmap(res_c, B):
    """H[mu, nu] = sum_k res_c[k] |B_k[mu, nu]|^2 (k = complex basis
    index; |.|^2 supports the complex bases of fit.observable_subbasis)."""
    B = np.asarray(B)
    return np.einsum("k,kij->ij", np.asarray(res_c, dtype=float),
                     np.abs(B) ** 2)


# ------------------------------------------------------------- sub-bases

def subspace_basis_for_entries(B, entries, sv_tol=1e-8):
    """Orthonormal basis of span{ P(E_ij) : (i,j) in entries } inside
    span{B_k} (same construction as parametrize.bright_orbit_basis, for an
    explicit entry list).  Returns (B_sub, coeff) with B_sub = coeff @ B."""
    B = np.asarray(B)
    nb = B.shape[0]
    entries = np.asarray(entries, dtype=int)
    G = B.reshape(nb, -1)[:, entries[:, 0] * B.shape[1] + entries[:, 1]].T
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    rank = int((s > sv_tol * (s[0] if s.size else 1.0)).sum())
    coeff = Vt[:rank]
    return np.tensordot(coeff, B, axes=(1, 0)), coeff


# ---------------------------------------------------- entry sensitivities

def entry_sensitivity(fm, ifreq, entries, angle_indices, T_base,
                      direction=DIRECTION_DEFAULT, eps=1e-6):
    """Raw per-entry sensitivities dS/dT[mu, nu] by central differences on
    the ENTRY (no basis involved): the forward map is holomorphic in T, so
    the complex derivative follows from a real entry perturbation,

        dS/dT_e = (S(T_base + eps E_e) - S(T_base - eps E_e)) / (2 eps).

    Returns (D, summary): D (nE, 8*n_angles) complex per-observable
    derivatives; summary dict with max_abs (nE,) = max_i |dS_i/dT_e| (the
    doc-par.-4 "per unit entry" sensitivity scale) and rms (nE,)."""
    angles = resolve_angles(angle_indices)
    entries = np.asarray(entries, dtype=int)
    D = np.empty((len(entries), 8 * len(angles)), dtype=complex)
    for e, (i, j) in enumerate(entries):
        Tp = np.array(T_base, dtype=complex); Tp[i, j] += eps  # noqa: E702
        Tm = np.array(T_base, dtype=complex); Tm[i, j] -= eps  # noqa: E702
        Sp = fm.predict(Tp, ifreq, angles, direction).ravel()
        Sm = fm.predict(Tm, ifreq, angles, direction).ravel()
        D[e] = (Sp - Sm) / (2.0 * eps)
    summary = dict(max_abs=np.abs(D).max(axis=1),
                   rms=np.sqrt((np.abs(D) ** 2).mean(axis=1)))
    return D, summary


def _named_entry_sets(modes):
    """Entry lists for the doc-par.-4 visibility claims."""
    idx = modes.index
    m33_11, m33_33, dipole = [], [], []
    for p in (ELECTRIC, MAGNETIC):
        for (mm, m1) in ((-3, +1), (+3, -1)):
            a, b = idx(3, mm, p), idx(1, m1, p)
            m33_11 += [(a, b), (b, a)]
        for mm in (-3, +3):
            a = idx(3, mm, p)
            m33_33.append((a, a))
        for m1 in (-1, +1):
            a = idx(1, m1, p)
            dipole.append((a, a))
    # even-m bright entries (measured bright mask, doc par. 3):
    even = [(idx(1, 0, ELECTRIC),) * 2, (idx(1, 0, MAGNETIC),) * 2,
            (idx(2, 0, ELECTRIC),) * 2,
            (idx(2, -2, ELECTRIC),) * 2, (idx(2, +2, ELECTRIC),) * 2,
            (idx(3, 0, MAGNETIC), idx(1, 0, MAGNETIC)),
            (idx(1, 0, MAGNETIC), idx(3, 0, MAGNETIC))]
    quad22 = [(idx(2, -2, ELECTRIC),) * 2, (idx(2, +2, ELECTRIC),) * 2]
    return dict(m33_11=m33_11, m33_33=m33_33, dipole=dipole,
                even_m=even, quad22=quad22)


def visibility_check(fm, ifreq, B_full, direction=DIRECTION_DEFAULT,
                     oblique_angles="campaign", eps=1e-6, verbose=True):
    """Verify the doc-par.-4 theta=0 visibility claims (module docstring).

    Uses base T = P(T_ref[ifreq]) (projected with the full basis, so the
    mod-4 m-grading of the resolvent is exact and the theta=0 darkness of
    even-m entries is structural, not noise-limited).

    Returns dict with per-class sensitivity tables at theta=0-only and at
    the oblique set, sub-basis Jacobian column norms, and pass/fail of:
      claim (i):  min(max-sens of m33_11, m33_33 classes at theta=0)
                  > max-sens of the dipole class at theta=0;
      claim (ii): even-m max-sens at theta=0 <= 1e-6 * (its oblique value)
                  and <= 1e-6 absolute.
    """
    modes = fm.modes
    T_base = par.unpack(par.pack(fm.data.T[ifreq], B_full), B_full)
    sets = _named_entry_sets(modes)
    normal = [0]
    obl = resolve_angles(oblique_angles)

    out = dict(ifreq=int(ifreq), lam_um=float(fm.lam_um[ifreq]),
               classes={}, direction=int(direction))
    for name, entries in sets.items():
        _, s0 = entry_sensitivity(fm, ifreq, entries, normal, T_base,
                                  direction, eps)
        _, s1 = entry_sensitivity(fm, ifreq, entries, obl, T_base,
                                  direction, eps)
        out["classes"][name] = dict(
            entries=entries,
            max_normal=s0["max_abs"], max_oblique=s1["max_abs"],
            class_max_normal=float(s0["max_abs"].max()),
            class_max_oblique=float(s1["max_abs"].max()))

    # sub-basis Jacobian column norms (the "J columns" view)
    colnorms = {}
    for name in ("m33_11", "m33_33", "even_m"):
        B_sub, _ = subspace_basis_for_entries(B_full, sets[name])
        Jn = jacobian(fm, ifreq, B_sub, normal, T_offset=T_base,
                      direction=direction)
        Jo = jacobian(fm, ifreq, B_sub, obl, T_offset=T_base,
                      direction=direction)
        colnorms[name] = dict(
            n_sub=B_sub.shape[0],
            normal=np.linalg.norm(Jn, axis=0),
            oblique=np.linalg.norm(Jo, axis=0))
    out["colnorms"] = colnorms

    c = out["classes"]
    claim1 = (min(c["m33_11"]["class_max_normal"],
                  c["m33_33"]["class_max_normal"])
              > c["dipole"]["class_max_normal"])
    ev0 = c["even_m"]["class_max_normal"]
    ev1 = c["even_m"]["class_max_oblique"]
    claim2 = (ev0 <= 1e-6 * ev1) and (ev0 <= 1e-6)
    out["claim1_pass"] = bool(claim1)
    out["claim2_pass"] = bool(claim2)

    if verbose:
        print("--- theta=0 visibility verification (doc par. 4) at "
              "ifreq %d, lam = %.2f um ---" % (ifreq, fm.lam_um[ifreq]))
        print("  per-entry |dS/dT| (max over complex observables), "
              "base = P(T_ref), direction = %+d:" % direction)
        for name in ("m33_11", "m33_33", "dipole", "even_m", "quad22"):
            d = c[name]
            labels = ", ".join(entry_label(modes, i, j)
                               for i, j in d["entries"][:4])
            print("    %-7s theta=0: max %.3e | oblique(%s): max %.3e   "
                  "[%s%s]" % (name, d["class_max_normal"],
                              str(oblique_angles), d["class_max_oblique"],
                              labels,
                              ", ..." if len(d["entries"]) > 4 else ""))
        for name, d in colnorms.items():
            print("    sub-basis %-7s (%d dirs) |J col| theta=0: max "
                  "%.3e | oblique: max %.3e"
                  % (name, d["n_sub"], d["normal"].max(),
                     d["oblique"].max()))
        print("  [%s] claim (i): m33 content visible at theta=0 "
              "(m33_11 max %.1f, m33_33 max %.1f vs dipole max %.1f)"
              % ("PASS" if claim1 else "FAIL",
                 c["m33_11"]["class_max_normal"],
                 c["m33_33"]["class_max_normal"],
                 c["dipole"]["class_max_normal"]))
        print("  [%s] claim (ii): even-m dark at theta=0 (max %.3e), "
              "lights up oblique (max %.3e)"
              % ("PASS" if claim2 else "FAIL", ev0, ev1))
    return out


# ---------------------------------------------------------------- figures

def _mode_ticklabels(modes):
    return [mode_label(modes, i) for i in range(modes.n)]


def plot_heatmap(H, modes, mask=None, path=HEATMAP_DEFAULT, title=""):
    """Labeled log-scale heatmap of H with bright entries outlined."""
    H = np.asarray(H, dtype=float)
    hmax = H.max() if H.max() > 0 else 1.0
    floor = hmax * 1e-8
    Hp = np.maximum(H, hmax * 1e-12)
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    im = ax.imshow(Hp, norm=LogNorm(vmin=floor, vmax=hmax),
                   cmap="viridis", interpolation="nearest")
    labels = _mode_ticklabels(modes)
    ax.set_xticks(range(modes.n))
    ax.set_yticks(range(modes.n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("column mode (l, m, pol)  [incident side]")
    ax.set_ylabel("row mode (l, m, pol)  [scattered side]")
    if mask is not None:
        for i, j in np.argwhere(mask):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                                   fill=False, edgecolor="red", lw=1.0))
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("H[mu,nu] = sum_k res_k |B_k[mu,nu]|^2   "
                 "(log; floor = max*1e-8)")
    ax.set_title(title or "Observability heatmap (bright entries outlined "
                          "in red)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_spectrum(s, lam, path=SPECTRUM_DEFAULT, title="", n_above=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.semilogy(np.arange(1, len(s) + 1), s, "o-", ms=3.5,
                label="singular values of J")
    ax.axhline(lam, color="crimson", ls="--",
               label="noise-referenced lambda = %.2e" % lam)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value  [dS per unit t]")
    if n_above is not None:
        ax.annotate("%d of %d above lambda" % (n_above, len(s)),
                    xy=(0.98, 0.95), xycoords="axes fraction",
                    ha="right", va="top")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title(title or "Jacobian singular-value spectrum")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ------------------------------------------------------------ orchestrator

def build_bases(fm, verify_numeric=False):
    """(B_full, meta, B_bright, bright_info) for fm's mode basis."""
    B, meta = par.build_c4v_reciprocity_basis(fm.modes,
                                              verify_numeric=verify_numeric)
    B_b, info = par.bright_orbit_basis(B, fm.data.T, threshold=1e-3,
                                       modes=fm.modes)
    return B, meta, B_b, info


def analyze(fm, ifreq, basis="full", angles="campaign",
            sigma=SIGMA_PLACEHOLDER, direction=DIRECTION_DEFAULT,
            B=None, bright_mask=None, heatmap_path=None, spectrum_path=None,
            npz_tag=None, do_visibility=True, B_full_for_vis=None,
            verbose=True):
    """Full observability analysis at one frequency; saves figures + npz.

    basis: 'bright' (10 complex dirs) or 'full' (68); angles: ANGLE_SETS
    name or index list.  B / bright_mask / B_full_for_vis may be passed to
    reuse prebuilt bases.  Figure paths default to the doc-named
    results/observability_heatmap.png / observability_spectrum.png.
    Returns dict (J, svd dict, res_c, H, paths, visibility)."""
    angle_list = resolve_angles(angles)
    angles_name = angles if isinstance(angles, str) else \
        ",".join(str(a) for a in angle_list)
    if B is None or bright_mask is None or (do_visibility
                                            and B_full_for_vis is None):
        B_full, _, B_b, info = build_bases(fm)
        if B is None:
            B = B_full if basis == "full" else B_b
        if bright_mask is None:
            bright_mask = info["mask"]
        if B_full_for_vis is None:
            B_full_for_vis = B_full
    nb = np.asarray(B).shape[0]

    t0 = time.time()
    J = jacobian(fm, ifreq, B, angle_list, direction=direction)
    t_jac = time.time() - t0
    holo = holomorphy_residual(J)
    sv = svd_resolution(J, sigma=sigma)
    res_c = fold_complex(sv["res"])
    H = entry_heatmap(res_c, B)

    tag = npz_tag or ("ifreq%d_%s_%s" % (ifreq, basis, angles_name))
    hp = heatmap_path or HEATMAP_DEFAULT
    sp = spectrum_path or SPECTRUM_DEFAULT
    title = ("Observability heatmap: ifreq %d (lam %.2f um), basis %s "
             "(%d complex dirs), angles %s, sigma %.1e (placeholder)"
             % (ifreq, fm.lam_um[ifreq], basis, nb, angles_name, sigma))
    plot_heatmap(H, fm.modes, bright_mask, hp, title)
    plot_spectrum(sv["s"], sv["lam"], sp,
                  "SV spectrum: ifreq %d, basis %s, angles %s"
                  % (ifreq, basis, angles_name), sv["n_above"])

    vis = None
    if do_visibility:
        vis = visibility_check(fm, ifreq, B_full_for_vis, direction,
                               verbose=verbose)

    npz_path = os.path.join(RESULTS_DIR, "observability_%s.npz" % tag)
    np.savez(npz_path, J=J, s=sv["s"], V=sv["V"], res=sv["res"],
             res_c=res_c, H=H, lam=sv["lam"], sigma=sigma,
             n_above=sv["n_above"], null_idx=sv["null_idx"],
             holomorphy_resid=holo, angle_indices=np.array(angle_list),
             basis=basis, ifreq=ifreq, direction=direction,
             bright_mask=bright_mask, t_jacobian_s=t_jac)

    if verbose:
        s = sv["s"]
        print("--- observability: ifreq %d (lam %.2f um), basis %s "
              "(%d complex = %d real params), angles %s (%d angles, %d "
              "packed obs) ---" % (ifreq, fm.lam_um[ifreq], basis, nb,
                                   2 * nb, angles_name, len(angle_list),
                                   J.shape[0]))
        print("  Jacobian: central FD, %.2f s (%.1f ms per column); "
              "holomorphy resid %.2e (FD noise level)"
              % (t_jac, 1e3 * t_jac / (2 * nb), holo))
        print("  SV spectrum: max %.3e, min %.3e, cond %.2e"
              % (s[0], s[-1], s[0] / s[-1] if s[-1] > 0 else np.inf))
        print("  lambda = sigma/sqrt(2) = %.3e (sigma = %.1e PLACEHOLDER "
              "until the par.-7 complex closure); %d of %d SVs above; "
              "%d real param directions with res < %.2f"
              % (sv["lam"], sigma, sv["n_above"], len(s),
                 len(sv["null_idx"]), sv["null_thresh"]))
        dark = np.nonzero(res_c < sv["null_thresh"])[0]
        if len(dark):
            print("  dark COMPLEX directions (res_c < %.2f):"
                  % sv["null_thresh"])
            for k in dark:
                amax = np.unravel_index(np.abs(B[k]).argmax(), B[k].shape)
                print("    k=%2d res_c=%.3f  dominant entry %s (|B|=%.2f)"
                      % (k, res_c[k],
                         entry_label(fm.modes, amax[0], amax[1]),
                         abs(B[k][amax])))
        else:
            print("  no dark complex directions (all res_c >= %.2f)"
                  % sv["null_thresh"])
        print("  figures: %s ; %s" % (hp, sp))
        print("  arrays:  %s" % npz_path)

    return dict(J=J, svd=sv, res_c=res_c, H=H, heatmap_path=hp,
                spectrum_path=sp, npz_path=npz_path, visibility=vis,
                holomorphy_resid=holo, t_jacobian_s=t_jac, basis=basis,
                angle_indices=angle_list, tag=tag)


# ----------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Observability map (doc par. 4 deliverable)")
    ap.add_argument("--ifreq", type=int, default=32)
    ap.add_argument("--basis", choices=("bright", "full"), default="full")
    ap.add_argument("--angles", default="campaign",
                    help="campaign | normal-only | starter | all17 | "
                         "comma list of indices 0-16")
    ap.add_argument("--sigma", type=float, default=SIGMA_PLACEHOLDER,
                    help="per-complex-observable noise floor "
                         "(PLACEHOLDER 3e-3 until the par.-7 closure)")
    ap.add_argument("--direction", type=int, choices=(-1, 1), default=-1)
    ap.add_argument("--no-visibility", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    from tmatrix.retrieval.forward import ForwardModel
    fm = ForwardModel()
    if args.ifreq not in fm.available_freqs:
        raise SystemExit("ifreq %d has no cached angles; available: %s"
                         % (args.ifreq, fm.available_freqs))
    analyze(fm, args.ifreq, basis=args.basis, angles=args.angles,
            sigma=args.sigma, direction=args.direction,
            do_visibility=not args.no_visibility, npz_tag=args.tag)


if __name__ == "__main__":
    main()
