"""Physical-constraint repair (denoising) of stored T-matrices.

Motivation, measured on the shipped tmat.h5 files (2026-08-17): extraction
error grows with multipole order -- the per-l diagonal-block norms of T decay
and then RISE with l at every stored frequency (a noise floor), the stored
reciprocity diagnostic sits at 1-18%, and half the singular values of
S = I + 2T exceed 1 (unphysical gain).  The periodic solve
a = (I - W T)^{-1} a_inc amplifies exactly those noisy high-l rows by
cond(I - W T), which reaches 3e6 on the subwavelength lattice (see
run_case.py's module docstring): raising --lmax 3 -> 5 made the S-parameter
error vs CST ~3x WORSE on the a,d;b,c benchmark.

Two exact constraints on any scatterer made of passive, reciprocal media are
available to project part of that noise out without touching the physics:

  reciprocity   T_{p l m, p' l' m'} = (-1)^{m+m'} T_{p' l' (-m'), p l (-m)}
                (parity basis of vswf.py; the map is convention-gated against
                 an analytic treams cluster by verify_denoise.py)
  passivity     every singular value of S = I + 2T is <= 1
                (equality for lossless; gated against a lossless treams
                 Mie sphere by verify_denoise.py)

enforce_reciprocity is the orthogonal projection onto the reciprocal
subspace (average of T with its reciprocal image).  enforce_passivity clips
the singular values of S at 1 -- the nearest passive S in the Frobenius
norm.  The clip commutes with the reciprocity symmetry (the reciprocal
image of an SVD is an SVD of the image with the same singular values), so
applying reciprocity first and passivity second leaves BOTH satisfied.
"""
import numpy as np


def reciprocity_permutation(modes):
    """(perm, sign): perm[i] = index of (l_i, -m_i, pol_i); sign[i] = (-1)^m_i.

    Raises if the basis is not closed under m -> -m (a truncated basis with
    whole l shells always is).
    """
    perm = np.empty(modes.n, dtype=int)
    for i in range(modes.n):
        j = modes.index(int(modes.l[i]), -int(modes.m[i]), int(modes.pol[i]))
        if j is None:
            raise ValueError(
                f"mode basis is not closed under m -> -m at "
                f"(l={modes.l[i]}, m={modes.m[i]}, pol={modes.pol[i]})")
        perm[i] = j
    sign = np.where(modes.m % 2 == 0, 1.0, -1.0)
    return perm, sign


def reciprocal_image(T, modes):
    """R(T)_{ij} = (-1)^{m_i+m_j} T_{neg(j), neg(i)} with neg the m -> -m
    index map.  Reciprocal T-matrices are exactly the fixed points of R,
    and R is a linear involution and an isometry.  T: (..., n, n)."""
    perm, sign = reciprocity_permutation(modes)
    return sign[:, None] * sign[None, :] * np.swapaxes(
        T[..., perm, :][..., :, perm], -1, -2)


def reciprocity_residual(T, modes):
    """||T - R(T)||_F / ||T||_F over the trailing two axes."""
    d = np.linalg.norm(T - reciprocal_image(T, modes), axis=(-2, -1))
    nT = np.linalg.norm(T, axis=(-2, -1))
    return d / np.where(nT > 0, nT, 1.0)


def enforce_reciprocity(T, modes):
    """Orthogonal projection onto the reciprocal subspace: (T + R(T)) / 2."""
    return 0.5 * (T + reciprocal_image(T, modes))


def enforce_passivity(T):
    """Clip the singular values of S = I + 2T at 1 and map back to T.

    Passivity of the constituent materials requires every singular value of
    the scattering matrix S = I + 2T to be <= 1 (no incident power ever
    amplified); the clip returns the nearest passive S in the Frobenius
    norm.  Supports stacked input, T: (..., n, n)."""
    T = np.asarray(T)
    eye = np.eye(T.shape[-1])
    U, s, Vh = np.linalg.svd(eye + 2.0 * T)
    S = np.einsum("...ij,...j,...jk->...ik", U, np.minimum(s, 1.0), Vh)
    return 0.5 * (S - eye)


def l_block_norms(T, modes):
    """(..., lmax) Frobenius norms of the diagonal per-l blocks of T.

    Physical multipole content of a subwavelength scatterer decays with l;
    a decay-then-rise profile marks the extraction noise floor."""
    out = []
    for L in range(1, modes.lmax + 1):
        sel = np.nonzero(modes.l == L)[0]
        out.append(np.linalg.norm(T[..., sel[:, None], sel[None, :]],
                                  axis=(-2, -1)))
    return np.stack(out, axis=-1)


def apply_denoise(T, modes, steps):
    """Apply the named repairs in the given order.

    steps: iterable of 'reciprocity' / 'passivity'.  T: (..., n, n)."""
    for s in steps:
        if s == "reciprocity":
            T = enforce_reciprocity(T, modes)
        elif s == "passivity":
            T = enforce_passivity(T)
        else:
            raise ValueError(f"unknown denoise step '{s}' "
                             f"(use 'reciprocity' and/or 'passivity')")
    return T
