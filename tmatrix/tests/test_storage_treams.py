"""Verification of the convention conversion and the tmat.h5 storage layer.

The conversion between the package convention (e^{+j omega t}, h^(2)) and
the tmat.h5 / treams convention (e^{-i omega t}, h^(1)) rests on the
antilinear coefficient map

    C(a)_{l,m} = (-1)^m conj(a_{l,-m})        (per polarization block),

from which T converts as T' = C T C^{-1}, i.e.
T'_{(l,m),(l',m')} = (-1)^{m+m'} conj(T_{(l,-m),(l',-m')}).

Two independent checks make this non-circular:

1. The coefficient map is verified against PHYSICAL FIELDS: for regular
   waves the radial functions are real, so conjugating a field must be the
   same as evaluating the mapped coefficients,
   conj(E[a]) = E[C(a)].  This tests the angular algebra (Condon-Shortley
   parity, the (-1)^m factor, m reversal) using only evaluators that are
   locked by the existing test suite.
2. The matrix formula is verified against the vector map:
   to_treams_convention(T) @ C(a) = C(T @ a) for random T and a.

Also covered: involution property, invariance for spheres, storage round
trip through an actual HDF5 file, and mode-table conformance.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst_tmatrix import vswf
from cst_tmatrix.quadrature import gauss_legendre_sphere, unit_vectors
from cst_tmatrix.storage import save_tmatrix, load_tmatrix
from cst_tmatrix.tmatrix_solve import sphere_reference_T

LMAX = 3
N = vswf.n_modes(LMAX)
HALF = N // 2
rng = np.random.default_rng(11)


def coeff_map(a):
    """C(a)_{l,m} = (-1)^m conj(a_{l,-m}), per polarization block."""
    pol, ns, ms = vswf.mode_list(LMAX)
    out = np.empty_like(a)
    for i, (p, n, m) in enumerate(zip(pol, ns, ms)):
        j = (0 if p == 0 else HALF) + vswf.block_index(n, -m)
        out[i] = (-1.0) ** m * np.conj(a[j])
    return out


def test_coefficient_map_against_physical_fields():
    """conj(E[a]) == E[C(a)] for regular waves (real radial functions)."""
    k = 2 * np.pi / 300e-6
    th, ph, w, _ = gauss_legendre_sphere(LMAX + 2)
    pts = 100e-6 * unit_vectors(th, ph)
    a = rng.normal(size=N) + 1j * rng.normal(size=N)
    E = vswf.evaluate_field(a, LMAX, "regular", k, pts)
    E_mapped = vswf.evaluate_field(coeff_map(a), LMAX, "regular", k, pts)
    assert np.max(np.abs(np.conj(E) - E_mapped)) < 1e-12 * np.max(np.abs(E))


def test_matrix_formula_matches_vector_map():
    T = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Tc = vswf.to_treams_convention(T, LMAX)
    for _ in range(4):
        a = rng.normal(size=N) + 1j * rng.normal(size=N)
        lhs = Tc @ coeff_map(a)
        rhs = coeff_map(T @ a)
        assert np.max(np.abs(lhs - rhs)) < 1e-12 * np.max(np.abs(rhs))


def test_involution_and_sphere_invariance():
    T = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    back = vswf.from_treams_convention(vswf.to_treams_convention(T, LMAX),
                                       LMAX)
    assert np.max(np.abs(back - T)) < 1e-14 * np.max(np.abs(T))
    # a sphere is diagonal and m-independent: conversion = conjugation
    Ts = sphere_reference_T(LMAX, 0.9, 2.0)
    assert np.max(np.abs(vswf.to_treams_convention(Ts, LMAX)
                         - np.conj(Ts))) < 1e-14


def test_storage_round_trip():
    freqs = np.array([0.5e12, 0.75e12])
    T = (rng.normal(size=(2, N, N)) + 1j * rng.normal(size=(2, N, N)))
    diags = {"residual": np.array([1e-3, 2e-3])}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "test_entry.tmat.h5"
        save_tmatrix(p, freqs, T, LMAX, name="round-trip test",
                     scatterer={"geometry": {"shape": "test",
                                             "radius": 40.0},
                                "geometry_units": {"radius": "um"},
                                "material": {"name": "test",
                                             "relative_permittivity": 4.0}},
                     diagnostics=diags)
        d2 = load_tmatrix(p)
    assert np.allclose(d2["frequencies"], freqs)
    assert np.max(np.abs(d2["tmatrix"] - T)) < 1e-13 * np.max(np.abs(T))
    assert d2["lmax"] == LMAX
    assert np.allclose(d2["diagnostics"]["residual"], diags["residual"])


def test_file_layout_conforms():
    """The written file contains the elements required by the format."""
    import h5py
    freqs = np.array([1.0e12])
    T = rng.normal(size=(1, N, N)) + 1j * rng.normal(size=(1, N, N))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "layout.tmat.h5"
        save_tmatrix(p, freqs, T, LMAX, name="layout test")
        with h5py.File(p, "r") as h:
            for key in ("tmatrix", "frequency", "modes/l", "modes/m",
                        "modes/polarization", "embedding", "computation"):
                assert key in h, f"missing {key}"
            assert h["frequency"].attrs["unit"] == "Hz"
            assert h.attrs["storage_format_version"]
            ls = h["modes/l"][...]
            ms = h["modes/m"][...]
            pols = [x.decode() if isinstance(x, bytes) else str(x)
                    for x in h["modes/polarization"][...]]
            assert len(ls) == N
            # interleaved ordering: l ascending, m = -l..l, electric first
            assert ls[0] == 1 and ms[0] == -1
            assert pols[0] == "electric" and pols[1] == "magnetic"
            assert set(pols) == {"electric", "magnetic"}
