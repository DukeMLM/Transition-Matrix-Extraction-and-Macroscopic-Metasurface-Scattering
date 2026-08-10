"""HDF5 storage of extracted T-matrices in the tmat.h5 data format.

Files written by this module follow the community data format for T-matrices
proposed in N. Asadova et al., J. Quant. Spectrosc. Radiat. Transfer 333,
109310 (2025), which is the native exchange format of the multiple-scattering
code treams (D. Beutel et al., Comput. Phys. Commun. 297, 109076 (2024)) and
of the Daphona T-matrix portal.  The essential elements are:

  /tmatrix                 (n_freq, N, N) complex; p = T a in the convention
                           of the format: e^{-i omega t} time dependence,
                           outgoing waves h_l^(1), Jackson-normalized vector
                           spherical waves with the Condon-Shortley phase
  /frequency               (n_freq,) float64, attribute unit = "Hz"
  /modes/l, /modes/m       (N,) integer multipole indices per matrix row
  /modes/polarization      (N,) strings, "electric" (N-type) / "magnetic"
                           (M-type) — parity basis
  /embedding/              relative permittivity and permeability of the
                           surrounding medium (vacuum here)
  /scatterer/              geometry and material metadata
  /computation/            method description, parameters, and the
                           extraction quality diagnostics of this package
  root attributes          name, description, keywords,
                           storage_format_version

Internally the package works in the e^{+j omega t} / h_l^(2) convention with
block mode ordering; `save_tmatrix` converts on write via
`vswf.to_treams_convention` and `vswf.treams_interleaved_order`, and
`load_tmatrix` converts back on read, so all analysis inside this package is
convention-consistent while the files on disk are interoperable.  The
conversion map is verified at the field level by tests/test_storage_treams.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT_VERSION = "v1"

CONVENTION_NOTE = (
    "T-matrix stored in the tmat.h5 convention: time dependence "
    "exp(-i omega t), outgoing radial functions h_l^(1), Jackson-normalized "
    "vector spherical waves with Condon-Shortley phase, parity basis; "
    "p = T a with modes as listed in /modes."
)


def save_tmatrix(path, frequencies_hz, T_internal, lmax, *, name: str,
                 description: str = "", keywords: str = "",
                 scatterer: dict | None = None,
                 computation: dict | None = None,
                 diagnostics: dict | None = None,
                 embedding: dict | None = None):
    """Write a T-matrix library entry in the tmat.h5 format.

    Parameters
    ----------
    frequencies_hz : (n_freq,) frequencies in Hz.
    T_internal : (n_freq, N, N) or (N, N) complex T-matrix in the PACKAGE
        convention (e^{+j omega t}, h^(2), block ordering); converted on
        write.
    lmax : multipole truncation order.
    name : short dataset name (root attribute, searchable metadata).
    scatterer : optional dict with keys such as "geometry" (dict) and
        "material" (dict with "name", "relative_permittivity",
        "relative_permeability").
    computation : optional dict of method parameters (stored under
        /computation as attributes/datasets).
    diagnostics : optional dict of per-frequency arrays (residual, cond,
        s_max_sv, unitarity, reciprocity, incident_deviation_max), stored
        under /computation/analysis.
    embedding : optional dict, default vacuum
        {"name": "Vacuum", "relative_permittivity": 1.0,
         "relative_permeability": 1.0}.
    """
    import h5py
    from .vswf import (n_modes, to_treams_convention,
                       treams_interleaved_order)

    T_internal = np.asarray(T_internal)
    frequencies_hz = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    if T_internal.ndim == 2:
        T_internal = T_internal[None, :, :]
    N = n_modes(lmax)
    if T_internal.shape != (frequencies_hz.size, N, N):
        raise ValueError(
            f"T shape {T_internal.shape} != ({frequencies_hz.size}, {N}, {N})")

    # convention conversion + interleaved (l, m; electric, magnetic) ordering
    T_out = to_treams_convention(T_internal, lmax)
    perm, l_arr, m_arr, pol_labels = treams_interleaved_order(lmax)
    T_out = T_out[:, perm][:, :, perm]

    emb = {"name": "Vacuum", "relative_permittivity": 1.0,
           "relative_permeability": 1.0, **(embedding or {})}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    str_t = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["name"] = name
        h.attrs["description"] = description or CONVENTION_NOTE
        h.attrs["keywords"] = keywords
        h.attrs["storage_format_version"] = FORMAT_VERSION
        h.attrs["created_with"] = "cst_tmatrix (CST Studio Suite driver)"

        h.create_dataset("tmatrix", data=T_out.astype(np.complex128))
        d = h.create_dataset("frequency", data=frequencies_hz)
        d.attrs["unit"] = "Hz"

        g = h.create_group("modes")
        g.create_dataset("l", data=l_arr)
        g.create_dataset("m", data=m_arr)
        g.create_dataset("polarization", data=pol_labels, dtype=str_t)

        g = h.create_group("embedding")
        g.attrs["name"] = str(emb.get("name", "Vacuum"))
        g.create_dataset("relative_permittivity",
                         data=complex(emb["relative_permittivity"]))
        g.create_dataset("relative_permeability",
                         data=complex(emb["relative_permeability"]))

        g = h.create_group("scatterer")
        sc = scatterer or {}
        if "material" in sc:
            gm = g.create_group("material")
            # a bare string ("TiO2 n=2.5") is accepted as the material
            # name: a metadata type mismatch must not discard a completed
            # extraction at the storage step.
            mat = sc["material"]
            if isinstance(mat, str):
                mat = {"name": mat}
            for k, v in mat.items():
                if isinstance(v, str):
                    gm.attrs[k] = v
                else:
                    gm.create_dataset(k, data=np.asarray(v))
        if "geometry" in sc:
            gg = g.create_group("geometry")
            geo = sc["geometry"]
            if isinstance(geo, str):
                geo = {"description": geo}
            # geometry_units: dict {param: unit}, or one string for all
            gu = sc.get("geometry_units", {})
            if isinstance(gu, str):
                gu = {k: gu for k in geo}
            for k, v in geo.items():
                if isinstance(v, str):
                    gg.attrs[k] = v
                else:
                    ds = gg.create_dataset(k, data=np.asarray(v))
                    unit = gu.get(k)
                    if unit:
                        ds.attrs["unit"] = unit
        for k, v in sc.items():
            if k in ("material", "geometry", "geometry_units"):
                continue
            try:
                g.attrs[k] = v if isinstance(v, str) else json.dumps(v)
            except TypeError:
                g.attrs[k] = str(v)

        g = h.create_group("computation")
        g.attrs["software"] = ("cst_tmatrix; CST Studio Suite "
                               "(frequency-domain solver, tetrahedral mesh)")
        g.attrs["method"] = (
            "Plane-wave illumination set; complex E and H exported on a "
            "spherical surface; simultaneous regular/outgoing decomposition "
            "per mode; T from the least-squares solution of F = T A.")
        for k, v in (computation or {}).items():
            try:
                g.attrs[k] = v if isinstance(v, (str, int, float)) \
                    else json.dumps(v, default=str)
            except TypeError:
                g.attrs[k] = str(v)
        if diagnostics:
            ga = g.create_group("analysis")
            ga.attrs["note"] = (
                "Extraction quality diagnostics, one value per frequency, "
                "computed in the package-internal convention (invariant "
                "under the convention conversion).")
            for k, v in diagnostics.items():
                ga.create_dataset(k, data=np.asarray(v))
    return path


def extraction_matches(path, frequencies_hz, lmax: int) -> bool:
    """True iff the tmat.h5 at `path` was extracted on this exact frequency
    grid and lmax.  Used to guard against a coarse-grid and a fine-grid run
    of the same design silently overwriting each other's library entry
    (see extract_tmatrix's overwrite guard in pipeline.py)."""
    import h5py

    path = Path(path)
    if not path.exists():
        return True                # nothing to collide with
    frequencies_hz = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    try:
        with h5py.File(path, "r") as h:
            existing_freq = np.atleast_1d(h["frequency"][...])
            existing_lmax = int(h["computation"].attrs.get("lmax", -1))
    except (OSError, KeyError):
        return False                # unreadable/foreign file: don't clobber silently
    return (existing_lmax == lmax
            and existing_freq.shape == frequencies_hz.shape
            and np.allclose(existing_freq, frequencies_hz, rtol=1e-9))


def merge_tmatrix_files(out_path, input_paths, name: str | None = None):
    """Merge per-band .tmat.h5 extractions of ONE scatterer into a single
    file by frequency-axis concatenation.

    Valid only when every input shares the same lmax (identical mode
    basis) and the frequency ranges do not overlap -- both verified, never
    assumed.  Same-lmax across bands is an interface requirement: the
    per-frequency T(f) blocks of one file must share one mode dimension,
    and a design family stored at one lmax is stackable downstream.
    Diagnostics arrays are concatenated alongside; per-band provenance is
    recorded in the computation metadata.

    Returns a dict with 'n_bands', 'n_freq', 'f_min_hz', 'f_max_hz'.
    """
    out_path = Path(out_path)
    parts = [load_tmatrix(p) for p in input_paths]
    lmax = parts[0]["lmax"]
    if any(d["lmax"] != lmax for d in parts):
        raise ValueError(f"lmax mismatch across inputs: "
                         f"{[d['lmax'] for d in parts]} -- merging different "
                         f"mode bases requires padding; refusing.")
    parts_sorted = sorted(range(len(parts)),
                          key=lambda i: float(np.min(parts[i]["frequencies"])))
    parts = [parts[i] for i in parts_sorted]
    paths_sorted = [Path(input_paths[i]) for i in parts_sorted]
    for a, b in zip(parts[:-1], parts[1:]):
        if float(np.max(a["frequencies"])) >= float(np.min(b["frequencies"])):
            raise ValueError("frequency ranges overlap between inputs -- "
                             "duplicate frequencies would corrupt the axis.")

    freqs = np.concatenate([d["frequencies"] for d in parts])
    T = np.concatenate([d["tmatrix"] for d in parts], axis=0)
    diag_keys = set(parts[0].get("diagnostics", {}) or {})
    if any(set(d.get("diagnostics", {}) or {}) != diag_keys for d in parts):
        raise ValueError("diagnostics keys differ across inputs.")
    diagnostics = {k: np.concatenate(
        [np.asarray(d["diagnostics"][k]) for d in parts])
        for k in sorted(diag_keys)}
    bands_meta = [{"file": p.name,
                   "f_min_hz": float(np.min(d["frequencies"])),
                   "f_max_hz": float(np.max(d["frequencies"])),
                   "n_freq": int(len(d["frequencies"]))}
                  for p, d in zip(paths_sorted, parts)]

    save_tmatrix(
        out_path, freqs, T, lmax,
        name=name or parts[0].get("name", out_path.stem),
        description=(f"merged from {len(parts)} band extractions; identical "
                     f"lmax={lmax} mode basis; per-band monitor/mesh -- see "
                     f"computation.bands"),
        keywords="cst,fem,plane-wave illumination,band-merged",
        computation={"lmax": lmax, "n_modes": T.shape[1],
                     "bands": bands_meta},
        diagnostics=diagnostics)

    # round-trip verification including the band boundaries
    chk = load_tmatrix(out_path)
    assert chk["tmatrix"].shape == T.shape
    assert np.allclose(chk["frequencies"], freqs)
    j = 0
    for d in parts[:-1]:
        j += len(d["frequencies"])
        assert np.allclose(chk["tmatrix"][j - 1], T[j - 1])
        assert np.allclose(chk["tmatrix"][j], T[j])
    return {"n_bands": len(parts), "n_freq": int(len(freqs)),
            "f_min_hz": float(freqs.min()), "f_max_hz": float(freqs.max())}


def load_tmatrix(path):
    """Read a tmat.h5 file and reconstruct the PACKAGE-convention T-matrix.

    The mode tables stored in the file are used explicitly (the format is
    self-describing), so files with any parity-basis mode ordering are
    accepted.  Returns a dict with keys: frequencies (Hz), tmatrix
    (n_freq, N, N; internal convention and block ordering), lmax, modes,
    name, description, diagnostics (may be empty).
    """
    import h5py
    from .vswf import (block_index, from_treams_convention, n_modes)

    out = {}
    with h5py.File(path, "r") as h:
        T_file = h["tmatrix"][...]
        if T_file.ndim == 2:
            T_file = T_file[None, :, :]
        freq = h["frequency"][...]
        unit = h["frequency"].attrs.get("unit", "Hz")
        scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9,
                 "THz": 1e12}.get(str(unit), None)
        if scale is None:
            raise ValueError(f"Unsupported frequency unit in file: {unit}")
        out["frequencies"] = np.atleast_1d(freq) * scale

        l_arr = h["modes/l"][...]
        m_arr = h["modes/m"][...]
        pol_raw = h["modes/polarization"][...]
        pol = [p.decode() if isinstance(p, bytes) else str(p)
               for p in pol_raw]
        lmax = int(np.max(l_arr))
        N = n_modes(lmax)
        if T_file.shape[-1] != N:
            raise ValueError(
                f"File has {T_file.shape[-1]} modes; expected {N} for "
                f"lmax={lmax} (monopole or partial mode sets not supported)")
        half = N // 2
        # permutation: file row -> internal index
        to_internal = np.empty(N, dtype=int)
        for i, (l, m, p) in enumerate(zip(l_arr, m_arr, pol)):
            if p == "magnetic":
                to_internal[i] = block_index(int(l), int(m))
            elif p == "electric":
                to_internal[i] = half + block_index(int(l), int(m))
            else:
                raise ValueError(
                    f"Unsupported polarization label '{p}' (parity basis "
                    f"expected: 'electric'/'magnetic')")
        T_blocked = np.empty_like(T_file)
        T_blocked[:, to_internal[:, None], to_internal[None, :]] = T_file
        out["tmatrix"] = from_treams_convention(T_blocked, lmax)
        out["lmax"] = lmax
        out["modes"] = {"l": l_arr, "m": m_arr, "polarization": pol}
        out["name"] = str(h.attrs.get("name", ""))
        out["description"] = str(h.attrs.get("description", ""))
        out["diagnostics"] = {}
        if "computation/analysis" in h:
            out["diagnostics"] = {k: h["computation/analysis"][k][...]
                                  for k in h["computation/analysis"]}
        out["computation"] = ({k: h["computation"].attrs[k]
                               for k in h["computation"].attrs}
                              if "computation" in h else {})
    return out
