"""Build the self-contained tmat.h5 web viewer.

Reads a tmat.h5 file, extracts everything (attributes, modes, geometry,
per-frequency complex T-matrix, extraction diagnostics, full HDF5 tree),
and injects it as JSON into tmat_view_template.html at the __DATA_JSON__
marker.  The output is a single ~1 MB HTML file with no external
dependencies: T-matrix heatmap (log10|T| / phase, frequency slider,
l-block structure, per-cell mode-pair tooltips), response spectrum,
stored diagnostics charts, geometry drawing, mode table, and file tree.

Usage:
    python build_view.py [tmat_h5_path] [out_html]
Defaults to the demo file and tmat_h5_view.html next to this script.
"""
import json
import os
import sys

import h5py
import numpy as np

from tmatrix.paths import BENCHMARK_SINGLE

HERE = os.path.dirname(os.path.abspath(__file__))   # package data (templates)
DEFAULT_H5 = os.path.join(BENCHMARK_SINGLE,
                          "saw_gold_wl15p0025um.tmat.h5")


def clean(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.complexfloating):
        return f"{v.real:g}{v.imag:+g}j"
    if isinstance(v, np.ndarray):
        return [clean(x) for x in v]
    return v if isinstance(v, (int, float, str, bool)) else str(v)


def sig5(x):
    return float(f"{x:.5g}")


def extract(path):
    with h5py.File(path, "r") as f:
        freq = f["frequency"][:]
        lam = 299792458.0 / freq * 1e6
        T = f["tmatrix"][:]
        data = {
            "root_attrs": {k: clean(v) for k, v in f.attrs.items()},
            "computation": {k: clean(v)
                            for k, v in f["computation"].attrs.items()},
            "analysis": {k: [sig5(x)
                             for x in f[f"computation/analysis/{k}"][:]]
                         for k in ["cond", "residual", "reciprocity",
                                   "unitarity", "incident_deviation_max",
                                   "s_max_sv"]},
            "freq_thz": [sig5(x) for x in freq / 1e12],
            "lam_um": [sig5(x) for x in lam],
            "modes": {
                "l": f["modes/l"][:].tolist(),
                "m": f["modes/m"][:].tolist(),
                "pol": [p.decode() if isinstance(p, bytes) else str(p)
                        for p in f["modes/polarization"][:]],
            },
            "geometry": {**{k: sig5(float(f[f"scatterer/geometry/{k}"][()]))
                            for k in ["r", "gap", "w", "w_ring", "thickness"]},
                         **{k: clean(v)
                            for k, v in f["scatterer/geometry"].attrs.items()},
                         **{k: clean(v)
                            for k, v in f["scatterer"].attrs.items()}},
            "material": {**{k: clean(v)
                            for k, v in f["scatterer/material"].attrs.items()},
                         "conductivity_S_per_m": sig5(float(
                             f["scatterer/material/conductivity_S_per_m"][()]))},
            "embedding": {
                "name": clean(f["embedding"].attrs.get("name", "")),
                "eps": clean(f["embedding/relative_permittivity"][()]),
                "mu": clean(f["embedding/relative_permeability"][()]),
            },
            "T_re": [[sig5(x) for x in Ti.real.ravel()] for Ti in T],
            "T_im": [[sig5(x) for x in Ti.imag.ravel()] for Ti in T],
        }
        tree = []

        def visit(name, obj):
            e = {"path": "/" + name,
                 "kind": ("dataset" if isinstance(obj, h5py.Dataset)
                          else "group"),
                 "attrs": {k: clean(v) for k, v in obj.attrs.items()}}
            if isinstance(obj, h5py.Dataset):
                e["shape"] = list(obj.shape)
                e["dtype"] = str(obj.dtype)
            tree.append(e)

        f.visititems(visit)
        data["tree"] = tree
    return data


def main():
    h5_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_H5
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        HERE, "tmat_h5_view.html")
    tpl = open(os.path.join(HERE, "tmat_view_template.html"),
               encoding="utf-8").read()
    assert "__DATA_JSON__" in tpl, "template marker missing"
    payload = json.dumps(extract(h5_path), separators=(",", ":"))
    html = tpl.replace("__DATA_JSON__", payload)
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out} ({len(html)/1e6:.2f} MB) from {h5_path}")


if __name__ == "__main__":
    main()
