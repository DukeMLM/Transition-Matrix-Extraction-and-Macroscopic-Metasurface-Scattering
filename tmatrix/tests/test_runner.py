"""Offline tests for the configuration-driven runner and the merge
function (no CST required)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cst_tmatrix import vswf
from cst_tmatrix.runner import (load_config, frequency_bands,
                                paper_preflight, extraction_config)
from cst_tmatrix.storage import (save_tmatrix, load_tmatrix,
                                 merge_tmatrix_files)


def write_config(tmp_path, cfg_dict):
    # an empty file stands in for the CST template: load_config checks
    # only its existence when no CST session is involved
    tpl = tmp_path / "geom.cst"
    tpl.write_bytes(b"")
    cfg_dict = dict(cfg_dict)
    cfg_dict.setdefault("template", str(tpl))
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg_dict), encoding="utf-8")
    return path


GOOD = {
    "_comment": "underscore keys must be ignored",
    "name": "unit_test_cell",
    "r_circ_um": 3.6,
    "frequencies": {"min_thz": 10.0, "max_thz": 34.0, "step_thz": 1.0,
                    "range_edges_thz": [10.0, 20.0, 34.0]},
    "extraction": {"lmax": 5, "symmetry_n_fold": 4,
                   "symmetry_mirror_planes_deg": [0.0, 45.0, 90.0, 135.0]},
    "mesh": {"steps_per_wave": [20, 8]},
    "solver": {"max_cpus": 6},
}


def test_good_config_parses_splits_and_preflights(tmp_path):
    cfg = load_config(write_config(tmp_path, GOOD))
    bands = frequency_bands(cfg)
    assert [len(fb) for _t, fb in bands] == [11, 14]
    assert bands[0][1][0] == pytest.approx(10e12)
    assert bands[1][1][0] == pytest.approx(21e12)   # no duplicate at 20
    mfs = paper_preflight(cfg, bands)               # must not raise
    assert len(mfs) == 2 and all(mf > 1 for _t, mf in mfs)
    ex = extraction_config(cfg)
    assert ex.symmetry_n_fold == 4 and ex.lmax == 5
    assert cfg["max_cpus"] == 6


def test_missing_key_is_a_readable_error(tmp_path):
    bad = {k: v for k, v in GOOD.items() if k != "r_circ_um"}
    with pytest.raises(SystemExit, match="r_circ_um"):
        load_config(write_config(tmp_path, bad))


def test_invalid_json_is_a_readable_error(tmp_path):
    tpl = tmp_path / "geom.cst"
    tpl.write_bytes(b"")
    path = tmp_path / "run.json"
    path.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        load_config(path)


def test_missing_template_is_a_readable_error(tmp_path):
    cfg_path = write_config(tmp_path, GOOD)
    (tmp_path / "geom.cst").unlink()
    with pytest.raises(SystemExit, match="template"):
        load_config(cfg_path)


def synthetic_file(path, freqs_hz, lmax, seed):
    rng = np.random.default_rng(seed)
    N = vswf.n_modes(lmax)
    T = (rng.normal(size=(len(freqs_hz), N, N))
         + 1j * rng.normal(size=(len(freqs_hz), N, N)))
    save_tmatrix(path, np.asarray(freqs_hz, dtype=float), T, lmax,
                 name="synthetic",
                 diagnostics={"residual": np.zeros(len(freqs_hz))})
    return T


def test_merge_concatenates_and_verifies(tmp_path):
    a = tmp_path / "a.tmat.h5"
    b = tmp_path / "b.tmat.h5"
    synthetic_file(a, [10e12, 11e12], lmax=1, seed=0)
    synthetic_file(b, [12e12, 13e12, 14e12], lmax=1, seed=1)
    out = tmp_path / "merged.tmat.h5"
    info = merge_tmatrix_files(out, [b, a])     # order-independent
    assert info["n_freq"] == 5 and info["n_bands"] == 2
    m = load_tmatrix(out)
    assert np.allclose(m["frequencies"],
                       [10e12, 11e12, 12e12, 13e12, 14e12])
    assert np.allclose(m["tmatrix"][:2], load_tmatrix(a)["tmatrix"])
    assert np.allclose(m["tmatrix"][2:], load_tmatrix(b)["tmatrix"])


def test_merge_refuses_lmax_mismatch_and_overlap(tmp_path):
    a = tmp_path / "a.tmat.h5"
    b = tmp_path / "b.tmat.h5"
    synthetic_file(a, [10e12, 11e12], lmax=1, seed=0)
    synthetic_file(b, [12e12, 13e12], lmax=2, seed=1)
    with pytest.raises(ValueError, match="lmax"):
        merge_tmatrix_files(tmp_path / "o1.h5", [a, b])
    c = tmp_path / "c.tmat.h5"
    synthetic_file(c, [11e12, 12e12], lmax=1, seed=2)   # overlaps a
    with pytest.raises(ValueError, match="overlap"):
        merge_tmatrix_files(tmp_path / "o2.h5", [a, c])
