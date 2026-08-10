"""Locating the CST Studio Suite python libraries.

Two machine-specific paths used to be hardcoded in five modules:

    E:\\cst\\AMD64\\python_cst_libraries    the CST python API (cst.interface)
    D:/Claude/auto_cst                     nir.cst_helpers, the proven VBA
                                           automation stack this repo builds on

Both are now read from the environment, falling back to those values, so the
code runs unchanged on the machine the campaign was measured on and is
portable off it.

Nothing here imports CST at module level: `--plan`, `--extract-only` and every
test must work on a machine with no CST installed.  Call ensure_on_path()
immediately before the `import cst.interface` that needs it.
"""
import os
import sys
from pathlib import Path

#: CST's own python libraries; ships inside the CST installation
CST_PYTHON_LIB = os.environ.get(
    "CST_PYTHON_LIB", r"E:\cst\AMD64\python_cst_libraries")

#: the auto_cst checkout that provides nir.cst_helpers (HistoryBuilder, ...)
AUTO_CST = Path(os.environ.get("AUTO_CST_DIR", r"D:/Claude/auto_cst"))


def ensure_on_path():
    """Put both directories on sys.path, nearest first.  Idempotent."""
    for p in (str(AUTO_CST), CST_PYTHON_LIB):
        if p not in sys.path:
            sys.path.insert(0, p)


def import_cst_interface():
    """Import and return `cst.interface`, with a diagnosis if it is missing."""
    ensure_on_path()
    try:
        import cst.interface as cstint          # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            f"cst.interface is not importable (looked in {CST_PYTHON_LIB}). "
            f"Solving needs CST on this machine; --plan and --extract-only do "
            f"not.  Set CST_PYTHON_LIB if it is installed elsewhere.  "
            f"Original error: {e}") from e
    return cstint


def import_cst_helpers():
    """Import the nir.cst_helpers names the builders use."""
    ensure_on_path()
    try:
        from nir.cst_helpers import (          # noqa: PLC0415
            HistoryBuilder, find_reflection_sparams, get_result_with_data,
            open_results, save_project_at)
    except ImportError as e:
        raise ImportError(
            f"nir.cst_helpers is not importable (looked in {AUTO_CST}). "
            f"Set AUTO_CST_DIR to the auto_cst checkout.  Original error: {e}"
        ) from e
    return dict(HistoryBuilder=HistoryBuilder,
                find_reflection_sparams=find_reflection_sparams,
                get_result_with_data=get_result_with_data,
                open_results=open_results,
                save_project_at=save_project_at)
