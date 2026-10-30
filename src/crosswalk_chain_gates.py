"""Read-only acceptance checks shared by the crosswalk Slurm stages."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _routes_by_upc(path):
    rows = pd.read_parquet(path, columns=["year", "upc", "brand_code_uc", "dollars"])
    if rows.duplicated(["year", "upc"]).any():
        raise ValueError(f"duplicate RMS (year, upc) keys in {path}")
    return rows.groupby(["year", "upc"]).agg(
        j=("brand_code_uc", "first"), dollars=("dollars", "sum"))


def check_rms_route_alignment(new_path, accepted_path, eval_path):
    """Require the build and Run 1 evaluation to cover the same UPC-years."""
    new = _routes_by_upc(new_path)
    accepted = _routes_by_upc(accepted_path)
    evaluated = pd.read_parquet(
        eval_path, columns=["year", "upc", "new_j"]
    ).set_index(["year", "upc"]).sort_index()
    if not evaluated.index.is_unique:
        raise ValueError("Run 1 evaluation has duplicate (year, upc) keys")
    if not new.index.equals(accepted.index):
        raise ValueError("new and accepted RMS UPC-year keys differ")
    if not new.index.equals(evaluated.index):
        raise ValueError("new RMS build and Run 1 evaluation UPC-year keys differ")
    if not np.allclose(new["dollars"], accepted["dollars"], rtol=1e-9,
                       atol=0, equal_nan=False):
        raise ValueError("new and accepted RMS UPC-year dollars differ")
    same_j = (new["j"].eq(evaluated["new_j"])
              | (new["j"].isna() & evaluated["new_j"].isna()))
    if not same_j.fillna(False).all():
        raise ValueError("new RMS build j differs from evaluated new_j")
    return len(new)


def require_current_law(provenance, *, source, panel_path, panel_sha,
                        producer_commit,
                        expected_vintage="crosswalk-2026-10-19"):
    """Reject a stale, wrong-vintage or unconverged B5 law before summarizing."""
    spec = provenance.get("spec") or {}
    data = provenance.get("data_provenance") or {}
    panel = (data.get("files") or {}).get("market_panel") or {}
    checks = {
        "embedded source": spec.get("source") == source,
        "data source": data.get("source") == source,
        "data vintage": data.get("vintage") == expected_vintage,
        "market panel path": (bool(panel.get("path")) and
                              os.path.realpath(panel["path"]) == os.path.realpath(panel_path)),
        "market panel sha256": bool(panel_sha) and panel.get("sha256") == panel_sha,
        "producer commit": provenance.get("producer_commit") == producer_commit,
        "optimizer convergence": (provenance.get("inference") or {}).get("converged") is True,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{source} law failed B5 gate: {', '.join(failed)}")
