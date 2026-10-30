"""Candidate physical sizes; no accepted route or brand assignment is changed.

Grounding and source precedence: .sucoder/ledger-rms-size-recovery.md.
TOTAL_SIZE candidates are evidence for review, not an automatic new vintage.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from crosswalk_eval import _sha

NIST_SOURCE = "https://www.nist.gov/document/2026-nist-handbook-44-appendix-c"
GRAMS_PER_OUNCE = 28.349523125
MASS_FACTORS = {"OZ": 1., "OUNCE": 1., "OUNCES": 1.,
                "POUND": 16., "POUNDS": 16.,
                "GRAM": 1. / GRAMS_PER_OUNCE, "GRAMS": 1. / GRAMS_PER_OUNCE}
SIZE_PATTERN = re.compile(r"\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s+(OZ|OUNCES?|POUNDS?|GRAMS?)\s*", re.I)


def mass_ounces(amount, unit):
    """Only explicit positive finite package mass; counts remain unknown."""
    if isinstance(amount, (bool, np.bool_)):
        return np.nan
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return np.nan
    factor = MASS_FACTORS.get(str(unit).strip().upper())
    if factor is None or not math.isfinite(value) or value <= 0:
        return np.nan
    value *= factor
    return value if math.isfinite(value) and value > 0 else np.nan


def total_size_ounces(text):
    """Reject ranges, counts and pack expressions rather than infer a size."""
    if not isinstance(text, str):
        return np.nan
    match = SIZE_PATTERN.fullmatch(text)
    return mass_ounces(*match.groups()) if match else np.nan


def candidate_map(routes, attributes):
    """Same-year UPC metadata; original recorded mass always has precedence."""
    if routes.duplicated(["year", "upc"]).any():
        raise ValueError("duplicate route year/upc")
    if attributes.duplicated(["year", "upc"]).any():
        raise ValueError("duplicate attribute year/upc")
    out = routes.merge(attributes[["year", "upc", "TOTAL_SIZE"]],
                       on=["year", "upc"], how="left", validate="one_to_one")
    primary = np.array([mass_ounces(a, u) for a, u in
                        zip(out.size1_amount, out.size1_units)])
    attr = np.array([total_size_ounces(s) for s in out.TOTAL_SIZE])
    recorded_oz = out.size1_units.fillna("").astype(str).str.upper().str.strip().eq("OZ")
    out["recorded_oz"] = np.where(recorded_oz, primary, np.nan)
    out["normalized_recorded_mass_oz"] = primary
    out["attribute_total_size_oz"] = attr
    out["candidate_oz"] = np.where(np.isfinite(primary), primary, attr)
    origin = np.full(len(out), "unresolved", dtype=object)
    origin[np.isfinite(attr)] = "same_year_TOTAL_SIZE_candidate"
    origin[np.isfinite(primary)] = "recorded_mass_conversion"
    origin[recorded_oz & np.isfinite(primary)] = "recorded_OZ"
    out["size_origin"] = origin
    both = np.isfinite(primary) & np.isfinite(attr)
    out["both_mass_sources"] = both
    out["source_abs_gap_oz"] = np.where(both, np.abs(primary - attr), np.nan)
    out["source_relative_gap"] = np.divide(np.abs(primary - attr), primary,
        out=np.full(len(out), np.nan), where=both)
    return out


def build(routes_path, rms_dir, output):
    os.umask(0o027)
    routes_path, rms_dir, output = Path(routes_path), Path(rms_dir), Path(output)
    routes = pd.read_parquet(routes_path)
    inputs, frames = {}, []
    for year in range(2021, 2025):
        path = rms_dir / str(year) / "productattributes_1344.parquet"
        frame = pd.read_parquet(path, columns=["upc", "TOTAL_SIZE"])
        frame["year"] = year
        frames.append(frame)
        inputs[str(year)] = dict(path=str(path.resolve()), sha256=_sha(path))
    result = candidate_map(routes, pd.concat(frames, ignore_index=True))
    old_gap = result.recorded_oz.isna() & result.dollars.gt(0)
    gaps = result.loc[old_gap]
    summary = gaps.groupby(["year", "size_origin"]).agg(
        upcs=("upc", "nunique"), pre_dma_filter_revenue=("dollars", "sum"))
    overlap = result.loc[result.both_mass_sources & result.dollars.gt(0)]
    # These are descriptive discrepancies, not an acceptance tolerance.
    agreement = {str(year): dict(upcs=len(group),
        abs_gap_quantiles={str(q): float(v) for q, v in group.source_abs_gap_oz.quantile(
            [0, .5, .9, .99, 1]).items()},
        relative_gap_quantiles={str(q): float(v) for q, v in group.source_relative_gap.quantile(
            [0, .5, .9, .99, 1]).items()}) for year, group in overlap.groupby("year")}
    output.mkdir(parents=True, exist_ok=False)
    result.to_parquet(output / "size_candidates.parquet", index=False)
    summary.to_parquet(output / "gap_sources.parquet")
    record = dict(status="candidate_metadata_only", producer=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), script_sha256=_sha(__file__),
        route_input=dict(path=str(routes_path.resolve()), sha256=_sha(routes_path)),
        attribute_inputs=inputs, conversion_reference=NIST_SOURCE,
        grams_per_ounce=GRAMS_PER_OUNCE, ounces_per_pound=16,
        universe="archived UPC dollars before DMA and final time filters",
        original_gap_upc_years=len(gaps),
        unresolved_upc_years=int(gaps.candidate_oz.isna().sum()),
        unresolved_pre_dma_filter_revenue=float(gaps.loc[gaps.candidate_oz.isna(), "dollars"].sum()),
        recovered_pre_dma_filter_revenue=float(gaps.loc[gaps.candidate_oz.notna(), "dollars"].sum()),
        source_agreement=agreement, argv=os.sys.argv)
    (output / "provenance.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(summary.to_string(), flush=True)
    print(json.dumps(record, indent=2), flush=True)
    return result, record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--rms-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.routes, args.rms_dir, args.output)
