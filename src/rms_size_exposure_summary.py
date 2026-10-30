"""Conditional size-coverage scenarios on unchanged adopted market accounting.

These scenarios do not promote candidate fields to an accepted panel, select
brands, or multiply every primary size by multi. See the measurement ledger.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from crosswalk_eval import _sha
from rms_size_market_audit import reconcile_uncovered

SCENARIOS = ("primary_and_catalog", "all_TOTAL_SIZE_candidates", "pack_screened_candidates")


def scenario_masses(candidates, findings):
    out = candidates.copy()
    out[SCENARIOS[0]] = out.followup_mass_oz.where(
        out.followup_mass_source.ne("same_year_TOTAL_SIZE_candidate"))
    out[SCENARIOS[1]] = out.followup_mass_oz
    out[SCENARIOS[2]] = out.followup_mass_oz
    screen = findings["fallback_screen"]
    if screen["future_descriptor_start_dates"]:
        raise ValueError("pack scenario requires temporally supported descriptor records")
    for row in screen["flagged_rows"]:
        take = out.year.eq(row["year"]) & out.upc.eq(row["upc"])
        if take.sum() != 1 or not out.loc[take, "size_origin"].eq("same_year_TOTAL_SIZE_candidate").all():
            raise ValueError("pack screen must bind a unique original TOTAL_SIZE fallback")
        if not np.isclose(out.loc[take, "candidate_oz"].iloc[0], row["candidate_oz"], rtol=0, atol=1e-12):
            raise ValueError("pack evidence and original candidate disagree")
        package = row["descriptor_package_oz"]
        if package is None:
            # Secondary count alone leaves package interpretation unvalidated.
            out.loc[take, SCENARIOS[2]] = np.nan
        else:
            multi = float(row["desc_multi"])
            if multi <= 1 or not np.isclose(package, row["candidate_oz"] * multi, rtol=0, atol=1e-12):
                raise ValueError("explicit pack scenario lacks matching count/mass arithmetic")
            out.loc[take, SCENARIOS[2]] = package
    return out


def market_scenario(details, baseline, candidates, name):
    if candidates.duplicated(["year", "upc"]).any():
        raise ValueError("duplicate candidate year/UPC")
    mass_columns = ["year", "upc", name, "catalog_mass_min_oz", "catalog_mass_max_oz",
                    "catalog_versions", "source_abs_gap_oz"]
    joined = details.merge(candidates[mass_columns], left_on=["file_year", "upc"], right_on=["year", "upc"],
                           how="left", validate="many_to_one", indicator=True)
    if joined._merge.ne("both").any():
        raise ValueError("focused market row lacks candidate metadata")
    joined = joined.loc[joined.category.ne("modeled_45")].copy()
    # Route dollars are a different, pre-DMA universe; always use movement revenue.
    missing = joined.uncovered_rows.gt(0)
    if not joined.loc[missing, "uncovered_rows"].eq(joined.loc[missing, "rows"]).all():
        raise ValueError("mixed original size coverage within a file-year/UPC cell")
    mass = joined[name]
    good = np.isfinite(mass) & mass.gt(0)
    remaining = missing & ~good
    recover = missing & good
    lo, hi = joined.catalog_mass_min_oz, joined.catalog_mass_max_oz
    bounded = remaining & np.isfinite(lo) & np.isfinite(hi) & lo.gt(0) & hi.ge(lo)
    joined["added_ounces"] = np.where(recover, joined.units * mass.fillna(0), 0.)
    joined["recovered_revenue"] = joined.revenue.where(recover, 0.)
    joined["remaining_revenue"] = joined.revenue.where(remaining, 0.)
    joined["remaining_rows"] = joined.uncovered_rows.where(remaining, 0)
    joined["unbounded_rows"] = joined.uncovered_rows.where(remaining & ~bounded, 0)
    joined["range_min_ounces"] = np.where(bounded, joined.units * lo.fillna(0), 0.)
    joined["range_max_ounces"] = np.where(bounded, joined.units * hi.fillna(0), 0.)
    joined["unowned_catalog_revenue"] = joined.revenue.where(joined.catalog_versions.notna(), 0.)
    joined["source_conflict_revenue"] = joined.revenue.where(joined.source_abs_gap_oz.gt(0), 0.)
    columns = ["added_ounces", "recovered_revenue", "remaining_revenue", "remaining_rows",
               "unbounded_rows", "range_min_ounces", "range_max_ounces", "unowned_catalog_revenue",
               "source_conflict_revenue"]
    agg = joined.groupby(["tcode", "m"])[columns].sum()
    if not agg.index.isin(baseline.index).all():
        raise ValueError("unexpected market outside original audit")
    agg = agg.reindex(baseline.index, fill_value=0)
    expected_missing = baseline.residual_revenue - baseline.residual_covered_revenue
    if not np.allclose(agg.recovered_revenue + agg.remaining_revenue, expected_missing, rtol=1e-12, atol=1e-6):
        raise ValueError("scenario changed the residual revenue universe")
    table = baseline[["residual_revenue", "selected_brand_excluded_revenue"]].join(agg)
    table["covered_ounces"] = baseline.residual_covered_ounces + agg.added_ounces
    complete = agg.remaining_rows.eq(0) & table.residual_revenue.gt(0) & table.covered_ounces.gt(0)
    table["conditional_point_price"] = (table.residual_revenue / table.covered_ounces).where(complete)
    qlo, qhi = table.covered_ounces + agg.range_min_ounces, table.covered_ounces + agg.range_max_ounces
    finite_range = agg.unbounded_rows.eq(0) & qlo.gt(0) & qhi.ge(qlo)
    table["catalog_conditional_price_lower"] = (table.residual_revenue / qhi).where(finite_range)
    table["catalog_conditional_price_upper"] = (table.residual_revenue / qlo).where(finite_range)
    table["scenario"] = name
    unknown_keys = joined.loc[remaining, ["file_year", "upc"]].drop_duplicates()
    summary = dict(scenario=name, markets=len(table), conditional_point_price_cells=int(complete.sum()),
        remaining_outside_revenue=float(agg.remaining_revenue.sum()),
        remaining_outside_rows=int(agg.remaining_rows.sum()),
        remaining_outside_upc_years=len(unknown_keys),
        catalog_interval_only_cells=int((finite_range & ~complete).sum()),
        unowned_catalog_outside_revenue=float(agg.unowned_catalog_revenue.sum()),
        full_numeric_price_without_reported_partition_flags=int((complete &
            table.selected_brand_excluded_revenue.eq(0) & agg.unowned_catalog_revenue.eq(0)).sum()))
    return table, summary


def run(scan_dir, consensus_dir, metadata_review, audit_dir, output):
    os.umask(0o027)
    scan_dir, consensus_dir, metadata_review, audit_dir, output = map(Path,
        (scan_dir, consensus_dir, metadata_review, audit_dir, output))
    if output.exists():
        raise FileExistsError(output)
    scan_meta = json.loads((scan_dir / "provenance.json").read_text())
    if scan_meta.get("status") != "complete":
        raise ValueError("focused movement scan is incomplete")
    inputs = {name: dict(path=str(p), sha256=_sha(p)) for name, p in dict(
        upc_markets=scan_dir / "upc_dma_quarter.parquet",
        followup_candidates=consensus_dir / "followup_candidates.parquet",
        review=metadata_review / "findings.json", baseline=audit_dir / "dma_quarter.parquet").items()}
    if inputs["baseline"]["sha256"] != scan_meta["inputs"]["prior_market_table"]["sha256"]:
        raise ValueError("market baseline differs from reconciled scan")
    details = pd.read_parquet(inputs["upc_markets"]["path"])
    baseline = pd.read_parquet(inputs["baseline"]["path"])
    check = reconcile_uncovered(details, baseline)
    findings = json.loads(Path(inputs["review"]["path"]).read_text())
    candidates = scenario_masses(pd.read_parquet(inputs["followup_candidates"]["path"]), findings)
    tables, summaries = [], []
    for name in SCENARIOS:
        table, summary = market_scenario(details, baseline, candidates, name)
        tables.append(table)
        summaries.append(summary)
    output.mkdir(parents=True)
    pd.concat(tables).to_parquet(output / "dma_quarter_scenarios.parquet")
    candidates.to_parquet(output / "mass_scenarios.parquet", index=False)
    # Preserve actual exposure of every focused UPC, including inside conflicts.
    totals = details.groupby(["file_year", "upc", "category"])[["revenue", "units", "rows"]].sum().reset_index()
    metadata = candidates.rename(columns={c: "inventory_" + c for c in candidates
                                         if c in totals.columns and c not in ("upc", "year")})
    totals.merge(metadata, left_on=["file_year", "upc"], right_on=["year", "upc"],
                 how="left", validate="many_to_one").to_parquet(output / "upc_exposure.parquet", index=False)
    record = dict(status="conditional_mass_scenarios_only", producer=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), script_sha256=_sha(__file__),
        inputs=inputs, reconciliation=check, scenarios=summaries,
        definitions="original residual revenue and brand routing unchanged; no accepted panel/law written",
        caution="TOTAL_SIZE and explicit descriptor-pack arithmetic are separate sensitivity assumptions; catalog price intervals require no other unbounded missing mass")
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("scan-dir", "consensus-dir", "metadata-review", "audit-dir", "output"):
        parser.add_argument("--" + flag, required=True)
    run(**vars(parser.parse_args()))
