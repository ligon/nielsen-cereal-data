"""Prepare conditional outside-model inputs using identified existing routes.

Reuses market_scenario, panel_cells and residual_table (measurement ledger).
No accepted panel, brand mapping or raw movement is rewritten.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from crosswalk_eval import _sha
from rms_outside_audit import panel_cells, residual_table
from rms_size_exposure_summary import market_scenario

KEYS = ["year", "upc", "category"]
KNOWN = "known_inside_recovered_mass"
UNKNOWN = "known_inside_still_unknown_mass"
UNRESOLVED = "catalog_partition_unresolved"
ACTIONS = {KNOWN, UNKNOWN, UNRESOLVED, "assortment_policy_retain_outside",
           "retain_existing_inside", "retain_existing_outside"}


def routed_inputs(inside, baseline, categories, details, candidates, decisions, mass_column):
    """Recover masses, then transfer quantified existing-brand purchases once."""
    before, _ = market_scenario(details, baseline, candidates, mass_column)
    if decisions.duplicated(KEYS).any() or not set(decisions.action).issubset(ACTIONS):
        raise ValueError("routing decisions require unique keys and recognized actions")
    if candidates.duplicated(["year", "upc"]).any():
        raise ValueError("duplicate mass candidate")
    # Inventory/exposure columns on metadata must never overwrite movement totals.
    metadata = candidates[["year", "upc", mass_column, "brand_code_uc", "catalog_versions"]]
    joined = details.merge(metadata, left_on=["file_year", "upc"], right_on=["year", "upc"],
                           how="left", validate="many_to_one", indicator=True)
    if joined._merge.ne("both").any():
        raise ValueError("movement row lacks mass metadata")
    joined = joined.drop(columns="_merge").merge(decisions[KEYS + ["action", "proposed_j"]],
        on=KEYS, how="left", validate="many_to_one")
    if joined.action.isna().any():
        raise ValueError("movement row lacks an explicit routing decision")
    existing = panel_cells(inside)
    j45 = set(existing.index.get_level_values("j"))
    missing = joined.uncovered_rows.gt(0) & joined.category.ne("modeled_45")
    good = np.isfinite(joined[mass_column]) & joined[mass_column].gt(0)
    recover = missing & good
    move = joined.action.eq(KNOWN)
    selected = joined.category.eq("selected_brand_excluded")
    if not joined.loc[selected, "action"].isin([KNOWN, UNKNOWN]).all():
        raise ValueError("selected-brand exclusions need an inside recovery/unknown decision")
    if (not joined.loc[move, "category"].eq("selected_brand_excluded").all()
            or not recover.loc[move].all()
            or not joined.loc[move, "proposed_j"].isin(j45).all()
            or not joined.loc[move, "proposed_j"].eq(joined.loc[move, "brand_code_uc"]).all()):
        raise ValueError("inside transfers require supported mass and the existing adopted j")
    if good.loc[joined.action.eq(UNKNOWN)].any():
        raise ValueError("unknown-inside decision disagrees with supported mass")
    catalog = joined.catalog_versions.notna() & joined.category.ne("modeled_45")
    if not joined.loc[catalog, "action"].isin([UNRESOLVED, "assortment_policy_retain_outside"]).all():
        raise ValueError("unowned catalog partitions need an explicit unresolved/assortment decision")
    excluded = joined.loc[selected].groupby(["tcode", "m"]).revenue.sum().reindex(baseline.index, fill_value=0)
    if not np.allclose(excluded, baseline.selected_brand_excluded_revenue, rtol=1e-12, atol=1e-6):
        raise ValueError("focused detail does not exhaust selected-brand exclusions")
    if not joined.loc[missing, "uncovered_rows"].eq(joined.loc[missing, "rows"]).all():
        raise ValueError("mixed original size coverage within a movement cell")

    cats = categories.copy(deep=True)
    original_cats = cats.copy(deep=True)
    levels = ["tcode", "m", "category"]
    for column, values in [
            ("covered_revenue", joined.revenue.where(recover, 0.)),
            ("covered_ounces", (joined.units * joined[mass_column].fillna(0)).where(recover, 0.)),
            ("uncovered_rows", -joined.uncovered_rows.where(recover, 0))]:
        delta = joined.assign(value=values).groupby(levels).value.sum()
        cats[column] = cats[column].add(delta, fill_value=0).reindex(cats.index)

    transfer = joined.loc[move].copy()
    transfer["servings"] = transfer.units * transfer[mass_column]
    transfer["j"] = transfer.proposed_j.astype(int)
    increments = transfer.groupby(["tcode", "m", "j"])[["revenue", "servings"]].sum()
    augmented_cells = existing.add(increments, fill_value=0).sort_index()
    for column, values in [("revenue", transfer.revenue), ("covered_revenue", transfer.revenue),
                           ("covered_ounces", transfer.servings), ("rows", transfer.rows)]:
        old = transfer.assign(value=values).groupby(levels).value.sum()
        new = transfer.assign(value=values, category="modeled_45").groupby(levels).value.sum()
        cats[column] = cats[column].sub(old, fill_value=0).add(new, fill_value=0).reindex(cats.index)
    for column in ["revenue", "covered_revenue", "covered_ounces", "rows", "uncovered_rows"]:
        if (cats[column] < -1e-6).any():
            raise ValueError("routing produced negative category accounting")
    audit = residual_table(cats, augmented_cells)
    unresolved = joined.loc[joined.action.eq(UNRESOLVED)].groupby(["tcode", "m"]).revenue.sum()
    audit["unresolved_partition_revenue"] = unresolved.reindex(audit.index, fill_value=0)
    remaining_inside = joined.loc[joined.action.eq(UNKNOWN)].groupby(["tcode", "m"]).revenue.sum()
    audit["selected_brand_excluded_revenue"] = remaining_inside.reindex(audit.index, fill_value=0)
    # Ounces used by market_scenario must agree with the same corrected categories.
    moved_q = transfer.groupby(["tcode", "m"]).servings.sum().reindex(audit.index, fill_value=0)
    np.testing.assert_allclose(audit.residual_covered_ounces, before.covered_ounces - moved_q,
                               rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(audit.module_revenue, baseline.module_revenue, rtol=1e-12, atol=1e-6)
    np.testing.assert_array_equal(cats.groupby(level=["tcode", "m"]).rows.sum(),
                                  original_cats.groupby(level=["tcode", "m"]).rows.sum())
    eligible = (audit.complete_residual_dollars_per_ounce.notna()
                & audit.selected_brand_excluded_revenue.eq(0)
                & audit.unresolved_partition_revenue.eq(0))
    summary = dict(markets=len(audit), complete_positive_prices=int(audit.complete_residual_dollars_per_ounce.notna().sum()),
        routing_and_mass_eligible_markets=int(eligible.sum()),
        transferred_revenue=float(transfer.revenue.sum()), transferred_ounces=float(transfer.servings.sum()),
        transferred_upc_years=len(transfer[["file_year", "upc"]].drop_duplicates()),
        remaining_inside_unknown_mass_revenue=float(remaining_inside.sum()),
        unresolved_partition_revenue=float(unresolved.sum()))
    return audit, cats, increments, summary


def run(*, market_panel, baseline, categories, details, masses, decisions, mass_column, output):
    os.umask(0o027)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    paths = {k: Path(v) for k, v in dict(market_panel=market_panel, baseline=baseline,
        categories=categories, details=details, masses=masses, decisions=decisions).items()}
    inputs = {k: dict(path=str(p.resolve()), sha256=_sha(p)) for k, p in paths.items()}
    frames = {k: pd.read_parquet(p) for k, p in paths.items()}
    audit, cats, increments, summary = routed_inputs(frames["market_panel"], frames["baseline"],
        frames["categories"], frames["details"], frames["masses"], frames["decisions"], mass_column)
    output.mkdir(parents=True)
    tables = {"audit": audit, "categories": cats, "inside_increments": increments}
    outputs = {}
    for name, frame in tables.items():
        path = output / (name + ".parquet")
        frame.to_parquet(path)
        outputs[name] = dict(path=str(path.resolve()), sha256=_sha(path))
    record = dict(version=1, status="conditional_identified_routing_inputs", inputs=inputs, outputs=outputs,
        producer=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=_sha(__file__), mass_column=mass_column, summary=summary,
        policy="existing adopted j only; unresolved catalog partitions/inside masses excluded as whole markets; assortment remains outside",
        accepted_panel_changed=False)
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("market-panel", "baseline", "categories", "details", "masses", "decisions", "mass-column", "output"):
        parser.add_argument("--" + flag, required=True)
    run(**vars(parser.parse_args()))
