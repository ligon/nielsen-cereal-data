"""Mass-only consensus among complete same-year catalog candidate sets.

The catalog universe is an explicit assumption, not observed movement-version
ownership. No brand is selected and no accepted route or panel is modified.
See .sucoder/ledger-rms-size-recovery.md section 8.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from crosswalk_eval import _sha
from rms_size_candidates import mass_ounces


def consensus(candidates):
    keys = ["year", "upc", "upc_ver_uc"]
    if candidates.duplicated(keys).any():
        raise ValueError("duplicate same-year catalog candidate version")
    records = []
    for (year, upc), group in candidates.groupby(["year", "upc"], sort=True):
        sizes = np.array([mass_ounces(a, u) for a, u in
                          zip(group.size1_amount, group.size1_units)])
        complete = np.isfinite(sizes).all()
        agree = complete and (sizes == sizes[0]).all()
        records.append(dict(year=int(year), upc=int(upc), catalog_versions=len(group),
            catalog_mass_status=("consensus" if agree else "conflict" if complete else "incomplete"),
            catalog_mass_oz=float(sizes[0]) if agree else np.nan,
            catalog_mass_min_oz=float(sizes.min()) if complete else np.nan,
            catalog_mass_max_oz=float(sizes.max()) if complete else np.nan,
            catalog_brand_count=group.brand_code_uc.nunique(dropna=False),
            movement_version_owned=False))
    return pd.DataFrame(records)


def extend_candidates(original, summary):
    out = original.merge(summary, on=["year", "upc"], how="left", validate="one_to_one")
    recover = out.candidate_oz.isna() & out.catalog_mass_oz.notna()
    out["followup_mass_oz"] = out.candidate_oz.where(~recover, out.catalog_mass_oz)
    out["followup_mass_source"] = out.size_origin.where(~recover, "same_year_catalog_mass_consensus")
    # Original finite masses take precedence over catalog disagreements.
    out["followup_mass_min_oz"] = out.followup_mass_oz.fillna(out.catalog_mass_min_oz)
    out["followup_mass_max_oz"] = out.followup_mass_oz.fillna(out.catalog_mass_max_oz)
    return out


def build(candidate_map, evidence, master, output):
    os.umask(0o027)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    evidence = Path(evidence)
    raw = json.loads(evidence.read_text())
    for record in raw["raw_inputs"].values():
        if _sha(record["path"]) != record["sha256"]:
            raise ValueError("raw annual evidence input changed")
    if raw["raw_inputs"]["rms_versions_2016.tsv"]["matches"]:
        raise ValueError("this recovery requires absent RMS version ownership")
    rows = [r["row"] for r in raw["raw_inputs"]["products_extra_2016.tsv"]["matches"]]
    catalog = pd.DataFrame(rows)[["panel_year", "upc", "upc_ver_uc"]].astype(int)
    catalog = catalog.rename(columns={"panel_year": "year"})
    if catalog.year.ne(2016).any() or catalog.empty:
        raise ValueError("expected the verified same-year 2016 catalog candidate set")
    master = Path(master)
    expected_master = raw["converted_verification"]["inputs"]["master"]["sha256"]
    if _sha(master) != expected_master:
        raise ValueError("catalog master changed since raw evidence verification")
    full = pd.read_parquet(master)
    wanted = ["upc", "upc_ver_uc", "size1_amount", "size1_units", "brand_code_uc", "brand_descr"]
    wanted += [c for c in ("multi", "upc_descr") if c in full]
    pairs = catalog.merge(full[wanted], on=["upc", "upc_ver_uc"], how="left", validate="one_to_one")
    summary = consensus(pairs)
    original = pd.read_parquet(candidate_map)
    keys = pd.MultiIndex.from_frame(original[["year", "upc"]])
    if not pd.MultiIndex.from_frame(summary[["year", "upc"]]).isin(keys).all():
        raise ValueError("catalog candidates are absent from archived routes")
    extended = extend_candidates(original, summary)
    positive_gap = original.recorded_oz.isna() & original.dollars.gt(0)
    recovered = positive_gap & original.candidate_oz.isna() & extended.followup_mass_oz.notna()
    # A left merge preserves original row order, but verify that invariant.
    pd.testing.assert_frame_equal(extended[original.columns], original, check_exact=True)
    output.mkdir(parents=True)
    pairs.to_parquet(output / "catalog_versions.parquet", index=False)
    summary.to_parquet(output / "catalog_consensus.parquet", index=False)
    extended.to_parquet(output / "followup_candidates.parquet", index=False)
    record = dict(status="mass_only_candidate_extension", producer=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), script_sha256=_sha(__file__),
        inputs={name: dict(path=str(Path(p).resolve()), sha256=_sha(p)) for name, p in
                dict(candidate_map=candidate_map, raw_evidence=evidence, master=master).items()},
        assumption="all versions in the raw same-year annual catalog form the candidate universe; movement ownership remains unknown",
        no_brand_selection=True, no_multi_factor="reuse existing units times size1 mass convention",
        recovered_positive_upc_years=int(recovered.sum()),
        recovered_pre_dma_filter_revenue=float(extended.loc[recovered, "dollars"].sum()),
        unresolved_positive_upc_years=int((positive_gap & extended.followup_mass_oz.isna()).sum()))
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(record, indent=2), flush=True)
    return extended


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("candidate-map", "evidence", "master", "output"):
        parser.add_argument("--" + flag, required=True)
    build(**vars(parser.parse_args()))
