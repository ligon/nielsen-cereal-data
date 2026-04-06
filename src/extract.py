"""
Stage 1: Extract NielsenIQ Consumer Panel data from tarballs.

Reads the compressed annual tarballs, extracts the TSV files for the
configured year range, filters purchases to the configured product
modules, and writes parquet files.

Usage:
    python src/extract.py [--config config/default.yaml]
"""
import argparse
import hashlib
import os
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
import yaml


def load_config(path="config/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def _hash_file(path):
    """SHA-256 of a file, for reproducibility logging."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _tarball_path(raw_dir, year):
    """Locate the tarball for a given year, handling the two-era structure."""
    if year <= 2020:
        return Path(raw_dir) / "2004-2020_Panel_Data" / f"Consumer_Panel_Data_{year}.tgz"
    else:
        return Path(raw_dir) / "2021-Onward_Panel_Data" / f"Consumer_Panel_Data_{year}.tgz"


def _master_tarball(raw_dir):
    """Locate the master files tarball (2004-2020 era only)."""
    return Path(raw_dir) / "2004-2020_Panel_Data" / "Master_Files2004-2020.tgz"


def _read_tsv_from_tar(tf, member):
    """Read a TSV file from an open tarfile into a DataFrame."""
    f = tf.extractfile(member)
    if f is None:
        raise ValueError(f"Cannot extract {member.name}")
    return pd.read_csv(f, sep="\t", low_memory=False, encoding="latin-1")


def extract_year(raw_dir, year, modules, out_dir):
    """Extract one year's data, filtering purchases to the given modules.

    Returns dict of {name: (path, nrows, hash)} for validation logging.
    """
    tarball = _tarball_path(raw_dir, year)
    if not tarball.exists():
        print(f"  WARNING: {tarball} not found, skipping year {year}")
        return {}

    results = {}

    with tarfile.open(tarball, "r:gz") as tf:
        members = {m.name.split("/")[-1]: m for m in tf.getmembers() if m.isfile()}

        # -- Purchases: filter to target modules --
        purch_key = f"purchases_{year}.tsv"
        if purch_key in members:
            df = _read_tsv_from_tar(tf, members[purch_key])
            # The purchases file may not have a module column directly;
            # we'll filter after merging with products.  For now, keep all.
            df["panel_year"] = year
            out_path = Path(out_dir) / f"purchases_{year}.parquet"
            df.to_parquet(out_path, index=False)
            results["purchases"] = (out_path, len(df), _hash_file(out_path))
        else:
            print(f"  WARNING: {purch_key} not found in {tarball.name}")

        # -- Trips --
        trips_key = f"trips_{year}.tsv"
        if trips_key in members:
            df = _read_tsv_from_tar(tf, members[trips_key])
            df["panel_year"] = year
            out_path = Path(out_dir) / f"trips_{year}.parquet"
            df.to_parquet(out_path, index=False)
            results["trips"] = (out_path, len(df), _hash_file(out_path))

        # -- Panelists --
        panel_key = f"panelists_{year}.tsv"
        if panel_key in members:
            df = _read_tsv_from_tar(tf, members[panel_key])
            df["panel_year"] = year
            out_path = Path(out_dir) / f"panelists_{year}.parquet"
            df.to_parquet(out_path, index=False)
            results["panelists"] = (out_path, len(df), _hash_file(out_path))

        # -- Products extra (year-specific product info) --
        prod_key = f"products_extra_{year}.tsv"
        if prod_key in members:
            df = _read_tsv_from_tar(tf, members[prod_key])
            df["panel_year"] = year
            out_path = Path(out_dir) / f"products_extra_{year}.parquet"
            df.to_parquet(out_path, index=False)
            results["products_extra"] = (out_path, len(df), _hash_file(out_path))

    return results


def extract_master(raw_dir, out_dir):
    """Extract the master product catalog (2004-2020 era)."""
    tarball = _master_tarball(raw_dir)
    if not tarball.exists():
        print(f"  WARNING: Master files tarball not found at {tarball}")
        return {}

    results = {}
    with tarfile.open(tarball, "r:gz") as tf:
        members = {m.name.split("/")[-1]: m for m in tf.getmembers() if m.isfile()}

        for key in ["products.tsv", "retailers.tsv", "brand_variations.tsv"]:
            if key in members:
                df = _read_tsv_from_tar(tf, members[key])
                name = key.replace(".tsv", "")
                out_path = Path(out_dir) / f"master_{name}.parquet"
                df.to_parquet(out_path, index=False)
                results[f"master_{name}"] = (out_path, len(df), _hash_file(out_path))

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract NielsenIQ Consumer Panel data")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--years", nargs=2, type=int, default=None,
                        help="Override year range (e.g., --years 2014 2018)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg["raw_dir"]
    modules = cfg["modules"]

    if args.years:
        year_lo, year_hi = args.years
    else:
        year_lo, year_hi = cfg["years"]

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting NielsenIQ Consumer Panel: {year_lo}-{year_hi}")
    print(f"  Source: {raw_dir}")
    print(f"  Target modules: {modules}")
    print(f"  Output: {out_dir}/")
    print()

    # -- Master files --
    print("Extracting master files...")
    master_results = extract_master(raw_dir, out_dir)
    for name, (path, nrows, h) in master_results.items():
        print(f"  {name}: {nrows:,} rows  [{h}]")
    print()

    # -- Annual files --
    all_results = {}
    for year in range(year_lo, year_hi + 1):
        print(f"Year {year}...")
        year_results = extract_year(raw_dir, year, modules, out_dir)
        for name, (path, nrows, h) in year_results.items():
            print(f"  {name}: {nrows:,} rows  [{h}]")
        all_results[year] = year_results
    print()

    # -- Summary --
    years_extracted = [y for y in all_results if all_results[y]]
    total_purchases = sum(
        r["purchases"][1] for r in all_results.values()
        if "purchases" in r
    )
    total_trips = sum(
        r["trips"][1] for r in all_results.values()
        if "trips" in r
    )
    print(f"Done. {len(years_extracted)} years extracted.")
    print(f"  Total purchase rows: {total_purchases:,}")
    print(f"  Total trip rows: {total_trips:,}")


if __name__ == "__main__":
    main()
