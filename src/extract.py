"""
Stage 1: Extract NielsenIQ Consumer Panel data from tarballs.

Handles two eras of NielsenIQ data layout:
  - 2004-2020: purchases_{year}.tsv, trips_{year}.tsv, panelists_{year}.tsv,
    products_extra_{year}.tsv.  Module codes live in the master products file;
    purchases must be joined to filter by module.
  - 2021+: purchase.tsv, trip.tsv, panelist.tsv, productdesc.tsv,
    productattributes.tsv.  Module codes are inline in purchase.tsv
    (product_module_code_hms) and productdesc.tsv.

Purchases are filtered to the configured product modules at extraction
time to avoid writing tens of gigabytes of non-cereal data.

Usage:
    python src/extract.py [--config config/default.yaml] [--years 2014 2018]
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path="config/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def _hash_file(path):
    """SHA-256 prefix for reproducibility logging."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _tarball_path(raw_dir, year):
    """Locate the tarball for a given year."""
    if year <= 2020:
        return Path(raw_dir) / "2004-2020_Panel_Data" / f"Consumer_Panel_Data_{year}.tgz"
    else:
        return Path(raw_dir) / "2021-Onward_Panel_Data" / f"Consumer_Panel_Data_{year}.tgz"


def _read_tsv_from_tar(tf, member, **kwargs):
    """Read a TSV from an open tarfile into a DataFrame."""
    f = tf.extractfile(member)
    if f is None:
        raise ValueError(f"Cannot extract {member.name}")
    return pd.read_csv(f, sep="\t", low_memory=False, encoding="latin-1",
                        **kwargs)


def _write_parquet(df, path):
    """Write parquet and return (path, nrows, hash)."""
    df.to_parquet(path, index=False)
    return (path, len(df), _hash_file(path))


def _find_member(members, *candidates):
    """Find first matching filename among candidates."""
    for c in candidates:
        if c in members:
            return members[c]
    return None


# ---------------------------------------------------------------------------
# Master files (2004-2020 era only)
# ---------------------------------------------------------------------------

def extract_master(raw_dir, out_dir):
    """Extract master product catalog.  Returns dict of results."""
    tarball = Path(raw_dir) / "2004-2020_Panel_Data" / "Master_Files2004-2020.tgz"
    if not tarball.exists():
        print(f"  WARNING: {tarball} not found")
        return {}

    import tarfile
    results = {}
    with tarfile.open(tarball, "r:gz") as tf:
        members = {m.name.split("/")[-1]: m for m in tf.getmembers() if m.isfile()}
        for key in ["products.tsv", "retailers.tsv", "brand_variations.tsv"]:
            if key in members:
                df = _read_tsv_from_tar(tf, members[key])
                name = key.replace(".tsv", "")
                results[f"master_{name}"] = _write_parquet(
                    df, Path(out_dir) / f"master_{name}.parquet"
                )
    return results


def _load_module_upcs(master_products_path, modules):
    """Load UPCs belonging to target modules from master products."""
    mp = pd.read_parquet(master_products_path)
    cereal = mp[mp["product_module_code"].isin(modules)]
    return cereal[["upc", "upc_ver_uc", "brand_code_uc", "brand_descr",
                    "size1_amount", "size1_units", "multi"]]


# ---------------------------------------------------------------------------
# Year extraction: 2004-2020 era
# ---------------------------------------------------------------------------

def _extract_year_legacy(tf, members, year, modules, module_upcs, out_dir):
    """Extract a 2004-2020 year.  Filter purchases via master UPC join."""
    results = {}

    # Purchases: join with module_upcs to filter
    purch_m = _find_member(members, f"purchases_{year}.tsv")
    if purch_m:
        df = _read_tsv_from_tar(tf, purch_m)
        n_raw = len(df)
        df = df.merge(module_upcs, on=["upc", "upc_ver_uc"], how="inner")
        df["panel_year"] = year
        results["purchases"] = _write_parquet(
            df, Path(out_dir) / f"purchases_{year}.parquet"
        )
        print(f"  purchases: {n_raw:,} raw -> {len(df):,} module-filtered")
    else:
        print(f"  WARNING: purchases_{year}.tsv not found")

    # Trips (keep all — we filter via purchase join downstream)
    trips_m = _find_member(members, f"trips_{year}.tsv")
    if trips_m:
        df = _read_tsv_from_tar(tf, trips_m)
        df["panel_year"] = year
        results["trips"] = _write_parquet(
            df, Path(out_dir) / f"trips_{year}.parquet"
        )

    # Panelists
    panel_m = _find_member(members, f"panelists_{year}.tsv")
    if panel_m:
        df = _read_tsv_from_tar(tf, panel_m)
        df["panel_year"] = year
        results["panelists"] = _write_parquet(
            df, Path(out_dir) / f"panelists_{year}.parquet"
        )

    # Products extra
    prod_m = _find_member(members, f"products_extra_{year}.tsv")
    if prod_m:
        df = _read_tsv_from_tar(tf, prod_m)
        df["panel_year"] = year
        results["products_extra"] = _write_parquet(
            df, Path(out_dir) / f"products_extra_{year}.parquet"
        )

    return results


# ---------------------------------------------------------------------------
# Year extraction: 2021+ era
# ---------------------------------------------------------------------------

def _extract_year_new(tf, members, year, modules, out_dir):
    """Extract a 2021+ year.  Filter purchases via inline module code."""
    results = {}

    # Purchases: filter on product_module_code_hms
    purch_m = _find_member(members, "purchase.tsv")
    if purch_m:
        df = _read_tsv_from_tar(tf, purch_m)
        n_raw = len(df)
        df = df[df["product_module_code_hms"].isin(modules)]
        results["purchases"] = _write_parquet(
            df, Path(out_dir) / f"purchases_{year}.parquet"
        )
        print(f"  purchases: {n_raw:,} raw -> {len(df):,} module-filtered")
    else:
        print(f"  WARNING: purchase.tsv not found")

    # Trips
    trips_m = _find_member(members, "trip.tsv")
    if trips_m:
        df = _read_tsv_from_tar(tf, trips_m)
        results["trips"] = _write_parquet(
            df, Path(out_dir) / f"trips_{year}.parquet"
        )

    # Panelists
    panel_m = _find_member(members, "panelist.tsv")
    if panel_m:
        df = _read_tsv_from_tar(tf, panel_m)
        results["panelists"] = _write_parquet(
            df, Path(out_dir) / f"panelists_{year}.parquet"
        )

    # Product description (has module codes — useful for product catalog)
    for key, name in [("productdesc.tsv", "productdesc"),
                      ("productattributes.tsv", "productattributes")]:
        m = _find_member(members, key)
        if m:
            df = _read_tsv_from_tar(tf, m)
            df["panel_year"] = year
            results[name] = _write_parquet(
                df, Path(out_dir) / f"{name}_{year}.parquet"
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_year(raw_dir, year, modules, module_upcs, out_dir):
    """Extract one year, dispatching to the correct era handler."""
    import tarfile

    tarball = _tarball_path(raw_dir, year)
    if not tarball.exists():
        print(f"  WARNING: {tarball} not found, skipping")
        return {}

    with tarfile.open(tarball, "r:gz") as tf:
        members = {m.name.split("/")[-1]: m for m in tf.getmembers()
                    if m.isfile()}

        if year <= 2020:
            return _extract_year_legacy(
                tf, members, year, modules, module_upcs, out_dir
            )
        else:
            return _extract_year_new(
                tf, members, year, modules, out_dir
            )


def main():
    parser = argparse.ArgumentParser(
        description="Extract NielsenIQ Consumer Panel data")
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
    print(f"  Modules: {modules}")
    print(f"  Output: {out_dir}/")
    print()

    # -- Master files (needed for 2004-2020 module filtering) --
    master_products_path = out_dir / "master_products.parquet"
    if not master_products_path.exists() or year_lo <= 2020:
        print("Extracting master files...")
        master_results = extract_master(raw_dir, out_dir)
        for name, (path, nrows, h) in master_results.items():
            print(f"  {name}: {nrows:,} rows  [{h}]")
        print()

    # Load module UPCs for 2004-2020 purchase filtering
    module_upcs = None
    if year_lo <= 2020 and master_products_path.exists():
        module_upcs = _load_module_upcs(master_products_path, modules)
        print(f"Module {modules} UPCs in master: {len(module_upcs):,}")
        print()

    # -- Annual files --
    all_results = {}
    for year in range(year_lo, year_hi + 1):
        print(f"Year {year}...")
        results = extract_year(raw_dir, year, modules, module_upcs, out_dir)
        for name, (path, nrows, h) in results.items():
            if "purchases" not in name:  # purchases already printed with filter info
                print(f"  {name}: {nrows:,} rows  [{h}]")
        all_results[year] = results
    print()

    # -- Summary --
    years_ok = [y for y, r in all_results.items() if r]
    total_purch = sum(
        r["purchases"][1] for r in all_results.values() if "purchases" in r
    )
    total_trips = sum(
        r["trips"][1] for r in all_results.values() if "trips" in r
    )
    print(f"Done. {len(years_ok)} years extracted ({min(years_ok)}-{max(years_ok)}).")
    print(f"  Cereal purchase rows: {total_purch:,}")
    print(f"  Total trip rows:      {total_trips:,}")


if __name__ == "__main__":
    main()
