"""
Build a brand-name crosswalk between the 2004-2020 and 2021+ NielsenIQ eras.

Nielsen changed their brand naming convention at the 2020/2021 boundary:
  - Legacy (2004-2020): abbreviated ("G M CHEERIOS", "KEL FROSTED FLAKES")
  - New (2021+): full names ("GENERAL MILLS CHEERIOS", "KELLOGG'S FROSTED FLAKES")

The brand_code_uc (legacy) and BRAND_cd (2021+) use different ID spaces
and cannot be matched directly.  Instead, we exploit UPCs that appear in
both eras: for each overlapping UPC, we observe both the legacy and new
brand names, giving a deterministic crosswalk.

This crosswalk is essential for:
  1. Carrying 2021+ nutrition data (from productattributes.tsv) back to
     legacy products
  2. Tracking the same brand across the full 2004-2024 sample
  3. Building a unified product panel with consistent identifiers

Usage:
    python src/brand_crosswalk.py [--config config/default.yaml]
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(path="config/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_crosswalk(raw_dir, modules):
    """Build the brand crosswalk from UPC overlap between eras.

    Returns a DataFrame with columns:
      brand_code_uc, brand_descr_legacy, brand_descr_2021
    """
    raw = Path(raw_dir)

    # Load master products (legacy era)
    mp = pd.read_parquet(raw / "master_products.parquet")
    mp_mod = mp[mp["product_module_code"].isin(modules)]
    legacy = (mp_mod[["upc", "brand_code_uc", "brand_descr"]]
              .drop_duplicates("upc")
              .rename(columns={"brand_descr": "brand_descr_legacy"}))

    # Find a 2021+ purchase file for the new-era brand names
    new_files = sorted(raw.glob("purchases_202*.parquet"))
    if not new_files:
        print("WARNING: No 2021+ purchase files found; crosswalk will be empty")
        return pd.DataFrame(columns=["brand_code_uc", "brand_descr_legacy",
                                      "brand_descr_2021"])

    # Use the earliest 2021+ file for maximum UPC overlap with legacy
    new_purch = pd.read_parquet(new_files[0])
    new_brands = (new_purch[["upc", "brand_descr_hms"]]
                  .drop_duplicates("upc")
                  .rename(columns={"brand_descr_hms": "brand_descr_2021"}))

    # Join on UPC
    matched = legacy.merge(new_brands, on="upc", how="inner")

    # Deduplicate to brand level (one legacy brand -> one 2021 name)
    crosswalk = (matched
                 .groupby("brand_code_uc")
                 .agg(brand_descr_legacy=("brand_descr_legacy", "first"),
                      brand_descr_2021=("brand_descr_2021", "first"),
                      n_upc_overlap=("upc", "count"))
                 .reset_index()
                 .sort_values("n_upc_overlap", ascending=False))

    return crosswalk


def main():
    parser = argparse.ArgumentParser(
        description="Build brand crosswalk between NielsenIQ eras")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    modules = cfg["modules"]
    raw_dir = Path("data/raw")
    clean_dir = Path("data/clean")
    clean_dir.mkdir(parents=True, exist_ok=True)

    crosswalk = build_crosswalk(raw_dir, modules)

    out_path = clean_dir / "brand_crosswalk.parquet"
    crosswalk.to_parquet(out_path, index=False)

    print(f"Brand crosswalk: {len(crosswalk)} brands matched via UPC overlap")
    print(f"Saved to {out_path}")
    print()

    # Show top brands
    print(f"{'Legacy name':40s}  {'2021+ name':40s}  UPCs")
    print("-" * 90)
    for _, row in crosswalk.head(30).iterrows():
        print(f"{row.brand_descr_legacy:40s}  {row.brand_descr_2021:40s}  "
              f"{row.n_upc_overlap:3d}")


if __name__ == "__main__":
    main()
