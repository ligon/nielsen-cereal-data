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

Since charter-crosswalk (2026-10-19) the 2021+ route is src/brand_route.py
(UPC first, then a name map learned from all of 2021-2024; rule in
docs/crosswalk-rule.org).  main() now writes that route's name map,
CLEAN/brand_name_map.parquet, and NO LONGER writes brand_crosswalk.parquet;
the builders no longer read it.  build_crosswalk() below is the old
(pre-2026-10-19) artifact's logic, kept only for comparison.

Usage:
    python src/brand_crosswalk.py [--config config/default.yaml]
        [--raw-dir data/raw] [--clean-dir data/clean] [--master PATH]
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
    # Must be >= 2021 (2020 is still legacy format)
    new_files = sorted(f for f in raw.glob("purchases_202*.parquet")
                       if int(f.stem.split("_")[1]) >= 2021)
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


def build_name_map(raw_dir, modules, master_path=None):
    """The 2021+ name map of the shared brand route (src/brand_route.py;
    docs/crosswalk-rule.org): one row per 2021-2024 Homescan name, its
    learned legacy code (dollar majority over all of 2021-2024 on
    legacy-coded UPCs, merged), the code after the curated overrides, and
    the evidence (dollars and UPC counts).  This is the crosswalk artifact
    of the new route; build_crosswalk above is the pre-2026-10-19 route,
    kept only so the before/after report can reproduce it."""
    from brand_route import BrandRoute
    raw = Path(raw_dir)
    br = BrandRoute.from_paths(raw, master_path or raw / "master_products.parquet",
                               tuple(modules))
    nm = br.name_map.copy()
    nm["j"] = nm["brand_descr_hms"].map(br._final_name)
    nm["override"] = nm["brand_descr_hms"].isin(list(br.overrides))
    return nm


def main():
    parser = argparse.ArgumentParser(
        description="Build the 2021+ brand name map (shared brand route)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--master", default=None,
                        help="product master (default RAW/master_products.parquet)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    modules = cfg["modules"]
    raw_dir = Path(args.raw_dir)
    clean_dir = Path(args.clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    nm = build_name_map(raw_dir, modules, args.master)
    out_path = clean_dir / "brand_name_map.parquet"
    nm.to_parquet(out_path, index=False)
    print(f"Brand name map: {len(nm)} 2021+ names, "
          f"{int(nm['j'].notna().sum())} placed; saved to {out_path}")
    top = nm.sort_values("new_dollars", ascending=False).head(30)
    print(f"{'2021+ name':32s} {'j':>8s} {'share':>6s} {'new-UPC $':>10s}")
    for _, r in top.iterrows():
        print(f"{r.brand_descr_hms:32s} {r.j:8.0f} {r.share:6.2f} "
              f"{r.new_dollars:10.0f}")


if __name__ == "__main__":
    main()
