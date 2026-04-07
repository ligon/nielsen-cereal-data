"""
Stage 2: Select top products and assemble nutrition attributes.

Reads the extracted purchase parquets, ranks brands by total
expenditure, selects the top N, and assembles nutrition data from:
  - 2021+ productattributes.tsv (has nutrition inline)
  - Master products + nutrition_overrides.yaml (for 2004-2020)

Usage:
    python src/select_products.py [--config config/default.yaml]
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(path="config/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_all_purchases(raw_dir):
    """Concatenate all per-year purchase parquets."""
    files = sorted(Path(raw_dir).glob("purchases_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No purchase parquets in {raw_dir}")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(out):,} cereal purchases from {len(files)} years")
    return out


def _get_brand_code(purchases):
    """Get the brand_code column, handling both eras.

    2004-2020: purchases were joined with master at extraction, so
    brand_code_uc is already present.
    2021+: purchase.tsv has brand_descr_hms but not brand_code_uc.
    We unify on brand_descr for selection.
    """
    if "brand_code_uc" in purchases.columns:
        return "brand_code_uc"
    elif "brand_descr_hms" in purchases.columns:
        return "brand_descr_hms"
    else:
        raise ValueError("Cannot find brand identifier in purchases")


def select_top_brands(purchases, n_products, master_products_path=None):
    """Rank brands by total expenditure, return top N.

    Returns DataFrame with brand_code_uc, brand_descr, total_spend, rank.
    """
    # For 2004-2020 purchases, brand_code_uc is present from the master join.
    # For 2021+, we use brand_descr_hms.  We need a unified brand key.

    # Compute total spend per brand_code_uc (available in legacy data)
    if "brand_code_uc" in purchases.columns:
        brand_spend = (
            purchases.groupby("brand_code_uc")["total_price_paid"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        brand_spend.columns = ["brand_code_uc", "total_spend"]

        # Get brand descriptions from master
        if master_products_path and Path(master_products_path).exists():
            mp = pd.read_parquet(master_products_path)
            mp_cereal = mp[["brand_code_uc", "brand_descr"]].drop_duplicates(
                "brand_code_uc"
            )
            brand_spend = brand_spend.merge(mp_cereal, on="brand_code_uc", how="left")
    else:
        # 2021+ era: use brand_descr_hms
        brand_spend = (
            purchases.groupby("brand_descr_hms")["total_price_paid"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        brand_spend.columns = ["brand_descr", "total_spend"]

    brand_spend["rank"] = range(1, len(brand_spend) + 1)
    brand_spend["share_of_total"] = (
        brand_spend["total_spend"] / brand_spend["total_spend"].sum()
    )

    top = brand_spend.head(n_products).copy()
    cumulative = top["share_of_total"].sum()

    print(f"\nTop {n_products} brands by expenditure:")
    print(f"  Cumulative expenditure share: {cumulative:.1%}")
    print(f"  Cutoff at rank {n_products}: "
          f"${top.iloc[-1]['total_spend']:,.0f} total spend")
    print()
    for _, row in top.head(10).iterrows():
        desc = row.get("brand_descr", row.get("brand_descr_hms", "?"))
        print(f"  {row['rank']:3d}. {desc:30s}  "
              f"${row['total_spend']:>12,.0f}  ({row['share_of_total']:.1%})")
    if n_products > 10:
        print(f"  ... ({n_products - 10} more)")

    return top, brand_spend


def _nutrition_from_2021(raw_dir, modules):
    """Extract nutrition from 2021+ productattributes files."""
    files = sorted(Path(raw_dir).glob("productattributes_*.parquet"))
    if not files:
        return None

    # Use the most recent year's attributes
    df = pd.read_parquet(files[-1])

    # Filter to cereal UPCs via productdesc
    desc_files = sorted(Path(raw_dir).glob("productdesc_*.parquet"))
    if desc_files:
        desc = pd.read_parquet(desc_files[-1])
        cereal_upcs = desc[desc["product_module_code"].isin(modules)]["upc"]
        df = df[df["upc"].isin(cereal_upcs)]

    # Extract nutrition columns
    nutrition_cols = {
        "SUGAR_GRAM": "sugar",
        "DIETARY_FIBER": "fiber",
        "PROTEIN_GRAM": "protein",
        "TOTAL_FAT_GRAM": "fat",
        "SODIUM": "sodium",
    }

    result = df[["upc"]].copy()
    for src, dst in nutrition_cols.items():
        if src in df.columns:
            result[dst] = pd.to_numeric(df[src], errors="coerce")

    # Also get brand from BRAND column
    if "BRAND" in df.columns:
        result["brand_descr"] = df["BRAND"]

    return result.dropna(subset=list(nutrition_cols.values()), how="all")


def main():
    parser = argparse.ArgumentParser(
        description="Select top products and assemble nutrition")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_products = cfg["n_products"]
    modules = cfg["modules"]

    raw_dir = Path("data/raw")
    clean_dir = Path("data/clean")
    clean_dir.mkdir(parents=True, exist_ok=True)

    # -- Load all purchases --
    purchases = _load_all_purchases(raw_dir)

    # -- Select top brands --
    master_path = raw_dir / "master_products.parquet"
    top_brands, all_brands = select_top_brands(
        purchases, n_products,
        master_products_path=str(master_path) if master_path.exists() else None,
    )

    # Save brand selection (full ranking for audit)
    all_brands.to_parquet(clean_dir / "brand_ranking.parquet", index=False)
    top_brands.to_parquet(clean_dir / "selected_brands.parquet", index=False)
    print(f"\nSaved brand ranking to {clean_dir}/brand_ranking.parquet")

    # -- Nutrition data --
    # Try 2021+ attributes first (richest source)
    nutrition_2021 = _nutrition_from_2021(raw_dir, modules)
    if nutrition_2021 is not None:
        print(f"\n2021+ nutrition data: {len(nutrition_2021):,} UPCs with nutrition")

    # Load overrides
    overrides_path = cfg.get("nutrition_overrides", "config/nutrition_overrides.yaml")
    if Path(overrides_path).exists():
        with open(overrides_path) as f:
            overrides = yaml.safe_load(f) or {}
        if overrides:
            print(f"Nutrition overrides: {len(overrides)} brands")

    # TODO: merge nutrition sources (2021+ attributes, FDC API, overrides)
    # and produce products_with_nutrition.parquet.  For now, save what
    # we have and let the market aggregation stage handle the join.

    print("\nStage 2 complete. Next: aggregate_markets.py")


if __name__ == "__main__":
    main()
