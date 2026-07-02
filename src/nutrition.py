"""
Parse nutrition data from 2021+ NielsenIQ productattributes and aggregate
to brand level.

The 2021+ productattributes file stores nutrition values as strings like
"2 GRAM", "NOT STATED", "0 TO 1 GRAM", "200 MILLIGRAM".  This module
parses those strings to numeric floats, joins to cereal purchases on UPC
(to filter to cereal and obtain the brand name), aggregates to brand
means, and maps back to legacy brand_code_uc via the UPC crosswalk.

Output: data/clean/nutrition.parquet
  Columns: brand_code_uc, brand_descr, sugar, fiber, protein, fat, sodium

Usage:
    python src/nutrition.py [--config config/default.yaml]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path="config/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Nutrition string parsing
# ---------------------------------------------------------------------------

# Pattern: optional number, optional "TO" second number, unit
_RANGE_RE = re.compile(
    r"^\s*(?P<lo>[\d.]+)\s+TO\s+(?P<hi>[\d.]+)\s+(?P<unit>GRAM|MILLIGRAM)\s*$",
    re.IGNORECASE,
)
_SINGLE_RE = re.compile(
    r"^\s*(?P<val>[\d.]+)\s+(?P<unit>GRAM|MILLIGRAM)\s*$",
    re.IGNORECASE,
)


def _parse_nutrition_value(s, target_unit="GRAM"):
    """Parse a single nutrition string to a float in the target unit.

    Parameters
    ----------
    s : str or NaN
        Raw string like "2 GRAM", "0 TO 1 GRAM", "NOT STATED", "200 MILLIGRAM".
    target_unit : str
        "GRAM" or "MILLIGRAM".  Values in the other unit are converted.

    Returns
    -------
    float or NaN
    """
    if pd.isna(s):
        return np.nan

    s = str(s).strip().upper()

    if "NOT STATED" in s or "NOT AVAILABLE" in s or s == "":
        return np.nan

    # Try range pattern first: "0 TO 1 GRAM" -> midpoint
    m = _RANGE_RE.match(s)
    if m:
        lo = float(m.group("lo"))
        hi = float(m.group("hi"))
        val = (lo + hi) / 2.0
        unit = m.group("unit").upper()
        return _convert_unit(val, unit, target_unit)

    # Single value: "2 GRAM" or "200 MILLIGRAM"
    m = _SINGLE_RE.match(s)
    if m:
        val = float(m.group("val"))
        unit = m.group("unit").upper()
        return _convert_unit(val, unit, target_unit)

    # Last resort: try to extract a bare number (shouldn't happen, but be safe)
    try:
        return float(s)
    except ValueError:
        return np.nan


def _convert_unit(val, from_unit, to_unit):
    """Convert between GRAM and MILLIGRAM."""
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    if from_unit == to_unit:
        return val
    if from_unit == "GRAM" and to_unit == "MILLIGRAM":
        return val * 1000.0
    if from_unit == "MILLIGRAM" and to_unit == "GRAM":
        return val / 1000.0
    return val  # unknown units, return as-is


def parse_nutrition_column(series, target_unit="GRAM"):
    """Parse a nutrition string column to float using vectorized string ops.

    Handles: "2 GRAM", "0 TO 1 GRAM", "200 MILLIGRAM", "NOT STATED", NaN.
    """
    s = series.astype(str).str.strip().str.upper()

    # Mark unparseable values
    invalid = s.isin(["", "NAN", "NONE"]) | s.str.contains("NOT STATED|NOT AVAILABLE",
                                                             na=False)
    result = pd.array([np.nan] * len(s), dtype="Float64")

    # --- Range pattern: "0 TO 1 GRAM" -> midpoint ---
    range_match = s.str.extract(
        r"^\s*(?P<lo>[\d.]+)\s+TO\s+(?P<hi>[\d.]+)\s+(?P<unit>GRAM|MILLIGRAM)\s*$",
        expand=True,
    )
    has_range = range_match["lo"].notna() & ~invalid
    if has_range.any():
        lo = pd.to_numeric(range_match.loc[has_range, "lo"])
        hi = pd.to_numeric(range_match.loc[has_range, "hi"])
        vals = (lo + hi) / 2.0
        units = range_match.loc[has_range, "unit"]
        if target_unit == "GRAM":
            vals[units == "MILLIGRAM"] /= 1000.0
        else:
            vals[units == "GRAM"] *= 1000.0
        result[has_range] = vals.values

    # --- Single-value pattern: "2 GRAM" or "200 MILLIGRAM" ---
    single_match = s.str.extract(
        r"^\s*(?P<val>[\d.]+)\s+(?P<unit>GRAM|MILLIGRAM)\s*$",
        expand=True,
    )
    has_single = single_match["val"].notna() & ~invalid & ~has_range
    if has_single.any():
        vals = pd.to_numeric(single_match.loc[has_single, "val"])
        units = single_match.loc[has_single, "unit"]
        if target_unit == "GRAM":
            vals[units == "MILLIGRAM"] /= 1000.0
        else:
            vals[units == "GRAM"] *= 1000.0
        result[has_single] = vals.values

    return pd.Series(result, index=series.index, dtype="float64")


# ---------------------------------------------------------------------------
# Brand crosswalk (inline, following src/brand_crosswalk.py logic)
# ---------------------------------------------------------------------------

def _build_crosswalk(raw_dir, modules):
    """Build brand_code_uc <-> brand_descr_hms crosswalk via UPC overlap.

    Returns DataFrame: brand_code_uc, brand_descr_legacy, brand_descr_2021
    """
    raw = Path(raw_dir)

    # Legacy era: master_products has brand_code_uc
    mp = pd.read_parquet(raw / "master_products.parquet")
    mp_mod = mp[mp["product_module_code"].isin(modules)]
    legacy = (mp_mod[["upc", "brand_code_uc", "brand_descr"]]
              .drop_duplicates("upc")
              .rename(columns={"brand_descr": "brand_descr_legacy"}))

    # 2021+ era: purchases have brand_descr_hms
    new_files = sorted(f for f in raw.glob("purchases_202*.parquet")
                       if int(f.stem.split("_")[1]) >= 2021)
    if not new_files:
        return pd.DataFrame(columns=["brand_code_uc", "brand_descr_legacy",
                                      "brand_descr_2021"])

    new_purch = pd.read_parquet(new_files[0], columns=["upc", "brand_descr_hms"])
    new_brands = (new_purch[["upc", "brand_descr_hms"]]
                  .drop_duplicates("upc")
                  .rename(columns={"brand_descr_hms": "brand_descr_2021"}))

    matched = legacy.merge(new_brands, on="upc", how="inner")

    crosswalk = (matched
                 .groupby("brand_code_uc")
                 .agg(brand_descr_legacy=("brand_descr_legacy", "first"),
                      brand_descr_2021=("brand_descr_2021", "first"),
                      n_upc_overlap=("upc", "count"))
                 .reset_index()
                 .sort_values("n_upc_overlap", ascending=False))

    return crosswalk


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Column mapping: source column -> (output name, target unit)
NUTRITION_COLS = {
    "SUGAR_GRAM":     ("sugar",   "GRAM"),
    "DIETARY_FIBER":  ("fiber",   "GRAM"),
    "PROTEIN_GRAM":   ("protein", "GRAM"),
    "TOTAL_FAT_GRAM": ("fat",     "GRAM"),
    # Parsed in the source unit (mg); aggregate_markets normalizes sodium to
    # grams so all five characteristics share per-serving-gram units.
    "SODIUM":         ("sodium",  "MILLIGRAM"),
}


def build_nutrition(raw_dir, modules):
    """Parse 2021+ nutrition, aggregate to brand, map to brand_code_uc.

    Returns DataFrame with: brand_code_uc, brand_descr, sugar, fiber,
    protein, fat, sodium.
    """
    raw = Path(raw_dir)

    # ---- 1. Load productattributes (only needed columns) ----
    pa_files = sorted(raw.glob("productattributes_*.parquet"))
    if not pa_files:
        raise FileNotFoundError(f"No productattributes files in {raw}")

    src_cols = ["upc"] + list(NUTRITION_COLS.keys())
    pa = pd.read_parquet(pa_files[-1], columns=src_cols)
    print(f"Loaded productattributes: {len(pa):,} UPCs")

    # ---- 2. Filter to cereal UPCs via 2021 purchases ----
    purch_files = sorted(f for f in raw.glob("purchases_202*.parquet")
                         if int(f.stem.split("_")[1]) >= 2021)
    if not purch_files:
        raise FileNotFoundError(f"No 2021+ purchase files in {raw}")

    purch = pd.read_parquet(purch_files[0], columns=["upc", "brand_descr_hms"])
    cereal_brands = purch[["upc", "brand_descr_hms"]].drop_duplicates("upc")
    print(f"Cereal UPCs from 2021 purchases: {len(cereal_brands):,}")

    pa_cereal = cereal_brands.merge(pa, on="upc", how="inner")
    print(f"Cereal UPCs with productattributes: {len(pa_cereal):,}")

    # ---- 3. Parse nutrition strings to numeric ----
    for src_col, (dst_col, target_unit) in NUTRITION_COLS.items():
        pa_cereal[dst_col] = parse_nutrition_column(
            pa_cereal[src_col], target_unit=target_unit
        )

    nutrient_names = [v[0] for v in NUTRITION_COLS.values()]

    # Report per-UPC parsing results
    n_complete = pa_cereal[nutrient_names].notna().all(axis=1).sum()
    n_any = pa_cereal[nutrient_names].notna().any(axis=1).sum()
    print(f"\nPer-UPC nutrition parsing:")
    print(f"  UPCs with all 5 nutrients: {n_complete:,} / {len(pa_cereal):,}")
    print(f"  UPCs with any nutrient:    {n_any:,} / {len(pa_cereal):,}")
    for col in nutrient_names:
        n_valid = pa_cereal[col].notna().sum()
        print(f"  {col:8s}: {n_valid:,} valid ({n_valid/len(pa_cereal):.0%})")

    # ---- 4. Aggregate to brand level (mean across UPCs) ----
    brand_nutrition = (pa_cereal
                       .groupby("brand_descr_hms")[nutrient_names]
                       .mean()
                       .reset_index()
                       .rename(columns={"brand_descr_hms": "brand_descr_2021"}))

    print(f"\nBrands with nutrition (2021 names): {len(brand_nutrition):,}")

    # ---- 5. Map to legacy brand_code_uc via crosswalk ----
    crosswalk = _build_crosswalk(raw_dir, modules)
    print(f"Brand crosswalk: {len(crosswalk):,} brands matched via UPC overlap")

    result = crosswalk.merge(brand_nutrition, on="brand_descr_2021", how="inner")

    # Choose the best brand description: use legacy name (shorter, consistent
    # with estimation data), but keep 2021 name for reference
    result = result.rename(columns={"brand_descr_legacy": "brand_descr"})
    result = result[["brand_code_uc", "brand_descr"] + nutrient_names]

    # Also produce a table for 2021-only brands (no legacy match)
    unmatched = brand_nutrition[
        ~brand_nutrition["brand_descr_2021"].isin(crosswalk["brand_descr_2021"])
    ].copy()

    print(f"\nFinal result: {len(result):,} brands with brand_code_uc + nutrition")
    if len(unmatched) > 0:
        print(f"  ({len(unmatched):,} brands in 2021 with no legacy crosswalk match)")

    return result, brand_nutrition


def validate_nutrition(df):
    """Print validation summary of the nutrition DataFrame."""
    nutrient_names = ["sugar", "fiber", "protein", "fat", "sodium"]

    n_complete = df[nutrient_names].notna().all(axis=1).sum()
    print(f"\n{'='*60}")
    print(f"VALIDATION: {len(df)} brands total, "
          f"{n_complete} with complete nutrition ({n_complete/len(df):.0%})")
    print(f"{'='*60}")

    print(f"\n{'Nutrient':>10s}  {'N valid':>8s}  {'Mean':>8s}  {'Std':>8s}  "
          f"{'Min':>8s}  {'Median':>8s}  {'Max':>8s}")
    print("-" * 72)
    for col in nutrient_names:
        s = df[col].dropna()
        unit = "mg" if col == "sodium" else "g"
        print(f"{col:>10s}  {len(s):8d}  {s.mean():8.1f}  {s.std():8.1f}  "
              f"{s.min():8.1f}  {s.median():8.1f}  {s.max():8.1f}  {unit}")

    # Flag suspicious values
    print()
    issues = []
    for col in nutrient_names:
        s = df[col].dropna()
        if col == "sodium":
            if s.max() > 2000:
                issues.append(f"  WARNING: {col} max = {s.max():.0f} mg (>2000)")
        else:
            if s.max() > 50:
                issues.append(f"  WARNING: {col} max = {s.max():.1f} g (>50)")
    if issues:
        for i in issues:
            print(i)
    else:
        print("  All nutrient ranges look reasonable.")


def main():
    parser = argparse.ArgumentParser(
        description="Parse 2021+ nutrition data and aggregate to brand level")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    modules = cfg["modules"]
    raw_dir = Path("data/raw")
    clean_dir = Path("data/clean")
    clean_dir.mkdir(parents=True, exist_ok=True)

    result, brand_nutrition_2021 = build_nutrition(raw_dir, modules)

    # Apply manual overrides
    overrides_path = cfg.get("nutrition_overrides", "config/nutrition_overrides.yaml")
    if Path(overrides_path).exists():
        with open(overrides_path) as f:
            overrides = yaml.safe_load(f) or {}
        nutrient_names = ["sugar", "fiber", "protein", "fat", "sodium"]
        n_applied = 0
        for brand_descr, vals in overrides.items():
            if not isinstance(vals, dict):
                continue
            mask = result["brand_descr"] == brand_descr
            if mask.any():
                for col in nutrient_names:
                    if col in vals:
                        result.loc[mask, col] = vals[col]
                reason = vals.get("reason", "")
                print(f"  Override: {brand_descr}  ({reason})")
                n_applied += 1
        if n_applied:
            print(f"  Applied {n_applied} nutrition overrides")

    validate_nutrition(result)

    # Save
    out_path = clean_dir / "nutrition.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nSaved {len(result)} brands to {out_path}")

    # Also save the full 2021-brand-level table (useful for 2021-only analyses)
    out_path_2021 = clean_dir / "nutrition_2021_brands.parquet"
    brand_nutrition_2021.to_parquet(out_path_2021, index=False)
    print(f"Saved {len(brand_nutrition_2021)} 2021-era brands to {out_path_2021}")

    # Show top brands
    print(f"\n{'Brand':40s}  {'Sugar':>6s}  {'Fiber':>6s}  {'Prot':>6s}  "
          f"{'Fat':>6s}  {'Na(mg)':>7s}")
    print("-" * 80)
    for _, row in result.head(20).iterrows():
        def _fmt(v):
            return f"{v:6.1f}" if pd.notna(v) else "   NaN"
        print(f"{row['brand_descr']:40s}  {_fmt(row['sugar'])}  "
              f"{_fmt(row['fiber'])}  {_fmt(row['protein'])}  "
              f"{_fmt(row['fat'])}  {_fmt(row['sodium']):>7s}")


if __name__ == "__main__":
    main()
