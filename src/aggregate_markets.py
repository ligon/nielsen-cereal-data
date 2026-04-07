"""
Stage 3: Aggregate household purchases to market-level panel.

Loads per-year purchase, trip, and panelist parquets (from Stage 1),
merges them, and aggregates to (quarter, DMA, brand) cells with
market shares, prices, and household counts.

The output market_panel.parquet is the main input for demand estimation.

Column conventions:
  - DMA codes are Nielsen DMA_Cd (e.g. 501 = New York).
  - Quarter labels: "Q1Y2014", "Q2Y2014", etc.
  - Brand identifier: brand_code_uc (numeric, from master products).
    For 2021+ data without brand_code_uc, the brand crosswalk maps
    brand_descr_hms back to brand_code_uc.

Usage:
    python src/aggregate_markets.py [--config config/default.yaml]
                                    [--years 2014 2018]
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="config/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_purchases(raw_dir, years):
    """Load and harmonize purchase parquets across eras.

    Returns a DataFrame with columns:
      trip_code_uc, upc, quantity, total_price_paid,
      brand_code_uc, brand_descr, size1_amount, size1_units, panel_year
    """
    dfs = []
    for year in years:
        path = raw_dir / f"purchases_{year}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue

        df = pd.read_parquet(path)
        df["panel_year"] = year

        if year <= 2020:
            # Legacy era: brand_code_uc and size1_amount already merged
            # from master at extraction time.
            keep = ["trip_code_uc", "upc", "quantity", "total_price_paid",
                    "brand_code_uc", "brand_descr", "size1_amount",
                    "size1_units", "panel_year"]
            # Ensure all columns exist
            for c in keep:
                if c not in df.columns:
                    raise ValueError(f"Missing column {c} in {path}")
            dfs.append(df[keep])

        else:
            # 2021+ era: different column names, size1_amount_hms is
            # encoded as integer-string in units of 0.001 (e.g. "0012200"
            # = 12.200 oz).
            out = pd.DataFrame()
            out["trip_code_uc"] = df["trip_code_uc"]
            out["upc"] = df["upc"]
            out["quantity"] = df["quantity"]
            out["total_price_paid"] = df["total_price_paid"]
            out["brand_code_uc"] = np.nan  # filled later via crosswalk
            out["brand_descr"] = df["brand_descr_hms"]
            out["size1_amount"] = (
                pd.to_numeric(df["size1_amount_hms"], errors="coerce") / 1000.0
            )
            out["size1_units"] = df["size1_unit_hms"].str.strip()
            out["panel_year"] = year
            dfs.append(out)

    if not dfs:
        raise FileNotFoundError(f"No purchase parquets found in {raw_dir}")

    purchases = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(purchases):,} purchases from "
          f"{purchases['panel_year'].nunique()} years")
    return purchases


def _load_trips(raw_dir, years):
    """Load trip parquets.  Keep only columns needed for the merge."""
    dfs = []
    for year in years:
        path = raw_dir / f"trips_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(
            path,
            columns=["trip_code_uc", "household_code", "purchase_date",
                      "panel_year"]
        )
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No trip parquets found in {raw_dir}")

    trips = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(trips):,} trips")
    return trips


def _load_panelists_with_dma(raw_dir, years, fips_to_dma=None):
    """Load panelist parquets and extract (household_code, panel_year, DMA_Cd).

    Legacy (2004-2020): DMA_Cd is a direct column (title-case: "DMA_Cd").
    New (2021+): DMA is missing; map via fips_state_code + fips_county_code.
    """
    dfs = []
    for year in years:
        path = raw_dir / f"panelists_{year}.parquet"
        if not path.exists():
            continue

        df = pd.read_parquet(path)

        if year <= 2020:
            # Legacy era: title-case column names
            out = pd.DataFrame()
            out["household_code"] = df["Household_Cd"]
            out["panel_year"] = df["Panel_Year"]
            out["DMA_Cd"] = df["DMA_Cd"]
            out["DMA_Name"] = df["DMA_Name"]
            out["projection_factor"] = df["Projection_Factor"]
            dfs.append(out)
        else:
            # 2021+ era: lowercase column names, no DMA column
            out = pd.DataFrame()
            out["household_code"] = df["household_code"]
            out["panel_year"] = df["panel_year"]
            out["projection_factor"] = df["projection_factor"]

            if fips_to_dma is not None:
                # Map FIPS county -> DMA
                fips = df[["fips_state_code", "fips_county_code"]].copy()
                mapped = fips.merge(fips_to_dma, how="left",
                                    left_on=["fips_state_code", "fips_county_code"],
                                    right_on=["fips_state_code", "fips_county_code"])
                out["DMA_Cd"] = mapped["DMA_Cd"].values
                out["DMA_Name"] = mapped["DMA_Name"].values
                n_miss = out["DMA_Cd"].isna().sum()
                if n_miss > 0:
                    warnings.warn(
                        f"{year}: {n_miss}/{len(out)} panelists could not be "
                        f"mapped to a DMA via FIPS county ({n_miss/len(out):.1%})"
                    )
            else:
                out["DMA_Cd"] = np.nan
                out["DMA_Name"] = np.nan
                warnings.warn(f"{year}: No FIPS->DMA mapping; all DMA values are NaN")

            dfs.append(out)

    if not dfs:
        raise FileNotFoundError(f"No panelist parquets found in {raw_dir}")

    panelists = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(panelists):,} panelist-years")
    return panelists


def _build_fips_to_dma(raw_dir):
    """Build a FIPS (state, county) -> DMA mapping from legacy panelist files.

    Uses all available legacy panelist files to maximize county coverage.
    """
    dfs = []
    for path in sorted(raw_dir.glob("panelists_*.parquet")):
        year = int(path.stem.split("_")[1])
        if year > 2020:
            continue
        df = pd.read_parquet(path, columns=["Fips_State_Cd", "Fips_County_Cd",
                                             "DMA_Cd", "DMA_Name"])
        dfs.append(df)

    if not dfs:
        return None

    all_fips = pd.concat(dfs, ignore_index=True)
    # Take the mode DMA for each FIPS county (should be unique, but be safe)
    mapping = (
        all_fips
        .groupby(["Fips_State_Cd", "Fips_County_Cd"])
        .agg(DMA_Cd=("DMA_Cd", lambda x: x.mode().iloc[0]),
             DMA_Name=("DMA_Name", lambda x: x.mode().iloc[0]))
        .reset_index()
        .rename(columns={"Fips_State_Cd": "fips_state_code",
                          "Fips_County_Cd": "fips_county_code"})
    )
    print(f"  Built FIPS->DMA mapping: {len(mapping)} counties -> "
          f"{mapping['DMA_Cd'].nunique()} DMAs")
    return mapping


# ---------------------------------------------------------------------------
# Brand unification
# ---------------------------------------------------------------------------

def _apply_brand_crosswalk(purchases, raw_dir, modules):
    """Fill in brand_code_uc for 2021+ purchases using the brand crosswalk.

    The crosswalk maps brand_descr_hms -> brand_code_uc via UPC overlap
    between eras.  If the crosswalk file doesn't exist, build it on the fly.
    """
    clean_dir = raw_dir.parent / "clean"
    xwalk_path = clean_dir / "brand_crosswalk.parquet"

    if xwalk_path.exists():
        xwalk = pd.read_parquet(xwalk_path)
    else:
        print("  Brand crosswalk not found; building from UPC overlap...")
        from brand_crosswalk import build_crosswalk
        xwalk = build_crosswalk(str(raw_dir), modules)

    if xwalk.empty:
        warnings.warn("Brand crosswalk is empty; 2021+ brands will be unmatched")
        return purchases

    # Map: brand_descr (2021+ names) -> brand_code_uc
    descr_to_code = (
        xwalk[["brand_code_uc", "brand_descr_2021"]]
        .drop_duplicates("brand_descr_2021")
        .rename(columns={"brand_descr_2021": "brand_descr"})
    )

    # Fill missing brand_code_uc from the crosswalk
    mask = purchases["brand_code_uc"].isna()
    if mask.any():
        lookup = purchases.loc[mask, ["brand_descr"]].merge(
            descr_to_code, on="brand_descr", how="left"
        )
        purchases.loc[mask, "brand_code_uc"] = lookup["brand_code_uc"].values
        n_filled = mask.sum() - purchases.loc[mask, "brand_code_uc"].isna().sum()
        n_still_missing = purchases["brand_code_uc"].isna().sum()
        print(f"  Brand crosswalk: filled {n_filled:,} of {mask.sum():,} "
              f"2021+ rows; {n_still_missing:,} still missing")

    return purchases


# ---------------------------------------------------------------------------
# Quarter computation
# ---------------------------------------------------------------------------

def _add_quarter(df):
    """Add a 't' column (quarter label) from purchase_date.

    purchase_date format: "YYYY-MM-DD" (string).
    Quarter label: "Q1Y2014", "Q2Y2014", etc.
    """
    dates = pd.to_datetime(df["purchase_date"], format="%Y-%m-%d")
    df["t"] = ("Q" + dates.dt.quarter.astype(str)
               + "Y" + dates.dt.year.astype(str))
    return df


# ---------------------------------------------------------------------------
# Product selection
# ---------------------------------------------------------------------------

def _select_top_brands(purchases, n_products, clean_dir):
    """Select top N brands by total expenditure.

    If data/clean/selected_brands.parquet exists, use it.
    Otherwise compute from purchases.
    """
    sel_path = clean_dir / "selected_brands.parquet"
    if sel_path.exists():
        top = pd.read_parquet(sel_path)
        if "brand_code_uc" in top.columns:
            brand_codes = top["brand_code_uc"].dropna().unique()
            print(f"  Using saved brand selection: {len(brand_codes)} brands")
            return brand_codes

    # Compute from purchases (only rows with valid brand_code_uc)
    valid = purchases.dropna(subset=["brand_code_uc"])
    spend = (
        valid
        .groupby("brand_code_uc")["total_price_paid"]
        .sum()
        .sort_values(ascending=False)
    )
    brand_codes = spend.head(n_products).index.values
    print(f"  Selected top {len(brand_codes)} brands by expenditure "
          f"(covers {spend.head(n_products).sum() / spend.sum():.1%} "
          f"of cereal spending)")
    return brand_codes


# ---------------------------------------------------------------------------
# Market size
# ---------------------------------------------------------------------------

def _compute_market_size(panelists, top_dmas, oz_per_day, frequency):
    """Compute market size per (t, DMA).

    Market size = n_households * days_per_period * oz_per_day * 1.05

    The 1.05 factor accounts for away-from-home cereal consumption not
    captured in the Nielsen panel.

    We count unique households per (panel_year, DMA) as the proxy for
    population.  For quarterly aggregation, days_per_period = 90 (approx).
    """
    if frequency == "quarterly":
        days = 90
    elif frequency == "monthly":
        days = 30
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    # Count households per (panel_year, DMA)
    hh_counts = (
        panelists[panelists["DMA_Cd"].isin(top_dmas)]
        .groupby(["panel_year", "DMA_Cd"])["household_code"]
        .nunique()
        .reset_index()
        .rename(columns={"household_code": "n_hh"})
    )

    hh_counts["market_size"] = (
        hh_counts["n_hh"] * days * oz_per_day * 1.05
    )

    return hh_counts


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def aggregate_markets(cfg, years=None):
    """Run the full aggregation pipeline.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration (from default.yaml).
    years : list of int, optional
        Override year range.  Default: from config.

    Returns
    -------
    pd.DataFrame
        Market panel with index (t, m, j).
    """
    raw_dir = Path("data/raw")
    clean_dir = Path("data/clean")
    clean_dir.mkdir(parents=True, exist_ok=True)

    if years is None:
        year_lo, year_hi = cfg["years"]
        years = list(range(year_lo, year_hi + 1))

    # Determine which years actually have data
    available_years = [y for y in years if (raw_dir / f"purchases_{y}.parquet").exists()]
    if not available_years:
        raise FileNotFoundError(
            f"No purchase files found for years {years} in {raw_dir}"
        )
    print(f"Aggregating markets for {len(available_years)} years: "
          f"{min(available_years)}-{max(available_years)}")

    n_products = cfg.get("n_products", 45)
    n_markets = cfg.get("n_markets", 65)
    oz_per_day = cfg.get("market_size_oz_per_day", 4.5)
    frequency = cfg.get("frequency", "quarterly")
    modules = cfg.get("modules", [1344])

    # ------------------------------------------------------------------
    # Step 1: Build FIPS->DMA mapping from legacy panelists
    # ------------------------------------------------------------------
    print("\n[1/7] Building FIPS->DMA mapping...")
    fips_to_dma = _build_fips_to_dma(raw_dir)

    # ------------------------------------------------------------------
    # Step 2: Load data
    # ------------------------------------------------------------------
    print("\n[2/7] Loading data...")
    purchases = _load_purchases(raw_dir, available_years)
    trips = _load_trips(raw_dir, available_years)
    panelists = _load_panelists_with_dma(raw_dir, available_years,
                                          fips_to_dma=fips_to_dma)

    # ------------------------------------------------------------------
    # Step 3: Apply brand crosswalk for 2021+ data
    # ------------------------------------------------------------------
    if purchases["brand_code_uc"].isna().any():
        print("\n[3/7] Applying brand crosswalk for 2021+ purchases...")
        purchases = _apply_brand_crosswalk(purchases, raw_dir, modules)
    else:
        print("\n[3/7] Brand crosswalk: not needed (all purchases have brand_code_uc)")

    # Drop purchases with no brand_code_uc (unmatchable 2021+ brands)
    n_before = len(purchases)
    purchases = purchases.dropna(subset=["brand_code_uc"])
    purchases["brand_code_uc"] = purchases["brand_code_uc"].astype(int)
    n_dropped = n_before - len(purchases)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} purchases with no brand_code_uc "
              f"({n_dropped/n_before:.1%})")

    # ------------------------------------------------------------------
    # Step 4: Merge purchases <- trips <- panelists
    # ------------------------------------------------------------------
    print("\n[4/7] Merging purchases with trips and panelists...")

    # Merge purchases with trips to get household_code and purchase_date
    # Only keep trip_code_uc values that appear in purchases to avoid
    # an enormous merge with 10M+ trip rows
    purchase_trips = set(purchases["trip_code_uc"].unique())
    trips_needed = trips[trips["trip_code_uc"].isin(purchase_trips)]
    print(f"  Trips matching cereal purchases: {len(trips_needed):,} "
          f"of {len(trips):,}")

    merged = purchases.merge(
        trips_needed[["trip_code_uc", "household_code", "purchase_date"]],
        on="trip_code_uc",
        how="inner"
    )
    n_unmatched = len(purchases) - len(merged)
    if n_unmatched > 0:
        warnings.warn(f"{n_unmatched:,} purchases had no matching trip")
    print(f"  After trip merge: {len(merged):,} rows")

    # Merge with panelists to get DMA
    # Panelist file is keyed by (household_code, panel_year)
    merged = merged.merge(
        panelists[["household_code", "panel_year", "DMA_Cd"]],
        on=["household_code", "panel_year"],
        how="inner"
    )
    print(f"  After panelist merge: {len(merged):,} rows")

    # Drop rows with missing DMA
    n_no_dma = merged["DMA_Cd"].isna().sum()
    if n_no_dma > 0:
        print(f"  Dropping {n_no_dma:,} rows with missing DMA ({n_no_dma/len(merged):.1%})")
        merged = merged.dropna(subset=["DMA_Cd"])
    merged["DMA_Cd"] = merged["DMA_Cd"].astype(int)

    # Add quarter label
    merged = _add_quarter(merged)

    # ------------------------------------------------------------------
    # Step 5: Select top products and top DMAs
    # ------------------------------------------------------------------
    print("\n[5/7] Selecting top products and markets...")

    # Top brands
    top_brand_codes = _select_top_brands(merged, n_products, clean_dir)
    merged = merged[merged["brand_code_uc"].isin(top_brand_codes)]
    print(f"  After brand filter: {len(merged):,} rows")

    # Top DMAs by household count
    hh_per_dma = (
        panelists
        .dropna(subset=["DMA_Cd"])
        .groupby("DMA_Cd")["household_code"]
        .nunique()
        .sort_values(ascending=False)
    )
    top_dmas = hh_per_dma.head(n_markets).index.values
    merged = merged[merged["DMA_Cd"].isin(top_dmas)]
    print(f"  After DMA filter (top {n_markets}): {len(merged):,} rows")

    # ------------------------------------------------------------------
    # Step 6: Aggregate to (t, m, j) cells
    # ------------------------------------------------------------------
    print("\n[6/7] Aggregating to market cells...")

    # Compute total ounces: quantity * size1_amount
    # Filter to OZ units only (drop CT, GRAM, POUND for now)
    oz_mask = merged["size1_units"].str.upper().str.strip() == "OZ"
    n_non_oz = (~oz_mask).sum()
    if n_non_oz > 0:
        print(f"  Dropping {n_non_oz:,} non-OZ purchases ({n_non_oz/len(merged):.1%})")
        merged = merged[oz_mask]

    merged["servings_oz"] = merged["quantity"] * merged["size1_amount"]

    # Group by (quarter, DMA, brand).
    # Also carry panel_year through for the market-size merge.  When a
    # quarter straddles a panel_year boundary (rare edge case with trips
    # filed in Dec of the prior calendar year), the modal panel_year
    # within each cell is used.
    agg = (
        merged
        .groupby(["t", "DMA_Cd", "brand_code_uc"])
        .agg(
            servings=("servings_oz", "sum"),
            revenue=("total_price_paid", "sum"),
            n_transactions=("total_price_paid", "count"),
            panel_year=("panel_year", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )

    # Price = revenue / servings (per oz)
    agg["p"] = agg["revenue"] / agg["servings"]

    # Replace inf/nan/zero prices with NaN and drop
    agg["p"] = agg["p"].replace([np.inf, -np.inf, 0.0], np.nan)
    bad_prices = agg["p"].isna().sum()
    if bad_prices > 0:
        print(f"  WARNING: {bad_prices} cells have undefined or zero prices (dropped)")
        agg = agg.dropna(subset=["p"])

    print(f"  Market cells: {len(agg):,}")
    print(f"  Quarters: {agg['t'].nunique()}")
    print(f"  DMAs: {agg['DMA_Cd'].nunique()}")
    print(f"  Brands: {agg['brand_code_uc'].nunique()}")

    # ------------------------------------------------------------------
    # Step 7: Compute market shares
    # ------------------------------------------------------------------
    print("\n[7/7] Computing market shares...")

    # Market size per (panel_year, DMA)
    mkt_size = _compute_market_size(panelists, top_dmas, oz_per_day, frequency)

    agg = agg.merge(
        mkt_size[["panel_year", "DMA_Cd", "market_size", "n_hh"]],
        on=["panel_year", "DMA_Cd"],
        how="left"
    )

    # Drop cells with missing market size (e.g. calendar year in quarter
    # label doesn't match any panel_year with household data)
    n_no_mkt = agg["market_size"].isna().sum()
    if n_no_mkt > 0:
        print(f"  Dropping {n_no_mkt:,} cells with no market size "
              f"({n_no_mkt/len(agg):.1%})")
        agg = agg.dropna(subset=["market_size"])

    # Shares
    agg["S"] = agg["servings"] / agg["market_size"]

    # Validate shares
    share_sum = agg.groupby(["t", "DMA_Cd"])["S"].sum()
    max_share_sum = share_sum.max()
    print(f"  Max inside-good share sum: {max_share_sum:.4f}")
    if max_share_sum >= 1.0:
        n_over = (share_sum >= 1.0).sum()
        warnings.warn(
            f"{n_over} (t, m) cells have inside shares summing to >= 1.0! "
            f"Max = {max_share_sum:.4f}. Check market_size_oz_per_day."
        )

    high_share = agg[agg["S"] > 0.3]
    if len(high_share) > 0:
        warnings.warn(
            f"{len(high_share)} cells have individual share > 0.3; "
            f"largest = {agg['S'].max():.4f}"
        )

    # Add brand description for readability
    brand_names = (
        merged[["brand_code_uc", "brand_descr"]]
        .drop_duplicates("brand_code_uc")
    )
    agg = agg.merge(brand_names, on="brand_code_uc", how="left")

    # Rename for output
    panel = agg.rename(columns={"DMA_Cd": "m", "brand_code_uc": "j"})

    # Select and order output columns
    out_cols = ["t", "m", "j", "S", "p", "brand_descr",
                "servings", "revenue", "n_transactions",
                "market_size", "n_hh", "panel_year"]
    panel = panel[out_cols].copy()

    # Set index
    panel = panel.set_index(["t", "m", "j"]).sort_index()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = clean_dir / "market_panel.parquet"
    panel.to_parquet(out_path)
    print(f"\nSaved market panel to {out_path}")
    print(f"  Shape: {panel.shape}")
    print(f"  Index: (t, m, j) with {panel.index.get_level_values('t').nunique()} "
          f"quarters x {panel.index.get_level_values('m').nunique()} DMAs "
          f"x {panel.index.get_level_values('j').nunique()} brands")

    # Summary stats
    print(f"\n  Share statistics:")
    print(f"    mean:   {panel['S'].mean():.6f}")
    print(f"    median: {panel['S'].median():.6f}")
    print(f"    max:    {panel['S'].max():.6f}")
    print(f"    min:    {panel['S'].min():.6f}")
    print(f"  Price ($/oz) statistics:")
    print(f"    mean:   {panel['p'].mean():.4f}")
    print(f"    median: {panel['p'].median():.4f}")
    print(f"    max:    {panel['p'].max():.4f}")
    print(f"    min:    {panel['p'].min():.4f}")

    # Inside-good share sum distribution
    inside = panel.groupby(level=["t", "m"])["S"].sum()
    print(f"  Inside-good share sum:")
    print(f"    mean:   {inside.mean():.4f}")
    print(f"    median: {inside.median():.4f}")
    print(f"    max:    {inside.max():.4f}")
    print(f"    min:    {inside.min():.4f}")

    return panel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Aggregate household purchases to market panel")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--years", nargs=2, type=int, default=None,
                        help="Override year range (e.g., --years 2014 2018)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.years:
        years = list(range(args.years[0], args.years[1] + 1))
    else:
        years = None

    aggregate_markets(cfg, years=years)


if __name__ == "__main__":
    main()
