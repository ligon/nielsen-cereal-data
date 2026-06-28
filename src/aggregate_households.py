"""
Stage 6: Aggregate to household-level panel.

Produces the (household, quarter, brand) panel needed for microdata
demand estimation.  Shares loading and merging logic with
aggregate_markets.py (Stage 5).

Outputs:
  data/clean/household_panel.parquet
    Index: (i, t, j) — household x quarter x brand
    Columns: S (household share), p (price $/oz), servings, revenue,
             DMA_Cd, panel_year

  data/clean/household_base_shares.parquet
    Index: (i, j) — household x brand
    Columns: S_base (predetermined share from base period), DMA_Cd

Usage:
    python src/aggregate_households.py [--config config/default.yaml]
                                       [--years 2014 2018]
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_markets import (
    load_config,
    _build_fips_to_dma,
    _load_all_data,
    _apply_brand_crosswalk,
    _select_top_brands,
    _add_quarter,
    _compute_market_size,
    _merge_nutrition,
)


# ---------------------------------------------------------------------------
# Household-level aggregation
# ---------------------------------------------------------------------------

def aggregate_households(cfg, years=None):
    """Run the household-level aggregation pipeline.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration (from default.yaml).
    years : list of int, optional
        Override year range.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        household_panel and household_base_shares.
    """
    raw_dir = Path("data/raw")
    clean_dir = Path("data/clean")
    clean_dir.mkdir(parents=True, exist_ok=True)

    if years is None:
        year_lo, year_hi = cfg["years"]
        years = list(range(year_lo, year_hi + 1))

    available_years = [y for y in years
                       if (raw_dir / f"purchases_{y}.parquet").exists()]
    if not available_years:
        raise FileNotFoundError(
            f"No purchase files found for years {years} in {raw_dir}")
    print(f"Aggregating households for {len(available_years)} years: "
          f"{min(available_years)}-{max(available_years)}")

    n_products = cfg.get("n_products", 45)
    n_markets = cfg.get("n_markets", 65)
    oz_per_day = cfg.get("market_size_oz_per_day", 4.5)
    frequency = cfg.get("frequency", "quarterly")
    modules = cfg.get("modules", [1344])
    base_period = cfg.get("base_period", "2014Q1")

    # ------------------------------------------------------------------
    # Steps 1-5: identical to market aggregation (shared loading code)
    # ------------------------------------------------------------------
    print("\n[1/8] Building FIPS->DMA mapping...")
    fips_to_dma = _build_fips_to_dma(raw_dir)

    print("\n[2/8] Loading data...")
    purchases, trips, panelists = _load_all_data(
        raw_dir, available_years, fips_to_dma
    )

    if purchases["brand_code_uc"].isna().any():
        print("\n[3/8] Applying brand crosswalk...")
        purchases = _apply_brand_crosswalk(purchases, raw_dir, modules)
    else:
        print("\n[3/8] Brand crosswalk: not needed")

    n_before = len(purchases)
    purchases = purchases.dropna(subset=["brand_code_uc"])
    purchases["brand_code_uc"] = purchases["brand_code_uc"].astype(int)
    n_dropped = n_before - len(purchases)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} purchases with no brand_code_uc "
              f"({n_dropped / n_before:.1%})")

    print("\n[4/8] Merging purchases with trips and panelists...")
    purchase_trips = set(purchases["trip_code_uc"].unique())
    trips_needed = trips[trips["trip_code_uc"].isin(purchase_trips)]
    print(f"  Trips matching cereal purchases: {len(trips_needed):,} "
          f"of {len(trips):,}")

    # Merge on (trip_code_uc, panel_year), NOT trip_code_uc alone -- the 2021+
    # HMS extracts re-use trip_code_uc as a per-year-sequential id (same int =
    # different trip each HMS year), so a trip_code_uc-only key fans every HMS
    # purchase out across all HMS years sharing the code (~3-4x).  Identical bug
    # + fix as aggregate_markets.py (see that file's note); gotcha-#18 class.
    merged = purchases.merge(
        trips_needed[["trip_code_uc", "panel_year", "household_code", "purchase_date"]],
        on=["trip_code_uc", "panel_year"], how="inner"
    )
    n_unmatched = len(purchases) - len(merged)
    if n_unmatched > 0:
        warnings.warn(f"{n_unmatched:,} purchases had no matching trip")
    print(f"  After trip merge: {len(merged):,} rows")

    merged = merged.merge(
        panelists[["household_code", "panel_year", "DMA_Cd"]],
        on=["household_code", "panel_year"], how="inner"
    )
    print(f"  After panelist merge: {len(merged):,} rows")

    n_no_dma = merged["DMA_Cd"].isna().sum()
    if n_no_dma > 0:
        print(f"  Dropping {n_no_dma:,} rows with missing DMA")
        merged = merged.dropna(subset=["DMA_Cd"])
    merged["DMA_Cd"] = merged["DMA_Cd"].astype(int)

    merged = _add_quarter(merged)

    print("\n[5/8] Selecting top products and markets...")
    top_brand_codes = _select_top_brands(merged, n_products, clean_dir)
    merged = merged[merged["brand_code_uc"].isin(top_brand_codes)]
    print(f"  After brand filter: {len(merged):,} rows")

    hh_per_dma = (
        panelists.dropna(subset=["DMA_Cd"])
        .groupby("DMA_Cd")["household_code"].nunique()
        .sort_values(ascending=False)
    )
    top_dmas = hh_per_dma.head(n_markets).index.values
    merged = merged[merged["DMA_Cd"].isin(top_dmas)]
    print(f"  After DMA filter (top {n_markets}): {len(merged):,} rows")

    # Filter to OZ
    oz_mask = merged["size1_units"].str.upper().str.strip() == "OZ"
    n_non_oz = (~oz_mask).sum()
    if n_non_oz > 0:
        print(f"  Dropping {n_non_oz:,} non-OZ purchases ({n_non_oz / len(merged):.1%})")
        merged = merged[oz_mask]

    merged["servings_oz"] = merged["quantity"] * merged["size1_amount"]

    # ------------------------------------------------------------------
    # Step 6: Aggregate to (household, quarter, brand) cells
    # ------------------------------------------------------------------
    print("\n[6/8] Aggregating to household cells...")

    grp_cols = ["household_code", "t", "brand_code_uc"]
    agg = (
        merged
        .groupby(grp_cols)
        .agg(
            servings=("servings_oz", "sum"),
            revenue=("total_price_paid", "sum"),
            n_transactions=("total_price_paid", "count"),
        )
        .reset_index()
    )

    # Dominant panel_year and DMA per cell
    cell_info = (
        merged
        .groupby(grp_cols + ["panel_year", "DMA_Cd"])["total_price_paid"]
        .sum()
        .reset_index()
        .sort_values("total_price_paid", ascending=False)
        .drop_duplicates(subset=grp_cols)
        [grp_cols + ["panel_year", "DMA_Cd"]]
    )
    agg = agg.merge(cell_info, on=grp_cols, how="left")

    # Price per oz
    agg["p"] = agg["revenue"] / agg["servings"]
    agg["p"] = agg["p"].replace([np.inf, -np.inf, 0.0], np.nan)
    bad_prices = agg["p"].isna().sum()
    if bad_prices > 0:
        print(f"  Dropping {bad_prices} cells with undefined prices")
        agg = agg.dropna(subset=["p"])

    print(f"  Household cells: {len(agg):,}")
    print(f"  Households: {agg['household_code'].nunique():,}")
    print(f"  Quarters: {agg['t'].nunique()}")
    print(f"  Brands: {agg['brand_code_uc'].nunique()}")

    # ------------------------------------------------------------------
    # Step 7: Compute household shares
    # ------------------------------------------------------------------
    print("\n[7/8] Computing household shares...")

    mkt_size = _compute_market_size(panelists, top_dmas, oz_per_day, frequency)

    # Household share = household servings / TOTAL market size -- the same
    # denominator the market panel uses, so household shares are bounded [0,1]
    # and directly comparable to market shares.  (The earlier
    # servings/(market_size/n_hh) per-household normalization produced S>1, max
    # 7.5; recoverable as S*n_hh if a per-household consumption index is wanted.)
    agg = agg.merge(
        mkt_size[["panel_year", "DMA_Cd", "market_size", "n_hh"]],
        on=["panel_year", "DMA_Cd"], how="left"
    )

    n_no_mkt = agg["market_size"].isna().sum()
    if n_no_mkt > 0:
        print(f"  Dropping {n_no_mkt:,} cells with no market size")
        agg = agg.dropna(subset=["market_size"])

    agg["S"] = agg["servings"] / agg["market_size"]

    # S_exp: household EXPENDITURE budget share across brands -- the K-aggregator
    # dependent object (paper notation S; we are deprecating W in favor of S --
    # see ledger).  Sums to 1 over j within (household, quarter).  `S` above is
    # the QUANTITY share (servings/market_size) for the BLP-style micro model.
    agg["S_exp"] = (
        agg["revenue"]
        / agg.groupby(["household_code", "t"])["revenue"].transform("sum")
    )

    # Validate
    hh_quarter_share = agg.groupby(["household_code", "t"])["S"].sum()
    print(f"  Household-quarter inside share sum:")
    print(f"    mean:   {hh_quarter_share.mean():.4f}")
    print(f"    median: {hh_quarter_share.median():.4f}")
    print(f"    max:    {hh_quarter_share.max():.4f}")

    n_over = (hh_quarter_share >= 1.0).sum()
    if n_over > 0:
        warnings.warn(
            f"{n_over} household-quarter cells have inside shares >= 1.0! "
            f"Max = {hh_quarter_share.max():.4f}"
        )

    # Rename for output
    panel = agg.rename(columns={
        "household_code": "i", "brand_code_uc": "j",
        "DMA_Cd": "m",
    })

    # ------------------------------------------------------------------
    # Merge nutrition (zeta) — FDC -> productattributes -> overrides
    # ------------------------------------------------------------------
    zeta_cols = ["sugar", "fiber", "protein", "fat", "sodium"]
    panel = _merge_nutrition(
        panel, zeta_cols,
        Path("config/fdc_nutrition.csv"),
        clean_dir / "nutrition.parquet",
        cfg,
    )

    # Z_i: household-level aggregates Z_il = sum_j S_ij * zeta_jl
    for col in zeta_cols:
        panel[f"Z_{col}"] = panel["S"] * panel[col]
    z_cols = [f"Z_{c}" for c in zeta_cols]
    z_hh = (
        panel.groupby(["i", "t"])[z_cols]
        .sum()
        .reset_index()
    )
    panel = panel.drop(columns=z_cols)
    panel = panel.merge(z_hh, on=["i", "t"], how="left")
    print(f"  Z_i computed: {z_cols}")

    # ------------------------------------------------------------------
    # Merge demographics
    # ------------------------------------------------------------------
    demog_path = clean_dir / "demographics_household.parquet"
    if demog_path.exists():
        demog = pd.read_parquet(demog_path)
        # demog is indexed by (i, t); reset for merge
        demog = demog.reset_index()
        # Keep ONLY the demographic variables; exclude index/metadata columns
        # (panel_year, m, market_ids, projection_factor) which would collide with
        # the panel's existing columns on the (i,t) merge -> _x/_y suffixes ->
        # KeyError downstream.
        _demog_meta = {"i", "t", "m", "panel_year", "market_ids", "projection_factor"}
        demog_cols = [c for c in demog.columns if c not in _demog_meta]
        panel = panel.merge(demog[["i", "t"] + demog_cols],
                            on=["i", "t"], how="left")
        n_with_demog = panel[demog_cols[0]].notna().sum()
        print(f"  Demographics merged: {n_with_demog:,} / {len(panel):,} cells")
    else:
        print("  WARNING: demographics_household.parquet not found")
        demog_cols = []

    out_cols = (["i", "t", "j", "S", "S_exp", "p", "m"]
                + zeta_cols + z_cols + demog_cols
                + ["servings", "revenue", "n_transactions", "panel_year"])
    panel = panel[out_cols].set_index(["i", "t", "j"]).sort_index()

    # ------------------------------------------------------------------
    # Step 8: Base period shares
    # ------------------------------------------------------------------
    print(f"\n[8/8] Computing base period shares (base={base_period})...")

    # Parse base_period: "2014Q1" or "Q1Y2014"
    if base_period[0] == "Q":
        base_t = base_period  # already in QxYyyyy format
    else:
        # "2014Q1" -> "Q1Y2014"
        year_str = base_period[:4]
        q_str = base_period[4:]
        base_t = f"{q_str}Y{year_str}"

    base = panel.loc[panel.index.get_level_values("t") == base_t].copy()
    if len(base) == 0:
        # Try the original format
        base = panel.loc[panel.index.get_level_values("t") == base_period].copy()
    if len(base) == 0:
        warnings.warn(
            f"No data for base period '{base_period}'; "
            f"base shares will be empty. Available quarters: "
            f"{sorted(panel.index.get_level_values('t').unique())[:5]}..."
        )
        base_shares = pd.DataFrame(
            columns=["i", "j", "S_base", "m"]).set_index(["i", "j"])
    else:
        base_shares = (
            base[["S", "m"]]
            .rename(columns={"S": "S_base"})
            .droplevel("t")
        )
        print(f"  Base period households: {base_shares.index.get_level_values('i').nunique():,}")
        print(f"  Base period cells: {len(base_shares):,}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_panel = clean_dir / "household_panel.parquet"
    panel.to_parquet(out_panel)
    print(f"\nSaved household panel to {out_panel}")
    print(f"  Shape: {panel.shape}")
    print(f"  Index: (i, t, j) with "
          f"{panel.index.get_level_values('i').nunique():,} households x "
          f"{panel.index.get_level_values('t').nunique()} quarters x "
          f"{panel.index.get_level_values('j').nunique()} brands")

    out_base = clean_dir / "household_base_shares.parquet"
    base_shares.to_parquet(out_base)
    print(f"Saved base shares to {out_base}")

    # Validation summary
    print(f"\n  Share statistics:")
    print(f"    mean:   {panel['S'].mean():.6f}")
    print(f"    median: {panel['S'].median():.6f}")
    print(f"    max:    {panel['S'].max():.6f}")

    products_per_hh_q = panel.groupby(level=["i", "t"]).size()
    print(f"  Products per household-quarter:")
    print(f"    mean:   {products_per_hh_q.mean():.1f}")
    print(f"    median: {products_per_hh_q.median():.0f}")
    print(f"    max:    {products_per_hh_q.max()}")

    pos_cells = (panel["S"] > 0).sum()
    print(f"  Positive-purchase cells: {pos_cells:,} / {len(panel):,} "
          f"({pos_cells / len(panel):.1%})")

    return panel, base_shares


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Aggregate to household-level panel")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--years", nargs=2, type=int, default=None,
                        help="Override year range (e.g., --years 2014 2018)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.years:
        years = list(range(args.years[0], args.years[1] + 1))
    else:
        years = None

    aggregate_households(cfg, years=years)


if __name__ == "__main__":
    main()
