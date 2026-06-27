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
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Parquet reads are I/O + decompression (releases the GIL), so threads help.
_N_WORKERS = min(24, os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="config/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Per-file readers (submitted to the shared thread pool)
# ---------------------------------------------------------------------------

def _read_one_purchase(raw_dir, year):
    """Read and harmonize one year's purchase parquet."""
    path = raw_dir / f"purchases_{year}.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df["panel_year"] = year

    if year <= 2020:
        keep = ["trip_code_uc", "upc", "quantity", "total_price_paid",
                "brand_code_uc", "brand_descr", "size1_amount",
                "size1_units", "panel_year"]
        for c in keep:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {path}")
        return df[keep]
    else:
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
        return out


def _read_one_trip(raw_dir, year):
    """Read one year's trip parquet (selected columns only)."""
    path = raw_dir / f"trips_{year}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(
        path,
        columns=["trip_code_uc", "household_code", "purchase_date",
                  "panel_year"]
    )


def _read_one_panelist(raw_dir, year, fips_to_dma):
    """Read and harmonize one year's panelist parquet."""
    path = raw_dir / f"panelists_{year}.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)

    if year <= 2020:
        out = pd.DataFrame()
        out["household_code"] = df["Household_Cd"]
        out["panel_year"] = df["Panel_Year"]
        out["DMA_Cd"] = df["DMA_Cd"]
        out["DMA_Name"] = df["DMA_Name"]
        out["projection_factor"] = df["Projection_Factor"]
        out["household_size"] = df["Household_Size"]
        return out
    else:
        out = pd.DataFrame()
        out["household_code"] = df["household_code"]
        out["panel_year"] = df["panel_year"]
        out["projection_factor"] = df["projection_factor"]
        out["household_size"] = (df["household_size"] if "household_size" in df.columns
                                 else df["Household_Size"])

        if fips_to_dma is not None:
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

        return out


def _read_fips_cols(path):
    """Read FIPS/DMA columns from one legacy panelist file."""
    return pd.read_parquet(path, columns=["Fips_State_Cd", "Fips_County_Cd",
                                           "DMA_Cd", "DMA_Name"])


# ---------------------------------------------------------------------------
# Bulk loader: one pool for all I/O
# ---------------------------------------------------------------------------

def _load_all_data(raw_dir, years, fips_to_dma):
    """Read purchases, trips, and panelists for all years in a single pool.

    Parquet reads are I/O + decompression (release the GIL), so one
    ThreadPoolExecutor with ~N_WORKERS threads keeps all cores busy
    across every file type simultaneously.

    Returns (purchases_df, trips_df, panelists_df).
    """
    with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
        # Submit every file read at once — ~63 reads for 21 years
        fut_purchases = {
            pool.submit(_read_one_purchase, raw_dir, y): y for y in years
        }
        fut_trips = {
            pool.submit(_read_one_trip, raw_dir, y): y for y in years
        }
        fut_panelists = {
            pool.submit(_read_one_panelist, raw_dir, y, fips_to_dma): y
            for y in years
        }

        def _collect(futures, label):
            dfs = [f.result() for f in futures]
            dfs = [d for d in dfs if d is not None]
            if not dfs:
                raise FileNotFoundError(
                    f"No {label} parquets found in {raw_dir}")
            out = pd.concat(dfs, ignore_index=True)
            print(f"  {label}: {len(out):,} rows from {len(dfs)} years")
            return out

        purchases = _collect(fut_purchases, "purchases")
        trips = _collect(fut_trips, "trips")
        panelists = _collect(fut_panelists, "panelists")

    return purchases, trips, panelists


def _build_fips_to_dma(raw_dir, pool=None):
    """Build a FIPS (state, county) -> DMA mapping from legacy panelist files.

    Uses all available legacy panelist files to maximize county coverage.
    If *pool* is provided, submits reads there; otherwise creates its own.
    """
    paths = [p for p in sorted(raw_dir.glob("panelists_*.parquet"))
             if int(p.stem.split("_")[1]) <= 2020]
    if not paths:
        return None

    if pool is not None:
        dfs = list(pool.map(_read_fips_cols, paths))
    else:
        with ThreadPoolExecutor(max_workers=_N_WORKERS) as own_pool:
            dfs = list(own_pool.map(_read_fips_cols, paths))

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
# Nutrition merge (FDC -> productattributes -> overrides)
# ---------------------------------------------------------------------------

def _merge_nutrition(panel, zeta_cols, fdc_path, nutrition_path, cfg):
    """Merge nutrition into panel using a three-tier fallback chain.

    1. FDC (config/fdc_nutrition.csv) — hand-verified, primary source
    2. Productattributes (data/clean/nutrition.parquet) — 2021+ NielsenIQ
    3. Overrides (config/nutrition_overrides.yaml) — manual entries for
       brands missing from both sources
    """
    fdc_path = Path(fdc_path)
    nutrition_path = Path(nutrition_path)

    if fdc_path.exists():
        fdc = pd.read_csv(fdc_path)[["brand_code_uc"] + zeta_cols]
        fdc = fdc.rename(columns={"brand_code_uc": "j"})
        panel = panel.merge(fdc, on="j", how="left")
        n_fdc = panel[zeta_cols[0]].notna().sum()
        print(f"  FDC nutrition: {n_fdc:,} / {len(panel):,} cells")

        # Fill gaps from productattributes
        if nutrition_path.exists():
            pa_nut = pd.read_parquet(nutrition_path)
            pa_lookup = (pa_nut.set_index("brand_code_uc")[zeta_cols]
                         .rename_axis("j"))
            for col in zeta_cols:
                mask = panel[col].isna()
                if mask.any():
                    panel.loc[mask, col] = (
                        panel.loc[mask, "j"].map(pa_lookup[col])
                    )
            n_after = panel[zeta_cols[0]].notna().sum()
            n_filled = n_after - n_fdc
            if n_filled > 0:
                print(f"  Productattributes fallback: {n_filled:,} more cells")
    elif nutrition_path.exists():
        nutrition = pd.read_parquet(nutrition_path)
        nut_join = (nutrition[["brand_code_uc"] + zeta_cols]
                    .rename(columns={"brand_code_uc": "j"}))
        panel = panel.merge(nut_join, on="j", how="left")
        n_with_zeta = panel[zeta_cols[0]].notna().sum()
        print(f"  Nutrition merged: {n_with_zeta:,} / {len(panel):,} cells")
    else:
        print("  WARNING: no nutrition data found; skipping zeta merge")
        for c in zeta_cols:
            panel[c] = np.nan

    # Apply manual overrides as final fallback
    # Overrides are keyed by brand_descr (brand NAME) and so apply only to the
    # market panel; the household panel is keyed by brand_code (no brand_descr),
    # where code-keyed FDC + productattributes already cover the kept brands.
    overrides_path = cfg.get("nutrition_overrides", "config/nutrition_overrides.yaml")
    if "brand_descr" in panel.columns and Path(overrides_path).exists():
        with open(overrides_path) as f:
            overrides = yaml.safe_load(f) or {}
        n_applied = 0
        for brand_descr, vals in overrides.items():
            if not isinstance(vals, dict):
                continue
            mask = panel["brand_descr"] == brand_descr
            if not mask.any():
                continue
            for col in zeta_cols:
                if col in vals and panel.loc[mask, col].isna().any():
                    panel.loc[mask & panel[col].isna(), col] = vals[col]
            if mask.any():
                n_applied += 1
        if n_applied:
            print(f"  Overrides applied: {n_applied} brands")

    n_final = panel[zeta_cols[0]].notna().sum()
    n_missing = len(panel) - n_final
    if n_missing > 0:
        _idcol = "brand_descr" if "brand_descr" in panel.columns else "j"
        missing_brands = panel.loc[panel[zeta_cols[0]].isna(), _idcol].unique()
        print(f"  Still missing {zeta_cols[0]}: {n_missing:,} cells "
              f"({list(missing_brands)})")

    return panel


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
            cached = top["brand_code_uc"].dropna().unique().tolist()
            # Guard: the cache (built by select_products.py) is NOT CTL-filtered
            # and may predate the current window.  `purchases` here is already
            # CTL-dropped, so intersect with the codes actually present -- a
            # stale/contaminated cache can then never reintroduce CTL or pick a
            # brand absent from this run.  Fall through to a fresh ranking if the
            # filtered cache is short of n_products.
            present = set(purchases["brand_code_uc"].dropna().unique())
            cached = [c for c in cached if c in present]
            if len(cached) >= n_products:
                print(f"  Using saved brand selection: {len(cached)} valid "
                      f"brands (CTL-guarded)")
                return np.array(cached[:n_products])
            print(f"  Saved selection has only {len(cached)} valid brands "
                  f"(<{n_products}); recomputing from CTL-dropped purchases")

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

    # Persons per (panel_year, DMA): sum household_size over UNIQUE households.
    # Intent is per-PERSON capacity x population (Nevo 2001), NOT households --
    # using a household count understated the market ~2.5x.
    _uniq = (
        panelists[panelists["DMA_Cd"].isin(top_dmas)]
        .dropna(subset=["household_size"])
        .drop_duplicates(subset=["panel_year", "DMA_Cd", "household_code"])
    )
    hh_counts = (
        _uniq.groupby(["panel_year", "DMA_Cd"])
        .agg(n_hh=("household_code", "nunique"),
             n_persons=("household_size", "sum"))
        .reset_index()
    )

    hh_counts["market_size"] = (
        hh_counts["n_persons"] * days * oz_per_day * 1.05
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
    # Step 2: Load all data (single thread pool for all file I/O)
    # ------------------------------------------------------------------
    print("\n[2/7] Loading data...")
    purchases, trips, panelists = _load_all_data(
        raw_dir, available_years, fips_to_dma
    )

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

    # Merge on (trip_code_uc, panel_year), NOT trip_code_uc alone.  The 2021+
    # HMS extracts re-use trip_code_uc as a PER-YEAR-SEQUENTIAL id (the same
    # integer is a different trip each HMS year: trips_2021 & trips_2022 share
    # 9.7M codes with 0% same-household), so a key of trip_code_uc alone fans
    # every HMS purchase out across all HMS years sharing the code (~3-4x
    # inflation of servings/transactions, surviving the panelist merge).  Adding
    # panel_year restricts each purchase to its own year's trip.  Legacy
    # (2004-2020) codes are globally unique -> this is a no-op there.  Same
    # incomplete-merge-key class as gotcha #18.
    merged = purchases.merge(
        trips_needed[["trip_code_uc", "panel_year", "household_code", "purchase_date"]],
        on=["trip_code_uc", "panel_year"],
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

    # Drop private label / store brands (CTL*) BEFORE selection + aggregation,
    # matching the original (Fally) analysis so the top-N are real brands and
    # both models share an identical, CTL-free product set.
    _ctl = merged["brand_descr"].astype(str).str.upper().str.startswith("CTL")
    _n_ctl = int(_ctl.sum())
    if _n_ctl:
        merged = merged[~_ctl]
        print(f"  Dropped {_n_ctl:,} private-label (CTL*) rows before selection")

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
    # Pure Cython path — no Python lambdas in the agg.
    grp_cols = ["t", "DMA_Cd", "brand_code_uc"]
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

    # Key market_size on the CALENDAR year parsed from the quarter label t, so
    # EVERY brand in a (t, DMA) cell gets the SAME market_size.  (Fixes the prior
    # per-(t,m,j) dominant-panel_year merge -> market_size varied by brand within
    # a cell; same s=(t,m) discipline class as K-Aggregators gotcha #17/#18.)
    agg["panel_year"] = agg["t"].str.extract(r"Y(\d{4})")[0].astype(int)
    agg = agg.merge(
        mkt_size[["panel_year", "DMA_Cd", "market_size", "n_hh", "n_persons"]],
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
    # S = QUANTITY inside-good share (servings / market_size) -- for BLP.
    agg["S"] = agg["servings"] / agg["market_size"]
    # W_exp = EXPENDITURE share (revenue_j / sum_k revenue_k) -- the K-aggregator
    # model's dependent object.  Sums to 1 over inside goods within (t, DMA).
    agg["W_exp"] = (
        agg["revenue"] / agg.groupby(["t", "DMA_Cd"])["revenue"].transform("sum")
    )

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

    # ------------------------------------------------------------------
    # Merge nutrition (zeta) and compute Z_agg
    # ------------------------------------------------------------------
    # FDC (config/fdc_nutrition.csv) is the primary source: hand-verified,
    # consistent serving sizes.  Fall back to nutrition.parquet (2021+
    # productattributes) for brands not in FDC.
    zeta_cols = ["sugar", "fiber", "protein", "fat", "sodium"]
    fdc_path = Path("config/fdc_nutrition.csv")
    nutrition_path = clean_dir / "nutrition.parquet"

    panel = _merge_nutrition(panel, zeta_cols, fdc_path, nutrition_path, cfg)

    # Z_agg: market-time aggregates Z_lt = sum_j S_jt * zeta_jl
    # These are share-weighted sums of characteristics per (t, m) cell.
    for col in zeta_cols:
        panel[f"Z_{col}"] = panel["S"] * panel[col]
    z_agg_cols = [f"Z_{c}" for c in zeta_cols]
    z_agg = (
        panel.groupby(["t", "m"])[z_agg_cols]
        .sum()
        .reset_index()
    )
    # Drop the per-cell S*zeta columns, merge back the (t,m) totals
    panel = panel.drop(columns=z_agg_cols)
    panel = panel.merge(z_agg, on=["t", "m"], how="left")
    print(f"  Z_agg computed: {z_agg_cols}")

    # pyblp / K-Aggregators compatible columns
    panel["market_ids"] = panel["t"] + "_DMA_" + panel["m"].astype(str)
    panel["product_ids"] = panel["j"].astype(str)
    panel["dma_cd"] = panel["m"]
    dates = pd.to_datetime(
        panel["t"].str.extract(r"Q(\d)Y(\d{4})").rename(columns={0: "q", 1: "y"})
        .assign(m=lambda d: d["q"].astype(int) * 3)
        .apply(lambda r: f"{r['y']}-{r['m']:02d}-01", axis=1)
    )
    panel["quarter"] = dates.dt.quarter

    # Rename to match pyblp / K-Aggregators conventions
    panel = panel.rename(columns={"S": "shares", "p": "prices"})

    # Select and order output columns
    out_cols = (["t", "m", "j", "market_ids", "product_ids",
                 "dma_cd", "quarter",
                 "shares", "prices", "brand_descr"]
                + zeta_cols + z_agg_cols
                + ["W_exp"]
                + ["servings", "revenue", "n_transactions",
                   "market_size", "n_hh", "n_persons", "panel_year"])
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
    print(f"    mean:   {panel['shares'].mean():.6f}")
    print(f"    median: {panel['shares'].median():.6f}")
    print(f"    max:    {panel['shares'].max():.6f}")
    print(f"    min:    {panel['shares'].min():.6f}")
    print(f"  Price ($/oz) statistics:")
    print(f"    mean:   {panel['prices'].mean():.4f}")
    print(f"    median: {panel['prices'].median():.4f}")
    print(f"    max:    {panel['prices'].max():.4f}")
    print(f"    min:    {panel['prices'].min():.4f}")

    # Inside-good share sum distribution
    inside = panel.groupby(level=["t", "m"])["shares"].sum()
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
