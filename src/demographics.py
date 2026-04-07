"""
Stage 7: Compute demographic aggregates.

Produces market-level and household-level demographic variables
for demand estimation (Pi regressors in BLP/K-Aggregator).

Outputs:
  data/clean/demographics_market.parquet
    Index: (t, m) — quarter x DMA
    Columns: demographic means (household_income, male_head_age, ...)

  data/clean/demographics_household.parquet
    Index: (i, t) — household x quarter
    Columns: demographic values for each household-quarter

Usage:
    python src/demographics.py [--config config/default.yaml]
                               [--years 2014 2018]
"""
import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_markets import load_config, _build_fips_to_dma

_N_WORKERS = min(24, os.cpu_count() or 1)

# Column name mapping: legacy (2004-2020) -> canonical (lowercase)
_LEGACY_COLS = {
    "Household_Cd": "household_code",
    "Panel_Year": "panel_year",
    "DMA_Cd": "DMA_Cd",
    "Projection_Factor": "projection_factor",
    "Household_Income": "household_income",
    "Household_Size": "household_size",
    "Male_Head_Age": "male_head_age",
    "Female_Head_Age": "female_head_age",
    "Male_Head_Education": "male_head_education",
    "Female_Head_Education": "female_head_education",
    "Male_Head_Employment": "male_head_employment",
    "Female_Head_Employment": "female_head_employment",
    "Marital_Status": "marital_status",
    "Race": "race",
    "Hispanic_Origin": "hispanic_origin",
    "Age_And_Presence_Of_Children": "age_and_presence_of_children",
    "Fips_State_Cd": "fips_state_code",
    "Fips_County_Cd": "fips_county_code",
}


def _read_one_panelist_demog(raw_dir, year, demog_cols, fips_to_dma):
    """Read one year's panelist file with demographic columns."""
    path = raw_dir / f"panelists_{year}.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)

    if year <= 2020:
        # Extract added a lowercase panel_year; drop it before renaming
        # the original Panel_Year to avoid duplicates.
        if "panel_year" in df.columns and "Panel_Year" in df.columns:
            df = df.drop(columns=["panel_year"])
        rename = {k: v for k, v in _LEGACY_COLS.items() if k in df.columns}
        out = df.rename(columns=rename)
    else:
        out = df.copy()
        # Map FIPS -> DMA for 2021+ panelists
        if fips_to_dma is not None and "DMA_Cd" not in out.columns:
            mapped = out[["fips_state_code", "fips_county_code"]].merge(
                fips_to_dma, how="left",
                on=["fips_state_code", "fips_county_code"])
            out["DMA_Cd"] = mapped["DMA_Cd"].values

    # Keep only the columns we need
    keep = ["household_code", "panel_year", "DMA_Cd", "projection_factor"]
    keep += [c for c in demog_cols if c in out.columns]
    out = out[[c for c in keep if c in out.columns]]
    return out


def _load_panelists_demog(raw_dir, years, demog_cols, fips_to_dma):
    """Load panelist demographics across all years (threaded)."""
    with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
        dfs = list(pool.map(
            lambda y: _read_one_panelist_demog(raw_dir, y, demog_cols, fips_to_dma),
            years
        ))
    dfs = [d for d in dfs if d is not None]
    if not dfs:
        raise FileNotFoundError(f"No panelist files found in {raw_dir}")
    return pd.concat(dfs, ignore_index=True)


def _quarter_from_panel_year(panel_year, frequency):
    """Generate quarter labels for a panel year.

    Each panel year spans 4 quarters (or 12 months).
    Returns a DataFrame with (panel_year, t) rows.
    """
    if frequency == "quarterly":
        return pd.DataFrame({
            "panel_year": panel_year,
            "t": [f"Q{q}Y{panel_year}" for q in range(1, 5)],
        })
    elif frequency == "monthly":
        return pd.DataFrame({
            "panel_year": panel_year,
            "t": [f"M{m:02d}Y{panel_year}" for m in range(1, 13)],
        })
    else:
        raise ValueError(f"Unknown frequency: {frequency}")


def compute_demographics(cfg, years=None):
    """Compute market-level and household-level demographics.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration.
    years : list of int, optional
        Override year range.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        demographics_market (t, m) and demographics_household (i, t).
    """
    raw_dir = Path("data/raw")
    clean_dir = Path("data/clean")
    clean_dir.mkdir(parents=True, exist_ok=True)

    if years is None:
        year_lo, year_hi = cfg["years"]
        years = list(range(year_lo, year_hi + 1))

    available_years = [y for y in years
                       if (raw_dir / f"panelists_{y}.parquet").exists()]
    if not available_years:
        raise FileNotFoundError(f"No panelist files for years {years}")

    n_markets = cfg.get("n_markets", 65)
    frequency = cfg.get("frequency", "quarterly")
    demog_cols = cfg.get("demographics", ["household_income", "male_head_age"])

    print(f"Computing demographics for {len(available_years)} years: "
          f"{min(available_years)}-{max(available_years)}")
    print(f"  Demographics: {demog_cols}")
    print(f"  Frequency: {frequency}")

    # ------------------------------------------------------------------
    # Load panelists with demographics
    # ------------------------------------------------------------------
    print("\n[1/4] Building FIPS->DMA mapping...")
    fips_to_dma = _build_fips_to_dma(raw_dir)

    print("\n[2/4] Loading panelist demographics...")
    panelists = _load_panelists_demog(
        raw_dir, available_years, demog_cols, fips_to_dma
    )
    print(f"  Loaded {len(panelists):,} panelist-years")

    # Check demographic column availability
    available_demog = [c for c in demog_cols if c in panelists.columns]
    missing_demog = [c for c in demog_cols if c not in panelists.columns]
    if missing_demog:
        warnings.warn(f"Demographics not found in data: {missing_demog}")
    print(f"  Available demographics: {available_demog}")

    # Filter to top DMAs
    panelists = panelists.dropna(subset=["DMA_Cd"])
    panelists["DMA_Cd"] = panelists["DMA_Cd"].astype(int)
    hh_per_dma = (
        panelists.groupby("DMA_Cd")["household_code"].nunique()
        .sort_values(ascending=False)
    )
    top_dmas = hh_per_dma.head(n_markets).index.values
    panelists = panelists[panelists["DMA_Cd"].isin(top_dmas)]
    print(f"  After DMA filter (top {n_markets}): "
          f"{panelists['household_code'].nunique():,} households")

    # Expand panelist-years to panelist-quarters
    # (demographics are constant within a panel year)
    quarter_maps = pd.concat(
        [_quarter_from_panel_year(y, frequency) for y in available_years],
        ignore_index=True,
    )
    panelists_qt = panelists.merge(quarter_maps, on="panel_year", how="inner")

    # ------------------------------------------------------------------
    # Market-level demographics: weighted mean per (t, m)
    # ------------------------------------------------------------------
    print("\n[3/4] Computing market-level demographics...")

    demog_market = (
        panelists_qt
        .groupby(["t", "DMA_Cd"])
        .apply(
            lambda g: pd.Series({
                col: np.average(
                    g[col].dropna(),
                    weights=g.loc[g[col].notna(), "projection_factor"]
                ) if g[col].notna().any() else np.nan
                for col in available_demog
            }),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={"DMA_Cd": "m"})
        .set_index(["t", "m"])
        .sort_index()
    )

    print(f"  Shape: {demog_market.shape}")
    print(f"  Quarters: {demog_market.index.get_level_values('t').nunique()}")
    print(f"  Markets: {demog_market.index.get_level_values('m').nunique()}")
    for col in available_demog:
        s = demog_market[col].dropna()
        print(f"  {col}: mean={s.mean():.1f}, median={s.median():.1f}, "
              f"std={s.std():.1f}")

    # ------------------------------------------------------------------
    # Household-level demographics: per (i, t)
    # ------------------------------------------------------------------
    print("\n[4/4] Computing household-level demographics...")

    demog_hh = (
        panelists_qt[["household_code", "t", "DMA_Cd", "panel_year",
                       "projection_factor"] + available_demog]
        .rename(columns={"household_code": "i", "DMA_Cd": "m"})
    )
    demog_hh["market_ids"] = demog_hh["t"] + "_DMA_" + demog_hh["m"].astype(int).astype(str)
    demog_hh = demog_hh.set_index(["i", "t"]).sort_index()

    print(f"  Shape: {demog_hh.shape}")
    print(f"  Households: {demog_hh.index.get_level_values('i').nunique():,}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_market = clean_dir / "demographics_market.parquet"
    demog_market.to_parquet(out_market)
    print(f"\nSaved market demographics to {out_market}")

    out_hh = clean_dir / "demographics_household.parquet"
    demog_hh.to_parquet(out_hh)
    print(f"Saved household demographics to {out_hh}")

    return demog_market, demog_hh


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 7: Compute demographic aggregates")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--years", nargs=2, type=int, default=None,
                        help="Override year range (e.g., --years 2014 2018)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.years:
        years = list(range(args.years[0], args.years[1] + 1))
    else:
        years = None

    compute_demographics(cfg, years=years)


if __name__ == "__main__":
    main()
