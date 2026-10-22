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
    Columns: demographic values for each household-quarter, plus
             household_size (NielsenIQ Household_Size: persons, 1-9, code 9
             = nine or more; the same raw column AM's market_size sums)

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

    # Keep only the columns we need.  household_size is always kept (not a
    # config demographic: it gets no market mean and is not a Pi regressor);
    # it goes to demographics_household for household-level micro moments.
    keep = ["household_code", "panel_year", "DMA_Cd", "projection_factor",
            "household_size"]
    keep += [c for c in demog_cols if c in out.columns and c not in keep]
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


# Column-name prefix for the cumulative bracket-share encoding.  The raw
# NielsenIQ ordinal CODES are themselves the brackets (no codebook needed),
# so the cumulative share at threshold v is the share of households with
# code >= v.  Any demographic not listed here falls back to a sanitized
# prefix derived from its name.
_THRESH_PREFIX = {
    "household_income": "inc",
    "male_head_age": "age",
}


def _thresh_prefix(col):
    return _THRESH_PREFIX.get(col, "".join(c for c in col if c.isalnum())[:6])


def compute_threshold_shares(panelists_qt, demog_cols):
    """Market-level CUMULATIVE BRACKET SHARES from the raw ordinal codes.

    For each demographic ``col`` and each threshold ``v`` in its sorted unique
    codes (the trivial lowest threshold, whose share is always 1, is dropped),
    the column ``{prefix}_ge_{v}`` is the projection_factor-weighted share of
    households in each market ``(t, m)`` whose code is ``>= v``.  The
    denominator is the projection-factor mass of households with a non-missing
    code for that demographic (so missing codes are excluded, matching the
    weighted-mean convention).

    Returns ``(shares_df, info)`` where ``shares_df`` is indexed ``(t, m)`` and
    ``info`` is a list of ``(col, prefix, [thresholds])`` tuples.
    """
    keys = ["t", "m"]
    info = []
    blocks = []
    pf_all = panelists_qt["projection_factor"].to_numpy(float)
    base = pd.DataFrame({
        "t": panelists_qt["t"].to_numpy(),
        "m": panelists_qt["DMA_Cd"].to_numpy(),
    })
    for col in demog_cols:
        if col not in panelists_qt.columns:
            continue
        prefix = _thresh_prefix(col)
        vals = panelists_qt[col].to_numpy()
        notna = ~pd.isna(vals)
        codes = np.sort(np.unique(vals[notna]))
        if codes.size < 2:
            continue
        thresholds = codes[1:]  # drop the trivial lowest (share == 1)
        bdf = base.copy()
        # Indicator * projection_factor for each threshold (0 where code is
        # missing); denominator = projection_factor where code is present.
        for v in thresholds:
            ind = np.where(notna & (vals >= v), pf_all, 0.0)
            bdf[f"{prefix}_ge_{int(v)}"] = ind
        denom_name = f"_denom_{prefix}"
        bdf[denom_name] = np.where(notna, pf_all, 0.0)
        g = bdf.groupby(keys).sum()
        denom = g.pop(denom_name)
        shares = g.div(denom, axis=0)
        blocks.append(shares)
        info.append((col, prefix, [int(v) for v in thresholds]))
    if not blocks:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays(
            [[], []], names=keys)), info
    out = pd.concat(blocks, axis=1).sort_index()
    return out, info


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


def compute_demographics(cfg, years=None, market_only=False,
                         raw_dir="data/raw", clean_dir="data/clean"):
    """Compute market-level and household-level demographics.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration.
    years : list of int, optional
        Override year range.
    market_only : bool, default False
        When True, skip the household-level computation and write, rebuilding
        ONLY demographics_market.parquet (leaves demographics_household.parquet
        untouched).

    Returns
    -------
    (pd.DataFrame, pd.DataFrame or None)
        demographics_market (t, m) and demographics_household (i, t) (None when
        ``market_only``).
    """
    # --raw-dir / --clean-dir: a new data vintage is written to its own
    # directory (as in AM/AH).  Defaults: the old relative paths.
    raw_dir = Path(raw_dir)
    clean_dir = Path(clean_dir)
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
    # Cumulative bracket shares (for FWL nuisance controls).  These keep
    # the existing mean columns and ADD one column per (demographic,
    # threshold).  Index is (t, m), aligned with demog_market.
    # ------------------------------------------------------------------
    print("\n[3b/4] Computing market-level cumulative bracket shares...")
    thresh_market, thresh_info = compute_threshold_shares(
        panelists_qt, available_demog)
    n_thresh = thresh_market.shape[1]
    demog_market = demog_market.join(thresh_market, how="left")
    print(f"  Threshold-share columns produced: {n_thresh}")
    for col, prefix, thr in thresh_info:
        print(f"    {col}: {len(thr)} thresholds "
              f"({prefix}_ge_{thr[0]} .. {prefix}_ge_{thr[-1]})")

    # ------------------------------------------------------------------
    # Household-level demographics: per (i, t)
    # ------------------------------------------------------------------
    if market_only:
        out_market = clean_dir / "demographics_market.parquet"
        demog_market.to_parquet(out_market)
        print(f"\nSaved market demographics to {out_market}")
        print("market_only=True: skipped household-level computation/write.")
        return demog_market, None

    print("\n[4/4] Computing household-level demographics...")

    demog_hh = (
        panelists_qt[["household_code", "t", "DMA_Cd", "panel_year",
                       "projection_factor", "household_size"]
                      + [c for c in available_demog if c != "household_size"]]
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
    parser.add_argument("--market-only", action="store_true",
                        help="Rebuild ONLY demographics_market.parquet "
                             "(skip household-level computation/write).")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="Homescan raw parquet directory (read)")
    parser.add_argument("--clean-dir", default="data/clean",
                        help="output directory (a new vintage dir)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.years:
        years = list(range(args.years[0], args.years[1] + 1))
    else:
        years = None

    compute_demographics(cfg, years=years, market_only=args.market_only,
                         raw_dir=args.raw_dir, clean_dir=args.clean_dir)


if __name__ == "__main__":
    main()
