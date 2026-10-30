"""
The brand route: module-1344 UPC -> brand j, for both sources.

ONE implementation, imported by the Homescan builders
(src/aggregate_markets.py, and src/aggregate_households.py through it) and
by the RMS builder (src/rms_market_panel.py), so the two sources cannot
drift.  The rule is stated in docs/crosswalk-rule.org (charter-crosswalk A1;
EL's decisions Q1 (a), Q2 (a), Q2b, Q3, 2026-10-19):

1. Legacy code.  A UPC with a row in the module-1344 product master keeps the
   brand_code_uc of its HIGHEST upc_ver_uc (the version rule).
2. Name map.  A UPC with no master row (new in 2021+) takes its Homescan
   2021+ name, mapped to the legacy code carrying the most 2021-2024
   Homescan dollars among legacy-coded UPCs of that name (learned once from
   all of 2021-2024, on merged codes).
3. Overrides (config/brand_name_overrides.csv) replace a name's learned
   code, for new UPCs only.  Empty j = unassigned.
4. Merge sets (config/brand_merge_sets.csv): j = merge(code), in every year
   and both sources (classic rows too: see ``merge_codes``).

A UPC's name in year y is the Homescan name with the most dollars on it in
year y; if Homescan did not record it in y, the nearest other 2021+ year
(tie -> earlier year), as rms_market_panel.py did (its 202-204).  The size
that goes with the name follows the same row; a UPC with no Homescan name
takes the master's size and description (the RMS fallback for UPCs no
household bought).

Inputs are explicit paths: the Homescan raw directory (purchases_<yr>) and
a module-1344 product master (either data/raw/master_products.parquet,
filtered to the modules, or rms/master_products_1344.parquet: identical on
module 1344).  Nothing here reads or writes data/clean.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config"
REVISED_FIRST = 2021
REVISED_LAST = 2024

MERGE_SETS = CONFIG / "brand_merge_sets.csv"
OVERRIDES = CONFIG / "brand_name_overrides.csv"
BRANDS_45 = CONFIG / "brands_45.csv"

# route labels (column ``j_route``)
LEGACY = "legacy_master"
NAME = "name_map"
OVERRIDE = "override"
OVERRIDE_NONE = "override_unassigned"
NO_NAME = "unassigned_no_name"
UNKNOWN = "unassigned_unknown_name"


# ---------------------------------------------------------------------------
# Committed configuration
# ---------------------------------------------------------------------------

def load_merge_sets(path=MERGE_SETS) -> dict[int, int]:
    """{code: j} for the merged codes (every other code is its own j)."""
    m = pd.read_csv(path)
    out = dict(zip(m["code"].astype(int), m["j"].astype(int)))
    assert not set(out) & set(out.values()), "merge targets must be terminal"
    return out


def load_overrides(path=OVERRIDES) -> dict[str, float]:
    """{brand_descr_hms: j or NaN (unassigned)}."""
    o = pd.read_csv(path, dtype={"brand_descr_hms": str},
                    keep_default_na=False)
    j = pd.to_numeric(o["j"].replace("", np.nan), errors="coerce")
    assert o["brand_descr_hms"].is_unique
    return dict(zip(o["brand_descr_hms"], j.astype(float)))


def load_brands_45(path=BRANDS_45) -> pd.DataFrame:
    """The pinned 45 (Q3): columns j, brand_descr, in committed order."""
    b = pd.read_csv(path)
    b["j"] = b["j"].astype("int64")
    assert b["j"].is_unique and len(b) == 45, path
    return b


def merge_codes(codes, merge=None) -> pd.Series:
    """j = merge(code), elementwise; NaN stays NaN.  Classic rows too."""
    merge = load_merge_sets() if merge is None else merge
    s = pd.Series(codes, copy=True)
    return s.replace(merge) if len(s) else s


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_master(path, modules=(1344,)) -> pd.DataFrame:
    """Module-`modules` rows of a product master (either file)."""
    cols = ["upc", "upc_ver_uc", "brand_code_uc", "brand_descr",
            "size1_amount", "size1_units"]
    path = Path(path)
    import pyarrow.parquet as pq
    names = pq.read_schema(path).names
    if "product_module_code" in names:
        m = pd.read_parquet(path, columns=cols + ["product_module_code"],
                            filters=[("product_module_code", "in",
                                      list(modules))])
        m = m.drop(columns="product_module_code")
    else:
        m = pd.read_parquet(path, columns=cols)
    return m


def legacy_codes(master: pd.DataFrame) -> pd.DataFrame:
    """Per upc: the highest master version's code, descr and size (rule 1)."""
    ml = (master.sort_values(["upc", "upc_ver_uc"])
                .drop_duplicates("upc", keep="last")
                .set_index("upc"))
    return ml.rename(columns={"brand_code_uc": "legacy_code",
                              "brand_descr": "legacy_descr",
                              "size1_amount": "legacy_size",
                              "size1_units": "legacy_units"})[
        ["upc_ver_uc", "legacy_code", "legacy_descr", "legacy_size",
         "legacy_units"]]


def hms_names(raw_dir, years=range(REVISED_FIRST, REVISED_LAST + 1)):
    """Homescan 2021+ purchases -> one row per (year, upc): the name (and
    its size) with the most dollars on the upc that year, plus the upc's
    dollars that year (all rows, all names)."""
    parts = []
    for y in years:
        p = Path(raw_dir) / f"purchases_{y}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["upc", "total_price_paid",
                                        "brand_descr_hms", "size1_amount_hms",
                                        "size1_unit_hms"])
        g = (d.groupby(["upc", "brand_descr_hms", "size1_amount_hms",
                        "size1_unit_hms"], dropna=False)
              .agg(dollars=("total_price_paid", "sum"),
                   n=("total_price_paid", "size"))
              .reset_index())
        tot = g.groupby("upc")["dollars"].sum()
        g = (g.sort_values(["upc", "dollars", "n", "brand_descr_hms"],
                           ascending=[True, False, False, True])
              .drop_duplicates("upc"))
        g["upc_dollars"] = g["upc"].map(tot)
        g["year"] = y
        parts.append(g)
    if not parts:
        raise FileNotFoundError(f"no purchases_2021+ files in {raw_dir}")
    h = pd.concat(parts, ignore_index=True)
    h["size1_amount"] = pd.to_numeric(h["size1_amount_hms"],
                                      errors="coerce") / 1000.0   # AM:73-76
    h["size1_units"] = h["size1_unit_hms"].astype(str).str.strip()
    return h[["year", "upc", "brand_descr_hms", "size1_amount", "size1_units",
              "upc_dollars"]]


def learn_name_map(names: pd.DataFrame, legacy: pd.Series,
                   merge: dict) -> pd.DataFrame:
    """Rule 2: each 2021+ name -> the merged legacy code with the most
    dollars among the name's legacy-coded UPCs (all of 2021-2024).

    names: rows (upc, brand_descr_hms, upc_dollars) per (year, upc).
    legacy: upc -> legacy code (highest version), raw.
    Returns one row per name: name_code, share (of the name's legacy-coded
    dollars), legacy_dollars, n_upc_legacy, new_dollars, n_upc_new.
    """
    n = names[["upc", "brand_descr_hms", "upc_dollars"]].copy()
    n["code"] = merge_codes(n["upc"].map(legacy), merge).to_numpy()
    leg = n.dropna(subset=["code"])
    g = (leg.groupby(["brand_descr_hms", "code"])
            .agg(d=("upc_dollars", "sum"), u=("upc", "nunique"))
            .reset_index()
            .sort_values(["brand_descr_hms", "d", "code"],
                         ascending=[True, False, True]))
    tot = g.groupby("brand_descr_hms").agg(legacy_dollars=("d", "sum"))
    top = g.drop_duplicates("brand_descr_hms").set_index("brand_descr_hms")
    out = tot.join(top.rename(columns={"code": "name_code"})[["name_code", "d"]])
    out["share"] = out["d"] / out["legacy_dollars"]
    out["n_upc_legacy"] = leg.groupby("brand_descr_hms")["upc"].nunique()
    new = n[n["code"].isna()]
    out = out.join(new.groupby("brand_descr_hms")
                      .agg(new_dollars=("upc_dollars", "sum"),
                           n_upc_new=("upc", "nunique")), how="outer")
    out[["legacy_dollars", "new_dollars"]] = (
        out[["legacy_dollars", "new_dollars"]].fillna(0.0))
    out[["n_upc_legacy", "n_upc_new"]] = (
        out[["n_upc_legacy", "n_upc_new"]].fillna(0).astype("int64"))
    out.index.name = "brand_descr_hms"
    return out.drop(columns="d").reset_index()


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

class BrandRoute:
    """The shared UPC -> j map.  Build once per process (``from_paths``);
    ``route(year, upcs)`` is per (year, upc)."""

    def __init__(self, master: pd.DataFrame, names: pd.DataFrame,
                 merge: dict | None = None, overrides: dict | None = None):
        self.merge = load_merge_sets() if merge is None else merge
        self.overrides = load_overrides() if overrides is None else overrides
        self.legacy = legacy_codes(master)
        self.names = names
        self.name_map = learn_name_map(names, self.legacy["legacy_code"],
                                       self.merge)
        nm = self.name_map.set_index("brand_descr_hms")["name_code"]
        self._name_code = nm
        ov = pd.Series(self.overrides, dtype=float)
        self._final_name = nm.copy()
        for k, v in ov.items():
            self._final_name.loc[k] = v

    @classmethod
    def from_paths(cls, raw_dir, master_path, modules=(1344,), **kw):
        return cls(load_master(master_path, modules), hms_names(raw_dir), **kw)

    def merge_codes(self, codes) -> pd.Series:
        return merge_codes(codes, self.merge)

    def names_for(self, year: int, upcs) -> pd.DataFrame:
        """Each upc's Homescan name for `year`: same year, else the nearest
        other 2021+ year (tie -> earlier)."""
        u = pd.Index(np.unique(np.asarray(upcs)))
        h = self.names[self.names["upc"].isin(u)].copy()
        h["dist"] = (h["year"] - year).abs() * 2 + (h["year"] > year)
        h = (h.sort_values(["upc", "dist"]).drop_duplicates("upc")
              .rename(columns={"year": "name_year"}))
        return h.drop(columns=["dist"]).set_index("upc").reindex(u)

    def route(self, year: int, upcs) -> pd.DataFrame:
        """One row per unique upc: j (merged; NaN = unassigned), j_route,
        legacy_code (raw, highest version), name_code (merged learned code
        of the upc's name, before overrides), brand_descr (the name, or the
        master's descr if Homescan never recorded the upc), size1_amount,
        size1_units (same source as brand_descr), name_year."""
        h = self.names_for(year, upcs)
        u = h.index
        out = pd.DataFrame(index=u)
        out.index.name = "upc"
        lg = self.legacy.reindex(u)
        out["legacy_code"] = lg["legacy_code"].to_numpy()
        out["name_year"] = h["name_year"].to_numpy()
        name = h["brand_descr_hms"]
        has_name = name.notna().to_numpy()
        out["brand_descr"] = np.where(has_name, name.to_numpy(),
                                      lg["legacy_descr"].to_numpy())
        out["size1_amount"] = np.where(has_name, h["size1_amount"].to_numpy(),
                                       lg["legacy_size"].to_numpy())
        out["size1_units"] = np.where(has_name, h["size1_units"].to_numpy(),
                                      lg["legacy_units"].to_numpy())
        out["name_code"] = name.map(self._name_code).to_numpy()
        final = name.map(self._final_name).to_numpy(dtype=float)
        is_ov = name.isin(list(self.overrides)).to_numpy()
        has_leg = out["legacy_code"].notna().to_numpy()
        known = name.isin(self._final_name.index).to_numpy()
        code = np.where(has_leg, out["legacy_code"].to_numpy(dtype=float),
                        final)
        out["j"] = self.merge_codes(code).to_numpy(dtype=float)
        out["j_route"] = np.select(
            [has_leg, ~has_name, is_ov & np.isnan(final), is_ov, known],
            [LEGACY, NO_NAME, OVERRIDE_NONE, OVERRIDE, NAME], UNKNOWN)
        out.loc[(out["j_route"] == NAME) & out["j"].isna(), "j_route"] = UNKNOWN
        return out.reset_index()


@lru_cache(maxsize=4)
def cached_route(raw_dir: str, master_path: str,
                 modules: tuple = (1344,)) -> BrandRoute:
    """One BrandRoute per (raw_dir, master) per process."""
    return BrandRoute.from_paths(raw_dir, master_path, modules)
