"""charter-crosswalk A2: one brand route for both sources.

Unit tests of src/brand_route.py on synthetic frames (the rule of
docs/crosswalk-rule.org), and the A2 integration test: for every 2021-2024
UPC that appears in both Homescan purchases and RMS (the RMS UPC set from
rms/diagnostics/upc_routes.parquet), the Homescan builder's call site
(aggregate_markets._apply_brand_route on the year's purchases, master =
data/raw/master_products.parquet) and the RMS builder's call site
(rms_market_panel.revised_upc_map, master = rms/master_products_1344.parquet)
assign the same j, or both assign none.

Run with the CerealDemand venv, serially or -n 4 at most:
    .../CerealDemand/.venv/bin/python -m pytest -q tests/test_brand_route.py
Paths: PANEL_RAW (Homescan raw, read), RMS_OUT (accepted RMS dir, read).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import brand_route as br  # noqa: E402

DATA = Path(os.environ.get("NIELSEN_DATA",
                           "/global/scratch/fsa/fc_jevons/ligon/Data"))
RMS_OUT = Path(os.environ.get("RMS_OUT", DATA / "NielsenScanner" / "rms"))
PANEL_RAW = Path(os.environ.get(
    "PANEL_RAW",
    "/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/raw"))
ROUTES = RMS_OUT / "diagnostics" / "upc_routes.parquet"


# --- the rule on synthetic data ----------------------------------------------

def _toy():
    master = pd.DataFrame({
        "upc":           [1,      1,      2,      3,      4],
        "upc_ver_uc":    [1,      2,      1,      1,      1],
        "brand_code_uc": [620554, 620485, 557579, 100,    200],
        "brand_descr":   ["POST SELECTS GG", "POST GG", "G M TRIX", "A", "B"],
        "size1_amount":  [10.0,   10.0,   11.0,   12.0,   13.0],
        "size1_units":   ["OZ"] * 5})
    names = pd.DataFrame({
        # year, upc, name, size, units, dollars
        "year": [2021, 2021, 2021, 2022, 2022, 2022, 2023, 2021, 2021],
        "upc":  [1,    2,    3,    3,    10,   11,   12,   4,    13],
        "brand_descr_hms": ["GG", "TRIX", "NAME A", "NAME A", "NAME A",
                            "TRIX", "GENERAL MILLS", "NAME A", "NAME B"],
        "size1_amount": [10.0, 11.0, 12.0, 12.0, 9.0, 8.0, 7.0, 13.0, 5.0],
        "size1_units": ["OZ"] * 9,
        "upc_dollars": [5.0, 7.0, 10.0, 10.0, 3.0, 4.0, 2.0, 30.0, 1.0]})
    return master, names


def _route():
    master, names = _toy()
    return br.BrandRoute(master, names,
                         merge={620485: 620554, 747763: 557579},
                         overrides={"GENERAL MILLS": np.nan,
                                    "NAME B": 100.0})


def test_version_rule_is_highest_version_then_merge():
    r = _route().route(2021, [1]).set_index("upc")
    assert r.loc[1, "legacy_code"] == 620485          # highest version
    assert r.loc[1, "j"] == 620554                    # merged (Q2 a)
    assert r.loc[1, "j_route"] == br.LEGACY


def test_name_map_is_dollar_majority_over_legacy_coded_upcs():
    rt = _route()
    nm = rt.name_map.set_index("brand_descr_hms")
    # NAME A: upc 3 (code 100) $20 over 2021-22, upc 4 (code 200) $30
    assert nm.loc["NAME A", "name_code"] == 200
    assert nm.loc["NAME A", "share"] == pytest.approx(0.6)
    r = rt.route(2022, [10, 3]).set_index("upc")
    assert r.loc[10, "j"] == 200 and r.loc[10, "j_route"] == br.NAME
    assert r.loc[3, "j"] == 100 and r.loc[3, "j_route"] == br.LEGACY


def test_overrides_apply_to_new_upcs_only():
    rt = _route()
    r = rt.route(2023, [12]).set_index("upc")
    assert np.isnan(r.loc[12, "j"]) and r.loc[12, "j_route"] == br.OVERRIDE_NONE
    r = rt.route(2021, [13]).set_index("upc")
    assert r.loc[13, "j"] == 100 and r.loc[13, "j_route"] == br.OVERRIDE


def test_new_upc_with_unknown_or_no_name_is_unassigned():
    r = _route().route(2021, [99]).set_index("upc")
    assert np.isnan(r.loc[99, "j"]) and r.loc[99, "j_route"] == br.NO_NAME


def test_nearest_year_name_and_master_fallback_descr():
    rt = _route()
    r = rt.route(2024, [11, 4]).set_index("upc")
    assert r.loc[11, "name_year"] == 2022 and r.loc[11, "brand_descr"] == "TRIX"
    assert r.loc[11, "j"] == 557579                   # TRIX -> code of upc 2
    master, _ = _toy()
    names = _toy()[1]
    rt2 = br.BrandRoute(master, names[names["upc"] != 4], merge={},
                        overrides={})
    r2 = rt2.route(2021, [4]).set_index("upc")
    assert r2.loc[4, "brand_descr"] == "B" and r2.loc[4, "size1_amount"] == 13.0
    assert r2.loc[4, "j"] == 200


def test_merge_codes_classic_rows():
    s = br.merge_codes(pd.Series([620485.0, 747763.0, 1.0, np.nan]))
    assert s.iloc[0] == 620554 and s.iloc[1] == 557579 and s.iloc[2] == 1
    assert np.isnan(s.iloc[3])


def test_committed_configs():
    m = br.load_merge_sets()
    assert m == {620485: 620554, 747763: 557579}
    o = br.load_overrides()
    assert set(o) == {"GENERAL MILLS", "KELLOGG'S"}
    assert all(np.isnan(v) for v in o.values())


# --- A2: both builders' call sites agree on every shared (year, upc) ---------

needs_data = pytest.mark.skipif(
    not (ROUTES.exists() and (PANEL_RAW / "purchases_2021.parquet").exists()),
    reason="Homescan raw or RMS upc_routes not available")


@needs_data
@pytest.mark.parametrize("year", [2021, 2022, 2023, 2024])
def test_a2_homescan_and_rms_builders_assign_the_same_j(year):
    import aggregate_markets as am
    import rms_market_panel as rmp
    assert rmp.PANEL_RAW.resolve() == PANEL_RAW.resolve(), \
        "set PANEL_CLEAN/PANEL_RAW so both builders read the same Homescan raw"

    # Homescan: the builder's reader + its brand-route call (AM:225).
    p = am._read_one_purchase(PANEL_RAW, year)
    p = am._apply_brand_route(p, PANEL_RAW, [1344])
    hms = (p.groupby("upc")["brand_code_uc"]
             .agg(["nunique", "first", "size"]))
    assert (hms["nunique"] <= 1).all()          # one j per (year, upc)
    hms_j = p.drop_duplicates("upc").set_index("upc")["brand_code_uc"]

    # RMS: the RMS builder's map for the year's movement UPCs.
    u = pd.read_parquet(ROUTES, columns=["upc", "year"])
    rms_upcs = u.loc[u["year"] == year, "upc"].unique()
    rm = rmp.revised_upc_map(year, rms_upcs).set_index("upc")["brand_code_uc"]

    both = hms_j.index.intersection(rm.index)
    assert len(both) > 1000
    a = hms_j.reindex(both).astype(float)
    b = rm.reindex(both).astype(float)
    same = (a == b) | (a.isna() & b.isna())
    bad = both[~same.to_numpy()]
    assert len(bad) == 0, (
        f"{year}: {len(bad)} shared UPCs differ, e.g. "
        f"{list(zip(bad[:5], a[bad[:5]], b[bad[:5]]))}")
