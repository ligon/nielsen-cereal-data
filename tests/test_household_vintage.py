"""Household brand set and household_size (fix/household-brands-hhsize-2026-10-22).

Defect 1: household_panel ranked brands without AM's CTL drop, so it held CTL
BR (536746) and lacked M-O-M GOLDEN PUFFS (591705).  Both builders now take
the pinned 45 (config/brands_45.csv; see test_pinned_brands.py).

Defect 2: demographics_household lacked household_size (NielsenIQ
Household_Size), which AM already sums into market_size.

The data tests read the vintage at HH_CLEAN (default: the
data/clean-hhfix-2026-10-22 vintage beside data/clean).  Run with the
CerealDemand venv, which has pytest:
    /global/scratch/fsa/fc_jevons/ligon/mirrors/CerealDemand/.venv/bin/python \
        -m pytest -q -p no:cacheprovider tests/test_household_vintage.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import aggregate_markets as am  # noqa: E402
import demographics as dg  # noqa: E402

HH_CLEAN = Path(os.environ.get(
    "HH_CLEAN",
    "/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/"
    "data/clean-hhfix-2026-10-22"))
DAYS, OZ_PER_DAY, AWAY = 90, 4.5, 1.05        # AM._compute_market_size
needs_vintage = pytest.mark.skipif(
    not (HH_CLEAN / "household_panel.parquet").exists(),
    reason=f"vintage not built at {HH_CLEAN}")


def test_panelist_reader_keeps_household_size_both_eras(tmp_path):
    legacy = pd.DataFrame({
        "Household_Cd": [1, 2], "Panel_Year": [2014, 2014],
        "panel_year": [2014, 2014], "DMA_Cd": [501, 501],
        "Projection_Factor": [100, 200], "Household_Size": [1, 9],
        "Household_Income": [10, 20], "Male_Head_Age": [0, 5]})
    revised = pd.DataFrame({
        "household_code": [3], "panel_year": [2021],
        "fips_state_code": [36], "fips_county_code": [61],
        "projection_factor": [300], "household_size": [4],
        "household_income": [21], "male_head_age": [9]})
    legacy.to_parquet(tmp_path / "panelists_2014.parquet")
    revised.to_parquet(tmp_path / "panelists_2021.parquet")
    f2d = pd.DataFrame({"fips_state_code": [36], "fips_county_code": [61],
                        "DMA_Cd": [501]})
    cols = ["household_income", "male_head_age"]
    a = dg._read_one_panelist_demog(tmp_path, 2014, cols, f2d)
    b = dg._read_one_panelist_demog(tmp_path, 2021, cols, f2d)
    assert a["household_size"].tolist() == [1, 9]
    assert b["household_size"].tolist() == [4]
    # listing it in the config demographics must not duplicate the column
    c = dg._read_one_panelist_demog(tmp_path, 2014, cols + ["household_size"],
                                    f2d)
    assert list(c.columns).count("household_size") == 1


@needs_vintage
def test_household_panel_brand_set_is_the_pin():
    hp = pd.read_parquet(HH_CLEAN / "household_panel.parquet", columns=["S"])
    js = set(hp.index.get_level_values("j").astype("int64"))
    assert js == set(am.load_brands_45()["j"])
    bs = pd.read_parquet(HH_CLEAN / "household_base_shares.parquet")
    bj = set(bs.index.get_level_values("j").astype("int64"))
    assert 536746 not in bj and 591705 in bj
    assert bj <= js


@needs_vintage
def test_household_panel_has_no_household_size_column():
    import pyarrow.parquet as pq
    names = pq.read_schema(HH_CLEAN / "household_panel.parquet").names
    assert "household_size" not in names


@needs_vintage
def test_household_ounces_sum_to_market_servings():
    """Per (t, m, j) cell, household ounces == market servings, all 45 brands
    (591705 included).  Today's data/clean: 99.85% on the 44 shared brands."""
    hp = pd.read_parquet(HH_CLEAN / "household_panel.parquet",
                         columns=["m", "servings"]).reset_index()
    h = hp.groupby(["t", "m", "j"])["servings"].sum()
    mp = pd.read_parquet(HH_CLEAN / "market_panel.parquet",
                         columns=["servings"])["servings"]
    both = pd.concat([h.rename("hh"), mp.rename("mk")], axis=1, join="inner")
    assert len(both) >= 0.99 * len(mp)
    exact = np.isclose(both["hh"], both["mk"], rtol=1e-9, atol=0)
    assert exact.mean() >= 0.998
    g = both.loc[both.index.get_level_values("j") == 591705]
    assert len(g) > 0
    assert np.isclose(g["hh"], g["mk"], rtol=1e-9).mean() >= 0.998


@needs_vintage
def test_demographics_household_household_size():
    dh = pd.read_parquet(HH_CLEAN / "demographics_household.parquet")
    hs = dh["household_size"]
    assert hs.notna().all()
    assert pd.api.types.is_integer_dtype(hs)
    assert hs.between(1, 9).all()
    # constant within (household, panel_year)
    assert (dh.groupby(["i", "panel_year"])["household_size"].nunique() == 1).all()


@needs_vintage
def test_household_size_sums_to_market_n_persons():
    """Charter-micro A1: per (panel_year, DMA), the sum of household_size over
    the panelist households equals AM's n_persons exactly."""
    dh = pd.read_parquet(HH_CLEAN / "demographics_household.parquet",
                         columns=["m", "panel_year", "household_size"]
                         ).reset_index()
    one = dh[dh["t"] == "Q1Y" + dh["panel_year"].astype(str)]
    assert not one.duplicated(["i", "panel_year"]).any()
    persons = one.groupby(["panel_year", "m"])["household_size"].sum()
    mp = pd.read_parquet(HH_CLEAN / "market_panel.parquet",
                         columns=["n_persons", "market_size"]).reset_index()
    mp["panel_year"] = mp["t"].str.split("Y").str[-1].astype(int)
    mk = (mp.drop_duplicates(["panel_year", "m"])
            .astype({"m": "int64"})
            .set_index(["panel_year", "m"]))
    np.testing.assert_allclose(mk["market_size"],
                               mk["n_persons"] * DAYS * OZ_PER_DAY * AWAY,
                               rtol=1e-12)
    joined = pd.concat([persons.rename("dh"), mk["n_persons"]], axis=1,
                       join="inner")
    assert len(joined) == len(mk)
    assert (joined["dh"] == joined["n_persons"]).all()
