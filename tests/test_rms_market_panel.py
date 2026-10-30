"""D3 contract test for market_panel_rms.parquet (charter-rms section 2, D3;
Amendment 1 items 1-2).

Run with the CerealDemand venv (the parquet is built with the same pandas):
    /global/scratch/fsa/fc_jevons/ligon/mirrors/CerealDemand/.venv/bin/python \
        -m pytest -q tests/test_rms_market_panel.py

"AM:<line>" cites src/aggregate_markets.py, the panel's builder, whose
definitions the RMS panel mirrors.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

DATA = Path(os.environ.get("NIELSEN_DATA",
                           "/global/scratch/fsa/fc_jevons/ligon/Data"))
OUT = Path(os.environ.get("RMS_OUT", DATA / "NielsenScanner" / "rms"))
RMS = OUT / "market_panel_rms.parquet"
PANEL_CLEAN = Path(os.environ.get(
    "PANEL_CLEAN",
    "/global/home/users/ligon/mirrors/nielsen-cereal-data/data/clean"))
PANEL = PANEL_CLEAN / "market_panel.parquet"
PANEL_RAW = PANEL_CLEAN.parent / "raw"
ZETA = ["sugar", "fiber", "protein", "fat", "sodium"]


@pytest.fixture(scope="module")
def rms():
    assert RMS.exists(), f"{RMS} missing: run src/rms_build.sh"
    return pd.read_parquet(RMS)


@pytest.fixture(scope="module")
def panel():
    return pd.read_parquet(PANEL)


@pytest.fixture(scope="module")
def flat(rms):
    return rms.reset_index()


# --- schema ---------------------------------------------------------------

def test_index_is_t_m_j_with_panel_dtypes(rms, panel):
    assert list(rms.index.names) == ["t", "m", "j"] == list(panel.index.names)
    for lvl in range(3):
        assert (rms.index.get_level_values(lvl).dtype
                == panel.index.get_level_values(lvl).dtype)
    assert rms.index.is_unique
    assert rms.index.is_monotonic_increasing


def test_identical_column_names_and_dtypes(rms, panel):
    assert list(rms.columns) == list(panel.columns)
    got = {c: str(d) for c, d in rms.dtypes.items()}
    want = {c: str(d) for c, d in panel.dtypes.items()}
    assert got == want


def test_rms_caveat_in_schema_metadata():
    md = pq.read_schema(RMS).metadata or {}
    assert b"rms_caveat" in md
    txt = md[b"rms_caveat"].decode()
    assert "B1" in txt and "Amendment 1" in txt
    assert "NOT meaningful" in txt
    for col in ("shares", "market_size", "Z_*"):
        assert col in txt
    assert b"pandas" in md          # index round-trips


# --- domain ---------------------------------------------------------------

def test_m_in_panel_dma_set(flat, panel):
    assert set(flat["m"]) <= set(panel.index.get_level_values("m"))


def test_t_in_q1y2006_to_q4y2024(flat):
    allowed = {f"Q{q}Y{y}" for y in range(2006, 2025) for q in range(1, 5)}
    ts = set(flat["t"])
    assert ts <= allowed
    assert ts == allowed, f"missing quarters: {sorted(allowed - ts)}"


def test_j_subset_of_panel_45(flat, panel):
    pj = set(panel.index.get_level_values("j"))
    assert len(pj) == 45
    assert set(flat["j"]) <= pj
    assert set(flat["product_ids"]) <= set(panel["product_ids"])


def test_market_size_equals_panel_for_same_t_m(flat, panel):
    p = (panel.reset_index()[["t", "m", "market_size", "n_hh", "n_persons"]]
         .drop_duplicates(["t", "m"]))
    r = flat[["t", "m", "market_size", "n_hh", "n_persons"]].drop_duplicates()
    assert not r.duplicated(["t", "m"]).any()      # one value per (t, m)
    j = r.merge(p, on=["t", "m"], suffixes=("_rms", "_panel"))
    assert len(j) > 0.9 * len(r)
    for c in ("market_size", "n_hh", "n_persons"):
        np.testing.assert_array_equal(j[f"{c}_rms"], j[f"{c}_panel"])
    # and every RMS (t, m) carries the panel's (calendar year, m) value (AM:675-683)
    py = (panel.reset_index()[["panel_year", "m", "market_size"]]
          .drop_duplicates(["panel_year", "m"]))
    k = flat[["panel_year", "m", "market_size"]].merge(
        py, on=["panel_year", "m"], how="left", suffixes=("", "_panel"))
    np.testing.assert_array_equal(k["market_size"], k["market_size_panel"])


# --- definitions ----------------------------------------------------------

def test_prices_are_revenue_over_servings(flat):            # AM:648-656
    np.testing.assert_allclose(flat["prices"],
                               flat["revenue"] / flat["servings"], rtol=1e-12)
    assert np.isfinite(flat["prices"]).all() and (flat["prices"] > 0).all()
    assert (flat["n_transactions"] > 0).all()


def test_shares_are_servings_over_market_size(flat):        # AM:692
    np.testing.assert_allclose(flat["shares"],
                               flat["servings"] / flat["market_size"],
                               rtol=1e-12)


def test_w_exp_is_within_market_expenditure_share(flat):    # AM:695-697
    tot = flat.groupby(["t", "m"])["revenue"].transform("sum")
    np.testing.assert_allclose(flat["W_exp"], flat["revenue"] / tot, rtol=1e-12)
    np.testing.assert_allclose(flat.groupby(["t", "m"])["W_exp"].sum(), 1.0,
                               rtol=1e-12)


def test_nutrients_and_brand_descr_are_the_panels(flat, panel):  # AM:271-355
    perj = panel.reset_index().groupby("j")[["brand_descr"] + ZETA].first()
    got = flat.set_index("j")[["brand_descr"] + ZETA]
    want = perj.loc[got.index]
    assert (got["brand_descr"].to_numpy() == want["brand_descr"].to_numpy()).all()
    np.testing.assert_array_equal(got[ZETA].to_numpy(), want[ZETA].to_numpy())


def test_z_aggregates(flat):                                 # AM:741-751
    for c in ZETA:
        z = (flat["shares"] * flat[c]).groupby([flat["t"], flat["m"]]).sum()
        got = flat.groupby(["t", "m"])[f"Z_{c}"].agg(["min", "max"])
        np.testing.assert_allclose(got["min"], got["max"], rtol=0)
        np.testing.assert_allclose(got["min"], z.loc[got.index], rtol=1e-10)


def test_id_columns(flat):                                   # AM:753-766
    assert (flat["market_ids"] == flat["t"] + "_DMA_" + flat["m"].astype(str)).all()
    assert (flat["product_ids"] == flat["j"].astype(str)).all()
    assert (flat["dma_cd"] == flat["m"]).all()
    assert (flat["quarter"] == flat["t"].str[1].astype(int)).all()
    assert (flat["panel_year"] == flat["t"].str[-4:].astype(int)).all()


# --- independent recomputation of cells from the D2 parquet -------------------

def _recompute(year, dma, q, upc_map):
    """Cells (j -> servings, revenue, n) for one (quarter, DMA), recomputed
    directly from the D2 movement parquet with pyarrow filters."""
    st = pd.read_parquet(OUT / str(year) / "stores.parquet")
    if "dma_code" in st.columns:
        stores = st.loc[st["dma_code"] == dma, "store_code_uc"]
    else:
        f2d = (pd.concat([pd.read_parquet(p, columns=[
                   "Fips_State_Cd", "Fips_County_Cd", "DMA_Cd"])
               for p in sorted(PANEL_RAW.glob("panelists_*.parquet"))
               if int(p.stem.split("_")[1]) <= 2020])
               .groupby(["Fips_State_Cd", "Fips_County_Cd"])["DMA_Cd"]
               .agg(lambda x: x.mode().iloc[0]).reset_index())
        st = st.merge(f2d, left_on=["fips_state_code", "fips_county_code"],
                      right_on=["Fips_State_Cd", "Fips_County_Cd"])
        stores = st.loc[st["DMA_Cd"] == dma, "store_code_uc"]
    lo = pd.Timestamp(year, 3 * q - 2, 1).date()
    hi = (pd.Timestamp(year, 3 * q - 2, 1) + pd.offsets.QuarterEnd(0)).date()
    mv = pq.read_table(
        OUT / str(year) / "movement_1344.parquet",
        columns=["store_code_uc", "upc", "week_end", "units", "prmult", "price"],
        filters=[("store_code_uc", "in", stores.astype("int32").tolist()),
                 ("week_end", ">=", lo), ("week_end", "<=", hi)]).to_pandas()
    mv = mv[(mv["prmult"] > 0) & (mv["price"] > 0)]
    mv = mv.merge(upc_map, on="upc", how="inner")
    mv = mv[~mv["brand_descr"].astype(str).str.upper().str.startswith("CTL")]
    mv = mv[mv["size1_units"].astype(str).str.upper().str.strip() == "OZ"]
    mv["servings"] = mv["units"] * mv["size1_amount"]                # AM:632
    mv["revenue"] = mv["units"] * mv["price"] / mv["prmult"]
    return mv.groupby("brand_code_uc").agg(
        servings=("servings", "sum"), revenue=("revenue", "sum"),
        n_transactions=("revenue", "size"))


def _check_cells(rms, year, dma, q, upc_map):
    got = rms.xs((f"Q{q}Y{year}", dma), level=["t", "m"])
    want = _recompute(year, dma, q, upc_map)
    want = want[want.index.isin(got.index)]
    assert len(got) >= 20
    assert set(got.index) == set(want.index)
    want = want.loc[got.index]
    np.testing.assert_allclose(got["servings"], want["servings"], rtol=1e-9)
    np.testing.assert_allclose(got["revenue"], want["revenue"], rtol=1e-9)
    np.testing.assert_array_equal(got["n_transactions"], want["n_transactions"])


def _classic_cell_upc_map(rv, master, js, shared_route):
    """Versioned master map, with Q2 merges only for a shared-route build."""
    um = rv.merge(master, on=["upc", "upc_ver_uc"])[
        ["upc", "brand_code_uc", "brand_descr", "size1_amount", "size1_units"]]
    if shared_route:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from brand_route import merge_codes
        um["brand_code_uc"] = merge_codes(um["brand_code_uc"])
    return um[um["brand_code_uc"].isin(js)]


@pytest.mark.parametrize("shared_route", [False, True])
def test_classic_cell_map_merges_before_45_filter(shared_route):
    rv = pd.DataFrame({"upc": [1, 2, 3], "upc_ver_uc": 1})
    master = rv.assign(brand_code_uc=[620485, 747763, 557396],
                       brand_descr=["Great Grains", "Classic Trix", "Cheerios"],
                       size1_amount=12.0, size1_units="OZ")
    um = _classic_cell_upc_map(rv, master, {620554, 557579, 557396}, shared_route)
    want = {1: 620554, 2: 557579, 3: 557396} if shared_route else {3: 557396}
    assert um.set_index("upc")["brand_code_uc"].to_dict() == want
    assert master["brand_code_uc"].tolist() == [620485, 747763, 557396]


def test_recompute_classic_cell(rms, panel):
    """Q2Y2016, New York: rms_versions -> master, with Q2 merges for a
    shared-route build.  Accepted name-route builds keep their raw codes."""
    rv = pd.read_parquet(OUT / "2016" / "rms_versions.parquet")
    master = pd.read_parquet(OUT / "master_products_1344.parquet")
    shared_route = "j_route" in pq.read_schema(OUT / "diagnostics" / "upc_routes.parquet").names
    um = _classic_cell_upc_map(rv, master, panel.index.get_level_values("j"),
                              shared_route)
    _check_cells(rms, 2016, 501, 2, um)


def _shared_route_mismatches(routes, year):
    """(number of the year's UPCs, number whose built j differs from j
    re-derived through src/brand_route.py from Homescan raw + the master).
    For a build on the shared brand route (upc_routes carries j_route)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from brand_route import BrandRoute
    br = BrandRoute.from_paths(PANEL_RAW, OUT / "master_products_1344.parquet")
    built = routes[routes["year"] == year].set_index("upc")["brand_code_uc"]
    ind = br.route(year, built.index.to_numpy()).set_index("upc")["j"]
    ind = ind.reindex(built.index)
    same = (built == ind) | (built.isna() & ind.isna())
    return len(built), int((~same).sum())


def test_recompute_revised_cell_and_name_route(rms, panel):
    """Q3Y2022, New York.  The upc map is the build's (diagnostics), but its
    2022 brand assignments are re-derived here: for an
    accepted (name-route) build from the Homescan purchases and the
    crosswalk (AM:245-250); for a build on the shared brand route
    (charter-crosswalk; upc_routes carries j_route) through
    the production src/brand_route.py from Homescan raw and the master, on
    every 2022 UPC.  The shared-route branch checks build consistency, not
    an independent implementation of the assignment rule."""
    routes = pd.read_parquet(OUT / "diagnostics" / "upc_routes.parquet")
    um = routes[(routes["year"] == 2022) & routes["brand_code_uc"].notna()].copy()
    um["brand_code_uc"] = um["brand_code_uc"].astype("int64")
    um = um[um["brand_code_uc"].isin(panel.index.get_level_values("j"))]

    if "j_route" in routes.columns:
        n, bad = _shared_route_mismatches(routes, 2022)
        assert n > 1000
        assert bad == 0
    else:
        h = pd.read_parquet(PANEL_RAW / "purchases_2022.parquet",
                            columns=["upc", "brand_descr_hms"]).drop_duplicates("upc")
        x = (pd.read_parquet(PANEL_CLEAN / "brand_crosswalk.parquet")
             .drop_duplicates("brand_descr_2021"))
        h = h.merge(x[["brand_descr_2021", "brand_code_uc"]],
                    left_on="brand_descr_hms", right_on="brand_descr_2021")
        same = um[um["route"] == "hms_same_year"].merge(h, on="upc",
                                                        suffixes=("", "_ind"))
        assert len(same) > 100
        assert (same["brand_code_uc"] == same["brand_code_uc_ind"]).all()
    _check_cells(rms, 2022, 501, 3,
                 um[["upc", "brand_code_uc", "brand_descr", "size1_amount",
                     "size1_units"]])
