"""
RMS stage D3: per-year module-1344 parquet (src/rms_convert.py) ->
market_panel_rms.parquet, a drop-in second source for the Homescan
market_panel.parquet (same columns, dtypes, (t, m, j) index).

Every definition mirrors the panel's builder, src/aggregate_markets.py
(cited as AM:<line> below).  What differs is only where a cell's raw
quantities come from: RMS store-week-UPC movement rows instead of Homescan
purchase records.

  unit of observation   one movement row (store, upc, week_end)
  quantity              units                  (Homescan `quantity`, AM:632)
  dollars               units * price / prmult (Homescan `total_price_paid`)
  t                     quarter of week_end    (AM:362-371 uses purchase_date)
  m                     store DMA: classic stores_<yr>.dma_code; 2021+ via
                        the panel's own FIPS->DMA map _build_fips_to_dma
                        (AM:188-218), imported, not re-implemented
  j                     brand_code_uc:
      classic  rms_versions_<yr> (upc, upc_ver_uc) joined to the product
               master on (upc, upc_ver_uc)  (mirrors extract.py:121)
      2021+    the panel's name route: upc -> brand_descr_hms (Homescan
               purchases of the SAME year first, then the nearest other
               2021+ year) -> brand crosswalk with drop_duplicates(
               'brand_descr_2021') in file order (AM:245-250); fallback for
               upcs no Homescan household bought: the classic master at the
               highest upc_ver_uc.  Size follows the same route
               (size1_amount_hms / 1000, AM:73-76; master size1_amount).
  filters               brand mapped (AM:526-528), DMA mapped (AM:578-582),
                        CTL* private label dropped (AM:591-595), j in the
                        panel's 45 product_ids, m in the panel's 65 DMAs
                        (the panel's sets, not re-selected), OZ only
                        (AM:626-631), t in Q1Y2006..Q4Y2024
  servings              units * size1_amount (AM:632; no `multi` factor)
  revenue, n_transactions  sum of dollars; number of movement rows (AM:636-645)
  prices                revenue / servings, inf/0 -> dropped (AM:648-656)
  market_size, n_hh, n_persons
                        the panel's values, joined on (calendar year of t, m)
                        (AM:675-689; Amendment 1 item 2)
  shares                servings / market_size (AM:692)
  W_exp                 revenue / sum_(t,m) revenue (AM:695-697)
  brand_descr, sugar..sodium
                        the panel's per-j values, copied (constant per j in
                        the panel; AM:271-355 and AM:714-718)
  Z_*                   sum_j shares * zeta_j within (t, m) (AM:741-751)
  market_ids, product_ids, dma_cd, quarter, panel_year   AM:756-766, 677

CAVEAT (charter-rms Amendment 1, blocker B1): the panel's market_size counts
unprojected Homescan sample persons (AM:441-458) while RMS servings are store
totals, so shares, market_size and Z_* are NOT meaningful for RMS (the
45-brand inside shares sum far above 1).  This is written into the parquet's
schema metadata under the key `rms_caveat`.

Also written (for the D4 validation report):
  OUT/diagnostics/funnel_by_year.parquet   module-1344 dollars by filter stage
  OUT/diagnostics/upc_routes.parquet       per (year, upc): mapping route,
                                           brand codes by route, dollars
  OUT/diagnostics/week_sources.parquet     which file each week_end came from
  docs/rms-d3-log.org                      build summary (committed)

Usage:  python src/rms_market_panel.py [--years 2006 2024] [--jobs 8]
            [--stores FILE --outdir DIR --log PATH]

--stores restricts the panel to a store set (a parquet with column
store_code_uc, e.g. from src/rms_store_sets.py); --outdir then takes the
panel and its diagnostics, so a variant never overwrites OUT's own.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

os.umask(0o027)

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("NIELSEN_DATA",
                           "/global/scratch/fsa/fc_jevons/ligon/Data"))
OUT = Path(os.environ.get("RMS_OUT", DATA / "NielsenScanner" / "rms"))
PANEL_CLEAN = Path(os.environ.get(
    "PANEL_CLEAN",
    "/global/home/users/ligon/mirrors/nielsen-cereal-data/data/clean"))
PANEL_RAW = Path(os.environ.get("PANEL_RAW", PANEL_CLEAN.parent / "raw"))

YEARS = range(2006, 2025)
REVISED_FIRST = 2021
T_LO, T_HI = 2006, 2024
ZETA = ["sugar", "fiber", "protein", "fat", "sodium"]
BATCH = 8_000_000

RMS_CAVEAT = (
    "charter-rms Amendment 1 (EL, 2026-10-19), blocker B1: market_size, "
    "n_hh and n_persons are the Homescan panel's values joined on (calendar "
    "year of t, m).  They count UNPROJECTED Homescan sample persons "
    "(aggregate_markets.py:441-458) while RMS servings are store totals, so "
    "`shares` = servings/market_size, `market_size` and `Z_*` are NOT "
    "meaningful for RMS (45-brand inside shares sum far above 1; S_0 < 0).  "
    "Meaningful: prices, servings, revenue, W_exp, n_transactions, "
    "brand_descr and the nutrient columns.  Use only with the K-aggregator "
    "estimators (homog, het); BLP-family estimators refuse source='rms'.  "
    "Coverage: RMS-reporting stores only (~43% of cereal dollars).")

OUT_COLS = (["t", "m", "j", "market_ids", "product_ids", "dma_cd", "quarter",
             "shares", "prices", "brand_descr"]
            + ZETA + [f"Z_{c}" for c in ZETA]
            + ["W_exp"]
            + ["servings", "revenue", "n_transactions",
               "market_size", "n_hh", "n_persons", "panel_year"])   # AM:768-774

FUNNEL = ["all", "brand_mapped", "dma_mapped", "not_ctl", "j_in_45",
          "m_in_65", "oz", "t_in_range"]


# ---------------------------------------------------------------------------
# The panel's sets and per-(year, m) / per-j constants (read-only)
# ---------------------------------------------------------------------------

def panel_frame():
    return pd.read_parquet(PANEL_CLEAN / "market_panel.parquet").reset_index()


def panel_sets(mp):
    j45 = np.sort(mp["j"].unique())
    m65 = np.sort(mp["m"].unique())
    return j45, m65


# ---------------------------------------------------------------------------
# UPC -> (brand, size) maps
# ---------------------------------------------------------------------------

def _master_latest():
    m = pd.read_parquet(OUT / "master_products_1344.parquet",
                        columns=["upc", "upc_ver_uc", "brand_code_uc",
                                 "brand_descr", "size1_amount", "size1_units"])
    return m


def classic_upc_map(y):
    """(upc) -> brand + size for classic year y via rms_versions."""
    rv = pd.read_parquet(OUT / str(y) / "rms_versions.parquet",
                         columns=["upc", "upc_ver_uc", "panel_year"])
    rv = rv[rv["panel_year"] == y] if "panel_year" in rv else rv
    master = _master_latest()
    j = rv.merge(master, on=["upc", "upc_ver_uc"], how="inner")
    # A upc must resolve to one version in a year; movement carries none.
    dup = j["upc"].duplicated(keep=False)
    j["route"] = np.where(dup, "classic_ambiguous_version", "classic_version")
    # Ambiguous: keep the highest version (logged via route + dollars).
    j = (j.sort_values(["upc", "upc_ver_uc"])
          .drop_duplicates("upc", keep="last"))
    j["master_code"] = j["brand_code_uc"]
    j["name_route_code"] = np.nan
    return j[["upc", "route", "brand_code_uc", "brand_descr",
              "size1_amount", "size1_units", "master_code", "name_route_code"]]


def _hms_names():
    """Homescan 2021+ purchases: (year, upc) -> brand_descr_hms, size."""
    parts = []
    for yy in range(REVISED_FIRST, T_HI + 1):
        p = PANEL_RAW / f"purchases_{yy}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["upc", "brand_descr_hms",
                                        "size1_amount_hms", "size1_unit_hms"])
        # within a year a upc has one brand_descr_hms (verified: 0 conflicts);
        # take the most frequent (name, size, unit) to be safe.
        d = (d.groupby(["upc", "brand_descr_hms", "size1_amount_hms",
                        "size1_unit_hms"], dropna=False).size()
              .rename("n").reset_index()
              .sort_values(["upc", "n"], ascending=[True, False])
              .drop_duplicates("upc"))
        d["hms_year"] = yy
        parts.append(d)
    h = pd.concat(parts, ignore_index=True)
    # AM:73-76: size1_amount_hms / 1000 in the unit size1_unit_hms (stripped)
    h["size1_amount"] = pd.to_numeric(h["size1_amount_hms"], errors="coerce") / 1000.0
    h["size1_units"] = h["size1_unit_hms"].astype(str).str.strip()
    h = h.rename(columns={"brand_descr_hms": "brand_descr"})
    return h[["upc", "hms_year", "brand_descr", "size1_amount", "size1_units"]]


def _descr_to_code():
    """AM:239-250, verbatim: crosswalk name -> brand_code_uc, first by file order."""
    xwalk = pd.read_parquet(PANEL_CLEAN / "brand_crosswalk.parquet")
    return (xwalk[["brand_code_uc", "brand_descr_2021"]]
            .drop_duplicates("brand_descr_2021")
            .rename(columns={"brand_descr_2021": "brand_descr"}))


def revised_upc_map(y, upcs):
    """(upc) -> brand + size for a 2021+ year y (panel name route)."""
    u = pd.DataFrame({"upc": np.unique(upcs)})
    h = _hms_names()
    h["dist"] = (h["hms_year"] - y).abs() * 2 + (h["hms_year"] > y)  # tie -> earlier
    h = h[h["upc"].isin(u["upc"])].sort_values(["upc", "dist"])
    h = h.drop_duplicates("upc")
    h["route"] = np.where(h["hms_year"] == y, "hms_same_year", "hms_other_year")
    d2c = _descr_to_code()
    h = h.merge(d2c, on="brand_descr", how="left")
    h["name_route_code"] = h["brand_code_uc"]

    master = _master_latest()
    ml = (master.sort_values(["upc", "upc_ver_uc"])
                .drop_duplicates("upc", keep="last"))   # rule: highest version
    ml = ml.rename(columns={"brand_code_uc": "master_code",
                            "brand_descr": "master_descr",
                            "size1_amount": "m_size", "size1_units": "m_units"})
    out = u.merge(h[["upc", "route", "brand_code_uc", "brand_descr",
                     "size1_amount", "size1_units", "name_route_code"]],
                  on="upc", how="left")
    out = out.merge(ml[["upc", "master_code", "master_descr", "m_size",
                        "m_units"]], on="upc", how="left")
    fb = out["route"].isna() & out["master_code"].notna()
    out.loc[fb, "route"] = "classic_master_fallback"
    out.loc[fb, "brand_code_uc"] = out.loc[fb, "master_code"]
    out.loc[fb, "brand_descr"] = out.loc[fb, "master_descr"]
    out.loc[fb, "size1_amount"] = out.loc[fb, "m_size"]
    out.loc[fb, "size1_units"] = out.loc[fb, "m_units"]
    out["route"] = out["route"].fillna("unmapped")
    return out[["upc", "route", "brand_code_uc", "brand_descr",
                "size1_amount", "size1_units", "master_code",
                "name_route_code"]]


# ---------------------------------------------------------------------------
# Store -> DMA
# ---------------------------------------------------------------------------

def store_dma(y):
    st = pd.read_parquet(OUT / str(y) / "stores.parquet")
    if "dma_code" in st.columns:                       # classic
        return st[["store_code_uc", "dma_code"]].rename(columns={"dma_code": "DMA_Cd"})
    sys.path.insert(0, str(HERE))
    from aggregate_markets import _build_fips_to_dma   # AM:188-218
    f2d = _build_fips_to_dma(PANEL_RAW)
    st = st.merge(f2d[["fips_state_code", "fips_county_code", "DMA_Cd"]],
                  on=["fips_state_code", "fips_county_code"], how="left")
    return st[["store_code_uc", "DMA_Cd"]]


# ---------------------------------------------------------------------------
# One year: movement -> partial (t, m, j) sums + diagnostics
# ---------------------------------------------------------------------------

def _week_keep(all_weeks):
    """Assign every week_end to exactly one source file.

    A week present in several files is kept from the file whose year equals
    the week_end's calendar year (else the earliest file).  Returns
    {file_year: set(week_end strings to keep)} and the overlap log.
    """
    owners = {}
    for fy, weeks in all_weeks.items():
        for w in weeks:
            owners.setdefault(w, []).append(fy)
    keep = {fy: set() for fy in all_weeks}
    overlaps = []
    for w, fys in owners.items():
        wy = int(w[:4])
        own = wy if wy in fys else min(fys)
        keep[own].add(w)
        if len(fys) > 1:
            overlaps.append({"week_end": w, "files": fys, "kept_from": own})
    return keep, overlaps


def year_cells(y, keep_weeks, j45, m65, stores=None):
    t0 = time.time()
    mv_path = OUT / str(y) / "movement_1344.parquet"
    pf = pq.ParquetFile(mv_path)
    if y < REVISED_FIRST:
        umap = classic_upc_map(y)
    else:
        upcs = pq.read_table(mv_path, columns=["upc"])["upc"].to_numpy()
        umap = revised_upc_map(y, upcs)
    umap = umap.reset_index(drop=True)
    uidx = pd.Index(umap["upc"].to_numpy())
    code = umap["brand_code_uc"].fillna(-1).astype("int64").to_numpy()
    ctl = (umap["brand_descr"].fillna("").astype(str).str.upper()
           .str.startswith("CTL").to_numpy())                    # AM:591-595
    oz = (umap["size1_units"].fillna("").astype(str).str.upper().str.strip()
          == "OZ").to_numpy()                                     # AM:626-631
    size = umap["size1_amount"].astype(float).fillna(0.0).to_numpy()
    in45 = np.isin(code, j45)
    # sentinel row for upcs absent from the map (index -1)
    code = np.append(code, -1)
    ctl, oz, in45 = (np.append(ctl, False), np.append(oz, False),
                     np.append(in45, False))
    size = np.append(size, 0.0)

    sd = store_dma(y)
    if stores is not None:   # --stores: other stores get no DMA -> dropped
        sd = sd[sd["store_code_uc"].isin(stores)]
    sidx = pd.Index(sd["store_code_uc"].to_numpy())
    dma = np.append(sd["DMA_Cd"].fillna(-1).astype("int64").to_numpy(), -1)
    in65_s = np.isin(dma, m65)

    kw = np.array(sorted(keep_weeks), dtype="datetime64[D]")
    funnel = dict.fromkeys(FUNNEL, 0.0)
    funnel_rows = dict.fromkeys(FUNNEL, 0)
    dropped_week_dollars = 0.0
    rev_45 = rev_dma = rev_65 = 0.0
    parts = []
    upc_rev = np.zeros(len(uidx) + 1)
    unmapped_rev = {}
    for b in pf.iter_batches(batch_size=BATCH,
                             columns=["store_code_uc", "upc", "week_end",
                                      "units", "prmult", "price"]):
        upc = b.column("upc").to_numpy()
        store = b.column("store_code_uc").to_numpy()
        wk = b.column("week_end").to_numpy(zero_copy_only=False).astype("datetime64[D]")
        units = b.column("units").to_numpy().astype("float64")
        prmult = b.column("prmult").to_numpy()
        price = b.column("price").to_numpy()
        ok = (prmult > 0) & (price > 0)
        wk_ok = np.isin(wk, kw)
        dropped_week_dollars += float(np.sum(
            units[ok & ~wk_ok] * price[ok & ~wk_ok] / prmult[ok & ~wk_ok]))
        ok &= wk_ok
        rev = np.where(ok, units * price / np.where(prmult > 0, prmult, 1), 0.0)

        ui = uidx.get_indexer(upc)
        si = sidx.get_indexer(store)
        c, d = code[ui], dma[si]
        upc_rev += np.bincount(ui + 1, weights=rev, minlength=len(uidx) + 1)[
            np.r_[1:len(uidx) + 1, 0]]
        miss = ui < 0
        if miss.any():
            s = pd.Series(rev[miss]).groupby(upc[miss]).sum()
            for k, v in s.items():
                unmapped_rev[k] = unmapped_rev.get(k, 0.0) + v

        yrs = wk.astype("datetime64[Y]").astype(int) + 1970
        mon = wk.astype("datetime64[M]").astype(int) % 12 + 1
        tcode = yrs * 10 + (mon - 1) // 3 + 1

        # cumulative funnel in the panel's filter order
        steps = [ok,
                 c >= 0,                       # AM:526-528
                 d >= 0,                       # AM:578-582
                 ~ctl[ui],                     # AM:591-595
                 in45[ui],                     # AM:603-605 (panel's 45)
                 in65_s[si],                   # AM:617-618 (panel's 65)
                 oz[ui],                       # AM:626-631
                 (yrs >= T_LO) & (yrs <= T_HI)]
        mask = np.ones(len(upc), dtype=bool)
        for name, s in zip(FUNNEL, steps):
            mask &= s
            funnel[name] += float(rev[mask].sum())
            funnel_rows[name] += int(mask.sum())
        rev_45 += float(rev[ok & in45[ui] & ~ctl[ui]].sum())
        rev_dma += float(rev[ok & (d >= 0)].sum())
        rev_65 += float(rev[ok & in65_s[si]].sum())

        k = mask
        if k.any():
            df = pd.DataFrame({
                "tcode": tcode[k], "m": d[k], "j": c[k],
                "servings": units[k] * size[ui[k]],               # AM:632
                "revenue": rev[k],
                "n_transactions": np.ones(int(k.sum()), dtype="int64"),
            })
            parts.append(df.groupby(["tcode", "m", "j"], sort=False).sum())

    cells = (pd.concat(parts).groupby(level=["tcode", "m", "j"]).sum()
             if parts else pd.DataFrame())
    umap["dollars"] = upc_rev[:len(uidx)]
    umap["year"] = y
    unm = pd.DataFrame({"upc": list(unmapped_rev.keys()),
                        "dollars": list(unmapped_rev.values())})
    if len(unm):
        unm["route"] = "not_in_map"
        unm["year"] = y
        umap = pd.concat([umap, unm], ignore_index=True)
    fun = {"year": y, **{f"dollars_{k}": v for k, v in funnel.items()},
           **{f"rows_{k}": v for k, v in funnel_rows.items()},
           "dollars_brand_in_45": rev_45, "dollars_dma_mapped": rev_dma,
           "dollars_dma_in_65": rev_65,
           "dollars_dropped_duplicate_weeks": dropped_week_dollars,
           "seconds": round(time.time() - t0, 1)}
    print(f"  {y}: {len(cells):,} partial cells, kept "
          f"{funnel['t_in_range'] / max(funnel['all'], 1):.1%} of $ "
          f"({fun['seconds']} s)", flush=True)
    return y, cells, fun, umap


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def build(years, jobs, outdir=OUT, stores_path=None):
    mp = panel_frame()
    j45, m65 = panel_sets(mp)
    print(f"Panel: {len(j45)} brands, {len(m65)} DMAs, {len(mp):,} cells")

    stats = {y: json.loads((OUT / str(y) / "stats.json").read_text())
             for y in years}
    all_weeks = {y: stats[y]["movement"]["weeks"] for y in years}
    keep, overlaps = _week_keep(all_weeks)
    print(f"Weeks: {sum(len(v) for v in all_weeks.values())} file-weeks, "
          f"{len(overlaps)} weeks in more than one file")

    stores = None
    if stores_path:
        stores = pd.read_parquet(stores_path)["store_code_uc"].to_numpy()
        print(f"Store set: {len(stores):,} stores from {stores_path}")

    res = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(year_cells, y, keep[y], j45, m65, stores)
                for y in years]
        for f in futs:
            y, cells, fun, umap = f.result()
            res[y] = (cells, fun, umap)

    # Combine files (a quarter can draw weeks from two files) -> (t, m, j)
    agg = (pd.concat([res[y][0] for y in years if len(res[y][0])])
           .groupby(level=["tcode", "m", "j"]).sum().reset_index())
    agg["t"] = ("Q" + (agg["tcode"] % 10).astype(str)
                + "Y" + (agg["tcode"] // 10).astype(str))
    agg = agg.drop(columns="tcode")
    log = {"cells_raw": len(agg)}

    # AM:648-656  price = revenue / servings; inf/nan/0 -> drop
    agg["p"] = agg["revenue"] / agg["servings"]
    agg["p"] = agg["p"].replace([np.inf, -np.inf, 0.0], np.nan)
    log["cells_bad_price"] = int(agg["p"].isna().sum())
    agg = agg.dropna(subset=["p"])

    # AM:675-689  market_size etc. on (calendar year of t, m), from the panel
    agg["panel_year"] = agg["t"].str.extract(r"Y(\d{4})")[0].astype(int)
    ms = (mp[["panel_year", "m", "market_size", "n_hh", "n_persons"]]
          .drop_duplicates(["panel_year", "m"]))
    agg = agg.merge(ms, on=["panel_year", "m"], how="left")
    log["cells_no_market_size"] = int(agg["market_size"].isna().sum())
    agg = agg.dropna(subset=["market_size"])

    agg["S"] = agg["servings"] / agg["market_size"]                   # AM:692
    agg["W_exp"] = (agg["revenue"]
                    / agg.groupby(["t", "m"])["revenue"].transform("sum"))  # AM:695-697

    # brand_descr + nutrients: the panel's per-j table (constant per j)
    perj = mp.groupby("j")[["brand_descr"] + ZETA].first().reset_index()
    panel = agg.merge(perj, on="j", how="left")

    # AM:741-751  Z_lt = sum_j S_jt * zeta_jl within (t, m)
    zc = [f"Z_{c}" for c in ZETA]
    for c in ZETA:
        panel[f"Z_{c}"] = panel["S"] * panel[c]
    z = panel.groupby(["t", "m"])[zc].sum().reset_index()
    panel = panel.drop(columns=zc).merge(z, on=["t", "m"], how="left")

    # AM:753-766
    panel["market_ids"] = panel["t"] + "_DMA_" + panel["m"].astype(str)
    panel["product_ids"] = panel["j"].astype(str)
    panel["dma_cd"] = panel["m"]
    panel["quarter"] = panel["t"].str.slice(1, 2).astype("int32")
    panel = panel.rename(columns={"S": "shares", "p": "prices"})
    panel["n_transactions"] = panel["n_transactions"].astype("int64")
    panel["panel_year"] = panel["panel_year"].astype("int64")
    panel = panel[OUT_COLS].copy()
    panel = panel.set_index(["t", "m", "j"]).sort_index()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "market_panel_rms.parquet"
    table = pa.Table.from_pandas(panel)
    md = dict(table.schema.metadata or {})
    md[b"rms_caveat"] = RMS_CAVEAT.encode()
    md[b"rms_build"] = json.dumps({
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": "nielsen-cereal-data src/rms_market_panel.py",
        "years": [min(years), max(years)],
        "stores": (str(stores_path) if stores_path else None),
        "n_stores": (int(len(stores)) if stores is not None else None)}).encode()
    table = table.replace_schema_metadata(md)
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, out)
    log["out"], log["stores"] = str(out), stores_path
    print(f"Saved {out}: {panel.shape}, "
          f"{panel.index.get_level_values('t').nunique()} quarters x "
          f"{panel.index.get_level_values('m').nunique()} DMAs x "
          f"{panel.index.get_level_values('j').nunique()} brands")

    # diagnostics
    dg = outdir / "diagnostics"
    dg.mkdir(exist_ok=True)
    fun = pd.DataFrame([res[y][1] for y in years])
    fun.to_parquet(dg / "funnel_by_year.parquet", index=False)
    routes = pd.concat([res[y][2] for y in years], ignore_index=True)
    routes["in_45"] = routes["brand_code_uc"].isin(j45)
    routes.to_parquet(dg / "upc_routes.parquet", index=False)
    ws = pd.DataFrame([{"file_year": fy, "week_end": w,
                        "kept": w in keep[fy]}
                       for fy, ws_ in all_weeks.items() for w in ws_])
    ws.to_parquet(dg / "week_sources.parquet", index=False)
    log["overlaps"] = overlaps
    return panel, fun, routes, log, mp


def write_log(panel, fun, routes, log, mp, path):
    L = ["#+title: RMS market panel build log (D3)",
         "#+date: " + time.strftime("%Y-%m-%d %H:%M %Z"), "",
         "Generated by =python src/rms_market_panel.py= (step 5 of "
         "=src/rms_build.sh=); do not edit by hand.",
         f"Output: ={log['out']}="
         + (f" (store set: ={log['stores']}=)" if log.get('stores') else ""), "",
         "#+begin_quote", RMS_CAVEAT, "#+end_quote", "",
         f"- shape {panel.shape}; "
         f"{panel.index.get_level_values('t').nunique()} quarters, "
         f"{panel.index.get_level_values('m').nunique()} DMAs, "
         f"{panel.index.get_level_values('j').nunique()} brands",
         f"- cells before price filter {log['cells_raw']:,}; dropped for "
         f"undefined/zero price {log['cells_bad_price']}; dropped for no "
         f"market_size {log['cells_no_market_size']}",
         f"- weeks present in more than one file: {len(log['overlaps'])}"
         + ("" if not log["overlaps"] else
            " (kept from the file of the week's calendar year): "
            + ", ".join(f"{o['week_end']} {o['files']}->{o['kept_from']}"
                        for o in log["overlaps"])),
         "", "* Module-1344 dollars by filter stage (cumulative, panel order)",
         "",
         "Stages: all valid rows (prmult>0, price>0, week kept) -> brand "
         "mapped -> store mapped to a DMA -> not CTL -> j in the panel's 45 -> "
         "m in the panel's 65 -> OZ -> t in range.  Shares are of =all=.  "
         "The last three columns are unconditional shares of =all=: mapped "
         "to one of the 45 brands, mapped to any DMA, in the 65 DMAs.", "",
         "| year | all $M | brand | DMA | not CTL | in 45 | in 65 | OZ | kept "
         "| 45 brands | any DMA | 65 DMAs |",
         "|-" * 12 + "|"]
    for _, r in fun.iterrows():
        a = r["dollars_all"]
        sh = [r[f"dollars_{k}"] / a for k in FUNNEL[1:]]
        L.append(f"| {int(r['year'])} | {a / 1e6:,.1f} | "
                 + " | ".join(f"{s:.3f}" for s in sh) + " | "
                 + f"{r['dollars_brand_in_45'] / a:.3f} | "
                 f"{r['dollars_dma_mapped'] / a:.3f} | "
                 f"{r['dollars_dma_in_65'] / a:.3f} |")
    L += ["", "* UPC mapping routes (share of the year's module-1344 dollars)",
          ""]
    rt = (routes.groupby(["year", "route"])["dollars"].sum()
          .unstack(fill_value=0.0))
    rt = rt.div(rt.sum(axis=1), axis=0)
    L.append("| year | " + " | ".join(rt.columns) + " |")
    L.append("|-" * (len(rt.columns) + 1) + "|")
    for y, r in rt.iterrows():
        L.append(f"| {y} | " + " | ".join(f"{v:.3f}" for v in r) + " |")
    L += ["", "* Cells per year (RMS vs panel, same years)", "",
          "| year | RMS cells | panel cells | RMS brands | panel brands | "
          "RMS DMAs |", "|-" * 6 + "|"]
    pr = panel.reset_index()
    for y in sorted(pr["panel_year"].unique()):
        a = pr[pr["panel_year"] == y]
        b = mp[mp["panel_year"] == y]
        L.append(f"| {y} | {len(a):,} | {len(b):,} | {a['j'].nunique()} | "
                 f"{b['j'].nunique()} | {a['m'].nunique()} |")
    ins = pr.groupby(["t", "m"])["shares"].sum()
    L += ["", "* 45-brand inside-share sums per (t, m) (B1; not meaningful "
          "for RMS)", "",
          f"min {ins.min():.2f}, median {ins.median():.2f}, max {ins.max():.2f} "
          f"over {len(ins):,} (t, m) cells."]
    q = [k for k in ins.index if k[0] == "Q2Y2016"]
    if q:
        s16 = ins.loc[q]
        L.append(f"Q2Y2016: min {s16.min():.2f}, median {s16.median():.2f}, "
                 f"max {s16.max():.2f} (Ground measured 3.6 / 31.4 / 78.8 on "
                 "a partial store map).")
    Path(path).write_text("\n".join(L) + "\n")
    print(f"log -> {path}")


def main():
    ap = argparse.ArgumentParser(description="RMS D3: market_panel_rms.parquet")
    ap.add_argument("--years", nargs=2, type=int, default=[2006, 2024])
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 8)))
    ap.add_argument("--log", default="docs/rms-d3-log.org")
    ap.add_argument("--stores", default=None,
                    help="parquet of store_code_uc: build on these stores only")
    ap.add_argument("--outdir", default=str(OUT),
                    help="where the panel and diagnostics go (default OUT)")
    a = ap.parse_args()
    if a.stores and Path(a.outdir).resolve() == OUT.resolve():
        ap.error("--stores needs an --outdir other than OUT")
    if a.stores and a.log == ap.get_default("log"):
        ap.error("--stores needs a --log other than the committed D3 log")
    years = list(range(a.years[0], a.years[1] + 1))
    panel, fun, routes, log, mp = build(years, a.jobs, a.outdir, a.stores)
    write_log(panel, fun, routes, log, mp, a.log)


if __name__ == "__main__":
    sys.exit(main())
