"""
Homescan vs RMS prices: what differs, and does coverage explain the gap?
(docs/rms-prices.org; follows up D4 section 2 of docs/rms-validation.org.)

Stage `hs` (Homescan purchase records, panel universe):
  Rebuilds the panel's (t, m, j) cells from the raw purchases exactly as
  src/aggregate_markets.py does (AM:<line>; the AM helpers are imported, not
  re-implemented): trips merge on (trip_code_uc, panel_year) (AM:556-566),
  household DMA (AM:568-582), quarter of purchase_date (AM:362-371), CTL*
  dropped (AM:587-595), the panel's 45 j and 65 m (taken from
  market_panel.parquet, not re-selected), OZ only (AM:626-631), servings =
  quantity x size1_amount (AM:632), prices = revenue / servings (AM:648-656).
  The all-purchases variant must reproduce the panel's prices on every cell
  (checked and reported).  Each purchase is tagged:
    store link   trip store_code_uc among the RMS stores with module-1344
                 movement that calendar year (diagnostics/stores_by_year)
    chain link   trip retailer_code among the retailer_codes of those stores
    covered      store link OR chain link
  Variants (cells re-aggregated from a subset of purchases or a different
  dollar concept): all / covered / uncovered / store-linked, gross and net
  of coupon_value, deal-flagged vs not, and a thinning placebo (a seeded
  random split of trips with the covered share of dollars, and its
  complement).  Each variant is compared with market_panel_rms.parquet on
  its common cells with D4 section 2's statistics (rms_validate helpers):
  corr of log prices pooled, within (t, m), and net of product and (t, m)
  FE; mean / median / sd of d = log p_RMS - log p_HS; and the two-way
  correlation by the Homescan cell's n_transactions.

Stage `match` (one classic year): Homescan purchases at RMS stores matched
  to RMS movement on (store_code_uc, upc, week_end = the Saturday on or
  after purchase_date): Homescan unit price vs RMS price / prmult.  Then
  the covered-store cells priced four ways (Homescan price; matched RMS
  store-week price; RMS store-quarter average for the same UPC; the RMS
  cell) to split the level gap into recording/imputation, week timing and
  store/UPC mix.

Stage `probe` (one year): price-ending digits and same-price-across-
  households shares by store class, to ask whether prices outside RMS are
  typed or imputed (inconclusive; docs/rms-prices.org 6.7).

Stage `thin` (thinning curve; docs/rms-theta-gap.org): the panel's cells
  rebuilt from random subsets of trips.  Each seed draws one uniform per
  trip and share s keeps the trips with u < s, so one seed's subsets are
  nested.  Keeping every trip must reproduce the panel (checked and
  reported).  Seeds are SEED + 1, SEED + 2, ..., distinct from the placebo's.

Outputs (Nielsen-derived; stay on the cluster, never committed):
  OUTDIR/hs_stats.json, OUTDIR/hs_cells.parquet, OUTDIR/match_<yr>.json,
  OUTDIR/probe_<yr>.json, OUTDIR/thin_stats.json, OUTDIR/thin_cells.parquet

Usage:  python src/rms_prices.py hs    --outdir DIR             (~2 min)
        python src/rms_prices.py match --outdir DIR [--year 2014]
        python src/rms_prices.py probe --outdir DIR [--year 2014]
        python src/rms_prices.py thin  --outdir DIR [--shares 0.25 0.5 0.75]
                                       [--nseeds 3]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

os.umask(0o027)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate_markets as AM                      # noqa: E402
from rms_validate import corr, demean, twoway       # noqa: E402

DATA = Path(os.environ.get("NIELSEN_DATA",
                           "/global/scratch/fsa/fc_jevons/ligon/Data"))
RMS = Path(os.environ.get("RMS_OUT", DATA / "NielsenScanner" / "rms"))
PANEL_CLEAN = Path(os.environ.get(
    "PANEL_CLEAN",
    "/global/home/users/ligon/mirrors/nielsen-cereal-data/data/clean"))
PANEL_RAW = Path(os.environ.get("PANEL_RAW", PANEL_CLEAN.parent / "raw"))
T_LO, T_HI = 2006, 2024
KEY = ["t", "m", "j"]
SEED = 20261019
THICK = [(1, 9), (10, 24), (25, 49), (50, 99), (100, 10**9)]


# ---------------------------------------------------------------------------
# Homescan purchases, panel universe
# ---------------------------------------------------------------------------

def rms_sets():
    """Per calendar year: RMS stores with cereal movement, and their chains."""
    sby = pd.read_parquet(RMS / "diagnostics" / "stores_by_year.parquet",
                          columns=["year", "store_code_uc"])
    stores, chains = {}, {}
    for y in range(T_LO, T_HI + 1):
        s = set(sby.loc[sby["year"] == y, "store_code_uc"].astype(int))
        st = pd.read_parquet(RMS / str(y) / "stores.parquet",
                             columns=["store_code_uc", "retailer_code"])
        st = st[st["store_code_uc"].isin(s)]
        stores[y] = s
        chains[y] = set(st["retailer_code"].dropna().astype(int)) - {0}
    return stores, chains


def load_year(y, fips_to_dma):
    """One panel year's module-1344 purchases with trip, DMA and quarter.

    Returns (all_1344, universe): all_1344 is every purchase that merged to
    a trip (for coverage shares of all cereal dollars); universe is after
    the panel's filters (45 j, 65 m, OZ) minus the j/m restriction, which
    the caller applies.
    """
    pur = AM._read_one_purchase(PANEL_RAW, y)
    ex = pd.read_parquet(PANEL_RAW / f"purchases_{y}.parquet",
                         columns=["coupon_value", "deal_flag_uc"])
    pur["coupon_value"] = pd.to_numeric(ex["coupon_value"],
                                        errors="coerce").fillna(0).to_numpy()
    pur["deal"] = (pd.to_numeric(ex["deal_flag_uc"], errors="coerce")
                   .fillna(0).astype(int).to_numpy())
    trips = pd.read_parquet(
        PANEL_RAW / f"trips_{y}.parquet",
        columns=["trip_code_uc", "household_code", "purchase_date",
                 "panel_year", "retailer_code", "store_code_uc"])
    trips = trips[trips["trip_code_uc"].isin(set(pur["trip_code_uc"]))]
    allc = pur.merge(trips, on=["trip_code_uc", "panel_year"], how="inner")

    if allc["brand_code_uc"].isna().any():                    # AM:519-528
        allc = AM._apply_brand_crosswalk(allc, PANEL_RAW, [1344])
    u = allc.dropna(subset=["brand_code_uc"]).copy()
    u["brand_code_uc"] = u["brand_code_uc"].astype(int)
    pan = AM._read_one_panelist(PANEL_RAW, y, fips_to_dma)
    u = u.merge(pan[["household_code", "panel_year", "DMA_Cd"]],
                on=["household_code", "panel_year"], how="inner")
    u = u.dropna(subset=["DMA_Cd"])
    u["DMA_Cd"] = u["DMA_Cd"].astype(int)
    u = AM._add_quarter(u)
    u = u[~u["brand_descr"].astype(str).str.upper().str.startswith("CTL")]
    u = u[u["size1_units"].astype(str).str.upper().str.strip() == "OZ"]
    u["servings"] = u["quantity"] * u["size1_amount"]
    return allc, u


def tag(df, stores, chains):
    """Calendar year of purchase; store / chain link to RMS that year."""
    yr = pd.to_datetime(df["purchase_date"], format="%Y-%m-%d").dt.year
    df["cal"] = yr.to_numpy()
    st = np.zeros(len(df), bool)
    ch = np.zeros(len(df), bool)
    sc = df["store_code_uc"].fillna(0).astype(np.int64).to_numpy()
    rc = df["retailer_code"].fillna(0).astype(np.int64).to_numpy()
    for y in range(T_LO, T_HI + 1):
        k = df["cal"].to_numpy() == y
        if k.any():
            st[k] = np.isin(sc[k], np.fromiter(stores[y], np.int64))
            ch[k] = np.isin(rc[k], np.fromiter(chains[y], np.int64))
    df["st"], df["ch"] = st, ch
    df["cov"] = st | ch
    return df


def cells(df, rev):
    """(t, m, j) unit values from a purchase subset (AM:633-656)."""
    g = (df.assign(_r=rev).groupby(KEY)
         .agg(servings=("servings", "sum"), revenue=("_r", "sum"),
              n_transactions=("_r", "count")).reset_index())
    g["prices"] = g["revenue"] / g["servings"]
    g["prices"] = g["prices"].replace([np.inf, -np.inf, 0.0], np.nan)
    return g.dropna(subset=["prices"])


def compare(hs, rms, cellset=None):
    """D4 section 2 statistics, Homescan variant vs RMS, common cells."""
    c = hs[KEY + ["prices", "n_transactions"]].merge(
        rms[KEY + ["prices"]], on=KEY, suffixes=("_h", "_r"))
    c = c[(c["prices_h"] > 0) & (c["prices_r"] > 0)]
    if cellset is not None:
        c = c.merge(cellset, on=KEY)
    if len(c) < 50:
        return {"n": int(len(c))}
    tm = pd.factorize(c["t"].astype(str) + "_" + c["m"].astype(str))[0]
    jj = pd.factorize(c["j"])[0]
    lh, lr = np.log(c["prices_h"].to_numpy()), np.log(c["prices_r"].to_numpy())
    wh, wr = twoway(lh, jj, tm), twoway(lr, jj, tm)
    d = lr - lh
    out = {"n": int(len(c)), "corr": corr(lh, lr),
           "corr_tm": corr(demean(lh, tm), demean(lr, tm)),
           "corr_2w": corr(wh, wr),
           "sd_2w_h": float(np.std(wh)), "sd_2w_r": float(np.std(wr)),
           "mean_d": float(d.mean()), "median_d": float(np.median(d)),
           "sd_d": float(d.std(ddof=1)),
           "n_trans_median": float(c["n_transactions"].median())}
    nt = c["n_transactions"].to_numpy()
    out["by_thick"] = {}
    for lo, hi in THICK:
        k = (nt >= lo) & (nt <= hi)
        out["by_thick"][f"{lo}-{hi if hi < 10**9 else 'inf'}"] = {
            "n": int(k.sum()),
            "corr_2w": corr(wh[k], wr[k]) if k.sum() > 50 else None,
            "mean_d": float(d[k].mean()) if k.sum() else None}
    yr = c["t"].astype(str).str.slice(3).astype(int).to_numpy()
    out["by_year"] = {int(y): {"n": int((yr == y).sum()),
                               "corr_2w": corr(wh[yr == y], wr[yr == y]),
                               "mean_d": float(d[yr == y].mean())}
                      for y in np.unique(yr)}
    return out


def share(num, den):
    den = float(den)
    return float(num) / den if den else float("nan")


def stage_hs(outdir):
    t0 = time.time()
    panel = pd.read_parquet(PANEL_CLEAN / "market_panel.parquet").reset_index()
    rms = pd.read_parquet(RMS / "market_panel_rms.parquet").reset_index()
    for d in (panel, rms):
        d["t"] = d["t"].astype(str)
        d["m"] = d["m"].astype(int)
        d["j"] = d["j"].astype(int)
    J, M = set(panel["j"]), set(panel["m"])
    stores, chains = rms_sets()
    fips_to_dma = AM._build_fips_to_dma(PANEL_RAW)

    tq = {f"Q{q}Y{y}" for y in range(T_LO, T_HI + 1) for q in range(1, 5)}
    parts, cover_all = [], []
    for y in range(T_LO - 1, T_HI + 1):
        allc, u = load_year(y, fips_to_dma)
        allc = tag(allc, stores, chains)
        cover_all.append(allc[["cal", "total_price_paid", "st", "ch", "cov"]])
        u = u[u["brand_code_uc"].isin(J) & u["DMA_Cd"].isin(M)]
        u = u[u["t"].isin(tq)]
        u = tag(u, stores, chains)
        u = u.rename(columns={"DMA_Cd": "m", "brand_code_uc": "j"})
        parts.append(u[KEY + ["servings", "total_price_paid", "coupon_value",
                              "deal", "quantity", "trip_code_uc", "panel_year",
                              "retailer_code", "store_code_uc", "cal", "st",
                              "ch", "cov"]])
        print(f"  {y}: universe {len(u):,} rows ({time.time() - t0:.0f} s)",
              flush=True)
    h = pd.concat(parts, ignore_index=True)
    ca = pd.concat(cover_all, ignore_index=True)
    ca = ca[(ca["cal"] >= T_LO) & (ca["cal"] <= T_HI)]
    del parts, cover_all

    S = {"rows": int(len(h))}
    gross = h["total_price_paid"].to_numpy(float)
    net = gross - h["coupon_value"].to_numpy(float)

    # --- check: all purchases, gross, reproduce the panel ----------------
    allg = cells(h, gross)
    chk = allg.merge(panel[KEY + ["prices", "revenue", "servings",
                                  "n_transactions"]],
                     on=KEY, how="outer", suffixes=("", "_panel"),
                     indicator=True)
    chk = chk[chk["t"].isin(tq)]
    both = chk[chk["_merge"] == "both"]
    S["check"] = {
        "cells_rebuilt": int((chk["_merge"] != "right_only").sum()),
        "cells_panel": int((chk["_merge"] != "left_only").sum()),
        "only_rebuilt": int((chk["_merge"] == "left_only").sum()),
        "only_panel": int((chk["_merge"] == "right_only").sum()),
        "max_abs_rel_price_diff": float(
            (both["prices"] / both["prices_panel"] - 1).abs().max()),
        "n_trans_mismatch": int((both["n_transactions"]
                                 != both["n_transactions_panel"]).sum())}
    print("check:", S["check"], flush=True)

    # --- coverage shares of cereal dollars -------------------------------
    def cov_rows(df, w):
        out = {}
        for y, g in df.groupby("cal"):
            ww = w.loc[g.index]
            out[int(y)] = {k: share(ww[g[k]].sum(), ww.sum())
                           for k in ("st", "ch", "cov")}
        tot = w.sum()
        out["all"] = {k: share(w[df[k]].sum(), tot) for k in ("st", "ch", "cov")}
        return out
    S["cover_all1344"] = cov_rows(ca, ca["total_price_paid"])
    S["cover_universe"] = cov_rows(h, h["total_price_paid"])

    # --- coupons and deals -----------------------------------------------
    def cd(g):
        gp = g["total_price_paid"].sum()
        return {"dollars": float(gp),
                "coupon_share": share(g["coupon_value"].sum(), gp),
                "coupon_used_share_dollars": share(
                    g.loc[g["coupon_value"] > 0, "total_price_paid"].sum(), gp),
                "deal_share_dollars": share(
                    g.loc[g["deal"] == 1, "total_price_paid"].sum(), gp),
                "deal_share_rows": float((g["deal"] == 1).mean()),
                "coupon_share_cov": share(
                    g.loc[g["cov"], "coupon_value"].sum(),
                    g.loc[g["cov"], "total_price_paid"].sum()),
                "coupon_share_unc": share(
                    g.loc[~g["cov"], "coupon_value"].sum(),
                    g.loc[~g["cov"], "total_price_paid"].sum()),
                "deal_share_cov": share(
                    g.loc[g["cov"] & (g["deal"] == 1), "total_price_paid"].sum(),
                    g.loc[g["cov"], "total_price_paid"].sum()),
                "deal_share_unc": share(
                    g.loc[~g["cov"] & (g["deal"] == 1), "total_price_paid"].sum(),
                    g.loc[~g["cov"], "total_price_paid"].sum())}
    h["year"] = h["t"].str.slice(3).astype(int)
    S["coupon_deal_by_year"] = {int(y): cd(g) for y, g in h.groupby("year")}
    S["coupon_deal_all"] = cd(h)
    S["coupon_deal_by_brand"] = {int(j): cd(g) for j, g in h.groupby("j")}
    S["coupon_deal_by_brand_cov"] = {
        int(j): share(g["cov"].mul(g["total_price_paid"]).sum(),
                      g["total_price_paid"].sum()) for j, g in h.groupby("j")}

    # coupon rate per cell: how much does it vary within (t, m), net of j?
    cr = (h.groupby(KEY)[["coupon_value", "total_price_paid"]].sum()
          .reset_index())
    cr = cr[cr["total_price_paid"] > 0]
    cr["l1c"] = np.log1p(-(cr["coupon_value"] / cr["total_price_paid"])
                         .clip(upper=0.99))
    tm = pd.factorize(cr["t"] + "_" + cr["m"].astype(str))[0]
    jj = pd.factorize(cr["j"])[0]
    S["coupon_cell_sd"] = {"sd_log1mc": float(cr["l1c"].std()),
                           "sd_log1mc_2w": float(np.std(twoway(
                               cr["l1c"].to_numpy(), jj, tm)))}

    # price endings: unit price (total_price_paid / quantity) ending in 9c
    up = np.round(gross / h["quantity"].to_numpy(float) * 100)
    nine = (up % 10 == 9)
    cls = np.where(h["st"], "store", np.where(h["ch"], "chain_only", "uncov"))
    S["ending9"] = {}
    for k in ("store", "chain_only", "uncov"):
        sel = cls == k
        S["ending9"][k] = {"rows": int(sel.sum()),
                           "share_9": float(nine[sel].mean()),
                           "share_99_or_49": float(np.isin(up[sel] % 100,
                                                           [99, 49]).mean())}
    S["ending9_by_era"] = {}
    for lo, hi in ((2006, 2010), (2011, 2015), (2016, 2020), (2021, 2024)):
        e = (h["year"] >= lo) & (h["year"] <= hi)
        S["ending9_by_era"][f"{lo}-{hi}"] = {
            k: float(nine[e.to_numpy() & (cls == k)].mean())
            for k in ("store", "chain_only", "uncov")}

    # --- variants vs RMS ---------------------------------------------------
    rng = np.random.default_rng(SEED)
    trip_key = h["trip_code_uc"].astype(np.int64) * 10000 + h["panel_year"]
    codes, inv = np.unique(trip_key.to_numpy(), return_inverse=True)
    u = rng.random(len(codes))[inv]
    p_cov = share(h.loc[h["cov"], "total_price_paid"].sum(),
                  h["total_price_paid"].sum())
    plac = u < p_cov
    S["placebo_p"] = p_cov

    covm, stm = h["cov"].to_numpy(), h["st"].to_numpy()
    dl = h["deal"].to_numpy() == 1
    V = {
        "all_gross": (np.ones(len(h), bool), gross),
        "all_net": (np.ones(len(h), bool), net),
        "covered_gross": (covm, gross),
        "covered_net": (covm, net),
        "uncovered_gross": (~covm, gross),
        "store_gross": (stm, gross),
        "chain_only_gross": (covm & ~stm, gross),
        "placebo_gross": (plac, gross),
        "placebo_compl_gross": (~plac, gross),
        "nodeal_gross": (~dl, gross),
        "deal_gross": (dl, gross),
    }
    VC = {}
    for name, (mask, rev) in V.items():
        VC[name] = cells(h[mask], rev[mask])
    S["variants"] = {name: compare(c, rms) for name, c in VC.items()}

    # same cells for the coverage split and its placebo
    inter = None
    for name in ("covered_gross", "uncovered_gross", "placebo_gross",
                 "placebo_compl_gross"):
        k = VC[name][KEY]
        inter = k if inter is None else inter.merge(k, on=KEY)
    S["variants_samecells"] = {
        name: compare(VC[name], rms, inter)
        for name in ("all_gross", "all_net", "covered_gross", "covered_net",
                     "uncovered_gross", "placebo_gross",
                     "placebo_compl_gross")}

    # covered vs uncovered Homescan prices, same cells (no RMS)
    cu = VC["covered_gross"][KEY + ["prices"]].merge(
        VC["uncovered_gross"][KEY + ["prices"]], on=KEY,
        suffixes=("_c", "_u"))
    S["covered_minus_uncovered_logp"] = float(
        (np.log(cu["prices_c"]) - np.log(cu["prices_u"])).mean())

    allcells = []
    for name, c in VC.items():
        allcells.append(c.assign(variant=name))
    pd.concat(allcells, ignore_index=True).to_parquet(
        outdir / "hs_cells.parquet")
    S["seconds"] = time.time() - t0
    (outdir / "hs_stats.json").write_text(json.dumps(S, indent=1,
                                                     default=float))
    print(f"wrote {outdir}/hs_stats.json ({S['seconds']:.0f} s)")


# ---------------------------------------------------------------------------
# Stage match: one classic year, store-week-UPC
# ---------------------------------------------------------------------------

def stage_match(outdir, year):
    t0 = time.time()
    rms = pd.read_parquet(RMS / "market_panel_rms.parquet").reset_index()
    rms["t"] = rms["t"].astype(str)
    rms["m"], rms["j"] = rms["m"].astype(int), rms["j"].astype(int)
    panel = pd.read_parquet(PANEL_CLEAN / "market_panel.parquet").reset_index()
    J, M = set(panel["j"].astype(int)), set(panel["m"].astype(int))
    stores, chains = rms_sets()
    fips_to_dma = AM._build_fips_to_dma(PANEL_RAW)

    us = []
    for y in (year - 1, year, year + 1):
        allc, u = load_year(y, fips_to_dma)
        us.append(u)
    u = pd.concat(us, ignore_index=True)
    u = u[u["t"].str.endswith(str(year))]
    u = u[u["brand_code_uc"].isin(J) & u["DMA_Cd"].isin(M)]
    u = tag(u, stores, chains)
    u = u.rename(columns={"DMA_Cd": "m", "brand_code_uc": "j"})
    pd_ = pd.to_datetime(u["purchase_date"], format="%Y-%m-%d")
    # Saturday on or after the purchase date (RMS week_end is a Saturday)
    u["week_end"] = (pd_ + pd.to_timedelta((5 - pd_.dt.dayofweek) % 7,
                                           unit="D")).dt.date
    u["upc"] = u["upc"].astype(np.int64)
    s = u[u["st"]].copy()
    s["store_code_uc"] = s["store_code_uc"].astype(np.int64)
    R = {"year": year, "hs_rows_universe": int(len(u)),
         "hs_rows_store_linked": int(len(s)),
         "hs_dollars_store_linked_share": share(
             s["total_price_paid"].sum(), u["total_price_paid"].sum())}
    print(f"  store-linked HS rows {len(s):,}", flush=True)

    # stream movement: keep (store, upc) pairs bought by panelists, and
    # aggregate every row to store-quarter-upc (units, dollars) for the mix
    pairs = s[["store_code_uc", "upc"]].drop_duplicates()
    want_st = pa.array(sorted(set(s["store_code_uc"])), pa.int32())
    want_upc = pa.array(sorted(set(s["upc"])), pa.int64())
    pf = pq.ParquetFile(RMS / str(year) / "movement_1344.parquet")
    keep, sqa = [], []
    for i in range(pf.num_row_groups):
        tb = pf.read_row_group(i, columns=["store_code_uc", "upc", "week_end",
                                           "units", "prmult", "price"])
        ok = pc.and_(pc.and_(pc.greater(tb["prmult"], 0),
                             pc.greater(tb["price"], 0)),
                     pc.greater(tb["units"], 0))
        tb = tb.filter(ok)
        m = pc.and_(pc.is_in(tb["store_code_uc"], value_set=want_st),
                    pc.is_in(tb["upc"], value_set=want_upc))
        sub = tb.filter(m).to_pandas()
        sub["store_code_uc"] = sub["store_code_uc"].astype(np.int64)
        keep.append(sub.merge(pairs, on=["store_code_uc", "upc"]))
        if i % 20 == 0:
            print(f"  rg {i}/{pf.num_row_groups} kept {sum(map(len, keep)):,}"
                  f" ({time.time() - t0:.0f} s)", flush=True)
    mv = pd.concat(keep, ignore_index=True)
    del keep
    mv["store_code_uc"] = mv["store_code_uc"].astype(np.int64)
    mv["unit_p"] = mv["price"] / mv["prmult"]
    mv["dollars"] = mv["units"] * mv["unit_p"]
    wk = pd.to_datetime(mv["week_end"])
    mv["t"] = "Q" + wk.dt.quarter.astype(str) + "Y" + wk.dt.year.astype(str)
    mv["week_end"] = wk.dt.date
    R["movement_rows_kept"] = int(len(mv))

    # store-week match (same week, and +1 week for late transmissions)
    s["hs_unit"] = s["total_price_paid"] / s["quantity"]
    mk = mv[["store_code_uc", "upc", "week_end", "unit_p", "prmult"]]
    a = s.merge(mk, on=["store_code_uc", "upc", "week_end"], how="left")
    s2 = s.copy()
    s2["week_end"] = (pd.to_datetime(s2["week_end"])
                      + pd.Timedelta(days=7)).dt.date
    b = s2.merge(mk, on=["store_code_uc", "upc", "week_end"], how="left")
    a["unit_p_next"] = b["unit_p"].to_numpy()
    R["matched_same_week_share"] = float(a["unit_p"].notna().mean())
    R["matched_next_week_only_share"] = float(
        (a["unit_p"].isna() & a["unit_p_next"].notna()).mean())
    mm = a[a["unit_p"].notna() & (a["hs_unit"] > 0)].copy()
    lr = np.log(mm["hs_unit"] / mm["unit_p"])
    absd = (mm["hs_unit"] - mm["unit_p"]).abs()
    R["match"] = {
        "n": int(len(mm)),
        "within_1c": float((absd <= 0.01 + 1e-9).mean()),
        "within_1c_dollarw": share(mm.loc[absd <= 0.01 + 1e-9,
                                          "total_price_paid"].sum(),
                                   mm["total_price_paid"].sum()),
        "within_1pct": float((lr.abs() <= 0.01).mean()),
        "mean_logratio": float(lr.mean()), "median_logratio": float(lr.median()),
        "sd_logratio": float(lr.std()),
        "p05": float(lr.quantile(.05)), "p25": float(lr.quantile(.25)),
        "p75": float(lr.quantile(.75)), "p95": float(lr.quantile(.95)),
        "prmult_gt1_share": float((mm["prmult"] > 1).mean()),
        "within_1c_next_week": float(
            ((mm["hs_unit"] - mm["unit_p_next"]).abs() <= 0.01 + 1e-9).mean()),
        "within_1c_coupon_gt0": float(
            (absd[mm["coupon_value"] > 0] <= 0.01 + 1e-9).mean()),
        "within_1c_deal1": float((absd[mm["deal"] == 1] <= 0.01 + 1e-9).mean()),
        "within_1c_deal0": float((absd[mm["deal"] == 0] <= 0.01 + 1e-9).mean()),
        "mean_logratio_deal1": float(lr[mm["deal"] == 1].mean()),
        "mean_logratio_deal0": float(lr[mm["deal"] == 0].mean()),
        "net_within_1c": float(((mm["hs_unit"]
                                 - mm["coupon_value"] / mm["quantity"]
                                 - mm["unit_p"]).abs() <= 0.01 + 1e-9).mean()),
    }
    edges = [-np.inf, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 0.6, 0.8, np.inf]
    h_ = np.histogram(lr.to_numpy(), bins=edges)[0] / len(lr)
    R["match"]["logratio_hist"] = {f"{edges[i]}..{edges[i + 1]}": float(h_[i])
                                   for i in range(len(h_))}
    R["match"]["quantity_gt1_share"] = float((mm["quantity"] > 1).mean())
    q1 = mm["quantity"] == 1
    R["match"]["within_1c_q1"] = float((absd[q1] <= 0.01 + 1e-9).mean())
    R["match"]["within_1c_qgt1"] = float((absd[~q1] <= 0.01 + 1e-9).mean())
    R["match"]["mean_logratio_q1"] = float(lr[q1].mean())
    R["match"]["mean_logratio_qgt1"] = float(lr[~q1].mean())
    print("match:", R["match"], flush=True)
    # is the non-matching fifth a few chains (NielsenIQ's documented
    # exceptions, where panelists enter prices) or spread over all of them?
    g = (mm.assign(ok=(absd <= 0.01 + 1e-9).to_numpy())
         .groupby("retailer_code")
         .agg(ok=("ok", "mean"), dol=("total_price_paid", "sum")))
    tot = g["dol"].sum()
    R["by_retailer"] = {
        "n_retailers": int(len(g)),
        "dollar_share_ok_ge90": share(g.loc[g["ok"] >= 0.9, "dol"].sum(), tot),
        "dollar_share_ok_50_90": share(
            g.loc[(g["ok"] >= 0.5) & (g["ok"] < 0.9), "dol"].sum(), tot),
        "dollar_share_ok_lt50": share(g.loc[g["ok"] < 0.5, "dol"].sum(), tot),
        "n_retailers_ok_lt50": int((g["ok"] < 0.5).sum()),
        "dollar_weighted_median_ok": float(
            g.sort_values("ok").assign(c=lambda x: x["dol"].cumsum() / tot)
            .query("c >= 0.5")["ok"].iloc[0])}
    print("by retailer:", R["by_retailer"], flush=True)

    # four pricings of the store-linked purchases, aggregated to (t, m, j)
    sq = (mv.groupby(["store_code_uc", "upc", "t"])[["units", "dollars"]]
          .sum().reset_index())
    sq["sq_unit"] = sq["dollars"] / sq["units"]
    a = a.merge(sq[["store_code_uc", "upc", "t", "sq_unit"]],
                on=["store_code_uc", "upc", "t"], how="left")
    ok = a["unit_p"].notna() & a["sq_unit"].notna() & (a["hs_unit"] > 0)
    a = a[ok]
    q = a["quantity"].to_numpy(float)
    pr = {"hs": a["total_price_paid"].to_numpy(float),
          "rms_week": q * a["unit_p"].to_numpy(float),
          "rms_storeqtr": q * a["sq_unit"].to_numpy(float)}
    cc = None
    for k, rev in pr.items():
        c = cells(a, rev)[KEY + ["prices", "n_transactions"]].rename(
            columns={"prices": f"p_{k}"})
        cc = c if cc is None else cc.merge(c.drop(columns="n_transactions"),
                                           on=KEY)
    cc = cc.merge(rms[KEY + ["prices"]].rename(columns={"prices": "p_rms"}),
                  on=KEY)
    L = {k: np.log(cc[f"p_{k}"].to_numpy()) for k in
         ("hs", "rms_week", "rms_storeqtr", "rms")}
    R["decomp_cells"] = int(len(cc))
    R["decomp_mean"] = {
        "rms_minus_hs": float((L["rms"] - L["hs"]).mean()),
        "recording (rms_week - hs)": float((L["rms_week"] - L["hs"]).mean()),
        "timing (rms_storeqtr - rms_week)": float(
            (L["rms_storeqtr"] - L["rms_week"]).mean()),
        "store/upc mix (rms - rms_storeqtr)": float(
            (L["rms"] - L["rms_storeqtr"]).mean())}
    tm = pd.factorize(cc["t"] + "_" + cc["m"].astype(str))[0]
    jj = pd.factorize(cc["j"])[0]
    W = {k: twoway(v, jj, tm) for k, v in L.items()}
    R["decomp_corr_2w_with_rms"] = {k: corr(W[k], W["rms"]) for k in W}
    R["decomp_corr_2w_with_hs"] = {k: corr(W[k], W["hs"]) for k in W}
    R["decomp_sd_2w"] = {k: float(np.std(W[k])) for k in W}
    nt = cc["n_transactions"].to_numpy()
    R["decomp_by_thick"] = {}
    for lo, hi in THICK:
        k = (nt >= lo) & (nt <= hi)
        if k.sum() > 50:
            R["decomp_by_thick"][f"{lo}-{hi}"] = {
                "n": int(k.sum()),
                **{f"corr_{x}_rms": corr(W[x][k], W["rms"][k])
                   for x in ("hs", "rms_week", "rms_storeqtr")}}
    R["seconds"] = time.time() - t0
    (outdir / f"match_{year}.json").write_text(json.dumps(R, indent=1,
                                                          default=float))
    print(f"wrote {outdir}/match_{year}.json ({R['seconds']:.0f} s)")


def stage_probe(outdir, year):
    """Are Homescan prices outside RMS typed by panelists or store-data?

    Two fingerprints, by store class (RMS store; RMS chain, other store;
    outside RMS with a store code; outside RMS without one; and the largest
    outside chain by cereal dollars, unnamed):
      - last cent digit of the unit price (typed shelf prices bunch on a
        chain's price endings; weekly averages spread over all ten digits);
      - among (store, upc, week) cells bought by >= 2 households, the share
        where every household's unit price is the same to the cent (a
        store-week average gives one price to all panelists that week).
    """
    stores, chains = rms_sets()
    fips_to_dma = AM._build_fips_to_dma(PANEL_RAW)
    panel = pd.read_parquet(PANEL_CLEAN / "market_panel.parquet").reset_index()
    J, M = set(panel["j"].astype(int)), set(panel["m"].astype(int))
    us = []
    for y in (year - 1, year, year + 1):
        _, u = load_year(y, fips_to_dma)
        us.append(u)
    u = pd.concat(us, ignore_index=True)
    u = u[u["t"].str.endswith(str(year))]
    u = u[u["brand_code_uc"].isin(J) & u["DMA_Cd"].isin(M)]
    u = tag(u, stores, chains)
    u = u[(u["quantity"] > 0) & (u["total_price_paid"] > 0)]
    sc = u["store_code_uc"].fillna(0).astype(np.int64)
    unc = ~u["cov"]
    big = (u[unc].groupby("retailer_code")["total_price_paid"].sum()
           .idxmax())
    cls = np.select(
        [u["st"], u["ch"], unc & (u["retailer_code"] == big),
         unc & (sc != 0), unc & (sc == 0)],
        ["rms_store", "rms_chain_other_store", "largest_outside_chain",
         "outside_with_store_code", "outside_no_store_code"], "other")
    u["cls"] = cls
    u["cents"] = np.round(u["total_price_paid"] / u["quantity"] * 100)
    u["digit"] = (u["cents"] % 10).astype(int)
    pd_ = pd.to_datetime(u["purchase_date"], format="%Y-%m-%d")
    u["week_end"] = (pd_ + pd.to_timedelta((5 - pd_.dt.dayofweek) % 7,
                                           unit="D")).dt.date
    P = {"year": year}
    tot = u["total_price_paid"].sum()
    for k, g in u.groupby("cls"):
        dig = g["digit"].value_counts(normalize=True).sort_index()
        cell = g[g["store_code_uc"].fillna(0) != 0].groupby(
            ["store_code_uc", "upc", "week_end"])
        agg = cell.agg(nh=("household_code", "nunique"),
                       lo=("cents", "min"), hi=("cents", "max"))
        agg = agg[agg["nh"] >= 2]
        P[k] = {"rows": int(len(g)),
                "dollar_share": share(g["total_price_paid"].sum(), tot),
                "store_code_share": float((g["store_code_uc"].fillna(0)
                                           != 0).mean()),
                "digit_dist": {int(i): float(v) for i, v in dig.items()},
                "top_digit_share": float(dig.max()),
                "multi_hh_cells": int(len(agg)),
                "same_price_share": (float((agg["lo"] == agg["hi"]).mean())
                                     if len(agg) else None)}
        print(k, {a: b for a, b in P[k].items() if a != "digit_dist"},
              flush=True)
    (outdir / f"probe_{year}.json").write_text(json.dumps(P, indent=1,
                                                          default=float))


# ---------------------------------------------------------------------------
# Stage thin: random trip subsets (thinning curve)
# ---------------------------------------------------------------------------

def stage_thin(outdir, shares, nseeds):
    t0 = time.time()
    panel = pd.read_parquet(PANEL_CLEAN / "market_panel.parquet").reset_index()
    panel["t"] = panel["t"].astype(str)
    J, M = set(panel["j"].astype(int)), set(panel["m"].astype(int))
    fips_to_dma = AM._build_fips_to_dma(PANEL_RAW)
    tq = {f"Q{q}Y{y}" for y in range(T_LO, T_HI + 1) for q in range(1, 5)}
    parts = []
    for y in range(T_LO - 1, T_HI + 1):
        _, u = load_year(y, fips_to_dma)
        u = u[u["brand_code_uc"].isin(J) & u["DMA_Cd"].isin(M)
              & u["t"].isin(tq)]
        u = u.rename(columns={"DMA_Cd": "m", "brand_code_uc": "j"})
        parts.append(u[KEY + ["servings", "total_price_paid", "trip_code_uc",
                              "panel_year"]])
        print(f"  {y}: universe {len(u):,} rows ({time.time() - t0:.0f} s)",
              flush=True)
    h = pd.concat(parts, ignore_index=True)
    del parts
    gross = h["total_price_paid"].to_numpy(float)

    # check: every trip kept reproduces the panel
    chk = cells(h, gross).merge(
        panel.loc[panel["t"].isin(tq), KEY + ["prices", "n_transactions"]],
        on=KEY, how="outer", suffixes=("", "_panel"), indicator=True)
    both = chk[chk["_merge"] == "both"]
    S = {"rows": int(len(h)), "shares": list(shares), "check": {
        "only_rebuilt": int((chk["_merge"] == "left_only").sum()),
        "only_panel": int((chk["_merge"] == "right_only").sum()),
        "max_abs_rel_price_diff": float(
            (both["prices"] / both["prices_panel"] - 1).abs().max()),
        "n_trans_mismatch": int((both["n_transactions"]
                                 != both["n_transactions_panel"]).sum())}}
    print("check:", S["check"], flush=True)

    trip_key = h["trip_code_uc"].astype(np.int64) * 10000 + h["panel_year"]
    codes, inv = np.unique(trip_key.to_numpy(), return_inverse=True)
    S["trips"] = int(len(codes))
    S["variants"], out = {}, []
    for k in range(1, nseeds + 1):
        u = np.random.default_rng(SEED + k).random(len(codes))[inv]
        for s in shares:
            keep = u < s
            name = f"thin{round(100 * s)}_s{k}"
            c = cells(h[keep], gross[keep])
            out.append(c.assign(variant=name))
            S["variants"][name] = {
                "seed": SEED + k, "share": s, "cells": int(len(c)),
                "rows": int(keep.sum()),
                "dollar_share": share(gross[keep].sum(), gross.sum())}
            print(f"  {name}: {len(c):,} cells, "
                  f"{S['variants'][name]['dollar_share']:.3f} of dollars",
                  flush=True)
    pd.concat(out, ignore_index=True).to_parquet(outdir / "thin_cells.parquet")
    S["seconds"] = time.time() - t0
    (outdir / "thin_stats.json").write_text(json.dumps(S, indent=1,
                                                       default=float))
    print(f"wrote {outdir}/thin_stats.json ({S['seconds']:.0f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stage", choices=["hs", "match", "probe", "thin"])
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--year", type=int, default=2014)
    ap.add_argument("--shares", nargs="+", type=float,
                    default=[0.25, 0.5, 0.75])
    ap.add_argument("--nseeds", type=int, default=3)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    if a.stage == "hs":
        stage_hs(a.outdir)
    elif a.stage == "match":
        stage_match(a.outdir, a.year)
    elif a.stage == "thin":
        stage_thin(a.outdir, a.shares, a.nseeds)
    else:
        stage_probe(a.outdir, a.year)


if __name__ == "__main__":
    main()
