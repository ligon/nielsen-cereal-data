"""
RMS stage D4: the validation report, RMS vs the Homescan panel.

Compares market_panel_rms.parquet (D3, src/rms_market_panel.py) with the
panel's market_panel.parquet on their common (t, m, j) cells and writes
docs/rms-validation.org.  It reports numbers only; whether the discrepancies
are acceptable is EL's call (charter-rms section 2, D4).

Charter D4:
  - per-year cell counts (panel, RMS, common, one-source-only);
  - correlation of log prices on common cells;
  - correlation of within-(t, m) log shares on common cells;
  - brand-years present in one source but not the other.
Amendment 1, item 3:
  (a) the per-(t, m) 45-brand inside-share sums for both sources, by year;
  (b) the option-(2) factor, RMS 45-brand servings / panel 45-brand
      servings per (t, m), by year;
  (c) brand-years created or lost by the panel's name-route crosswalk
      (2021+; e.g. Trix 557579 vs 747763);
  (d) per year, the share of module-1344 revenue mapped to the 45 brands
      and the share mapped to a DMA.
Plus, if present, the D6 homog comparison (CerealDemand
src/rms_homog_compare.py -> <rms>/laws/homog_compare.json).

Inputs (read-only):
  <rms>/market_panel_rms.parquet, <rms>/diagnostics/{funnel_by_year,
  upc_routes}.parquet, <panel clean>/market_panel.parquet

Usage:  python src/rms_validate.py [--out docs/rms-validation.org]
                                   [--d6 <homog_compare.json>]
                                   [--gate-crosswalk]
        (make rms-validate)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("NIELSEN_DATA",
                           "/global/scratch/fsa/fc_jevons/ligon/Data"))
RMS = Path(os.environ.get("RMS_OUT", DATA / "NielsenScanner" / "rms"))
PANEL_CLEAN = Path(os.environ.get(
    "PANEL_CLEAN",
    "/global/home/users/ligon/mirrors/nielsen-cereal-data/data/clean"))
T_LO, T_HI = 2006, 2024
REVISED_FIRST = 2021
KEY = ["t", "m", "j"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def year_of(t):
    """'Q3Y2016' -> 2016 (the panel's quarter label, AM:362-371)."""
    return t.astype(str).str.slice(3).astype(int)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def demean(x, g):
    """x minus its group mean (g: integer codes)."""
    s = np.bincount(g, weights=x)
    n = np.bincount(g)
    return x - (s / n)[g]


def twoway(x, g1, g2, tol=1e-12, maxit=1000):
    """Alternating projections: x net of g1 and g2 fixed effects (FWL)."""
    y = np.asarray(x, float) - np.mean(x)
    for _ in range(maxit):
        z = demean(demean(y, g1), g2)
        if np.max(np.abs(z - y)) < tol:
            return z
        y = z
    return y


def q(x, p):
    return float(np.quantile(np.asarray(x, float), p)) if len(x) else float("nan")


def f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:,.{nd}f}"


FLAGS = {}   # cross-section facts collected for the summary


def money(v):
    return f"${v / 1e6:,.2f}M" if v >= 1e5 else f"${v:,.0f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("-" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load():
    p = pd.read_parquet(PANEL_CLEAN / "market_panel.parquet").reset_index()
    r = pd.read_parquet(RMS / "market_panel_rms.parquet").reset_index()
    meta = pq.read_schema(RMS / "market_panel_rms.parquet").metadata or {}
    caveat = meta.get(b"rms_caveat", b"").decode()
    for d in (p, r):
        d["year"] = year_of(d["t"])
    fun = pd.read_parquet(RMS / "diagnostics" / "funnel_by_year.parquet")
    routes = pd.read_parquet(RMS / "diagnostics" / "upc_routes.parquet")
    return p, r, caveat, fun, routes


def common_cells(p, r):
    cols = ["prices", "shares", "servings", "revenue", "W_exp", "market_size"]
    c = p[KEY + ["year", "brand_descr"] + cols].merge(
        r[KEY + cols], on=KEY, suffixes=("_p", "_r"), how="inner")
    c["tm"] = pd.factorize(c["t"].astype(str) + "_" + c["m"].astype(str))[0]
    c["jj"] = pd.factorize(c["j"])[0]
    c["lp_p"], c["lp_r"] = np.log(c["prices_p"]), np.log(c["prices_r"])
    c["ls_p"], c["ls_r"] = np.log(c["shares_p"]), np.log(c["shares_r"])
    # within-(t, m) over the COMMON cells, so both sources see the same brands
    for s in ("p", "r"):
        c[f"ls_tm_{s}"] = demean(c[f"ls_{s}"].to_numpy(), c["tm"].to_numpy())
        c[f"lp_tm_{s}"] = demean(c[f"lp_{s}"].to_numpy(), c["tm"].to_numpy())
        c[f"ls_2w_{s}"] = twoway(c[f"ls_{s}"].to_numpy(), c["jj"].to_numpy(),
                                 c["tm"].to_numpy())
        c[f"lp_2w_{s}"] = twoway(c[f"lp_{s}"].to_numpy(), c["jj"].to_numpy(),
                                 c["tm"].to_numpy())
    c["dlp"] = c["lp_r"] - c["lp_p"]
    return c


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def sec_cells(p, r, c):
    rows = []
    allp = set(map(tuple, p[KEY].astype(str).to_numpy()))
    allr = set(map(tuple, r[KEY].astype(str).to_numpy()))
    tot = np.zeros(5, dtype=np.int64)
    for y in range(int(p["year"].min()), T_HI + 1):
        np_ = int((p["year"] == y).sum())
        nr = int((r["year"] == y).sum())
        nc = int((c["year"] == y).sum())
        if y >= T_LO:
            tot += [np_, nr, nc, np_ - nc, nr - nc]
        rows.append([y, f"{np_:,}", f"{nr:,}", f"{nc:,}", f"{np_ - nc:,}",
                     f"{nr - nc:,}",
                     f(nc / np_ if np_ else float("nan")),
                     f(nc / nr if nr else float("nan"))])
    rows.append([f"{T_LO}-{T_HI}", *[f"{x:,}" for x in tot],
                 f(tot[2] / tot[0]), f(tot[2] / tot[1])])
    assert len(allp & allr) == len(c)
    L = ["* 1. Cells per year",
         "",
         "A cell is one (t, m, j).  /common/ = in both files; /panel-only/ and",
         "/RMS-only/ are cells one source has and the other lacks (a brand with",
         "no recorded purchase, or no store sale, in that DMA-quarter).  The",
         f"panel starts in 2004; RMS in {T_LO}.  Totals are over {T_LO}-{T_HI}.",
         ""]
    L += table(["year", "panel", "RMS", "common", "panel-only", "RMS-only",
                "common / panel", "common / RMS"], rows)
    return L


def sec_panel_only(p, r, routes):
    """Panel-only cells grouped by brand-year, with a version-route check:
    where did last year's UPCs of the brand go this year in RMS?"""
    m = p[p["year"] >= T_LO].merge(r[KEY], on=KEY, how="left", indicator=True)
    po = m[m["_merge"] == "left_only"]
    g = po.groupby(["year", "j", "brand_descr"]).agg(
        cells=("t", "size"), quarters=("t", "nunique")).reset_index()
    g = g[g["cells"] >= 10].sort_values(["year", "j"])
    rows = []
    for _, x in g.iterrows():
        y, j = int(x["year"]), int(x["j"])
        prev = routes[(routes["year"] == y - 1) & (routes["brand_code_uc"] == j)]
        now = routes[(routes["year"] == y) & routes["upc"].isin(prev["upc"])]
        el = now[now["brand_code_uc"] != j]
        lab = [("unmapped" if pd.isna(v) else str(int(v))) + " " + str(n or "")
               for v, n in zip(el["brand_code_uc"], el["brand_descr"])]
        dest = (pd.Series(el["dollars"].to_numpy(), index=lab, dtype=float)
                .groupby(level=0).sum().sort_values(ascending=False))
        stay = now.loc[now["brand_code_uc"] == j, "dollars"].sum()
        rows.append([y, j, x["brand_descr"], f"{int(x['cells']):,}",
                     int(x["quarters"]), f(stay / 1e6, 2),
                     "; ".join(f"{k.strip()} ${v / 1e6:,.2f}M"
                               for k, v in dest.head(2).items()) or "-"])
    L = ["** Panel-only cells by brand-year (>= 10 cells)", "",
         "For each such brand-year: the RMS dollars, this year, of the UPCs",
         "that RMS assigned to the brand LAST year, split into those still",
         "assigned to it and those now assigned elsewhere (classic years: via",
         "this year's =rms_versions= (upc, upc_ver_uc) -> master; the panel",
         "takes the version from Homescan's own product records instead).",
         ""]
    for row in rows:
        if row[6] != "-":
            FLAGS.setdefault("notes", []).append((
                "section 1",
                f"{row[0]} {row[1]} {row[2]}: {row[3]} panel-only cells; "
                f"last year's RMS UPCs of the brand now map to {row[6]}"))
    L += table(["year", "j", "brand_descr", "panel-only cells", "quarters",
                "last year's UPCs, still j $M", "... now elsewhere (top 2)"],
               rows) if rows else ["None."]
    mr = r.merge(p[KEY], on=KEY, how="left", indicator=True)
    ro = mr[mr["_merge"] == "left_only"]
    rows = []
    for y, g in ro.groupby("year"):
        top3 = g.groupby("brand_descr").size().sort_values(ascending=False)
        rows.append([int(y), f"{len(g):,}", "; ".join(
            f"{k} {v}" for k, v in top3.head(3).items())])
    L += ["", "** RMS-only cells: the three brands with the most, by year", "",
          "RMS-only cells are mostly small brands in DMA-quarters where no",
          "sample household bought them, plus the brand-years in section 4.",
          ""]
    L += table(["year", "RMS-only cells", "top brands (cells)"], rows)
    nim = routes[routes["route"] == "not_in_map"]
    if len(nim):
        tot = routes.groupby("year")["dollars"].sum()
        rows = [[int(y), len(d), f(d["dollars"].sum() / 1e6, 2),
                 f(d["dollars"].sum() / tot[y], 4),
                 ", ".join(str(u) for u in d.sort_values(
                     "dollars", ascending=False)["upc"].head(5))]
                for y, d in nim.groupby("year")]
        for row in rows:
            FLAGS.setdefault("notes", []).append((
                "section 1", f"{row[0]}: {row[1]} movement UPCs ({row[2]} $M) "
                "have no row in that year's rms_versions and are dropped"))
        L += ["", "Movement UPCs with no row in that year's =rms_versions= (dropped",
              "in D3 as unmapped):", ""]
        L += table(["year", "UPCs", "$M", "share of year $", "UPCs (top 5)"],
                   rows)
    return L


def sec_prices(c):
    rows = []
    for y, g in list(c.groupby("year")) + [(f"{T_LO}-{T_HI}", c)]:
        rows.append([y, f"{len(g):,}",
                     f(corr(g["lp_p"], g["lp_r"])),
                     f(corr(g["lp_tm_p"], g["lp_tm_r"])),
                     f(corr(g["lp_2w_p"], g["lp_2w_r"])),
                     f(g["dlp"].mean()), f(g["dlp"].median()),
                     f(g["dlp"].std())])
    L = ["* 2. Log prices on common cells",
         "",
         "=prices= = revenue / servings in both files (AM:648-656), where",
         "=servings= is ounces (quantity x size1_amount, AM:632): Homescan",
         "=total_price_paid= per ounce bought by sample households vs RMS",
         "units x price / prmult per ounce sold.  Columns: Pearson correlation of log",
         "prices (the charter's measure); the same within (t, m) (log price",
         "minus its common-cell (t, m) mean: relative prices across brands);",
         "and net of product and (t, m) fixed effects (alternating projections",
         "over the pooled common cells; the variation the K-aggregator's",
         "two-way FWL step keeps).  d = log p_RMS - log p_panel.",
         ""]
    L += table(["year", "common cells", "corr log p", "corr within (t,m)",
                "corr two-way", "mean d", "median d", "sd d"], rows)
    return L


def sec_shares(c):
    rows = []
    for y, g in list(c.groupby("year")) + [(f"{T_LO}-{T_HI}", c)]:
        rows.append([y, f"{len(g):,}",
                     f(corr(g["ls_tm_p"], g["ls_tm_r"])),
                     f(corr(g["ls_2w_p"], g["ls_2w_r"])),
                     f(corr(g["W_exp_p"], g["W_exp_r"]))])
    L = ["* 3. Within-(t, m) log shares on common cells",
         "",
         "log(shares) demeaned within each (t, m) over the COMMON cells, so",
         "both sources are compared on the same brand set in each market.",
         "Since shares = servings / market_size and market_size is constant",
         "within (t, m) (and identical in the two files; Amendment 1 item 2),",
         "the demeaning removes log(market_size): this is exactly the",
         "correlation of within-(t, m) log servings and does NOT depend on the",
         "B1 denominator.  Also shown: the same net of product and (t, m)",
         "fixed effects, and the correlation of =W_exp= (revenue / (t, m)",
         "revenue over each source's own cells; AM:695-697), the expenditure",
         "share the K-aggregator estimators read.",
         ""]
    L += table(["year", "common cells", "corr within (t,m) log share",
                "corr two-way log share", "corr W_exp"], rows)
    return L


def sec_brands(p, r, c):
    names = (pd.concat([p[["j", "brand_descr"]], r[["j", "brand_descr"]]])
             .drop_duplicates("j").set_index("j")["brand_descr"])
    pr = p[p["year"] >= T_LO]
    bp = pr.groupby(["year", "j"]).agg(cells=("t", "size"),
                                      rev=("revenue", "sum"))
    br = r.groupby(["year", "j"]).agg(cells=("t", "size"),
                                     rev=("revenue", "sum"))
    only_r = br.index.difference(bp.index)
    only_p = bp.index.difference(br.index)
    L = ["* 4. Brand-years present in one source but not the other",
         "",
         f"Over {T_LO}-{T_HI} (the years both cover).  A brand-year is present",
         "if the brand has at least one cell that year.  The panel's own",
         "2004-2005 brand-years have no RMS counterpart by construction.",
         ""]
    rows = [[y, j, names.get(j, "?"), f"{br.loc[(y, j), 'cells']:,}",
             f"{br.loc[(y, j), 'rev']:,.0f}"] for y, j in only_r]
    L += [f"** In RMS, not in the panel: {len(only_r)}", ""]
    L += table(["year", "j", "brand_descr", "RMS cells", "RMS revenue $"],
               rows) if rows else ["None."]
    rows = [[y, j, names.get(j, "?"), f"{bp.loc[(y, j), 'cells']:,}",
             f"{bp.loc[(y, j), 'rev']:,.0f}"] for y, j in only_p]
    L += ["", f"** In the panel, not in RMS: {len(only_p)}", ""]
    L += table(["year", "j", "brand_descr", "panel cells",
                "panel revenue $ (sample)"], rows) if rows else ["None."]

    # brands absent from each source, per year
    j45 = sorted(set(p["j"]))
    rows = []
    for y in range(T_LO, T_HI + 1):
        ap = sorted(set(j45) - set(pr.loc[pr["year"] == y, "j"]))
        ar = sorted(set(j45) - set(r.loc[r["year"] == y, "j"]))
        if ap or ar:
            rows.append([y, "; ".join(f"{j} {names.get(j, '?')}" for j in ap) or "-",
                         "; ".join(f"{j} {names.get(j, '?')}" for j in ar) or "-"])
    L += ["", "** Which of the 45 brands are absent, by year", "",
          "Years not listed have all 45 brands in both sources.", ""]
    L += table(["year", "absent from the panel", "absent from RMS"], rows)

    # present but vestigial: < 5% of the brand's median annual revenue
    def vest(b):
        med = b["rev"].groupby(level="j").transform("median")
        return b[b["rev"] < 0.05 * med]
    vp, vr = vest(bp), vest(br)
    keys = sorted(set(vp.index) | set(vr.index))
    rows = []
    for y, j in keys:
        rows.append([y, j, names.get(j, "?"),
                     f"{int(bp['cells'].get((y, j), 0)):,}",
                     f(bp['rev'].get((y, j), 0.0) / bp['rev'].xs(j, level='j').median(), 3)
                     if (y, j) in bp.index else "absent",
                     f"{int(br['cells'].get((y, j), 0)):,}",
                     f(br['rev'].get((y, j), 0.0) / br['rev'].xs(j, level='j').median(), 3)
                     if (y, j) in br.index else "absent"])
    for row in rows:
        if row[4] != "absent" and row[6] != "absent" and (
                float(row[4]) < 0.05) != (float(row[6]) < 0.05):
            side = "panel" if float(row[4]) < 0.05 else "RMS"
            FLAGS.setdefault("notes", []).append((
                "section 4", f"{row[0]} {row[1]} {row[2]}: vestigial in the "
                f"{side} only (panel {row[3]} cells, ratio {row[4]}; RMS "
                f"{row[5]} cells, ratio {row[6]})"))
    L += ["", "** Present but vestigial", "",
          "Brand-years present in a source with revenue below 5% of that",
          "brand's median annual revenue in that source (a few stray cells,",
          "e.g. a brand the name route has dropped).  Ratio = the year's",
          "revenue / the brand's median annual revenue, per source.", ""]
    L += table(["year", "j", "brand_descr", "panel cells", "panel ratio",
                "RMS cells", "RMS ratio"], rows) if rows else ["None."]

    # per-brand comparison on common cells + revenue shares
    rs = r.groupby("j")["revenue"].sum()
    ps = pr.groupby("j")["revenue"].sum()
    rs, ps = rs / rs.sum(), ps / ps.sum()
    rows = []
    for j, g in c.groupby("j"):
        rows.append([j, names.get(j, "?"), f"{len(g):,}",
                     f(corr(g["lp_p"], g["lp_r"])),
                     f(g["dlp"].mean()),
                     f(corr(g["ls_tm_p"], g["ls_tm_r"])),
                     f(100 * ps.get(j, 0.0), 2), f(100 * rs.get(j, 0.0), 2)])
    L += ["", "** Per brand (common cells)", "",
          "corr log p and mean d (= log p_RMS - log p_panel) within the brand;",
          "corr of within-(t, m) log shares within the brand; and the brand's",
          f"share of each source's 45-brand revenue over {T_LO}-{T_HI} (panel",
          "revenue is the unprojected sample's).  A large |mean d| or a revenue",
          "share far from the panel's flags a brand or size mapping to check.",
          ""]
    L += table(["j", "brand_descr", "common cells", "corr log p", "mean d",
                "corr within (t,m) log share", "panel rev %", "RMS rev %"],
               rows)
    return L


def _dist_rows(s, by):
    rows = []
    for y, g in s.groupby(by):
        v = g.to_numpy()
        rows.append([y, f"{len(v):,}", f(v.min()), f(q(v, .1)), f(q(v, .5)),
                     f(q(v, .9)), f(v.max())])
    return rows


def sec_share_sums(p, r):
    sp = p[p["year"] >= T_LO].groupby(["year", "t", "m"])["shares"].sum()
    sr = r.groupby(["year", "t", "m"])["shares"].sum()
    hdr = ["year", "(t, m)", "min", "p10", "median", "p90", "max"]
    L = ["* 5. (a) 45-brand inside-share sums per (t, m)",
         "",
         "sum_j shares over each source's own cells in a (t, m).  Both files",
         "divide by the same panel market_size (unprojected Homescan sample",
         "persons x 90 x 4.5 x 1.05; AM:423-460), so the RMS sums are store",
         "totals over a sample-person denominator: NOT meaningful (B1).",
         "Shown so the size of the problem is on record.", "",
         "** Panel", ""]
    L += table(hdr, _dist_rows(sp, "year"))
    L += ["", "** RMS", ""]
    rows = _dist_rows(sr, "year")
    n_neg = sr.groupby("year").apply(lambda v: int((v >= 1).sum()))
    for row in rows:
        row.append(f"{n_neg.loc[row[0]]:,}")
    L += table(hdr + ["(t, m) with S_0 <= 0"], rows)
    L += ["", f"Pooled {T_LO}-{T_HI}: panel median {f(sp.median(), 4)} "
          f"(max {f(sp.max(), 4)}); RMS median {f(sr.median(), 2)} "
          f"(min {f(sr.min(), 2)}, max {f(sr.max(), 2)}); "
          f"{int((sr >= 1).sum()):,} of {len(sr):,} RMS (t, m) have S_0 <= 0."]
    return L, sp, sr


def sec_factor(p, r, c, sp, sr):
    vp = p[p["year"] >= T_LO].groupby(["year", "t", "m"])["servings"].sum()
    vr = r.groupby(["year", "t", "m"])["servings"].sum()
    fac = (vr / vp).dropna()
    # identity check: the factor equals the ratio of the share sums
    chk = float(np.nanmax(np.abs(fac / (sr / sp).reindex(fac.index) - 1)))
    cp = c.groupby(["year", "t", "m"])[["servings_p", "servings_r"]].sum()
    fc = cp["servings_r"] / cp["servings_p"]
    rows = []
    for y, g in fac.groupby("year"):
        v = g.to_numpy()
        rows.append([y, f"{len(v):,}", f(v.min(), 1), f(q(v, .1), 1),
                     f(q(v, .5), 1), f(q(v, .9), 1), f(v.max(), 1),
                     f(float(np.std(np.log(v))), 2),
                     f(float(fc.loc[y].median()), 1)])
    v = fac.to_numpy()
    rows.append([f"{T_LO}-{T_HI}", f"{len(v):,}", f(v.min(), 1),
                 f(q(v, .1), 1), f(q(v, .5), 1), f(q(v, .9), 1),
                 f(v.max(), 1), f(float(np.std(np.log(v))), 2),
                 f(float(fc.median()), 1)])
    L = ["* 6. (b) The option-(2) factor per (t, m)",
         "",
         "factor = RMS 45-brand servings / panel 45-brand servings in the",
         "same (t, m), each summed over that source's own cells.  Option (2)",
         "would set market_size_RMS = market_size_panel x factor, so the RMS",
         "inside-share sum, and S_0, would equal the panel's.  Because both",
         "files carry the same market_size, the factor is also the ratio of",
         f"the two share sums in section 5 (max relative gap {chk:.1e}).",
         "Last two columns: the within-year sd of log factor across (t, m),",
         "and the median factor computed on common cells only.",
         ""]
    L += table(["year", "(t, m)", "min", "p10", "median", "p90", "max",
                "sd log factor", "median, common cells"], rows)
    return L


def sec_crosswalk(p, r, routes):
    names = (pd.concat([p[["j", "brand_descr"]], r[["j", "brand_descr"]]])
             .drop_duplicates("j").set_index("j")["brand_descr"])
    j45 = set(p["j"].astype(int))
    rv = routes[routes["year"] >= REVISED_FIRST].copy()
    # A build on the shared brand route (src/brand_route.py; its upc_routes
    # carry j_route) has brand_code_uc already merged (Q2: 620485 -> 620554,
    # 747763 -> 557579).  Count j's merge set as j on the master side too
    # (charter-crosswalk A7), so a merge is not reported as LOST / CREATED.
    # An accepted (name-route) build has no j_route and is reported as before.
    merged = "j_route" in rv.columns
    FLAGS["shared_route"] = merged
    if merged:
        import sys
        sys.path.insert(0, str(HERE))
        from brand_route import merge_codes
        rv["master_code"] = merge_codes(rv["master_code"]).to_numpy()
    legacy = rv[rv["master_code"].notna()]
    wrong = legacy[legacy["brand_code_uc"].isna()
                   | (legacy["brand_code_uc"] != legacy["master_code"])]
    FLAGS["legacy_route_mismatches"] = [
        (int(x.year), int(x.upc)) for x in wrong.itertuples()]
    used = rv["brand_code_uc"]
    mast = rv["master_code"]
    rv["used45"] = used.isin(j45)
    rv["mast45"] = mast.isin(j45)
    name_route = rv["route"].isin(["hms_same_year", "hms_other_year"])
    L = ["* 7. (c) Brand-years created or lost by the panel's name-route crosswalk (2021+)",
         "",
         "From 2021 the panel (and so RMS, which mirrors it) assigns a UPC its",
         "brand through the Homescan brand NAME (brand_descr_hms) and the",
         "panel's brand crosswalk, deduplicated by name in file order",
         "(AM:225-264; drop_duplicates('brand_descr_2021') at AM:245-250).",
         "The counterfactual here is the classic product master's brand code",
         "for the same UPC (highest upc_ver_uc), which exists only for UPCs",
         "sold before 2021.  Dollars are all valid module-1344 dollars of the",
         "UPC in the year (every store; before the DMA, CTL, 45-brand, 65-DMA",
         "and OZ filters), from =rms/diagnostics/upc_routes.parquet=.",
         ""]
    if merged:
        L += ["This build uses the shared brand route instead (src/brand_route.py;",
              "rule: docs/crosswalk-rule.org): a UPC's legacy master code first,",
              "the 2021-24 name map only for UPCs with no master code, then the",
              "merge sets.  Both sides below count j's merge set as j (620485 in",
              "620554, 747763 in 557579), so a LOST / CREATED / shrunk / grown",
              "row is a route difference, not a merge (charter-crosswalk A7).",
              "The column /name route/ then counts UPCs that carry a Homescan",
              "name, whichever rule placed them.", ""]
    # brand-year totals under each rule
    tot = rv.groupby("year")["dollars"].sum()
    rows = []
    for y, g in rv.groupby("year"):
        has_m = g["master_code"].notna()
        agree = has_m & (g["brand_code_uc"] == g["master_code"])
        moved = has_m & g["brand_code_uc"].notna() & ~agree
        unx = has_m & g["brand_code_uc"].isna()
        rows.append([y, f(tot[y] / 1e6, 1),
                     f(g.loc[name_route[g.index], "dollars"].sum() / tot[y]),
                     f(g.loc[has_m, "dollars"].sum() / tot[y]),
                     f(g.loc[agree, "dollars"].sum() / tot[y]),
                     f(g.loc[moved, "dollars"].sum() / tot[y]),
                     f(g.loc[unx, "dollars"].sum() / tot[y]),
                     f(g.loc[unx & g["mast45"], "dollars"].sum() / tot[y]),
                     f(g.loc[~has_m, "dollars"].sum() / tot[y])])
    L += ["** Agreement of the two rules (share of the year's module-1344 dollars)", ""]
    L += table(["year", "all $M", "name route", "has a master code",
                "same brand", "different brand", "name not in crosswalk",
                "... of which master brand in the 45",
                "new UPCs (no master code)"], rows)

    # per brand-year, on the SAME UPCs (those with a classic-master code):
    # dollars to j under each rule; plus name-route dollars from new UPCs
    hm = rv[rv["master_code"].notna()]
    nw = rv[rv["master_code"].isna()]
    by = pd.concat([
        hm[hm["mast45"]].groupby(["year", "master_code"])["dollars"].sum()
        .rename_axis(["year", "j"]).rename("master"),
        hm[hm["used45"]].groupby(["year", "brand_code_uc"])["dollars"].sum()
        .rename_axis(["year", "j"]).rename("name_same"),
        nw[nw["used45"]].groupby(["year", "brand_code_uc"])["dollars"].sum()
        .rename_axis(["year", "j"]).rename("name_new"),
    ], axis=1).fillna(0.0).sort_index()
    moved = hm[hm["brand_code_uc"] != hm["master_code"]].assign(
        dest=lambda d: d["brand_code_uc"].map(
            lambda x: "not in crosswalk" if pd.isna(x) else str(int(x))))
    lost_to = (moved[moved["mast45"]]
               .groupby(["year", "master_code", "dest"])["dollars"].sum())
    gained_from = (moved[moved["used45"]]
                   .groupby(["year", "brand_code_uc", "master_code"])["dollars"]
                   .sum())
    cp = p.groupby(["year", "j"]).size()
    cr = r.groupby(["year", "j"]).agg(n=("t", "size"), rev=("revenue", "sum"))

    def top(s, key, fmt):
        if key not in s.index.droplevel(2):
            return "-"
        x = s.loc[key].sort_values(ascending=False)
        x = x[x >= 5e3].head(3)
        return "; ".join(f"{fmt(k)} ${v / 1e6:,.2f}M" for k, v in x.items()) or "-"

    rows = []
    for (y, j), v in by.iterrows():
        ms, ns, nn = v["master"], v["name_same"], v["name_new"]
        if ms >= 1e5 and ns + nn < 0.01 * ms:
            tag = "LOST"
        elif ns >= 1e5 and ms < 0.01 * ns:
            tag = "CREATED"
        elif max(ms, ns) > 0 and abs(ns - ms) / max(ms, ns) > 0.10:
            tag = "shrunk" if ns < ms else "grown"
        else:
            continue
        key = (y, j)
        rows.append([y, int(j), names.get(int(j), "?"), tag,
                     f(ms / 1e6, 2), f(ns / 1e6, 2), f(nn / 1e6, 2),
                     top(lost_to, key, str), top(gained_from, key,
                                                 lambda k: str(int(k))),
                     f"{int(cp.get((y, int(j)), 0)):,}",
                     f"{int(cr['n'].get((y, int(j)), 0)):,} / "
                     + money(cr['rev'].get((y, int(j)), 0.0))])
        if tag in ("LOST", "CREATED"):
            FLAGS.setdefault("xwalk", []).append(
                (y, int(j), names.get(int(j), "?"), tag,
                 top(lost_to, key, str) if tag == "LOST"
                 else top(gained_from, key, lambda k: str(int(k)))))
    L += ["", "** Brand-years the name route moves by more than 10% (same UPCs)", "",
          "Compared on the SAME UPCs (those with a classic-master code):",
          "/master $M/ = their dollars whose master brand is j; /name route",
          "$M/ = their dollars the name route assigns to j.  /new UPCs $M/ =",
          "dollars of UPCs with no master code (first sold in 2021+) that the",
          "name route assigns to j; the master rule cannot place them at all.",
          "LOST: the name route (same + new UPCs) keeps < 1% of the master",
          "dollars; CREATED: the master rule gives j < 1% of what the name",
          "route moves in; shrunk / grown: the same UPCs move by > 10%.",
          "/lost to/, /gained from/: the top other brand codes (or 'not in",
          "crosswalk'), flows >= $5k.  Last two columns: the brand-year's cells",
          "in the panel, and its cells / kept revenue in market_panel_rms",
          "(a few cells and dollars survive a LOST brand through the",
          "classic-master fallback for UPCs no Homescan household bought).",
          ""]
    L += table(["year", "j", "brand_descr", "change", "master $M",
                "name route $M", "new UPCs $M", "lost to", "gained from",
                "panel cells", "RMS cells / $"],
               rows) if rows else ["None."]
    FLAGS["crosswalk_changes"] = [(int(row[0]), row[1], row[3]) for row in rows]
    return L


def unallowed_crosswalk_changes(changes, allow):
    """D4 section 7 rows lacking an explicit, reasoned D4_7 exception."""
    from crosswalk_checks import allowed

    return [(year, j, tag) for year, j, tag in changes
            if allowed(allow, "D4_7", "rms", j, year) is None]


def sec_funnel(fun):
    rows = []
    for _, x in fun.sort_values("year").iterrows():
        a = x["dollars_all"]
        rows.append([int(x["year"]), f(a / 1e6, 1),
                     f(x["dollars_brand_in_45"] / a),
                     f(x["dollars_dma_any"] / a),
                     f(x["dollars_dma_in_65"] / a),
                     f(x["dollars_t_in_range"] / a)])
    L = ["* 8. (d) Share of module-1344 revenue mapped, per year",
         "",
         "Of all valid module-1344 dollars (units x price / prmult, prmult > 0,",
         "price > 0) in the year's RMS movement: the share whose UPC maps to",
         "one of the panel's 45 brands (and is not CTL private label); the",
         "share whose store maps to a DMA (classic: stores.dma_code; 2021+:",
         "the panel's FIPS->DMA map, AM:188-218); the share in the panel's 65",
         "DMAs; and the share kept in market_panel_rms (all filters, incl. OZ).",
         "From =rms/diagnostics/funnel_by_year.parquet=; the full filter",
         "funnel is in =docs/rms-d3-log.org=.",
         ""]
    L += table(["year", "all $M", "to the 45 brands", "to a DMA",
                "to the 65 DMAs", "kept"], rows)
    return L


def sec_d6(path):
    L = ["* 9. D6 (stretch): homog on RMS beside the panel homog law", ""]
    if not path or not Path(path).exists():
        return L + ["Not run (no =homog_compare.json=)."]
    rows = json.loads(Path(path).read_text())
    L += ["Two-step clustered GMM, s = (t, m) clusters, K = 2, LF instruments,",
          "two-way FE: the default =Spec(\"homog\")= with only =source= (and,",
          "in the last row if present, the window) changed.  theta is in",
          "original units, theta = exp(phi) / sigma_logp",
          "(=homog._rescale_theta=); SE by the delta method on the clustered",
          "Sigma_theta (=law.functional_se=).  J_cluster is the N-scaled",
          "cluster Hansen J at theta-hat; p is its cluster-wild bootstrap",
          "p-value (2,000 draws, so the floor is 1/2001 = 0.0005).  The panel",
          "law is read read-only from the shared mirror's =var/laws=; the RMS",
          "laws are in =Data/NielsenScanner/rms/laws/= (group-only), produced",
          "by CerealDemand =src/rms_homog_compare.py= (branch =rms=).",
          ""]
    L += ["Slugs: =homog= is the declared panel law (full window); =homog.rms=",
          "is =Spec(\"homog\", source=\"rms\")=; =homog.win2006-2024.baseQ1Y2006=,",
          "if present, is the PANEL on the RMS window and base quarter, to",
          "separate the source effect from the window effect.  Only the first",
          "is a declared law; the others are not in the 29 and live only in",
          "the RMS store.  sigma_logp is the sample sd used to standardize log",
          "price (theta_std = exp(phi)); eig = eigenvalues of Gamma in",
          "original units.",
          ""]
    L += table(["law", "source", "years", "base", "N", "(t, m) clusters",
                "theta", "SE", "theta_std", "sigma_logp", "J_cluster", "dof",
                "p (wild)", "eig 1", "eig 2", "producer"],
               [[x["slug"], x["source"],
                 f"{x['years'][0]}-{x['years'][1]}" if x.get("years") else "?",
                 x["base_period"], f"{x['N']:,}", f"{x['n_clusters']:,}",
                 f(x["theta_orig"], 4), f(x["theta_orig_se"], 4),
                 f(x["theta_std"], 4), f(x["sigma_logp"], 4),
                 f(x["J_cluster"], 1), x["J_cluster_dof"],
                 f(x["J_cluster_pvalue"], 4),
                 f(x["eigenvalues_orig"][0], 3),
                 f"{x['eigenvalues_orig'][1]:.2e}" if len(x["eigenvalues_orig"]) > 1 else "",
                 x.get("producer_commit")] for x in rows])
    return L


def sec_summary(p, r, c, sp, sr, fun, d6):
    pr = p[p["year"] >= T_LO]
    bp = set(map(tuple, pr[["year", "j"]].drop_duplicates().to_numpy()))
    br = set(map(tuple, r[["year", "j"]].drop_duplicates().to_numpy()))
    vp = pr.groupby(["year", "t", "m"])["servings"].sum()
    vr = r.groupby(["year", "t", "m"])["servings"].sum()
    fac = (vr / vp).dropna()
    a = fun["dollars_all"]
    s45 = fun["dollars_brand_in_45"] / a
    sdma = fun["dollars_dma_any"] / a
    L = ["* 0. Summary (numbers only; sections 1-9 have the detail)", "",
         f"- Common cells, {T_LO}-{T_HI}: {len(c):,} = "
         f"{len(c) / len(pr):.1%} of the panel's and {len(c) / len(r):.1%} of "
         "RMS's cells (section 1).",
         f"- Log prices, common cells, pooled: corr {corr(c['lp_p'], c['lp_r']):.3f}; "
         f"within (t, m) {corr(c['lp_tm_p'], c['lp_tm_r']):.3f}; net of product "
         f"and (t, m) FE {corr(c['lp_2w_p'], c['lp_2w_r']):.3f}.  RMS prices "
         f"are higher by {c['dlp'].mean():.3f} log points on average "
         f"(per-year means {c.groupby('year')['dlp'].mean().min():.3f} to "
         f"{c.groupby('year')['dlp'].mean().max():.3f}) (section 2).",
         f"- Within-(t, m) log shares, common cells, pooled: corr "
         f"{corr(c['ls_tm_p'], c['ls_tm_r']):.3f} (per year "
         f"{c.groupby('year').apply(lambda g: corr(g['ls_tm_p'], g['ls_tm_r'])).min():.3f}"
         f" to {c.groupby('year').apply(lambda g: corr(g['ls_tm_p'], g['ls_tm_r'])).max():.3f});"
         f" net of product and (t, m) FE {corr(c['ls_2w_p'], c['ls_2w_r']):.3f}; "
         f"W_exp {corr(c['W_exp_p'], c['W_exp_r']):.3f} (section 3).",
         f"- Brand-years in RMS only: {len(br - bp)}; in the panel only: "
         f"{len(bp - br)} (section 4; the RMS-only ones are a few stray cells"
         " of brands the name route has dropped).",
         f"- (a) 45-brand inside-share sums: panel median {sp.median():.4f}; RMS "
         f"median {sr.median():.1f} (min {sr.min():.2f}); S_0 <= 0 in "
         f"{int((sr >= 1).sum()):,} of {len(sr):,} RMS (t, m) (section 5; B1).",
         f"- (b) Option-(2) factor RMS/panel 45-brand servings per (t, m): median "
         f"{fac.median():,.0f}, p10-p90 {fac.quantile(.1):,.0f}-"
         f"{fac.quantile(.9):,.0f}, sd of log factor {np.log(fac).std():.2f} "
         "(section 6).",
         f"- (d) Module-1344 revenue mapped to the 45 brands: {s45.min():.3f}-"
         f"{s45.max():.3f} per year; to a DMA: {sdma.min():.3f}-{sdma.max():.3f}"
         " (section 8)."]
    xw = FLAGS.get("xwalk", [])
    if xw:
        L += ["- (c) Brand-years the 2021+ name route LOSES or CREATES vs the "
              "classic master (section 7):"]
        for y, j, n, tag, other in xw:
            L += [f"  - {y} {j} {n}: {tag} ({'to' if tag == 'LOST' else 'from'} {other})"]
    for k, txt in FLAGS.get("notes", []):
        L += [f"- {txt} ({k})"]
    if d6 and Path(d6).exists():
        rows = json.loads(Path(d6).read_text())
        L += ["- D6 homog (section 9): " + "; ".join(
            f"{x['slug']} theta {x['theta_orig']:.4f} (SE {x['theta_orig_se']:.4f}),"
            f" J_cluster {x['J_cluster']:.1f} on {x['J_cluster_dof']} dof"
            for x in rows) + "."]
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(HERE.parent / "docs" / "rms-validation.org"))
    ap.add_argument("--d6", default=str(RMS / "laws" / "homog_compare.json"))
    ap.add_argument("--gate-crosswalk", action="store_true",
                    help="fail on missing shared route, mismatched legacy UPCs, or unallow-listed section 7 rows")
    a = ap.parse_args()
    FLAGS.clear()
    t0 = time.time()
    p, r, caveat, fun, routes = load()
    c = common_cells(p, r)

    L = ["#+title: RMS vs Homescan market panel --- validation report (D4)",
         f"#+date: {time.strftime('%Y-%m-%d %H:%M %Z')}",
         "",
         "Generated by =python src/rms_validate.py= (=make rms-validate=); do",
         "not edit by hand.  charter-rms section 2 D4 and Amendment 1 item 3.",
         "Numbers only: whether the discrepancies are acceptable is EL's call.",
         "",
         f"- RMS: =market_panel_rms.parquet= ({len(r):,} cells, "
         f"{r['t'].nunique()} quarters x {r['m'].nunique()} DMAs x "
         f"{r['j'].nunique()} brands), built by =src/rms_market_panel.py=.",
         f"- Panel: =data/clean/market_panel.parquet= ({len(p):,} cells, "
         f"{p['t'].nunique()} quarters, {int(p['year'].min())}-{int(p['year'].max())}).",
         f"- Common (t, m, j) cells, {T_LO}-{T_HI}: {len(c):,}.",
         "- Coverage: RMS describes the RMS-reporting stores only (~43% of",
         "  cereal dollars; one non-reporting chain is 21% alone); the panel",
         "  is the full-coverage check (charter section 0).",
         "- Quarters: RMS assigns t by the week's end date (Saturday), the panel",
         "  by purchase date (AM:362-371).  A week that straddles a quarter end",
         "  falls in the later quarter, so some RMS quarters hold 12 or 14 weeks",
         "  (e.g. Q1Y2006 12, Q3Y2006 14).  RMS servings, revenue and",
         "  n_transactions levels move by up to about 8% for that reason alone;",
         "  prices and W_exp are ratios and are not affected.",
         "- AM:<line> cites =src/aggregate_markets.py= (the panel builder).",
         "",
         "#+begin_quote",
         caveat or "(rms_caveat missing from the parquet metadata)",
         "#+end_quote",
         ""]
    L += sec_cells(p, r, c) + [""]
    L += sec_panel_only(p, r, routes) + [""]
    L += sec_prices(c) + [""]
    L += sec_shares(c) + [""]
    L += sec_brands(p, r, c) + [""]
    s5, sp, sr = sec_share_sums(p, r)
    L += s5 + [""]
    L += sec_factor(p, r, c, sp, sr) + [""]
    L += sec_crosswalk(p, r, routes) + [""]
    L += sec_funnel(fun) + [""]
    L += sec_d6(a.d6) + [""]
    head = L.index("* 1. Cells per year")
    L[head:head] = sec_summary(p, r, c, sp, sr, fun, a.d6) + [""]
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out} ({len(L)} lines, {time.time() - t0:.0f} s)")
    if a.gate_crosswalk:
        from crosswalk_checks import load_allowlist

        bad = unallowed_crosswalk_changes(FLAGS["crosswalk_changes"], load_allowlist())
        wrong = FLAGS["legacy_route_mismatches"]
        if not FLAGS["shared_route"] or bad or wrong:
            raise SystemExit(
                "D4 crosswalk gate failed: "
                f"shared_route={FLAGS['shared_route']}, "
                f"unallow-listed section 7 rows={bad[:10]} ({len(bad)} total), "
                f"legacy UPC mismatches={wrong[:10]} ({len(wrong)} total)")
        print("D4 crosswalk gate passed: no unallow-listed section 7 rows "
              "or legacy UPC mismatches")


if __name__ == "__main__":
    main()
