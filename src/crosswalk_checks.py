"""
charter-crosswalk A4 / A5 / A6: the route's acceptance checks, per source,
per j and year.  Thresholds are EL's (Q8): retention >= 0.95, leakage <=
0.05, continuity ratios in [0.5, 2].  Exceptions only through
config/crosswalk_allowlist.csv (Q6), each with a reason.

Inputs (two adapters, both -> a brand x year dollar frame):
  --eval DIR   the UPC-dollar evaluation of src/crosswalk_eval.py
               (DIR/{hms,rms}_upc.parquet); --route old|new picks today's
               route or the new one.  A4, A5 and A6 run here.
  --panel PATH a market panel parquet (market_panel.parquet or
               market_panel_rms.parquet, today's or a new vintage);
               --source hms|rms.  A5 runs here (A4/A6 need UPC dollars).

A4 (mechanism), 2021-2024, every j of the 45:
  retention = of the dollars on UPCs whose legacy code (A1 version rule,
              merged) is j, the share the route assigns to j  (>= 0.95);
  leakage   = of the legacy-coded dollars the route assigns to j, the share
              whose legacy code is not j                       (<= 0.05).
  Non-CTL dollars only.  A j with no legacy-coded dollars in a year has
  nothing to retain (reported as NaN, not a failure).
A5 (outcome): s_jy = j's share of the year's 45-brand revenue (dollars);
  (i)   s_jy / s_j,y-1 in [0.5, 2], every consecutive pair of years;
  (ii)  s_jy / median(s_j,2017..2020) in [0.5, 2], y = 2021..2024;
  (iii) j present every year: >= 1 cell (panel) or dollars > 0 (UPC level).
  Years: Homescan 2004-2024, RMS 2006-2024.
A6 (coverage), 2021-2024: share of non-CTL module-1344 dollars with some
  brand code >= today's (the old route in the same frame, and the charter's
  measured values); the 45-brand share is reported beside it.

Exit status 1 if any failure is not allow-listed.
Usage:
  python src/crosswalk_checks.py --eval DIR --route new [--source hms rms]
  python src/crosswalk_checks.py --panel PATH --source rms
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from brand_route import load_brands_45, CONFIG  # noqa: E402

ALLOWLIST = CONFIG / "crosswalk_allowlist.csv"
RET_MIN, LEAK_MAX, LO, HI = 0.95, 0.05, 0.5, 2.0
FIRST_YEAR = {"hms": 2004, "rms": 2006}
LAST_YEAR = 2024
NEW_YEARS = [2021, 2022, 2023, 2024]
BASE_YEARS = [2017, 2018, 2019, 2020]
# charter A6: today's non-CTL coverage (measured, ledger section 0)
A6_TODAY = {"hms": {2021: 0.958, 2022: 0.921, 2023: 0.886, 2024: 0.875},
            "rms": {2021: 0.959, 2022: 0.914, 2023: 0.878, 2024: 0.872}}


def j45():
    return load_brands_45()["j"].to_numpy()


def names45():
    b = load_brands_45()
    return dict(zip(b["j"], b["brand_descr"]))


def _ctl(s):
    return s.fillna("").astype(str).str.upper().str.startswith("CTL")


# --- adapters ---------------------------------------------------------------

def load_eval(eval_dir, source):
    return pd.read_parquet(Path(eval_dir) / f"{source}_upc.parquet")


def route_cols(route):
    assert route in ("old", "new"), route
    return f"{route}_j", f"name_{route}"


def brand_year_from_eval(frame, route):
    """j x year dollars (45 only, non-CTL) under `route`."""
    jc, nc = route_cols(route)
    f = frame[~_ctl(frame[nc]) & frame[jc].isin(j45())]
    by = f.groupby([jc, "year"])["dollars"].sum().unstack("year")
    by.index = by.index.astype("int64")
    by.index.name = "j"
    return by


def brand_year_from_panel(path):
    """j x year revenue (sum over the year's (t, m) cells), and the number
    of cells per (j, year)."""
    mp = pd.read_parquet(path, columns=["revenue"]).reset_index()
    mp["year"] = mp["t"].str[-4:].astype(int)
    g = mp.groupby(["j", "year"])["revenue"]
    return g.sum().unstack("year"), g.size().unstack("year")


# --- allow-list ---------------------------------------------------------------

def load_allowlist(path=ALLOWLIST):
    a = pd.read_csv(path)
    a["j"] = a["j"].astype("int64")
    return a


def allowed(allow, criterion, source, j, year):
    a = allow[(allow["criterion"] == criterion)
              & allow["source"].isin([source, "any"])
              & (allow["j"] == int(j))
              & (allow["year_lo"] <= year) & (allow["year_hi"] >= year)]
    return a["reason"].iloc[0] if len(a) else None


# --- A4 ---------------------------------------------------------------------

def a4(frame, route, years=NEW_YEARS):
    jc, nc = route_cols(route)
    f = frame[frame["year"].isin(years) & ~_ctl(frame[nc])
              & frame["legacy_j"].notna()]
    rows = []
    for (y), d in f.groupby("year"):
        for j in j45():
            leg = d["legacy_j"] == j
            to = d[jc] == j
            dl = d.loc[leg, "dollars"].sum()
            dt = d.loc[to, "dollars"].sum()
            ret = d.loc[leg & to, "dollars"].sum() / dl if dl > 0 else np.nan
            leak = d.loc[to & ~leg, "dollars"].sum() / dt if dt > 0 else np.nan
            rows.append({"j": int(j), "year": int(y), "legacy_dollars": dl,
                         "assigned_legacy_dollars": dt,
                         "retention": ret, "leakage": leak})
    r = pd.DataFrame(rows)
    r["ret_fail"] = r["retention"] < RET_MIN
    r["leak_fail"] = r["leakage"] > LEAK_MAX
    return r


# --- A5 ---------------------------------------------------------------------

def a5(by, present, source):
    """by: j x year dollars; present: j x year bool (>= 1 cell / $ > 0)."""
    years = list(range(FIRST_YEAR[source], LAST_YEAR + 1))
    by = by.reindex(index=j45(), columns=years)
    present = present.reindex(index=j45(), columns=years).fillna(False)
    s = by.fillna(0.0) / by.sum(axis=0)
    med = s[BASE_YEARS].median(axis=1)
    rows = []
    for j in s.index:
        for y in years:
            if not present.loc[j, y]:
                rows.append({"j": int(j), "year": y, "test": "iii",
                             "value": 0.0})
            if y > years[0]:
                a, b = s.loc[j, y], s.loc[j, y - 1]
                if a > 0 or b > 0:
                    v = a / b if b > 0 else np.inf
                    if not (LO <= v <= HI):
                        rows.append({"j": int(j), "year": y, "test": "i",
                                     "value": v})
            if y in NEW_YEARS:
                v = s.loc[j, y] / med[j] if med[j] > 0 else np.nan
                if not (LO <= v <= HI):
                    rows.append({"j": int(j), "year": y, "test": "ii",
                                 "value": v})
    return pd.DataFrame(rows, columns=["j", "year", "test", "value"]), s


def a5_eval(frame, route, source):
    by = brand_year_from_eval(frame, route)
    return a5(by, by.fillna(0) > 0, source)


def a5_panel(path, source):
    by, cells = brand_year_from_panel(path)
    return a5(by, cells.fillna(0) > 0, source)


# --- A6 ---------------------------------------------------------------------

def a6(frame, years=NEW_YEARS):
    rows = []
    f = frame[frame["year"].isin(years)]
    for y, d in f.groupby("year"):
        r = {"year": int(y)}
        for route in ("old", "new"):
            jc, nc = route_cols(route)
            nn = d[~_ctl(d[nc])]
            tot = nn["dollars"].sum()
            r[f"cov_{route}"] = nn.loc[nn[jc].notna(), "dollars"].sum() / tot
            r[f"s45_{route}"] = nn.loc[nn[jc].isin(j45()), "dollars"].sum() / tot
            r[f"nonctl_dollars_{route}"] = tot
        rows.append(r)
    return pd.DataFrame(rows)


# --- failures (allow-list applied) --------------------------------------------

def failures(source, route=None, frame=None, panel=None, allow=None):
    """Long table of every check failure: criterion, j, year, detail,
    allowed (reason or None)."""
    allow = load_allowlist() if allow is None else allow
    out = []
    if frame is not None:
        r4 = a4(frame, route)
        for _, r in r4[r4["ret_fail"] | r4["leak_fail"]].iterrows():
            what = []
            if r["ret_fail"]:
                what.append(f"retention {r['retention']:.3f}")
            if r["leak_fail"]:
                what.append(f"leakage {r['leakage']:.3f}")
            out.append({"criterion": "A4", "j": r["j"], "year": r["year"],
                        "detail": ", ".join(what)})
        f5, _ = a5_eval(frame, route, source)
    else:
        f5, _ = a5_panel(panel, source)
    for _, r in f5.iterrows():
        out.append({"criterion": "A5", "j": r["j"], "year": r["year"],
                    "detail": f"({r['test']}) {r['value']:.2f}"})
    if frame is not None:
        r6 = a6(frame)
        for _, r in r6.iterrows():
            today = A6_TODAY[source][r["year"]]
            if r[f"cov_{route}"] < min(r["cov_old"], today) - 5e-4:
                out.append({"criterion": "A6", "j": -1, "year": r["year"],
                            "detail": f"coverage {r[f'cov_{route}']:.3f} < "
                                      f"today {min(r['cov_old'], today):.3f}"})
    df = pd.DataFrame(out, columns=["criterion", "j", "year", "detail"])
    df["allowed"] = [allowed(allow, c, source, j, y) if j >= 0 else None
                     for c, j, y in zip(df["criterion"], df["j"], df["year"])]
    nm = names45()
    df["brand"] = df["j"].map(nm).fillna("")
    return df


def report(df, title, evaluated=("A4", "A5", "A6")):
    bad = df[df["allowed"].isna()]
    print(f"== {title}: {len(df)} flags, {len(bad)} not allow-listed")
    for c in ["A4", "A5", "A6"]:
        if c not in evaluated:
            print(f"  {c} not evaluated")
            continue
        d = bad[bad["criterion"] == c]
        if len(d):
            print(f"  {c} FAIL:")
            for (j, b), g in d.groupby(["j", "brand"]):
                items = "; ".join(f"{y} {t}" for y, t in zip(g["year"], g["detail"]))
                print(f"    {j} {b}: {items}")
        else:
            print(f"  {c} pass")
    al = df[df["allowed"].notna()]
    if len(al):
        print(f"  allow-listed: " + "; ".join(
            f"{b} {y} {t}" for b, y, t in zip(al["brand"], al["year"],
                                               al["detail"])))
    return len(bad)


def main():
    ap = argparse.ArgumentParser(description="charter-crosswalk A4/A5/A6")
    ap.add_argument("--eval", default=None, help="UPC-dollar evaluation dir")
    ap.add_argument("--route", default="new", choices=["old", "new"])
    ap.add_argument("--panel", default=None, help="market panel parquet")
    ap.add_argument("--source", nargs="+", default=["hms", "rms"])
    a = ap.parse_args()
    nbad = 0
    if a.panel:
        assert len(a.source) == 1, "--panel takes one --source"
        df = failures(a.source[0], panel=a.panel)
        nbad += report(df, f"{a.source[0]} panel {a.panel} (A5)", evaluated=("A5",))
    else:
        assert a.eval, "--eval DIR or --panel PATH"
        for s in a.source:
            fr = load_eval(a.eval, s)
            df = failures(s, route=a.route, frame=fr)
            nbad += report(df, f"{s} UPC-dollar evaluation, {a.route} route")
            r6 = a6(fr)
            print("  A6 coverage (old -> new; 45-brand share old -> new):")
            for _, r in r6.iterrows():
                print(f"    {r['year']}: {r['cov_old']:.3f} -> {r['cov_new']:.3f}"
                      f"  (charter today {A6_TODAY[s][r['year']]:.3f});"
                      f"  45-share {r['s45_old']:.3f} -> {r['s45_new']:.3f}")
    print("RESULT:", "FAIL" if nbad else "PASS")
    sys.exit(1 if nbad else 0)


if __name__ == "__main__":
    main()
