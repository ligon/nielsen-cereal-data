"""
charter-crosswalk A8: the before/after report in the evaluation directory,
from the UPC-dollar evaluation (src/crosswalk_eval.py) for both sources.

One command:  make crosswalk-report   (re-runs the evaluation, then this)
or            python src/crosswalk_report.py --eval DIR --out PATH

Contents (charter A8): A4/A5/A6 old vs new; per-year 45-brand totals;
brand-year dollars old vs new for all 45 x years; changed pre-2021
brand-years with the reason; the D4-section-7-style LOST / CREATED / shrunk
/ grown rows (2021+, same UPCs) under each route; the top 30 2021+ names
still unassigned; each j's largest name-only (new-UPC) placements; the
route mix.  Aggregates only; no retailer or chain is named.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crosswalk_checks as cc  # noqa: E402
from brand_route import load_merge_sets  # noqa: E402

SRC_LABEL = {"hms": "Homescan (unprojected sample $, purchases_<yr>)",
             "rms": "RMS (all module-1344 $, every store, upc_routes)"}
UNIT = {"hms": (1e3, "$k"), "rms": (1e6, "$M")}
YEARS = {"hms": list(range(2004, 2025)), "rms": list(range(2006, 2025))}
NEW = [2021, 2022, 2023, 2024]


def table(header, rows):
    out = ["| " + " | ".join(map(str, header)) + " |",
           "|" + "|".join("-" for _ in header) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return out


def fnum(x, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-" if x is None or np.isnan(x) else "inf"
    return f"{x:,.{d}f}"


def nonctl(f, route):
    jc, nc = cc.route_cols(route)
    return f[~cc._ctl(f[nc])]


def section_checks(fr, s):
    L = [f"** A4 / A5 / A6, old vs new: {s}", ""]
    rows = []
    res = {}
    for route in ("old", "new"):
        df = cc.failures(s, route=route, frame=fr)
        res[route] = df
        bad = df[df["allowed"].isna()]
        for c in ("A4", "A5", "A6"):
            d = bad[bad["criterion"] == c]
            rows.append([route, c, len(d), d["j"].nunique() if c != "A6" else "",
                         "; ".join(sorted(set(d["brand"]))) or "-"])
    L += table(["route", "criterion", "failing (j, year)", "brands",
                "which"], rows)
    L += [""]
    for route in ("old", "new"):
        bad = res[route][res[route]["allowed"].isna()]
        if len(bad):
            L += [f"Failures, {route} route (not allow-listed):", ""]
            r = []
            for (c, j, b), g in bad.groupby(["criterion", "j", "brand"]):
                r.append([c, j, b, "; ".join(f"{y}: {t}" for y, t in
                                              zip(g["year"], g["detail"]))])
            L += table(["criterion", "j", "brand", "year: value"], r) + [""]
    al = res["new"][res["new"]["allowed"].notna()]
    if len(al):
        L += ["Allow-listed under the new route (config/crosswalk_allowlist.csv, Q6):",
              ""]
        r = [[c, j, b, ", ".join(str(y) for y in g["year"].unique())]
             for (c, j, b), g in al.groupby(["criterion", "j", "brand"])]
        L += table(["criterion", "j", "brand", "years"], r) + [""]
    a4 = cc.a4(fr, "old").merge(cc.a4(fr, "new"), on=["j", "year"],
                                suffixes=("_old", "_new"))
    a4 = a4[(a4["ret_fail_old"] | a4["leak_fail_old"] | a4["ret_fail_new"]
             | a4["leak_fail_new"])]
    if len(a4):
        nm = cc.names45()
        L += ["A4 values where either route fails (retention / leakage):", ""]
        L += table(["j", "brand", "year", "retention old", "retention new",
                     "leakage old", "leakage new"],
                   [[r.j, nm[r.j], r.year, fnum(r.retention_old, 3),
                     fnum(r.retention_new, 3), fnum(r.leakage_old, 3),
                     fnum(r.leakage_new, 3)] for r in a4.itertuples()]) + [""]
    r6 = cc.a6(fr)
    L += ["A6 coverage: share of non-CTL module-1344 dollars with a brand "
          "code (charter's 'today' beside it); 45-brand share beside it.", ""]
    L += table(["year", "today (charter)", "old route", "new route",
                "45-share old", "45-share new"],
               [[int(r.year), fnum(cc.A6_TODAY[s][int(r.year)], 3),
                 fnum(r.cov_old, 3), fnum(r.cov_new, 3), fnum(r.s45_old, 3),
                 fnum(r.s45_new, 3)] for r in r6.itertuples()]) + [""]
    return L


def section_totals(fr, s):
    k, u = UNIT[s]
    o = cc.brand_year_from_eval(fr, "old").reindex(columns=YEARS[s])
    n = cc.brand_year_from_eval(fr, "new").reindex(columns=YEARS[s])
    L = [f"** Per-year 45-brand totals ({u}): {s}", ""]
    L += table(["year", "old", "new", "new/old"],
               [[y, fnum(o[y].sum() / k), fnum(n[y].sum() / k),
                 fnum(n[y].sum() / o[y].sum(), 3)] for y in YEARS[s]])
    return L + [""], o, n


def section_brand_years(o, n, s):
    k, u = UNIT[s]
    nm = cc.names45()
    o = o.reindex(cc.j45())
    n = n.reindex(cc.j45())
    L = [f"** Brand-year dollars, old vs new, all 45 x {YEARS[s][0]}-2024: {s}",
         "", f"New-route dollars ({u}); then new/old where it differs from 1 "
         "by more than 0.0005 ('.' = unchanged, 'new' = old had none).", ""]
    L += table(["j", "brand"] + [str(y)[2:] for y in YEARS[s]],
               [[j, nm[j]] + [fnum(v / k) if v == v else "-"
                              for v in n.loc[j]] for j in n.index]) + [""]
    L += [f"Old-route dollars ({u}):", ""]
    L += table(["j", "brand"] + [str(y)[2:] for y in YEARS[s]],
               [[j, nm[j]] + [fnum(v / k) if v == v else "-"
                              for v in o.loc[j]] for j in o.index]) + [""]

    def ratio(a, b):
        if not (b == b and b > 0):
            return "new" if a == a and a > 0 else "."
        r = (a if a == a else 0.0) / b
        return "." if abs(r - 1) <= 5e-4 else fnum(r, 2)
    rr = [[j, nm[j]] + [ratio(n.loc[j, y], o.loc[j, y]) for y in YEARS[s]]
          for j in n.index]
    rr = [r for r in rr if any(c != "." for c in r[2:])]
    L += ["new/old, brands with any change:", ""]
    L += table(["j", "brand"] + [str(y)[2:] for y in YEARS[s]], rr) + [""]
    return L


def section_pre2021(fr, o, n, s):
    k, u = UNIT[s]
    merge = load_merge_sets()
    nm = cc.names45()
    pre = [y for y in YEARS[s] if y < 2021]
    d = (n[pre].fillna(0) - o.reindex(n.index)[pre].fillna(0))
    ch = d[(d.abs() > 1e-6)]
    cells = int((d.abs() > 1e-6).sum().sum())
    L = [f"** Changed pre-2021 brand-years: {s}", "",
         f"{cells} brand-years before 2021 change (of {len(n) * len(pre)}).", ""]
    rows = []
    for j in ch.index[(d.abs() > 1e-6).any(axis=1)]:
        src = [c for c, t in merge.items() if t == j]
        why = (f"merge set: code(s) {', '.join(map(str, src))} -> {j} "
               "(Q2 (a) / Q2b)" if src else "UNEXPECTED (not a merge)")
        yrs = [y for y in pre if abs(d.loc[j, y]) > 1e-6]
        rows.append([j, nm.get(j, "?"), len(yrs), f"{yrs[0]}-{yrs[-1]}",
                     fnum(d.loc[j, pre].sum() / k), why])
    L += table(["j", "brand", "brand-years", "years", f"added {u}", "reason"],
               rows) + [""]
    return L


def section_d4(fr, s):
    """D4 section 7 pattern (rms_validate.py:491-581), counting j's merge set
    as j: on legacy-coded UPCs, dollars whose legacy j is j vs dollars the
    route gives j; plus name-only dollars from new UPCs."""
    k, u = UNIT[s]
    nm = cc.names45()
    J = set(cc.j45())
    f = fr[fr["year"].isin(NEW)]
    L = [f"** LOST / CREATED / shrunk / grown, 2021-2024: {s}", "",
         "As D4 section 7, but counting j's merge set as j.  On the SAME UPCs",
         "(those with a legacy master code): /legacy/ = their dollars whose",
         "legacy j is j; /route/ = their dollars the route gives j; /new UPCs/",
         "= name-only dollars the route gives j.  LOST: route (same + new) <",
         "1% of legacy; CREATED: legacy < 1% of route; shrunk / grown: same",
         "UPCs move by > 10%.  Non-CTL dollars.", ""]
    rows = []
    for route in ("old", "new"):
        jc, nc = cc.route_cols(route)
        g = f[~cc._ctl(f[nc])]
        hm = g[g["legacy_j"].notna()]
        nw = g[g["legacy_j"].isna()]
        by = pd.concat([
            hm[hm["legacy_j"].isin(J)].groupby(["year", "legacy_j"])["dollars"]
            .sum().rename_axis(["year", "j"]).rename("legacy"),
            hm[hm[jc].isin(J)].groupby(["year", jc])["dollars"].sum()
            .rename_axis(["year", "j"]).rename("route"),
            nw[nw[jc].isin(J)].groupby(["year", jc])["dollars"].sum()
            .rename_axis(["year", "j"]).rename("new"),
        ], axis=1).fillna(0.0).sort_index()
        thr = 1e5 if s == "rms" else 100.0
        for (y, j), v in by.iterrows():
            ms, ns, nn = v["legacy"], v["route"], v["new"]
            if ms >= thr and ns + nn < 0.01 * ms:
                tag = "LOST"
            elif ns >= thr and ms < 0.01 * ns:
                tag = "CREATED"
            elif max(ms, ns) > 0 and abs(ns - ms) / max(ms, ns) > 0.10:
                tag = "shrunk" if ns < ms else "grown"
            else:
                continue
            rows.append([route, int(y), int(j), nm.get(int(j), "?"), tag,
                         fnum(ms / k, 2), fnum(ns / k, 2), fnum(nn / k, 2)])
    n_new = sum(1 for r in rows if r[0] == "new")
    L += [f"Rows: old route {len(rows) - n_new}, new route {n_new}.", ""]
    L += table(["route", "year", "j", "brand", "tag", f"legacy {u}",
                f"route {u}", f"new UPCs {u}"], rows) + [""]
    return L


def section_unassigned(fr, s):
    k, u = UNIT[s]
    f = nonctl(fr, "new")
    f = f[f["year"].isin(NEW) & f["new_j"].isna()]
    t = (f.groupby(["name_new", "year"], dropna=False)["dollars"].sum()
          .unstack("year").reindex(columns=NEW).fillna(0.0))
    t["all"] = t.sum(axis=1)
    t = t.sort_values("all", ascending=False).head(30)
    L = [f"** Top 30 2021+ names still unassigned (new route; {u}): {s}", "",
         "Names with no legacy-coded UPC (line extensions outside the 45 stay",
         "out by design), names the overrides unassign, and RMS UPCs with no",
         "Homescan name ('(no name)').", ""]
    L += table(["name"] + NEW + ["2021-24"],
               [[n if n == n else "(no name)"] + [fnum(v / k, 2) for v in r]
                for n, r in t.iterrows()]) + [""]
    return L


def section_name_only(fr, s):
    k, u = UNIT[s]
    nm = cc.names45()
    f = nonctl(fr, "new")
    f = f[f["year"].isin(NEW) & f["new_j"].isin(cc.j45())]
    tot = f.groupby("new_j")["dollars"].sum()
    no = f[f["legacy_j"].isna()]
    g = no.groupby(["new_j", "name_new"])["dollars"].sum()
    L = [f"** Name-only placements per j (new UPCs, 2021-24; {u}): {s}", "",
         "Each j's largest names among UPCs with no legacy code, which reach j",
         "only through the name map (or an override): review these.  /share/ =",
         "name-only dollars / all of j's 2021-24 dollars.", ""]
    rows = []
    for j in cc.j45():
        if j not in g.index.get_level_values(0):
            rows.append([j, nm[j], "0.000", "-"])
            continue
        x = g.loc[j].sort_values(ascending=False)
        rows.append([j, nm[j], fnum(x.sum() / tot[j], 3),
                     "; ".join(f"{n} {fnum(v / k, 2)}" for n, v in
                               x.head(3).items())])
    L += table(["j", "brand", "share", f"top names ({u}, 2021-24)"], rows)
    return L + [""]


def section_routes(fr, s):
    f = nonctl(fr, "new")
    f = f[f["year"].isin(NEW)]
    t = f.groupby(["year", "j_route"])["dollars"].sum().unstack("j_route")
    t = t.div(t.sum(axis=1), axis=0).fillna(0.0)
    L = [f"** New route mix (share of non-CTL dollars): {s}", ""]
    L += table(["year"] + list(t.columns),
               [[y] + [fnum(v, 3) for v in r] for y, r in t.iterrows()])
    return L + [""]


def section_orientation(o, n, s):
    nm = cc.names45()
    base = [2017, 2018, 2019, 2020]
    so = o.fillna(0) / o.sum()
    sn = n.fillna(0) / n.sum()
    L = [f"** Orientation: the charter's expected moves (A8; approximation, "
         f"not targets): {s}", "",
         "s / median(s, 2017-20) for 2021-24, old -> new, and the new/old",
         "dollar ratio.", ""]
    rows = []
    for j in [620458, 557579, 579277, 557461, 557426, 579923, 620554]:
        mo, mn = so.loc[j, base].median(), sn.loc[j, base].median()
        rows.append([j, nm[j]] + [
            f"{fnum(so.loc[j, y] / mo, 2)} -> {fnum(sn.loc[j, y] / mn, 2)} "
            f"(x{fnum(n.loc[j, y] / o.loc[j, y], 2) if o.loc[j, y] > 0 else 'new'})"
            for y in NEW])
    L += table(["j", "brand"] + NEW, rows)
    return L + [""]


def main():
    ap = argparse.ArgumentParser(description="charter-crosswalk A8 report")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", help="report path (default: --eval/crosswalk-before-after.org)")
    ap.add_argument("--vintage", choices=["run1", "combined"], default="run1",
                    help="report introduction; UPC-dollar tables use --eval")
    a = ap.parse_args()
    if a.out is None:
        a.out = str(Path(a.eval) / "crosswalk-before-after.org")
    meta = json.loads((Path(a.eval) / "meta.json").read_text())
    title = ("Crosswalk before/after (charter-crosswalk A8, Run 1)"
             if a.vintage == "run1" else
             "Crosswalk before/after (combined xw2 vintage, 2026-10-23)")
    scope = ("the UPC-dollar level: no panel is rebuilt in Run 1 (A7 deferred)."
             if a.vintage == "run1" else
             "the UPC-dollar level.  Panel-level gates run on this vintage separately.")
    L = [f"#+TITLE: {title}",
         "",
         "Generated by =make crosswalk-report= (=src/crosswalk_eval.py= then",
         "=src/crosswalk_report.py=).  Today's route (old) vs the new brand",
         "route (=src/brand_route.py=; rule: =docs/crosswalk-rule.org=), at",
         scope,
         "Old = what today's builders did; new = what the rewired builders' own",
         "call sites (aggregate_markets._apply_brand_route,",
         "rms_market_panel.revised_upc_map) assign.  Dollars include every",
         "trip / store / DMA and all sizes (no DMA, OZ or panelist filter),",
         "so levels differ from the panels; the comparison is like for like.",
         "Aggregates only.",
         "",
         f"- evaluation: ={Path(a.eval).name}= under the vintage dir; code commit "
         f"{meta['commit'][:7]}",
         f"- accepted crosswalk (old route) sha256 {meta['old_crosswalk_sha256'][:16]}...",
         f"- upc_routes sha256 {meta['upc_routes_sha256'][:16]}...",
         f"- rows: Homescan {meta['rows']['hms']:,} (year, upc, name), "
         f"RMS {meta['rows']['rms']:,} (year, upc)",
         "",
         "Household panel (ledger G1): the builders now take the pinned 45 in",
         "aggregate_households.py too, so CTL BR (536746) leaves the household",
         "panel and M-O-M GOLDEN PUFFS (591705) enters it, apart from the route;",
         "household_base_shares and the Bartik parquets move for those two codes.",
         ""]
    frames = {s: cc.load_eval(a.eval, s) for s in ("hms", "rms")}
    L += ["* Summary", ""]
    rows = []
    for s, fr in frames.items():
        for route in ("old", "new"):
            df = cc.failures(s, route=route, frame=fr)
            bad = df[df["allowed"].isna()]
            rows.append([s, route] + [
                len(bad[bad["criterion"] == c]) for c in ("A4", "A5", "A6")]
                + ["; ".join(f"{b} {c} {y}" for b, c, y in
                             zip(bad["brand"], bad["criterion"], bad["year"]))
                   if route == "new" else "(see below)"])
    L += table(["source", "route", "A4 fails", "A5 fails", "A6 fails",
                "new-route residuals (not allow-listed)"], rows)
    L += ["",
          "The table counts only failures without a reasoned entry in",
          "config/crosswalk_allowlist.csv.  EL authorized KASHI GO LEAN",
          "CRUNCH! (579277) for Homescan 2024 and RMS 2023-24 on 2026-10-19:",
          "its decline is genuine, renamed UPCs retain the legacy code, and",
          "no misrouted UPC was found.  Its underlying A5 flags remain visible",
          "in the allow-listed tables below.", ""]
    for s in ("hms", "rms"):
        fr = frames[s]
        L += [f"* {SRC_LABEL[s]}", ""]
        L += section_checks(fr, s)
        T, o, n = section_totals(fr, s)
        L += T
        L += section_orientation(o, n, s)
        L += section_pre2021(fr, o, n, s)
        L += section_d4(fr, s)
        L += section_unassigned(fr, s)
        L += section_name_only(fr, s)
        L += section_routes(fr, s)
        L += section_brand_years(o, n, s)
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
