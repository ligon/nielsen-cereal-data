"""
charter-crosswalk Run 1: the UPC-dollar evaluation of the brand route,
old (today's) vs new (src/brand_route.py), for both sources.

No panel is built.  The route is evaluated on every module-1344 dollar:

  Homescan  data/raw/purchases_<yr>.parquet, 2004-2024, one year at a time,
            read with the builder's own reader (aggregate_markets.
            _read_one_purchase); unprojected sample dollars
            (total_price_paid), no trip / panelist / DMA / OZ filters.
            old j: 2004-2020 the row's brand_code_uc; 2021+ the accepted
                   crosswalk's name lookup, AM:245-250 as it was
                   (drop_duplicates('brand_descr_2021') in file order).
            new j: aggregate_markets._apply_brand_route (the builder's call).
  RMS       rms/diagnostics/upc_routes.parquet: per (year, upc) all valid
            module-1344 dollars, every store, before the DMA/CTL/45/OZ
            filters (as in D4 section 7), 2006-2024.
            old j: its brand_code_uc (the accepted build's route).
            new j: classic years merge(brand_code_uc); 2021+
                   rms_market_panel.revised_upc_map (the builder's call).

Per (source, year, upc) the output carries dollars, the name used for the
CTL drop (old and new), legacy_j (the A1 legacy code, merged: classic years
the year's own version code, 2021+ the highest master version), old_j,
new_j and the new route label.  A4/A5/A6 (src/crosswalk_checks.py) and the
A8 report (src/crosswalk_report.py) read these frames.

Usage (Makefile target crosswalk-eval):
  python src/crosswalk_eval.py --raw RAW --rms RMS_DIR --old-crosswalk PATH
                               --out OUTDIR
Writes OUTDIR/{hms_upc,rms_upc,name_map}.parquet and OUTDIR/meta.json
(umask 027: group-only).  Nielsen data; never leaves the cluster.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.umask(0o027)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MIRROR = Path("/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data")
D_RAW = MIRROR / "data" / "raw"
D_RMS = Path("/global/scratch/fsa/fc_jevons/ligon/Data/NielsenScanner/rms")
D_OLD_XWALK = MIRROR / "data" / "clean" / "brand_crosswalk.parquet"
D_OUT = MIRROR / "data" / "clean-crosswalk-2026-10-19" / "run1-eval"

COLS = ["source", "year", "upc", "dollars", "name_old", "name_new",
        "legacy_j", "old_j", "new_j", "j_route"]


def _ctl(s):
    return s.fillna("").astype(str).str.upper().str.startswith("CTL")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def old_name_lookup(old_crosswalk):
    """AM:245-250 as it was before this charter: name -> first code in file
    order."""
    x = pd.read_parquet(old_crosswalk)
    return (x[["brand_code_uc", "brand_descr_2021"]]
            .drop_duplicates("brand_descr_2021")
            .set_index("brand_descr_2021")["brand_code_uc"])


def hms_frame(raw, old_crosswalk, years=range(2004, 2025)):
    import aggregate_markets as am
    from brand_route import cached_route, merge_codes, REVISED_FIRST
    raw = Path(raw)
    route = cached_route(str(raw), str(raw / "master_products.parquet"),
                         (1344,))
    old = old_name_lookup(old_crosswalk)
    legacy = route.legacy["legacy_code"]
    parts = []
    for y in years:
        p = am._read_one_purchase(raw, y)
        if p is None:
            continue
        if y < REVISED_FIRST:
            p["old_j"] = p["brand_code_uc"].astype(float)
            p["legacy_j"] = merge_codes(p["old_j"]).to_numpy()
        else:
            p["old_j"] = p["brand_descr"].map(old).astype(float).to_numpy()
            p["legacy_j"] = merge_codes(p["upc"].map(legacy)).to_numpy()
        new = am._apply_brand_route(p.copy(), raw, [1344], route=route)
        p["new_j"] = new["brand_code_uc"].astype(float).to_numpy()
        if y >= REVISED_FIRST:
            r = route.route(y, p["upc"].unique()).set_index("upc")["j_route"]
            p["j_route"] = p["upc"].map(r).to_numpy()
        else:
            p["j_route"] = "classic_row_code"
        g = (p.groupby(["upc", "brand_descr", "legacy_j", "old_j", "new_j",
                        "j_route"], dropna=False)["total_price_paid"].sum()
              .rename("dollars").reset_index())
        g["year"] = y
        parts.append(g)
        print(f"  hms {y}: {len(p):,} rows -> {len(g):,} (upc) rows",
              flush=True)
    h = pd.concat(parts, ignore_index=True)
    h["source"] = "hms"
    h["name_old"] = h["brand_descr"]
    h["name_new"] = h["brand_descr"]
    return h[COLS]


def rms_frame(rms_dir):
    import rms_market_panel as rmp
    from brand_route import merge_codes, REVISED_FIRST
    u = pd.read_parquet(Path(rms_dir) / "diagnostics" / "upc_routes.parquet")
    u = u.groupby(["year", "upc"], as_index=False).agg(
        dollars=("dollars", "sum"), route=("route", "first"),
        brand_code_uc=("brand_code_uc", "first"),
        brand_descr=("brand_descr", "first"),
        master_code=("master_code", "first"))
    u["old_j"] = u["brand_code_uc"].astype(float)
    u["name_old"] = u["brand_descr"]
    parts = []
    for y, d in u.groupby("year"):
        d = d.copy()
        if y < REVISED_FIRST:
            d["new_j"] = merge_codes(d["old_j"]).to_numpy()
            d["legacy_j"] = merge_codes(d["master_code"].astype(float)).to_numpy()
            d["name_new"] = d["name_old"]
            d["j_route"] = np.where(d["route"] == "not_in_map", "not_in_map",
                                    "classic_version")
        else:
            m = rmp.revised_upc_map(int(y), d["upc"].to_numpy()).set_index("upc")
            d["new_j"] = d["upc"].map(m["brand_code_uc"]).astype(float).to_numpy()
            d["legacy_j"] = merge_codes(
                d["upc"].map(m["master_code"]).astype(float)).to_numpy()
            d["name_new"] = d["upc"].map(m["brand_descr"]).to_numpy()
            d["j_route"] = d["upc"].map(m["j_route"]).to_numpy()
        parts.append(d)
        print(f"  rms {y}: {len(d):,} upcs", flush=True)
    r = pd.concat(parts, ignore_index=True)
    r["source"] = "rms"
    return r[COLS]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raw", default=str(D_RAW))
    ap.add_argument("--rms", default=str(D_RMS))
    ap.add_argument("--old-crosswalk", default=str(D_OLD_XWALK))
    ap.add_argument("--out", default=str(D_OUT))
    a = ap.parse_args()
    out = Path(a.out)
    assert out.resolve() != (MIRROR / "data" / "clean").resolve()
    out.mkdir(parents=True, exist_ok=True)
    dirs = [out] + ([out.parent] if out.parent.name.startswith(
        "clean-crosswalk-") else [])
    for d in dirs:                       # group fc_jevons, setgid (data terms)
        try:
            os.chown(d, -1, __import__("grp").getgrnam("fc_jevons").gr_gid)
            os.chmod(d, 0o2750)
        except (KeyError, PermissionError):
            pass
    os.environ.setdefault("PANEL_CLEAN", str(Path(a.raw).parent / "clean"))
    os.environ.setdefault("PANEL_RAW", str(a.raw))
    os.environ.setdefault("RMS_OUT", str(a.rms))
    t0 = time.time()
    h = hms_frame(a.raw, a.old_crosswalk)
    h.to_parquet(out / "hms_upc.parquet", index=False)
    r = rms_frame(a.rms)
    r.to_parquet(out / "rms_upc.parquet", index=False)
    from brand_crosswalk import build_name_map
    build_name_map(a.raw, [1344]).to_parquet(out / "name_map.parquet",
                                             index=False)
    head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    meta = {"commit": head, "raw": a.raw, "rms": a.rms,
            "old_crosswalk": a.old_crosswalk,
            "old_crosswalk_sha256": _sha(a.old_crosswalk),
            "upc_routes_sha256": _sha(Path(a.rms) / "diagnostics" /
                                      "upc_routes.parquet"),
            "seconds": round(time.time() - t0, 1),
            "rows": {"hms": len(h), "rms": len(r)}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
