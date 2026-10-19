"""
RMS (Nielsen Retail Scanner) stage D2: module-1344 movement + annual tables
-> group-only parquet, one year per call, plus a per-year stats.json.

Two input formats:

classic (2006-2020)
    Raw TSVs already streamed out of the monoliths by rms_extract_classic.sh:
      extracted/classic/<yr>/nielsen_extracts/RMS/<yr>/Movement_Files/*/1344_<yr>.tsv
      extracted/classic/<yr>/nielsen_extracts/RMS/<yr>/Annual_Files/
          stores_<yr>.tsv, rms_versions_<yr>.tsv, products_extra_<yr>.tsv
    The product master (all versions) is Master_Files_2006-2020/Latest/
    products.tsv; it is identical to the Homescan master the panel uses
    (19,556 module-1344 (upc, upc_ver_uc) rows, brand and size agree 100%).

revised (2021-2024)
    Movement streamed (never unpacked) from NielsenScanner/<yr>_cereal.tgz,
    whose single member is the CEREAL AND GRANOLA category file (its name
    varies by year, so it is not named here); annual tables streamed from
    Incoming/2021-Onward_Scanner_Data/<yr>_File_Select_.../01_Annual_Files/
    {stores,productdesc,productattributes}.tgz.  Module 1344 is selected by
    productdesc.product_module_code read as a STRING and compared after
    strip() (AUDIT-2022.org: a numeric read mixes str/int and the filter
    silently keeps 75 upcs).  The revised format ships no UPC-version table.

Outputs (OUT = Data/NielsenScanner/rms, umask 027 -> o-rwx):
    OUT/master_products_1344.parquet          classic master, module 1344, all versions
    OUT/<yr>/movement_1344.parquet            store_code_uc int32, upc int64,
                                              week_end date32, units int32,
                                              prmult int32, price float64,
                                              feature int8, display int8
    OUT/<yr>/stores.parquet                   full stores table for the year
    OUT/<yr>/rms_versions.parquet             classic only (full table)
    OUT/<yr>/products_extra_1344.parquet      classic only, master-1344 upcs
    OUT/<yr>/productdesc_1344.parquet         revised only, module-1344 rows
    OUT/<yr>/productattributes_1344.parquet   revised only, module-1344 upcs
    OUT/<yr>/stats.json                       rows, stores, upcs, dollars, weeks

Dollars = sum(units * price / prmult) over rows with prmult > 0 and
price > 0 (RMS price is the price of prmult units).  Rows with prmult <= 0
or price <= 0 are kept in the parquet (raw) but counted separately.

Usage:
    python src/rms_convert.py master
    python src/rms_convert.py year 2006 [--force]
    python src/rms_convert.py log  [--out docs/rms-d2-log.org]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq

os.umask(0o027)

DATA = Path(os.environ.get("NIELSEN_DATA",
                           "/global/scratch/fsa/fc_jevons/ligon/Data"))
SCANNER = DATA / "NielsenScanner"
CLASSIC = Path(os.environ.get("RMS_CLASSIC", SCANNER / "extracted" / "classic"))
REVISED_ANNUAL = DATA / "Incoming" / "2021-Onward_Scanner_Data"
MASTER_TSV = SCANNER / "extracted" / "Master_Files_2006-2020" / "Latest" / "products.tsv"
OUT = Path(os.environ.get("RMS_OUT", SCANNER / "rms"))

CLASSIC_YEARS = range(2006, 2021)
REVISED_YEARS = range(2021, 2025)
MODULE = "1344"

BLOCK = 128 << 20  # CSV block size (bytes) -> ~3-4M movement rows per batch

MOVE_IN_TYPES = {
    "store_code_uc": pa.int32(), "upc": pa.int64(), "week_end": pa.string(),
    "units": pa.int32(), "prmult": pa.int32(), "price": pa.float64(),
    "feature": pa.int8(), "display": pa.int8(),
}
MOVE_SCHEMA = pa.schema([
    ("store_code_uc", pa.int32()), ("upc", pa.int64()),
    ("week_end", pa.date32()), ("units", pa.int32()),
    ("prmult", pa.int32()), ("price", pa.float64()),
    ("feature", pa.int8()), ("display", pa.int8()),
])


def _tsv_parse():
    # Nielsen TSVs are unquoted; a stray '"' in a description must not open
    # a quoted field.
    return pv.ParseOptions(delimiter="\t", quote_char=False)


def _write_parquet(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return path


def _stream(cmd):
    """Popen a shell pipeline whose stdout is a byte stream."""
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            bufsize=1 << 20, executable="/bin/bash")


def _finish(proc, what):
    proc.stdout.close()
    rc = proc.wait()
    # tar/pigz get SIGPIPE only if we stopped reading early; we never do.
    if rc != 0:
        raise RuntimeError(f"{what}: stream exited with {rc}")


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def _week_end_to_date(col, fmt):
    ts = pc.strptime(col, format=fmt, unit="s", error_is_null=False)
    return pc.cast(ts, pa.date32())


class _MoveStats:
    def __init__(self):
        self.rows = 0
        self.dollars = 0.0
        self.units = 0
        self.bad_prmult = 0
        self.bad_price = 0
        self.feature_nonnull = 0
        self.display_nonnull = 0
        self.stores = []
        self.upcs = []
        self.weeks = []

    def add(self, t):
        n = t.num_rows
        if n == 0:
            return
        self.rows += n
        units = t["units"]
        price = t["price"]
        prmult = t["prmult"]
        ok_m = pc.greater(prmult, 0)
        ok_p = pc.greater(price, 0)
        ok = pc.and_(ok_m, ok_p)
        self.bad_prmult += n - pc.sum(pc.cast(ok_m, pa.int64())).as_py()
        self.bad_price += n - pc.sum(pc.cast(ok_p, pa.int64())).as_py()
        rev = pc.divide(pc.multiply(pc.cast(units, pa.float64()), price),
                        pc.cast(prmult, pa.float64()))
        self.dollars += pc.sum(pc.filter(rev, ok)).as_py() or 0.0
        self.units += pc.sum(pc.cast(units, pa.int64())).as_py() or 0
        self.feature_nonnull += n - t["feature"].null_count
        self.display_nonnull += n - t["display"].null_count
        self.stores.append(pc.unique(t["store_code_uc"]).to_numpy())
        self.upcs.append(pc.unique(t["upc"]).to_numpy())
        self.weeks.append(pc.unique(t["week_end"]).to_numpy(zero_copy_only=False))

    def summary(self):
        stores = np.unique(np.concatenate(self.stores)) if self.stores else []
        upcs = np.unique(np.concatenate(self.upcs)) if self.upcs else []
        weeks = (np.unique(np.concatenate(self.weeks)) if self.weeks
                 else np.array([], dtype="datetime64[D]"))
        return {
            "rows": int(self.rows),
            "stores": int(len(stores)),
            "upcs": int(len(upcs)),
            "dollars": float(self.dollars),
            "units": int(self.units),
            "rows_prmult_le0": int(self.bad_prmult),
            "rows_price_le0": int(self.bad_price),
            "feature_nonnull_share": (self.feature_nonnull / self.rows
                                      if self.rows else None),
            "display_nonnull_share": (self.display_nonnull / self.rows
                                      if self.rows else None),
            "n_weeks": int(len(weeks)),
            "week_end_min": str(weeks.min()) if len(weeks) else None,
            "week_end_max": str(weeks.max()) if len(weeks) else None,
            "weeks": [str(w) for w in weeks],
        }


def _convert_movement(source, week_fmt, out_path, keep_upcs=None):
    """Stream a movement TSV (path or byte stream) to parquet.

    If keep_upcs (int64 array) is given, rows are filtered to those upcs and
    the pre-filter totals are returned as well (category totals).
    """
    reader = pv.open_csv(
        source,
        read_options=pv.ReadOptions(block_size=BLOCK, use_threads=True),
        parse_options=_tsv_parse(),
        convert_options=pv.ConvertOptions(column_types=MOVE_IN_TYPES),
    )
    names = reader.schema.names
    missing = [c for c in MOVE_SCHEMA.names if c not in names]
    if missing:
        raise ValueError(f"movement lacks columns {missing}: {names}")
    keep = pa.array(np.asarray(keep_upcs, dtype="int64")) if keep_upcs is not None else None
    tmp = out_path.with_suffix(".parquet.tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(tmp, MOVE_SCHEMA, compression="zstd")
    kept, cat = _MoveStats(), _MoveStats()
    try:
        for batch in reader:
            t = pa.Table.from_batches([batch])
            t = t.set_column(t.schema.get_field_index("week_end"), "week_end",
                             _week_end_to_date(t["week_end"], week_fmt))
            t = t.select(MOVE_SCHEMA.names).cast(MOVE_SCHEMA)
            if keep is not None:
                cat.add(t)
                t = t.filter(pc.is_in(t["upc"], value_set=keep))
            kept.add(t)
            if t.num_rows:
                writer.write_table(t)
    finally:
        writer.close()
    os.replace(tmp, out_path)
    return kept.summary(), (cat.summary() if keep is not None else None)


# ---------------------------------------------------------------------------
# Small tables
# ---------------------------------------------------------------------------

def _str_table(source, encoding="utf8", filt=None):
    """Read a TSV (path or byte stream) with EVERY column as string.

    The header is read first so all columns can be forced to string: type
    inference would strip leading zeros (store_zip3, upc) and can flip a
    column's type between blocks (the module-code trap in AUDIT-2022.org).
    An optional filter is applied per block, so large tables stream.
    """
    f = open(source, "rb") if isinstance(source, (str, Path)) else source
    try:
        hdr = f.readline().decode("latin1").rstrip("\r\n").split("\t")
        reader = pv.open_csv(
            f,
            read_options=pv.ReadOptions(block_size=BLOCK, encoding=encoding,
                                        column_names=hdr),
            parse_options=_tsv_parse(),
            convert_options=pv.ConvertOptions(
                column_types={c: pa.string() for c in hdr},
                strings_can_be_null=False),
        )
        schema = pa.schema([(c, pa.string()) for c in hdr])
        parts = []
        for batch in reader:
            t = pa.Table.from_batches([batch])
            if filt is not None:
                t = filt(t)
            if t.num_rows:
                parts.append(t)
    finally:
        if f is not source:
            f.close()
    return pa.concat_tables(parts) if parts else schema.empty_table()


def _to_int(t, cols, typ=pa.int64()):
    for c in cols:
        if c in t.schema.names:
            s = pc.utf8_trim_whitespace(t[c])
            s = pc.if_else(pc.equal(s, ""), pa.scalar(None, pa.string()), s)
            t = t.set_column(t.schema.get_field_index(c), c, pc.cast(s, typ))
    return t


def _to_float(t, cols):
    return _to_int(t, cols, pa.float64())


def build_master(force=False):
    """Classic master (Latest/products.tsv) -> module-1344 rows, all versions."""
    out = OUT / "master_products_1344.parquet"
    if out.exists() and not force:
        print(f"master: exists ({out})")
        return out
    t = _str_table(str(MASTER_TSV), encoding="latin1",
                   filt=lambda t: t.filter(pc.equal(
                       pc.utf8_trim_whitespace(t["product_module_code"]), MODULE)))
    t = _to_int(t, ["upc", "upc_ver_uc", "product_module_code",
                    "brand_code_uc", "multi", "size1_change_flag_uc"])
    t = _to_float(t, ["product_group_code", "department_code",
                      "size1_code_uc", "size1_amount"])
    _write_parquet(t, out)
    print(f"master: {t.num_rows:,} module-1344 rows -> {out}")
    return out


def _master_upcs():
    return pq.read_table(OUT / "master_products_1344.parquet",
                         columns=["upc"])["upc"].to_numpy()


# ---------------------------------------------------------------------------
# One year
# ---------------------------------------------------------------------------

def _classic_year(y, ydir):
    base = CLASSIC / str(y) / "nielsen_extracts" / "RMS" / str(y)
    ann = base / "Annual_Files"
    moves = sorted(base.glob(f"Movement_Files/*/{MODULE}_{y}.tsv"))
    if len(moves) != 1:
        raise FileNotFoundError(f"{y}: expected one 1344 movement file, got {moves}")
    stats = {"year": y, "format": "classic", "movement_source": str(moves[0])}

    # stores: keep zip3 and descriptors as strings; codes as ints
    st = _str_table(str(ann / f"stores_{y}.tsv"))
    st = _to_int(st, ["store_code_uc", "year", "parent_code", "retailer_code",
                      "fips_state_code", "fips_county_code", "dma_code"])
    _write_parquet(st, ydir / "stores.parquet")
    stats["stores_table_rows"] = st.num_rows
    stats["stores_table_dma_null"] = st["dma_code"].null_count

    rv = _str_table(str(ann / f"rms_versions_{y}.tsv"))
    rv = _to_int(rv, ["upc", "upc_ver_uc", "panel_year"])
    _write_parquet(rv, ydir / "rms_versions.parquet")
    stats["rms_versions_rows"] = rv.num_rows
    stats["rms_versions_dup_upc"] = int(rv.num_rows - len(pc.unique(rv["upc"])))

    m_upcs = pa.array(_master_upcs())
    pe = _str_table(str(ann / f"products_extra_{y}.tsv"), encoding="latin1",
                    filt=lambda t: t.filter(pc.is_in(
                        pc.cast(t["upc"], pa.int64()), value_set=m_upcs)))
    pe = _to_int(pe, ["upc", "upc_ver_uc", "panel_year"])
    _write_parquet(pe, ydir / "products_extra_1344.parquet")
    stats["products_extra_1344_rows"] = pe.num_rows

    kept, _ = _convert_movement(str(moves[0]), "%Y%m%d",
                                ydir / "movement_1344.parquet")
    stats["movement"] = kept
    return stats


def _revised_paths(y):
    ann = REVISED_ANNUAL / f"{y}_File_Select_Supercategory_MovementData" / "01_Annual_Files"
    return ann, SCANNER / f"{y}_cereal.tgz"


def _tar_stream(tgz):
    # Every revised archive holds exactly one member; -O concatenates
    # members, so the member name (which varies by year) is not needed.
    return _stream(f"set -o pipefail; pigz -dc '{tgz}' | tar -xOf -")


def _revised_year(y, ydir):
    ann, cereal = _revised_paths(y)
    stats = {"year": y, "format": "revised", "movement_source": str(cereal)}

    p = _tar_stream(ann / "stores.tgz")
    st = _str_table(p.stdout)
    _finish(p, f"{y} stores")
    st = _to_int(st, ["store_code_uc", "year", "retailer_code", "parent_code",
                      "fips_state_code", "fips_county_code", "acv"])
    _write_parquet(st, ydir / "stores.parquet")
    stats["stores_table_rows"] = st.num_rows

    p = _tar_stream(ann / "productdesc.tgz")
    pdsc = _str_table(p.stdout, filt=lambda t: t.filter(pc.equal(
        pc.utf8_trim_whitespace(t["product_module_code"]), MODULE)))
    _finish(p, f"{y} productdesc")
    pdsc = _to_int(pdsc, ["upc", "year"])
    _write_parquet(pdsc, ydir / "productdesc_1344.parquet")
    upcs = np.unique(pdsc["upc"].to_numpy())
    stats["productdesc_1344_rows"] = pdsc.num_rows
    stats["productdesc_1344_upcs"] = int(len(upcs))
    if len(upcs) < 1000:
        raise RuntimeError(f"{y}: only {len(upcs)} module-1344 upcs; "
                           "module filter is not dtype-safe?")

    keep = pa.array(upcs)
    p = _tar_stream(ann / "productattributes.tgz")
    pa_t = _str_table(p.stdout, filt=lambda t: t.filter(pc.is_in(
        pc.cast(t["upc"], pa.int64()), value_set=keep)))
    _finish(p, f"{y} productattributes")
    pa_t = _to_int(pa_t, ["upc"])
    _write_parquet(pa_t, ydir / "productattributes_1344.parquet")
    stats["productattributes_1344_rows"] = pa_t.num_rows

    p = _tar_stream(cereal)
    kept, cat = _convert_movement(p.stdout, "%Y-%m-%d",
                                  ydir / "movement_1344.parquet", keep_upcs=upcs)
    _finish(p, f"{y} movement")
    stats["movement"] = kept
    cat.pop("weeks", None)
    stats["category_movement"] = cat
    return stats


def convert_year(y, force=False):
    ydir = OUT / str(y)
    sp = ydir / "stats.json"
    if sp.exists() and not force:
        print(f"{y}: done already ({sp})")
        return json.loads(sp.read_text())
    ydir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats = _classic_year(y, ydir) if y in CLASSIC_YEARS else _revised_year(y, ydir)
    stats["seconds"] = round(time.time() - t0, 1)
    stats["outputs"] = {f.name: f.stat().st_size
                        for f in sorted(ydir.glob("*.parquet"))}
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stats, indent=1))
    os.replace(tmp, sp)
    mv = stats["movement"]
    print(f"{y}: rows {mv['rows']:,} stores {mv['stores']:,} upcs {mv['upcs']:,} "
          f"dollars {mv['dollars']:,.0f} weeks {mv['week_end_min']}..{mv['week_end_max']} "
          f"({stats['seconds']} s)")
    return stats


# ---------------------------------------------------------------------------
# Per-year log (committed: docs/rms-d2-log.org)
# ---------------------------------------------------------------------------

def write_log(out_path):
    rows = []
    for y in list(CLASSIC_YEARS) + list(REVISED_YEARS):
        sp = OUT / str(y) / "stats.json"
        if not sp.exists():
            rows.append((y, None))
            continue
        rows.append((y, json.loads(sp.read_text())))
    L = []
    L.append("#+title: RMS module-1344 extraction log (D2)")
    L.append("#+date: " + time.strftime("%Y-%m-%d %H:%M %Z"))
    L.append("")
    L.append("Generated by =python src/rms_convert.py log= (called from "
             "=src/rms_build.sh=); do not edit by hand.")
    L.append(f"Parquet root: ={OUT}= (group-only, o-rwx).")
    L.append("Dollars = sum(units x price / prmult) over rows with prmult > 0 "
             "and price > 0.  Stores and UPCs are distinct counts in the "
             "module-1344 movement.  Weeks are distinct =week_end= dates.")
    L.append("")
    L.append("| year | format | rows | stores | UPCs | dollars | weeks | "
             "first week_end | last week_end | prmult<=0 | price<=0 | "
             "feature non-null | movement parquet (MB) |")
    L.append("|-" * 13 + "|")
    tot = dict(rows=0, dollars=0.0)
    for y, s in rows:
        if s is None:
            L.append(f"| {y} | MISSING |" + " |" * 11)
            continue
        m = s["movement"]
        tot["rows"] += m["rows"]
        tot["dollars"] += m["dollars"]
        fs = m["feature_nonnull_share"]
        L.append(
            f"| {y} | {s['format']} | {m['rows']:,} | {m['stores']:,} | "
            f"{m['upcs']:,} | {m['dollars']:,.0f} | {m['n_weeks']} | "
            f"{m['week_end_min']} | {m['week_end_max']} | "
            f"{m['rows_prmult_le0']:,} | {m['rows_price_le0']:,} | "
            f"{(fs if fs is not None else float('nan')):.3f} | "
            f"{s['outputs'].get('movement_1344.parquet', 0) / 2**20:,.0f} |")
    L.append(f"| total | | {tot['rows']:,} | | | {tot['dollars']:,.0f} |"
             + " |" * 7)
    L.append("")
    L.append("* Annual tables")
    L.append("")
    L.append("| year | stores rows | stores w/o dma_code | rms_versions rows | "
             "dup upc in rms_versions | product rows (1344) | "
             "productattributes rows (1344) |")
    L.append("|-" * 7 + "|")
    for y, s in rows:
        if s is None:
            continue
        prod = s.get("products_extra_1344_rows", s.get("productdesc_1344_rows"))
        L.append(f"| {y} | {s['stores_table_rows']:,} | "
                 f"{s.get('stores_table_dma_null', 'n/a (fips->DMA)')} | "
                 f"{s.get('rms_versions_rows', 'n/a')} | "
                 f"{s.get('rms_versions_dup_upc', 'n/a')} | {prod:,} | "
                 f"{s.get('productattributes_1344_rows', 'n/a')} |")
    L.append("")
    rev = [(y, s) for y, s in rows if s and s.get("category_movement")]
    if rev:
        L.append("* Revised format (2021+): module 1344 within the CEREAL AND "
                 "GRANOLA category file")
        L.append("")
        L.append("| year | category rows | category dollars | 1344 rows | "
                 "1344 dollars | 1344 share of $ | productdesc 1344 UPCs |")
        L.append("|-" * 7 + "|")
        for y, s in rev:
            c, m = s["category_movement"], s["movement"]
            L.append(f"| {y} | {c['rows']:,} | {c['dollars']:,.0f} | "
                     f"{m['rows']:,} | {m['dollars']:,.0f} | "
                     f"{m['dollars'] / c['dollars']:.3f} | "
                     f"{s['productdesc_1344_upcs']:,} |")
        L.append("")
    L.append("* Outputs per year")
    L.append("")
    for y, s in rows:
        if s is None:
            continue
        outs = ", ".join(f"={k}= {v / 2**20:,.1f} MB"
                         for k, v in s["outputs"].items())
        L.append(f"- {y} ({s['seconds']} s): {outs}")
    Path(out_path).write_text("\n".join(L) + "\n")
    print(f"log -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("master")
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("year")
    a.add_argument("years", nargs="+", type=int)
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("log")
    a.add_argument("--out", default="docs/rms-d2-log.org")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "master":
        build_master(force=args.force)
    elif args.cmd == "year":
        for y in args.years:
            convert_year(y, force=args.force)
    else:
        write_log(args.out)


if __name__ == "__main__":
    sys.exit(main())
