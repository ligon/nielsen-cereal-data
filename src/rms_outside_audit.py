"""Audit other-cereal revenue and recorded ounce sizes without changing a panel.

Reuse the adopted vintage's saved UPC maps and week owners, and its producer's
store-DMA mapping (.sucoder/ledger-outside-cereal.md sections 2-4). Outputs are
diagnostics, not an estimation-ready outside good. Nielsen data stays on cluster.
"""
import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import rms_market_panel as rmp
from crosswalk_eval import _sha

CATEGORIES = ("modeled_45", "other_named", "private_label", "unmapped",
              "selected_brand_excluded")
MOVEMENT_COLS = ["store_code_uc", "upc", "week_end", "units", "prmult", "price"]


def combine(parts, levels):
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).groupby(level=levels, sort=True).sum()


def panel_cells(panel):
    """Use existing panel keys, revenue and ounces as the cellwise oracle."""
    p = panel.reset_index()
    p["tcode"] = (p["t"].str.extract(r"Y(\d+)")[0].astype(int) * 10
                  + p["t"].str.slice(1, 2).astype(int))
    if p.duplicated(["tcode", "m", "j"]).any():
        raise ValueError("duplicate panel cells")
    return p.set_index(["tcode", "m", "j"])[["revenue", "servings"]].sort_index()


def week_owners(weeks):
    kept = weeks.loc[weeks["kept"].eq(True)].copy()
    if kept["week_end"].duplicated().any():
        raise ValueError("retained week has multiple file owners")
    return {int(y): np.array(g["week_end"], dtype="datetime64[D]")
            for y, g in kept.groupby("file_year")}


def aggregate_batch(rows, routes, store_map, keep, j45, m65, accepted,
                    detail_callback=None):
    """Classify movement, keeping the original 45-brand selection separate.

    Complete residual price is never formed here. Recorded size coverage and
    same-row covered revenue/ounces are independent of brand assignment.
    """
    if not routes["upc"].is_unique or not store_map["store_code_uc"].is_unique:
        raise ValueError("UPC and store maps must have unique keys")
    n = len(rows)
    ui = pd.Index(routes["upc"]).get_indexer(rows["upc"])
    si = pd.Index(store_map["store_code_uc"]).get_indexer(rows["store_code_uc"])
    codes = np.r_[routes["brand_code_uc"].fillna(-1).astype(int), -1][ui]
    ctl = np.r_[routes["brand_descr"].fillna("").astype(str).str.upper()
                .str.startswith("CTL").to_numpy(), False][ui]
    oz = np.r_[routes["size1_units"].fillna("").astype(str).str.upper()
               .str.strip().eq("OZ").to_numpy(), False][ui]
    sizes = np.r_[routes["size1_amount"].fillna(0).astype(float), 0.][ui]
    dma = np.r_[store_map["DMA_Cd"].fillna(-1).astype(int), -1][si]
    units, price, mult = (rows[c].to_numpy(dtype=float)
                          for c in ("units", "price", "prmult"))
    positive = (units > 0) & (price > 0) & (mult > 0)
    finite = np.isfinite(units) & np.isfinite(price) & np.isfinite(mult)
    wk = rows["week_end"].to_numpy().astype("datetime64[D]")
    year = wk.astype("datetime64[Y]").astype(int) + 1970
    month = wk.astype("datetime64[M]").astype(int) % 12 + 1
    tc = year * 10 + (month - 1) // 3 + 1
    valid_week = np.isin(wk, keep)
    selected_dma = np.isin(dma, m65)
    selected = (positive & finite & valid_week & selected_dma
                & (year >= 2006) & (year <= 2024))
    revenue = np.zeros(n)
    valid = positive & finite
    revenue[valid] = units[valid] * price[valid] / mult[valid]
    if not np.isfinite(revenue[selected]).all():
        raise ValueError("nonfinite valid movement revenue")
    in45 = np.isin(codes, j45)
    candidate = selected & in45 & ~ctl & oz
    keys = pd.MultiIndex.from_arrays([tc, dma, codes], names=accepted.names)
    modeled = candidate & (accepted.get_indexer(keys) >= 0)
    category = np.full(n, "other_named", dtype=object)
    category[codes < 0] = "unmapped"
    category[ctl] = "private_label"
    category[in45 & ~modeled] = "selected_brand_excluded"
    category[modeled] = "modeled_45"
    known = oz & np.isfinite(sizes) & (sizes > 0)
    covered = selected & known
    ounces = np.zeros(n)
    ounces[covered] = units[covered] * sizes[covered]
    if not np.isfinite(ounces[covered]).all():
        raise ValueError("nonfinite covered ounces")
    frame = pd.DataFrame({
        "tcode": tc[selected], "m": dma[selected], "category": category[selected],
        "revenue": revenue[selected], "rows": np.ones(selected.sum(), dtype=int),
        "covered_revenue": np.where(known[selected], revenue[selected], 0.),
        "covered_ounces": ounces[selected],
        "uncovered_rows": (~known[selected]).astype(int),
    })
    if detail_callback is not None:
        # Optional diagnostics inherit exactly the selected rows/categories.
        # Give the callback its own frame; the original aggregate is unchanged.
        detail_callback(frame.assign(upc=rows["upc"].to_numpy()[selected],
                                     units=units[selected]))
    markets = frame.groupby(["tcode", "m", "category"], sort=False).sum()
    cells = pd.DataFrame({"tcode": tc[candidate], "m": dma[candidate],
                          "j": codes[candidate], "revenue": revenue[candidate],
                          "servings": units[candidate] * sizes[candidate]})
    candidates = cells.groupby(["tcode", "m", "j"], sort=False).sum()
    stats = dict(rows=n, selected_rows=int(selected.sum()),
                 positive_nonfinite_rows=int((positive & ~finite).sum()),
                 missing_route_selected_rows=int((selected & (ui < 0)).sum()),
                 dollars_unselected_dma=float(revenue[positive & finite & valid_week
                                                       & ~selected_dma].sum()),
                 dollars_duplicate_week=float(revenue[positive & finite & ~valid_week].sum()))
    return markets, candidates, stats


def reconcile(candidates, expected, rtol=1e-12, atol=1e-6):
    """Mirror the producer's final invalid-price-cell filter, then compare cells."""
    got = candidates.copy()
    price = got["revenue"] / got["servings"]
    good = np.isfinite(price) & price.ne(0)
    dropped = int((~good).sum())
    got = got.loc[good]
    if set(got.index) != set(expected.index):
        raise ValueError("reproduced modeled cell keys differ from adopted panel")
    got = got.reindex(expected.index)
    result = dict(cells=len(got), producer_bad_price_cells=dropped)
    for col in ("revenue", "servings"):
        gap = np.abs(got[col] - expected[col])
        result[col + "_max_abs_gap"] = float(gap.max()) if len(gap) else 0.
        result[col + "_max_relative_gap"] = float(
            (gap / expected[col].abs().clip(lower=1e-30)).max()) if len(gap) else 0.
        if not np.allclose(got[col], expected[col], rtol=rtol, atol=atol):
            raise ValueError(f"modeled {col} differs from adopted panel: {result}")
    return result


def residual_table(categories, expected):
    totals = categories.groupby(level=["tcode", "m"]).sum(numeric_only=True)
    modeled = expected.groupby(level=["tcode", "m"])["revenue"].sum()
    residual = categories.loc[
        categories.index.get_level_values("category") != "modeled_45"]
    residual = residual.groupby(level=["tcode", "m"]).sum(numeric_only=True)
    idx = totals.index.union(modeled.index)
    table = totals[["revenue"]].reindex(idx, fill_value=0).rename(
        columns={"revenue": "module_revenue"})
    table["modeled_revenue"] = modeled.reindex(idx, fill_value=0)
    table["residual_revenue"] = table["module_revenue"] - table["modeled_revenue"]
    for col in ("covered_revenue", "covered_ounces", "uncovered_rows", "rows"):
        table["residual_" + col] = residual[col].reindex(idx, fill_value=0)
    observed = residual["revenue"].reindex(idx, fill_value=0)
    if not np.allclose(table["residual_revenue"], observed, rtol=1e-12, atol=1e-5):
        raise ValueError("residual category totals do not reconcile")
    table["residual_share"] = table["residual_revenue"] / table["module_revenue"]
    table["recorded_ounce_revenue_coverage"] = (
        table["residual_covered_revenue"] / observed.where(observed > 0))
    table["covered_subset_dollars_per_ounce"] = (
        table["residual_covered_revenue"] / table["residual_covered_ounces"].where(
            table["residual_covered_ounces"] > 0))
    # Missing-size rows with even tiny revenue cannot certify a complete price.
    complete = table["residual_uncovered_rows"].eq(0) & observed.gt(0)
    table["complete_residual_dollars_per_ounce"] = (
        table["covered_subset_dollars_per_ounce"].where(complete))
    excluded = categories.xs("selected_brand_excluded", level="category") if (
        "selected_brand_excluded" in categories.index.get_level_values("category")) else None
    table["selected_brand_excluded_revenue"] = (
        excluded["revenue"].reindex(idx, fill_value=0) if excluded is not None else 0.)
    table["t"] = [f"Q{t % 10}Y{t // 10}" for t, _m in table.index]
    table["year"] = [t // 10 for t, _m in table.index]
    return table


def audit_year(year, rms_dir, routes, stores, keep, j45, m65, accepted, batch_size):
    started = time.monotonic()
    path = Path(rms_dir) / str(year) / "movement_1344.parquet"
    groups, cells, counters = [], [], {}
    pf = pq.ParquetFile(path)
    # One scan process per CPU, rather than one Arrow thread pool per process.
    for batch in pf.iter_batches(batch_size=batch_size, columns=MOVEMENT_COLS,
                                 use_threads=False):
        g, c, stats = aggregate_batch(batch.to_pandas(), routes, stores,
                                     keep, j45, m65, accepted)
        groups.append(g)
        cells.append(c)
        for name, val in stats.items():
            counters[name] = counters.get(name, 0) + val
    counters.update(file_year=year, seconds=time.monotonic() - started,
                    movement_sha256=_sha(path), movement_path=str(path.resolve()),
                    stores_sha256=_sha(Path(rms_dir) / str(year) / "stores.parquet"))
    return year, combine(groups, ["tcode", "m", "category"]), combine(
        cells, ["tcode", "m", "j"]), counters


def run(rms_dir, panel_raw, output, jobs=4, batch_size=500_000):
    os.umask(0o027)
    pa.set_cpu_count(1)
    rms_dir, output = Path(rms_dir).resolve(), Path(output)
    available = len(os.sched_getaffinity(0))
    if jobs < 1 or jobs > available or batch_size < 1:
        raise ValueError("jobs/batch size must fit the current CPU allocation")
    panel_path = rms_dir / "market_panel_rms.parquet"
    metadata = pq.read_schema(panel_path).metadata or {}
    build = json.loads(metadata.get(b"rms_build", b"{}"))
    if build.get("stores") is not None:
        raise ValueError("restricted-store panel requires an explicit matching store-set audit")
    routes_path = rms_dir / "diagnostics" / "upc_routes.parquet"
    weeks_path = rms_dir / "diagnostics" / "week_sources.parquet"
    routes, weeks = pd.read_parquet(routes_path), pd.read_parquet(weeks_path)
    expected = panel_cells(pd.read_parquet(panel_path))
    j45 = np.sort(expected.index.get_level_values("j").unique())
    m65 = np.sort(expected.index.get_level_values("m").unique())
    if len(j45) != 45 or len(m65) != 65:
        raise ValueError("audit requires the adopted 45-brand, 65-DMA universe")
    owners = week_owners(weeks)
    years = sorted(owners)
    if years != list(range(2006, 2025)):
        raise ValueError("expected movement file years 2006-2024")
    output.mkdir(parents=True, exist_ok=False)
    (output / "by_file_year").mkdir()
    rmp.OUT, rmp.PANEL_RAW = rms_dir, Path(panel_raw).resolve()
    # Reuse producer geography, and persist the exact maps actually consumed.
    maps = {}
    for year in years:
        maps[year] = rmp.store_dma(year)
        maps[year].to_parquet(output / "by_file_year" / f"stores-{year}.parquet", index=False)
    inputs = {"market_panel": dict(path=str(panel_path), sha256=_sha(panel_path)),
              "upc_routes": dict(path=str(routes_path), sha256=_sha(routes_path)),
              "week_sources": dict(path=str(weeks_path), sha256=_sha(weeks_path)),
              "fips_panelists": {str(p.resolve()): _sha(p) for p in
                  sorted(Path(panel_raw).glob("panelists_*.parquet"))
                  if int(p.stem.split("_")[-1]) <= 2020}}
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    meta = dict(producer=pin, script_sha256=_sha(__file__), inputs=inputs,
                rms_build=build, jobs=jobs, batch_size=batch_size, pid=os.getpid(),
                python=os.sys.version, pandas=pd.__version__, numpy=np.__version__,
                pyarrow=pa.__version__, argv=os.sys.argv, file_years=years,
                definitions=".sucoder/ledger-outside-cereal.md", status="running")
    (output / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    grouped, candidate_parts, statistics = [], [], []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(audit_year, y, rms_dir, routes.loc[routes.year == y],
                   maps[y], owners[y], j45, m65, expected.index, batch_size) for y in years]
        for future in as_completed(futures):
            y, group, cells, stats = future.result()
            group.to_parquet(output / "by_file_year" / f"categories-{y}.parquet")
            cells.to_parquet(output / "by_file_year" / f"modeled-{y}.parquet")
            (output / "by_file_year" / f"stats-{y}.json").write_text(
                json.dumps(stats, indent=2) + "\n")
            grouped.append(group)
            candidate_parts.append(cells)
            statistics.append(stats)
            print(f"file {y}: {stats['rows']:,} rows in {stats['seconds']:.1f}s", flush=True)
    if any(s["positive_nonfinite_rows"] or s["missing_route_selected_rows"] for s in statistics):
        raise ValueError("nonfinite positive movement or missing saved route; inspect file stats")
    categories = combine(grouped, ["tcode", "m", "category"])
    candidates = combine(candidate_parts, ["tcode", "m", "j"])
    comparison = reconcile(candidates, expected)
    table = residual_table(categories, expected)
    grid = pd.MultiIndex.from_product(
        [[y * 10 + q for y in range(2006, 2025) for q in range(1, 5)], m65],
        names=["tcode", "m"])
    if set(table.index) != set(grid):
        raise ValueError("audit does not cover all 4940 DMA-quarter cells")
    if ((table.residual_revenue < -1e-5).any()
            or ~np.isfinite(table[["module_revenue", "modeled_revenue", "residual_revenue"]]).all().all()):
        raise ValueError("invalid residual revenue")
    categories.to_parquet(output / "categories.parquet")
    candidates.to_parquet(output / "reproduced_45.parquet")
    table.to_parquet(output / "dma_quarter.parquet")
    totals = table[["module_revenue", "modeled_revenue", "residual_revenue",
                    "residual_covered_revenue", "residual_covered_ounces",
                    "residual_uncovered_rows", "selected_brand_excluded_revenue"]].groupby(table.year).sum()
    totals["residual_share"] = totals.residual_revenue / totals.module_revenue
    totals["recorded_ounce_revenue_coverage"] = totals.residual_covered_revenue / totals.residual_revenue
    totals.to_parquet(output / "annual.parquet")
    meta.update(status="complete", reconciliation=comparison,
                complete_price_cells=int(table.complete_residual_dollars_per_ounce.notna().sum()),
                n_markets=len(table), file_statistics=sorted(statistics, key=lambda s: s["file_year"]))
    (output / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(dict(reconciliation=comparison, n_markets=len(table),
                          complete_price_cells=meta["complete_price_cells"])), flush=True)
    return table, categories, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rms-dir", required=True)
    parser.add_argument("--panel-raw", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=500_000)
    args = parser.parse_args()
    run(args.rms_dir, args.panel_raw, args.output, args.jobs, args.batch_size)
