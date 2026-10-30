"""Measure size gaps in the adopted market universe using existing row rules.

Save units/revenue independently of any new mass or brand choice. Grounding:
.sucoder/ledger-rms-size-recovery.md section 8. No estimation panel is written.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from crosswalk_eval import _sha
from rms_outside_audit import aggregate_batch, combine, panel_cells, week_owners

LEVELS = ["tcode", "m", "category", "upc"]


def scan_year(year, rms_dir, routes, stores, keep, j45, m65, accepted,
              focus, batch_size, expected_hash):
    path = Path(rms_dir) / str(year) / "movement_1344.parquet"
    parts = []
    stats = dict(file_year=year, input_rows=0, focus_rows=0, selected_rows=0)
    began = time.monotonic()

    def collect(frame):
        frame["uncovered_revenue"] = frame.revenue - frame.covered_revenue
        parts.append(frame.groupby(LEVELS, sort=False).sum(numeric_only=True))

    columns = ["store_code_uc", "upc", "week_end", "units", "prmult", "price"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                 columns=columns, use_threads=False):
        rows = batch.to_pandas()
        stats["input_rows"] += len(rows)
        rows = rows.loc[rows.upc.isin(focus)]
        stats["focus_rows"] += len(rows)
        if rows.empty:
            continue
        _, _, native = aggregate_batch(rows, routes, stores, keep, j45, m65,
                                       accepted, detail_callback=collect)
        if native["positive_nonfinite_rows"] or native["missing_route_selected_rows"]:
            raise ValueError(f"invalid focused movement in file {year}: {native}")
        stats["selected_rows"] += native["selected_rows"]
    digest = _sha(path)
    if digest != expected_hash:
        raise ValueError(f"movement input changed since original audit: {year}")
    stats.update(seconds=time.monotonic() - began, movement_sha256=digest,
                 movement_path=str(path.resolve()))
    grouped = combine(parts, LEVELS)
    if grouped.empty:
        grouped = pd.DataFrame(columns=LEVELS + ["units", "revenue", "rows",
            "covered_revenue", "covered_ounces", "uncovered_rows", "uncovered_revenue"]).set_index(LEVELS)
    grouped["file_year"] = year
    return grouped.reset_index(), stats


def reconcile_uncovered(details, baseline):
    residual = details.loc[details.category.ne("modeled_45")]
    got = residual.groupby(["tcode", "m"])[["uncovered_rows", "uncovered_revenue"]].sum()
    got = got.reindex(baseline.index, fill_value=0)
    np.testing.assert_array_equal(got.uncovered_rows, baseline.residual_uncovered_rows)
    expected = baseline.residual_revenue - baseline.residual_covered_revenue
    if not np.allclose(got.uncovered_revenue, expected, rtol=1e-12, atol=1e-6):
        raise ValueError("focused unknown-size revenue fails original same-universe audit")
    return dict(markets=len(baseline), uncovered_rows=int(got.uncovered_rows.sum()),
                uncovered_revenue=float(got.uncovered_revenue.sum()),
                max_abs_revenue_gap=float((got.uncovered_revenue - expected).abs().max()))


def run(rms_dir, audit_dir, candidate_map, output, jobs=4, batch_size=500_000):
    os.umask(0o027)
    pa.set_cpu_count(1)
    if not 1 <= jobs <= len(os.sched_getaffinity(0)) or batch_size < 1:
        raise ValueError("scan jobs/batch size must fit the allocated slice")
    rms_dir, audit_dir, output = map(Path, (rms_dir, audit_dir, output))
    if output.exists():
        raise FileExistsError(output)
    archive = audit_dir.parent
    archive_hashes = dict(line.rstrip().split("  ", 1)[::-1] for line in
                          (archive / "SHA256SUMS").read_text().splitlines())
    for name in ("provenance.json", "dma_quarter.parquet"):
        path = audit_dir / name
        if archive_hashes.get(str(path.relative_to(archive))) != _sha(path):
            raise ValueError(f"sealed prior audit changed: {name}")
    old = json.loads((audit_dir / "provenance.json").read_text())
    if old.get("status") != "complete" or old.get("rms_build", {}).get("stores") is not None:
        raise ValueError("requires the completed unrestricted-store adopted audit")
    inputs = {name: dict(path=str(p.resolve()), sha256=_sha(p)) for name, p in dict(
        market_panel=rms_dir / "market_panel_rms.parquet",
        upc_routes=rms_dir / "diagnostics/upc_routes.parquet",
        week_sources=rms_dir / "diagnostics/week_sources.parquet",
        candidate_map=Path(candidate_map), prior_audit=audit_dir / "provenance.json",
        prior_market_table=audit_dir / "dma_quarter.parquet").items()}
    for name in ("market_panel", "upc_routes", "week_sources"):
        if inputs[name]["sha256"] != old["inputs"][name]["sha256"]:
            raise ValueError(f"adopted input changed: {name}")
    candidates = pd.read_parquet(candidate_map)
    if candidates.duplicated(["year", "upc"]).any():
        raise ValueError("candidate map has duplicate file-year/UPC")
    missing = candidates.recorded_oz.isna() & candidates.dollars.gt(0)
    conflict = candidates.both_mass_sources & candidates.source_abs_gap_oz.gt(0) & candidates.dollars.gt(0)
    focus = candidates.loc[missing | conflict, ["year", "upc"]]
    routes = pd.read_parquet(rms_dir / "diagnostics/upc_routes.parquet")
    if not pd.MultiIndex.from_frame(focus).isin(pd.MultiIndex.from_frame(routes[["year", "upc"]])).all():
        raise ValueError("focused candidate keys are absent from adopted routes")
    expected = panel_cells(pd.read_parquet(rms_dir / "market_panel_rms.parquet"))
    j45 = np.sort(expected.index.get_level_values("j").unique())
    m65 = np.sort(expected.index.get_level_values("m").unique())
    if (len(j45), len(m65)) != (45, 65):
        raise ValueError("expected 45 adopted brands and 65 DMAs")
    owners = week_owners(pd.read_parquet(rms_dir / "diagnostics/week_sources.parquet"))
    if sorted(owners) != list(range(2006, 2025)):
        raise ValueError("expected every adopted file-year 2006-2024")
    stores = {}
    inputs["archived_store_maps"] = {}
    for year in owners:
        path = audit_dir / "by_file_year" / f"stores-{year}.parquet"
        digest = _sha(path)
        if archive_hashes.get(str(path.relative_to(archive))) != digest:
            raise ValueError(f"archived producer store map changed: {year}")
        inputs["archived_store_maps"][str(year)] = dict(path=str(path), sha256=digest)
        stores[year] = pd.read_parquet(path)
    output.mkdir(parents=True)
    (output / "by_file_year").mkdir()
    meta = dict(status="running", producer=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), script_sha256=_sha(__file__),
        classifier_sha256=_sha(Path(__file__).with_name("rms_outside_audit.py")), inputs=inputs,
        jobs=jobs, batch_size=batch_size, original_gap_upc_years=int(missing.sum()),
        source_conflict_upc_years=int(conflict.sum()), focus_upc_years=len(focus),
        scope="unchanged adopted categories, stores and week ownership; candidate masses applied only after aggregation")
    (output / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    old_stats = {s["file_year"]: s for s in old["file_statistics"]}
    parts, statistics = [], []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        tasks = [pool.submit(scan_year, year, rms_dir, routes.loc[routes.year.eq(year)],
            stores[year], owners[year], j45, m65, expected.index,
            focus.loc[focus.year.eq(year), "upc"].to_numpy(), batch_size,
            old_stats[year]["movement_sha256"]) for year in sorted(owners)]
        for future in as_completed(tasks):
            detail, stats = future.result()
            year = stats["file_year"]
            detail.to_parquet(output / "by_file_year" / f"upcs-{year}.parquet", index=False)
            (output / "by_file_year" / f"stats-{year}.json").write_text(json.dumps(stats, indent=2) + "\n")
            parts.append(detail)
            statistics.append(stats)
            print(f"file {year}: {stats['input_rows']:,} input rows; {stats['selected_rows']:,} selected focus rows", flush=True)
    details = pd.concat(parts, ignore_index=True)
    baseline = pd.read_parquet(audit_dir / "dma_quarter.parquet")
    check = reconcile_uncovered(details, baseline)
    details.to_parquet(output / "upc_dma_quarter.parquet", index=False)
    meta.update(status="complete", reconciliation=check,
                file_statistics=sorted(statistics, key=lambda s: s["file_year"]))
    (output / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(check, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("rms-dir", "audit-dir", "candidate-map", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=500_000)
    run(**vars(parser.parse_args()))
