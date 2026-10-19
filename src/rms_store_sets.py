"""
RMS store presence by year: which stores sell module-1344 cereal each year.

The RMS store universe is not fixed: 33.9k stores report cereal in 2017,
46.1k in 2018, 51.2k in 2021 (docs/rms-d2-log.org).  A balanced store set
(stores with cereal movement in EVERY year of a range) removes that churn,
for the theta-gap decomposition (CerealDemand src/rms_theta_gap.py).

The stores_<yr> tables list a near-static universe (38,285 stores in every
year 2006-2010), so presence is read from the movement itself: a store is
present in a year if it has at least one module-1344 movement row there with
prmult > 0 and price > 0.  (This is D3's `ok` filter as it stood when the
balanced set was built, before D3 also required units > 0.  Zero-unit rows number 2 in 2020, 408 in 2021 and 16,548 in
2024; they carry no dollars.)

Writes (group-only):
  OUT/diagnostics/stores_by_year.parquet   year, store_code_uc, rows, dollars
  OUT/store_sets/balanced_<lo>-<hi>.parquet store_code_uc (present every year)

Usage:  python src/rms_store_sets.py [--years 2006 2024] [--jobs 4]
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

os.umask(0o027)

DATA = Path(os.environ.get("NIELSEN_DATA",
                           "/global/scratch/fsa/fc_jevons/ligon/Data"))
OUT = Path(os.environ.get("RMS_OUT", DATA / "NielsenScanner" / "rms"))
BATCH = 8_000_000


def year_stores(y):
    t0 = time.time()
    pf = pq.ParquetFile(OUT / str(y) / "movement_1344.parquet")
    parts = []
    for b in pf.iter_batches(batch_size=BATCH,
                             columns=["store_code_uc", "units", "prmult", "price"]):
        store = b.column("store_code_uc").to_numpy()
        units = b.column("units").to_numpy().astype("float64")
        prmult = b.column("prmult").to_numpy()
        price = b.column("price").to_numpy()
        ok = (prmult > 0) & (price > 0)
        rev = np.where(ok, units * price / np.where(prmult > 0, prmult, 1), 0.0)
        parts.append(pd.DataFrame({"store_code_uc": store[ok], "rows": 1,
                                   "dollars": rev[ok]})
                     .groupby("store_code_uc").sum())
    s = pd.concat(parts).groupby(level=0).sum().reset_index()
    s.insert(0, "year", y)
    print(f"  {y}: {len(s):,} stores ({time.time() - t0:.0f} s)", flush=True)
    return s


def main():
    ap = argparse.ArgumentParser(description="RMS store presence by year")
    ap.add_argument("--years", nargs=2, type=int, default=[2006, 2024])
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 4)))
    a = ap.parse_args()
    lo, hi = a.years
    years = list(range(lo, hi + 1))
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        by_year = pd.concat(list(ex.map(year_stores, years)), ignore_index=True)
    (OUT / "diagnostics").mkdir(exist_ok=True)
    by_year.to_parquet(OUT / "diagnostics" / "stores_by_year.parquet", index=False)

    n_years = by_year.groupby("store_code_uc")["year"].nunique()
    balanced = n_years[n_years == len(years)].index
    (OUT / "store_sets").mkdir(exist_ok=True)
    out = OUT / "store_sets" / f"balanced_{lo}-{hi}.parquet"
    pd.DataFrame({"store_code_uc": balanced}).to_parquet(out, index=False)

    share = (by_year[by_year["store_code_uc"].isin(balanced)]
             .groupby("year")["dollars"].sum()
             / by_year.groupby("year")["dollars"].sum())
    print(f"Balanced {lo}-{hi}: {len(balanced):,} stores -> {out}")
    print("Share of each year's module-1344 dollars in the balanced set:")
    print(share.round(3).to_string())


if __name__ == "__main__":
    main()
