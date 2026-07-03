"""Build ``config/cpi_us_quarterly.csv`` --- the US all-items CPI deflator.

Quarterly, rebased so 2014Q1 = 1.0 (the ``config/default.yaml`` ``cpi_base``),
for price deflation of the nominal nielsen prices.

Source: FRED series ``USACPALTT01IXNBQ`` --- OECD "Consumer Price Index: All
Items: Total for the United States", index, quarterly, not seasonally adjusted.
This is the exact series the project's legacy (Nevo) pipeline used.  The OECD
MEI was discontinued at 2023Q4, so quarters past it are SPLICED IN from FRED
``CPIAUCNS`` (monthly all-items NSA, quarterly-averaged over complete
quarters, rebased at the 2023/2024 seam) --- the splice this docstring used to
prescribe, implemented in :func:`_splice_current` once the full-panel
level-price models (blp-ces) actually demanded 2024 deflators.

Deflation only bites for level-price BLP (see K-Aggregators / CerealDemand
DESIGN.org): for the log-price K-aggregator models it is a market x time FE
no-op, so this deflator exists for ``estimate_blp``.

    python src/build_cpi.py            # fetch FRED -> config/cpi_us_quarterly.csv
"""
from __future__ import annotations

import io
import os
import urllib.request

import pandas as pd

_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USACPALTT01IXNBQ"
# The current monthly series used to extend past the discontinued OECD one.
_SPLICE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS"
_BASE = "Q1Y2014"
_START_YEAR = 2004
_MONTH_TO_Q = {1: 1, 4: 2, 7: 3, 10: 4}
_OUT = os.path.join(os.path.dirname(__file__), "..", "config", "cpi_us_quarterly.csv")


def _splice_current(out: pd.DataFrame, url: str = _SPLICE_URL) -> pd.DataFrame:
    """Extend past the discontinued OECD series with CPIAUCNS quarterly means.

    Quarterly averages of the monthly all-items NSA index, restricted to
    COMPLETE quarters (3 observed months), rescaled so the spliced series
    matches the OECD-based deflator at the last common quarter (the
    2023/2024 seam).  The ``Q1Y2014 = 1.0`` base is untouched (the seam
    rescaling is applied only to appended quarters).
    """
    with urllib.request.urlopen(url, timeout=60) as resp:
        m = pd.read_csv(io.BytesIO(resp.read()))
    m.columns = ["date", "index"]
    m["date"] = pd.to_datetime(m["date"])
    m = m.dropna(subset=["index"])
    m["year"] = m["date"].dt.year
    m["q"] = m["date"].dt.quarter
    qa = m.groupby(["year", "q"]).agg(
        index=("index", "mean"), months=("index", "size")).reset_index()
    qa = qa[qa["months"] == 3]                      # complete quarters only
    qa["t"] = "Q" + qa["q"].astype(str) + "Y" + qa["year"].astype(str)
    seam = out["t"].iloc[-1]                        # last OECD quarter
    if seam not in set(qa["t"]):
        raise ValueError(f"seam quarter {seam} missing from the splice series")
    scale = (float(out.loc[out["t"] == seam, "cpi"].iloc[0])
             / float(qa.loc[qa["t"] == seam, "index"].iloc[0]))
    ext = qa[(~qa["t"].isin(set(out["t"])))
             & (qa["year"] >= int(seam.split("Y")[1]))].copy()
    ext["cpi"] = ext["index"] * scale
    return pd.concat([out, ext[["t", "cpi"]]], ignore_index=True)


def build(url: str = _FRED_URL, start_year: int = _START_YEAR) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = pd.read_csv(io.BytesIO(resp.read()))
    raw.columns = ["date", "index"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["q"] = raw["date"].dt.month.map(_MONTH_TO_Q)
    raw = raw.dropna(subset=["q", "index"])
    raw["t"] = ("Q" + raw["q"].astype(int).astype(str)
                + "Y" + raw["date"].dt.year.astype(str))
    base = raw.loc[raw["t"] == _BASE, "index"]
    if base.empty:
        raise ValueError(f"base quarter {_BASE} not in the FRED series")
    raw["cpi"] = raw["index"] / float(base.iloc[0])
    out = (raw[raw["date"].dt.year >= start_year][["t", "cpi"]]
           .reset_index(drop=True))
    out = _splice_current(out)
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    out.to_csv(_OUT, index=False)
    return out


if __name__ == "__main__":
    df = build()
    print(df.head(3).to_string(index=False))
    print("  ...")
    print(df.tail(3).to_string(index=False))
    print(f"\nwrote {os.path.relpath(_OUT)}  ({len(df)} quarters, base {_BASE}=1.0)")
