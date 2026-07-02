"""Build ``config/cpi_us_quarterly.csv`` --- the US all-items CPI deflator.

Quarterly, rebased so 2014Q1 = 1.0 (the ``config/default.yaml`` ``cpi_base``),
for price deflation of the nominal nielsen prices.

Source: FRED series ``USACPALTT01IXNBQ`` --- OECD "Consumer Price Index: All
Items: Total for the United States", index, quarterly, not seasonally adjusted.
This is the exact series the project's legacy (Nevo) pipeline used.  The OECD MEI
was discontinued, so the series ends 2023Q4; the four 2024 quarters need a
current series spliced in (e.g. FRED ``CPIAUCNS`` rebased at the 2023/2024 seam)
before a *full-panel* level-price BLP is run.  The 2014-2019 BLP window and the
2004-2023 span are fully covered.

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
_BASE = "Q1Y2014"
_START_YEAR = 2004
_MONTH_TO_Q = {1: 1, 4: 2, 7: 3, 10: 4}
_OUT = os.path.join(os.path.dirname(__file__), "..", "config", "cpi_us_quarterly.csv")


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
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    out.to_csv(_OUT, index=False)
    return out


if __name__ == "__main__":
    df = build()
    print(df.head(3).to_string(index=False))
    print("  ...")
    print(df.tail(3).to_string(index=False))
    print(f"\nwrote {os.path.relpath(_OUT)}  ({len(df)} quarters, base {_BASE}=1.0)")
