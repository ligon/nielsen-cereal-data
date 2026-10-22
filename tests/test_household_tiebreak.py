"""The household cell's panel year (aggregate_households._dominant_panel_year).

A household x quarter x brand cell at a calendar-year boundary takes
purchases from two adjacent panel-year files (NielsenIQ panel years are
Sunday-Saturday years; e.g. panel year 2015 starts 2014-12-28).  The cell
gets the panel_year/DMA with the larger spend; on an exact spend tie the old
code's unstable sort let unrelated rows decide.  The rule is now a total
order: spend (cents) desc, then panel_year == year(t), then panel_year asc,
then DMA_Cd asc.

Run with the CerealDemand venv, which has pytest:
    /global/scratch/fsa/fc_jevons/ligon/mirrors/CerealDemand/.venv/bin/python \
        -m pytest -q -p no:cacheprovider tests/test_household_tiebreak.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import aggregate_households as ah  # noqa: E402

G = ["household_code", "t", "brand_code_uc"]


def _rows(rows):
    return pd.DataFrame(rows, columns=G + ["panel_year", "DMA_Cd",
                                           "total_price_paid"])


# one row per purchase.  Cells:
#   A  Q4Y2014, tie 3.99 vs 3.99 (py2014 DMA 501 | py2015 DMA 602: a mover)
#   B  Q1Y2021, tie 2.50 vs 2.50 (py2020 runs to 2021-01-02)
#   C  Q4Y2014, py2015 outspends py2014: spend wins over the calendar year
#   D  Q4Y2009, tie reached by different purchase counts (1.00+2.99 = 3.99)
#   E  Q4Y2008, float noise: 0.1+0.2 vs 0.3 is a tie in cents
#   F  one panel year only (Q4Y2003 rows live in panel year 2004)
PURCHASES = _rows([
    (1, "Q4Y2014", 557396, 2014, 501, 3.99),
    (1, "Q4Y2014", 557396, 2015, 602, 3.99),
    (2, "Q1Y2021", 620517, 2020, 693, 2.50),
    (2, "Q1Y2021", 620517, 2021, 693, 2.50),
    (3, "Q4Y2014", 557548, 2014, 505, 2.00),
    (3, "Q4Y2014", 557548, 2015, 505, 4.00),
    (4, "Q4Y2009", 579866, 2009, 803, 1.00),
    (4, "Q4Y2009", 579866, 2009, 803, 2.99),
    (4, "Q4Y2009", 579866, 2010, 803, 3.99),
    (5, "Q4Y2008", 591688, 2008, 505, 0.30),
    (5, "Q4Y2008", 591688, 2009, 505, 0.10),
    (5, "Q4Y2008", 591688, 2009, 505, 0.20),
    (6, "Q4Y2003", 557396, 2004, 501, 5.00),
])
EXPECT = {  # household -> (panel_year, DMA_Cd)
    1: (2014, 501), 2: (2021, 693), 3: (2015, 505), 4: (2009, 803),
    5: (2008, 505), 6: (2004, 501),
}


def _got(df):
    out = ah._dominant_panel_year(df, G)
    assert not out.duplicated(G).any()
    return {r.household_code: (r.panel_year, r.DMA_Cd)
            for r in out.itertuples()}


def test_rule():
    assert _got(PURCHASES) == EXPECT


def test_invariant_to_row_order_and_unrelated_rows():
    """The defect: on a tie, the winner moved with row order (which changes
    when unrelated rows are added or dropped).  Every permutation, with or
    without filler rows from other cells, must give the same answer (checked
    against the first permutation, independently of the rule)."""
    rng = np.random.default_rng(20261022)
    filler = _rows([(100 + k, "Q4Y2014", 557396, 2014 + (k % 2), 501,
                     float(k % 7) + 0.99) for k in range(2000)])
    first = None
    for i in range(100):
        df = PURCHASES if i % 2 else pd.concat([PURCHASES, filler])
        df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        got = _got(df.reset_index(drop=True))
        got = {h: got[h] for h in EXPECT}
        first = first or got
        assert got == first
