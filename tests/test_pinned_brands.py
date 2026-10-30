"""charter-crosswalk A3: the 45 are pinned (Q3, EL 2026-10-19).

config/brands_45.csv is the selection; aggregate_markets._select_top_brands
(AM, and aggregate_households.py through it) returns it instead of
re-ranking and fails hard when a pinned brand is missing.  The market
panel's j set equals the list (today's file now; PANEL_CLEAN may point at a
new vintage).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import aggregate_markets as am  # noqa: E402
import brand_route as br  # noqa: E402

# charter-crosswalk Appendix A: the current 45 (j)
APPENDIX_A = [557396, 557426, 557434, 557439, 557442, 557461, 557491, 557501,
              557517, 557523, 557548, 557549, 557579, 557586, 557587, 579277,
              579824, 579866, 579870, 579877, 579905, 579916, 579923, 579996,
              579998, 580002, 580025, 580031, 580036, 580037, 591688, 591702,
              591705, 620458, 620469, 620481, 620512, 620517, 620537, 620554,
              623809, 623816, 623841, 623870, 623878]
PANEL_CLEAN = Path(os.environ.get(
    "PANEL_CLEAN",
    "/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean"))


def _purchases(codes):
    return pd.DataFrame({"brand_code_uc": np.repeat(codes, 2).astype(float),
                         "total_price_paid": 1.0})


def test_list_is_appendix_a():
    b = br.load_brands_45()
    assert sorted(b["j"]) == sorted(APPENDIX_A)
    assert 536746 not in set(b["j"])          # CTL BR (ledger G1)
    assert 591705 in set(b["j"])              # M-O-M GOLDEN PUFFS
    assert b["brand_descr"].notna().all() and b["brand_descr"].is_unique


def test_select_returns_the_pin_not_a_ranking():
    # 536746 outspends everything here; a re-ranking would pick it.
    p = pd.concat([_purchases(APPENDIX_A),
                   pd.DataFrame({"brand_code_uc": [536746.0] * 50,
                                 "total_price_paid": 1e6})])
    got = am._select_top_brands(p, 45, None)
    assert sorted(got) == sorted(APPENDIX_A)


def test_select_fails_on_missing_pinned_brand():
    with pytest.raises(ValueError, match="no purchases"):
        am._select_top_brands(_purchases(APPENDIX_A[1:]), 45, None)


def test_select_fails_on_n_products_mismatch():
    with pytest.raises(ValueError, match="n_products"):
        am._select_top_brands(_purchases(APPENDIX_A), 40, None)


def test_households_use_the_same_selection():
    import aggregate_households as ah
    assert ah._select_top_brands is am._select_top_brands
    assert ah._apply_brand_route is am._apply_brand_route


@pytest.mark.skipif(not (PANEL_CLEAN / "market_panel.parquet").exists(),
                    reason="market panel not available")
def test_market_panel_j_set_and_descr_equal_the_list():
    mp = pd.read_parquet(PANEL_CLEAN / "market_panel.parquet",
                         columns=["brand_descr"]).reset_index()
    assert set(mp["j"].unique()) == set(APPENDIX_A)
    d = mp.groupby("j")["brand_descr"].first()
    pin = am._pinned_brand_descr().set_index("brand_code_uc")["brand_descr"]
    assert (d.reindex(pin.index) == pin).all()
