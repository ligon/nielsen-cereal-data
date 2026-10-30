"""Catalog consensus recovers mass without choosing a movement version/brand."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rms_size_consensus import consensus, extend_candidates


def pairs():
    return pd.DataFrame(dict(year=[2016]*6, upc=[1, 1, 2, 2, 3, 3],
        upc_ver_uc=[1, 2]*3, size1_amount=[13., 13., 26.8, 26.8, .625, 32.7],
        size1_units=["OZ"]*6, brand_code_uc=[10, 20, 30, 40, 50, 50]))


def test_unanimous_mass_and_brand_routing_are_separate():
    out = consensus(pairs()).set_index("upc")
    assert out.loc[1, "catalog_mass_oz"] == 13.
    assert out.loc[2, "catalog_mass_oz"] == 26.8
    assert out.loc[1, "catalog_brand_count"] == 2
    assert not out.movement_version_owned.any()
    assert "brand_code_uc" not in out
    assert out.loc[3, "catalog_mass_status"] == "conflict"
    assert np.isnan(out.loc[3, "catalog_mass_oz"])
    assert out.loc[3, "catalog_mass_min_oz"] == .625
    assert out.loc[3, "catalog_mass_max_oz"] == 32.7


def test_missing_candidate_mass_cannot_be_silently_dropped():
    frame = pairs().iloc[:2].copy()
    frame.loc[1, "size1_amount"] = np.nan
    row = consensus(frame).iloc[0]
    assert row.catalog_mass_status == "incomplete"
    assert np.isnan(row.catalog_mass_oz) and np.isnan(row.catalog_mass_min_oz)


def test_distinct_positive_masses_are_not_rounded_to_agreement():
    frame = pairs().iloc[:2].copy()
    frame.loc[1, "size1_amount"] += 1e-10
    assert consensus(frame).iloc[0].catalog_mass_status == "conflict"


def test_duplicate_catalog_version_is_an_error():
    with pytest.raises(ValueError, match="duplicate"):
        consensus(pd.concat([pairs(), pairs().iloc[:1]]))


def test_extension_preserves_original_primary_and_brand_columns():
    original = pd.DataFrame(dict(year=[2016]*4, upc=[1, 2, 3, 4],
        candidate_oz=[7., np.nan, np.nan, np.nan], size_origin=["recorded_OZ"]+['unresolved']*3,
        brand_code_uc=[100., np.nan, np.nan, np.nan]))
    out = extend_candidates(original, consensus(pairs()))
    pd.testing.assert_frame_equal(out[original.columns], original, check_exact=True)
    assert out.followup_mass_oz.iloc[:2].tolist() == [7., 26.8]
    assert np.isnan(out.followup_mass_oz.iloc[2]) and np.isnan(out.followup_mass_oz.iloc[3])
    assert out.followup_mass_min_oz.iloc[0] == 7.
    assert out.followup_mass_source.iloc[1] == "same_year_catalog_mass_consensus"
