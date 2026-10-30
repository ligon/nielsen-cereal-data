"""Actual movement dollars and honest unknown-mass/interval accounting."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rms_size_exposure_summary import market_scenario, scenario_masses, SCENARIOS


def world():
    baseline = pd.DataFrame(dict(residual_revenue=[100.], residual_covered_revenue=[70.],
        residual_covered_ounces=[10.], selected_brand_excluded_revenue=[0.]),
        index=pd.MultiIndex.from_tuples([(20161, 501)], names=["tcode", "m"]))
    details = pd.DataFrame(dict(file_year=[2016, 2016], upc=[1, 2], tcode=[20161]*2,
        m=[501]*2, category=["unmapped"]*2, units=[2., 1.], revenue=[20., 10.],
        rows=[1, 1], uncovered_rows=[1, 1]))
    candidates = pd.DataFrame(dict(year=[2016, 2016], upc=[1, 2],
        followup_mass_oz=[5., np.nan], followup_mass_source=["recorded_mass_conversion", "unresolved"],
        size_origin=["recorded_mass_conversion", "unresolved"], candidate_oz=[5., np.nan],
        catalog_mass_min_oz=[np.nan, 1.], catalog_mass_max_oz=[np.nan, 9.],
        catalog_versions=[np.nan, 2.], source_abs_gap_oz=[np.nan, np.nan],
        dollars=[9999., 8888.], rows=[1000, 2000], units=[10000., 20000.]))
    return details, baseline, candidates


def test_actual_universe_price_interval_and_inventory_columns_do_not_leak():
    details, baseline, candidates = world()
    findings = dict(fallback_screen=dict(future_descriptor_start_dates=0, flagged_rows=[]))
    candidates = scenario_masses(candidates, findings)
    table, summary = market_scenario(details, baseline, candidates, SCENARIOS[0])
    row = table.iloc[0]
    assert row.covered_ounces == 20.
    assert row.remaining_revenue == 10.
    assert np.isnan(row.conditional_point_price)
    assert row.catalog_conditional_price_lower == pytest.approx(100/29)
    assert row.catalog_conditional_price_upper == pytest.approx(100/21)
    assert summary["remaining_outside_upc_years"] == 1
    assert summary["unowned_catalog_outside_revenue"] == 10.


def test_count_only_unknown_mass_prevents_a_finite_price_interval():
    details, baseline, candidates = world()
    candidates[SCENARIOS[0]] = candidates.followup_mass_oz
    candidates.loc[1, ["catalog_mass_min_oz", "catalog_mass_max_oz"]] = np.nan
    table, _ = market_scenario(details, baseline, candidates, SCENARIOS[0])
    assert table.catalog_conditional_price_lower.isna().all()
    assert table.conditional_point_price.isna().all()


def test_explicit_pack_is_a_separate_scenario_and_secondary_count_is_withheld():
    _, _, candidates = world()
    candidates["followup_mass_oz"] = [3.9, 7.2]
    candidates["candidate_oz"] = [3.9, 7.2]
    candidates["size_origin"] = "same_year_TOTAL_SIZE_candidate"
    candidates["followup_mass_source"] = "same_year_TOTAL_SIZE_candidate"
    flags = [dict(year=2016, upc=1, candidate_oz=3.9, desc_multi="6", descriptor_package_oz=23.4),
             dict(year=2016, upc=2, candidate_oz=7.2, desc_multi="1", descriptor_package_oz=None)]
    findings = dict(fallback_screen=dict(future_descriptor_start_dates=0, flagged_rows=flags))
    out = scenario_masses(candidates, findings)
    pd.testing.assert_frame_equal(out[candidates.columns], candidates, check_exact=True)
    assert out[SCENARIOS[0]].isna().all()
    assert out[SCENARIOS[1]].tolist() == [3.9, 7.2]
    assert out[SCENARIOS[2]].iloc[0] == 23.4
    assert np.isnan(out[SCENARIOS[2]].iloc[1])
    findings["fallback_screen"]["future_descriptor_start_dates"] = 1
    with pytest.raises(ValueError, match="temporally"):
        scenario_masses(candidates, findings)


def test_missing_focused_revenue_cannot_be_reported_as_complete():
    details, baseline, candidates = world()
    candidates[SCENARIOS[0]] = candidates.followup_mass_oz
    with pytest.raises(ValueError, match="residual revenue universe"):
        market_scenario(details.iloc[:1], baseline, candidates, SCENARIOS[0])
