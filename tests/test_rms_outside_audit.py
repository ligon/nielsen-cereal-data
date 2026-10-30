"""Other-cereal accounting, coverage and fixed-vintage reconciliation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import rms_outside_audit as audit


@pytest.fixture
def world():
    routes = pd.DataFrame({
        "upc": [1, 2, 3, 4, 5, 6],
        "brand_code_uc": [10, 20, 30, np.nan, 10, 20],
        "brand_descr": ["INSIDE", "OTHER", "CTL CEREAL", None, "INSIDE", "OTHER"],
        "size1_units": ["OZ", " oz ", "OZ", None, "LB", "OZ"],
        "size1_amount": [10., 4., 2., np.nan, 2., 0.]})
    stores = pd.DataFrame({"store_code_uc": [1, 2], "DMA_Cd": [501, 999]})
    rows = pd.DataFrame({
        "upc": [1, 2, 3, 4, 5, 6, 1, 1, 1],
        "store_code_uc": [1, 1, 1, 1, 1, 1, 2, 1, 1],
        "week_end": pd.to_datetime(["2020-03-28"] * 7 + ["2020-03-21", "2020-03-28"]),
        "units": [2., 3., 1., 1., 1., 1., 1., 1., -1.],
        "price": [5., 4., 2., 7., 3., 1., 99., 88., 5.],
        "prmult": [1., 2., 1., 1., 1., 1., 1., 1., 1.]})
    expected = pd.DataFrame({"revenue": [10.], "servings": [20.]},
        index=pd.MultiIndex.from_tuples([(20201, 501, 10)], names=["tcode", "m", "j"]))
    keep = np.array(["2020-03-28"], dtype="datetime64[D]")
    return rows, routes, stores, keep, expected


def evaluate(world, rows=None):
    full, routes, stores, keep, expected = world
    return audit.aggregate_batch(full if rows is None else rows, routes, stores,
                                  keep, [10], [501], expected.index)


def test_residual_categories_use_same_stores_and_hold_incomplete_price(world):
    groups, candidates, stats = evaluate(world)
    assert stats["selected_rows"] == 6
    assert stats["dollars_unselected_dma"] == 99.
    assert stats["dollars_duplicate_week"] == 88.
    assert stats["missing_route_selected_rows"] == 0
    assert groups.xs("private_label", level="category").revenue.sum() == 2.
    assert groups.xs("unmapped", level="category").revenue.sum() == 7.
    expected = world[-1]
    assert audit.reconcile(candidates, expected)["cells"] == 1
    table = audit.residual_table(groups, expected)
    row = table.iloc[0]
    assert row.module_revenue == 29.
    assert row.modeled_revenue == 10.
    assert row.residual_revenue == 19.
    assert row.residual_covered_revenue == 8.
    assert row.residual_covered_ounces == 14.
    assert row.residual_uncovered_rows == 3
    assert row.selected_brand_excluded_revenue == 3.
    assert row.recorded_ounce_revenue_coverage == pytest.approx(8/19)
    assert row.covered_subset_dollars_per_ounce == pytest.approx(8/14)
    assert np.isnan(row.complete_residual_dollars_per_ounce)


def test_complete_price_only_when_exact_residual_rows_have_sizes(world):
    groups, _cells, _stats = evaluate(world, world[0].iloc[:3])
    row = audit.residual_table(groups, world[-1]).iloc[0]
    assert row.residual_revenue == 8.
    assert row.recorded_ounce_revenue_coverage == 1.
    assert row.complete_residual_dollars_per_ounce == pytest.approx(8/14)


def test_market_without_residual_does_not_invent_an_outside_price(world):
    groups, _cells, _stats = evaluate(world, world[0].iloc[:1])
    row = audit.residual_table(groups, world[-1]).iloc[0]
    assert row.residual_revenue == 0.
    assert np.isnan(row.recorded_ounce_revenue_coverage)
    assert np.isnan(row.complete_residual_dollars_per_ounce)


def test_positive_revenue_with_tiny_missing_size_still_withholds_price(world):
    rows = world[0].iloc[:4].copy()
    rows.loc[3, "price"] = 1e-20
    groups, _cells, _stats = evaluate(world, rows)
    row = audit.residual_table(groups, world[-1]).iloc[0]
    # Floating revenue coverage can round to 1, but missing-row accounting cannot.
    assert row.residual_uncovered_rows == 1
    assert np.isnan(row.complete_residual_dollars_per_ounce)


def test_streaming_batch_splits_do_not_change_accounting(world):
    groups, candidates, _stats = evaluate(world)
    parts = [evaluate(world, world[0].iloc[s]) for s in (slice(0, 2), slice(2, 6), slice(6, None))]
    pd.testing.assert_frame_equal(groups.sort_index(), audit.combine(
        [p[0] for p in parts], ["tcode", "m", "category"]).sort_index())
    pd.testing.assert_frame_equal(candidates.sort_index(), audit.combine(
        [p[1] for p in parts], ["tcode", "m", "j"]).sort_index())


def test_detail_callback_inherits_filters_and_preserves_original_results(world):
    rows, routes, stores, keep, expected = world
    frames = []
    got = audit.aggregate_batch(rows, routes, stores, keep, [10], [501],
                                expected.index, detail_callback=frames.append)
    reference = evaluate(world)
    for a, b in zip(got[:2], reference[:2]):
        pd.testing.assert_frame_equal(a, b)
    assert got[2] == reference[2]
    detail = frames[0]
    assert len(detail) == 6 and detail.units.sum() == 9.
    pd.testing.assert_frame_equal(detail.drop(columns=["upc", "units"]).groupby(
        ["tcode", "m", "category"]).sum(), got[0].sort_index())
    def poison(frame):
        frame["revenue"] = -100.
    untouched = audit.aggregate_batch(rows, routes, stores, keep, [10], [501],
                                     expected.index, detail_callback=poison)
    pd.testing.assert_frame_equal(untouched[0], reference[0])


def test_focused_market_reconciliation_catches_dropped_unknown_rows(world):
    from rms_size_market_audit import reconcile_uncovered
    frames = []
    rows, routes, stores, keep, expected = world
    groups, _, _ = audit.aggregate_batch(rows, routes, stores, keep, [10], [501],
                                        expected.index, detail_callback=frames.append)
    detail = frames[0]
    detail["uncovered_revenue"] = detail.revenue - detail.covered_revenue
    baseline = audit.residual_table(groups, expected)
    assert reconcile_uncovered(detail, baseline)["uncovered_rows"] == 3
    with pytest.raises(AssertionError):
        reconcile_uncovered(detail.loc[detail.upc.ne(4)], baseline)
    wrong = detail.copy()
    wrong.loc[wrong.upc.eq(4), "uncovered_revenue"] += 1.
    with pytest.raises(ValueError, match="revenue"):
        reconcile_uncovered(wrong, baseline)


def test_focused_scan_keeps_file_year_separate_from_calendar_year(world, tmp_path):
    from rms_size_market_audit import scan_year
    from crosswalk_eval import _sha
    rows, routes, stores, keep, expected = world
    rows = pd.concat([rows, rows.iloc[[3]].assign(week_end=pd.Timestamp("2021-01-02")),
                      rows.iloc[[3]].assign(week_end=pd.Timestamp("2025-01-04"))])
    path = tmp_path / "2020" / "movement_1344.parquet"
    path.parent.mkdir()
    rows.to_parquet(path, index=False)
    keep = np.r_[keep, np.array(["2021-01-02", "2025-01-04"], dtype="datetime64[D]")]
    detail, stats = scan_year(2020, tmp_path, routes, stores, keep, [10], [501],
                             expected.index, np.array([2, 4]), 3, _sha(path))
    assert stats["selected_rows"] == 3
    assert set(detail.file_year) == {2020}
    assert set(detail.tcode) == {20201, 20211}
    assert detail.units.sum() == 5.


def test_reconciliation_rejects_missing_cells_and_changed_revenue(world):
    _groups, candidates, _stats = evaluate(world)
    altered = candidates.copy()
    altered.iloc[0, altered.columns.get_loc("revenue")] += .01
    with pytest.raises(ValueError, match="modeled revenue"):
        audit.reconcile(altered, world[-1])
    with pytest.raises(ValueError, match="cell keys"):
        audit.reconcile(candidates.iloc[:0], world[-1])


def test_zero_ounce_selected_cell_is_residual_filter_exclusion(world):
    rows, routes, stores, keep, expected = world
    routes = routes.copy()
    routes.loc[routes.upc == 1, "size1_amount"] = 0.
    empty = expected.iloc[:0]
    groups, candidates, _stats = audit.aggregate_batch(
        rows.iloc[:1], routes, stores, keep, [10], [501], empty.index)
    assert audit.reconcile(candidates, empty)["producer_bad_price_cells"] == 1
    row = audit.residual_table(groups, empty).iloc[0]
    assert row.residual_revenue == row.selected_brand_excluded_revenue == 10.
    assert row.residual_uncovered_rows == 1


def test_unarchived_upc_is_visible_and_never_dropped_from_module_total(world):
    rows = world[0].iloc[:1].copy()
    rows["upc"] = 999
    groups, _cells, stats = evaluate(world, rows)
    assert stats["missing_route_selected_rows"] == 1
    assert groups.xs("unmapped", level="category").revenue.sum() == 10.


def test_positive_nonfinite_input_is_flagged(world):
    rows = world[0].iloc[:1].copy()
    rows["price"] = np.inf
    _groups, _cells, stats = evaluate(world, rows)
    assert stats["positive_nonfinite_rows"] == 1
    assert stats["selected_rows"] == 0


def test_saved_week_ownership_rejects_duplicate_kept_week():
    weeks = pd.DataFrame({"file_year": [2019, 2020],
                          "week_end": ["2020-01-04"] * 2, "kept": [False, True]})
    owners = audit.week_owners(weeks)
    assert list(owners) == [2020]
    weeks["kept"] = True
    with pytest.raises(ValueError, match="multiple file owners"):
        audit.week_owners(weeks)


def test_quarter_uses_week_end_and_panel_conversion_matches(world):
    panel = pd.DataFrame({"t": ["Q1Y2020"], "m": [501], "j": [10],
                          "revenue": [10.], "servings": [20.]}).set_index(["t", "m", "j"])
    pd.testing.assert_frame_equal(audit.panel_cells(panel), world[-1])


def test_duplicate_route_key_fails_before_join_multiplication(world):
    rows, routes, stores, keep, expected = world
    with pytest.raises(ValueError, match="unique keys"):
        audit.aggregate_batch(rows, pd.concat([routes, routes.iloc[:1]]),
                              stores, keep, [10], [501], expected.index)
