"""The combined-vintage comparison permits only the household fixes."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from crosswalk_hhfix_compare import compare


def _panel(j, panel_year, m, share):
    idx = pd.MultiIndex.from_tuples([(1, "Q4Y2014", j)], names=["i", "t", "j"])
    return pd.DataFrame({"panel_year": [panel_year], "m": [m],
                         "S": [share], "servings": [2.0]}, index=idx)


def _fixtures(tmp_path):
    dirs = [tmp_path / name for name in ("accepted", "xw1", "xw2", "rms1", "rms2")]
    for d in dirs:
        d.mkdir()
    accepted, xw1, xw2, rms1, rms2 = dirs
    _panel(536746, 2015, 602, 0.1).to_parquet(accepted / "household_panel.parquet")
    _panel(557579, 2015, 602, 0.1).to_parquet(xw1 / "household_panel.parquet")
    _panel(557579, 2014, 501, 0.2).to_parquet(xw2 / "household_panel.parquet")
    pd.DataFrame({"S_base": [0.1]}).to_parquet(xw1 / "household_base_shares.parquet")
    pd.DataFrame({"S_base": [0.1]}).to_parquet(xw2 / "household_base_shares.parquet")
    for name in ("market_panel.parquet", "demographics_market.parquet"):
        pd.DataFrame({"value": [1]}).to_parquet(xw1 / name)
        pd.DataFrame({"value": [1]}).to_parquet(xw2 / name)
    pd.DataFrame({"value": [1]}).to_parquet(rms1 / "market_panel_rms.parquet")
    pd.DataFrame({"value": [1]}).to_parquet(rms2 / "market_panel_rms.parquet")
    pd.DataFrame({"income": [3]}).to_parquet(xw1 / "demographics_household.parquet")
    pd.DataFrame({"income": [3], "household_size": pd.Series([4], dtype="int64")}
                 ).to_parquet(xw2 / "demographics_household.parquet")
    return dirs


def test_compare_reports_accepted_and_isolates_household_changes(tmp_path):
    dirs = _fixtures(tmp_path)
    report = compare(*(str(d) for d in dirs))
    assert "Accepted-only brands: [536746]" in report
    assert "xw2-only brands: [557579]" in report
    assert "household_size non-null rows | 0 | 0 | 1" in report
    assert "=panel_year= | 0 | 1" in report


def test_compare_rejects_non_household_changes(tmp_path):
    dirs = _fixtures(tmp_path)
    xw2 = dirs[2]
    pd.DataFrame({"value": [2]}).to_parquet(xw2 / "market_panel.parquet")
    with pytest.raises(AssertionError, match="market_panel"):
        compare(*dirs)
