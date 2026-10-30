"""Explicit mass units and same-year candidate-source ownership."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rms_size_candidates import candidate_map, mass_ounces, total_size_ounces


@pytest.mark.parametrize("amount,unit,expected", [
    (12, "OZ", 12), (.5, "POUND", 8), (28.349523125, "GRAM", 1),
    (2, " ounce ", 2), (453.59237, "GRAM", 16)])
def test_explicit_dimensional_mass_conversion(amount, unit, expected):
    assert mass_ounces(amount, unit) == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("value", [None, "", "12 COUNT", "12 CT", "0 OUNCE",
    "-2 OUNCE", "10 TO 12 OUNCE", "6 X 2 OUNCE", "12", "inf OUNCE", "12 FLUID OUNCE",
    ". OUNCE", "..45 OUNCE"])
def test_no_count_range_expression_or_unknown_mass_inference(value):
    assert np.isnan(total_size_ounces(value))


def test_total_size_single_mass_and_invalid_primary():
    assert total_size_ounces(" 12.5 OUNCE ") == 12.5
    assert total_size_ounces("1 POUND") == 16
    for amount in (None, 0, -1, np.nan, np.inf, True):
        assert np.isnan(mass_ounces(amount, "OZ"))


@pytest.mark.parametrize("text,expected", [(".45 OUNCE", .45), ("1. OZ", 1.)])
def test_explicit_leading_and_trailing_decimal_mass(text, expected):
    assert total_size_ounces(text) == expected


def test_primary_precedence_conflicts_and_same_year_join():
    routes = pd.DataFrame(dict(year=[2021, 2021, 2021, 2020], upc=[1, 2, 3, 3],
        size1_amount=[12., .5, 6., 6.], size1_units=["OZ", "POUND", "CT", "CT"]))
    attrs = pd.DataFrame(dict(year=[2021]*3, upc=[1, 2, 3],
                              TOTAL_SIZE=["13 OUNCE", "8 OUNCE", "9 OUNCE"]))
    result = candidate_map(routes, attrs)
    assert result.candidate_oz.iloc[:3].tolist() == [12., 8., 9.]
    assert result.size_origin.iloc[:3].tolist() == [
        "recorded_OZ", "recorded_mass_conversion", "same_year_TOTAL_SIZE_candidate"]
    assert result.source_abs_gap_oz.iloc[0] == 1.
    assert result.size1_amount.tolist() == routes.size1_amount.tolist()
    assert result.size1_units.tolist() == routes.size1_units.tolist()
    assert np.isnan(result.candidate_oz.iloc[3])  # no future-year borrowing
    with pytest.raises(ValueError, match="duplicate attribute"):
        candidate_map(routes, pd.concat([attrs, attrs.iloc[:1]]))
    with pytest.raises(ValueError, match="duplicate route"):
        candidate_map(pd.concat([routes, routes.iloc[:1]]), attrs)
