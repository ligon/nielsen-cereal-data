"""D4 section 7 (src/rms_validate.py sec_crosswalk) counts j's merge set as j
on a build that used the shared brand route (charter-crosswalk A7; Q2 merge
sets 620485 -> 620554 Great Grains, 747763 -> 557579 Trix), and reports an
accepted name-route build exactly as before.  Synthetic frames: no Nielsen
data is read.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import rms_validate as rv  # noqa: E402

TRIX, TRIX_NEW, GG, GG_OLD, CHEERIOS = 557579, 747763, 620554, 620485, 557396


def _frames(shared_route: bool):
    p = pd.DataFrame({"t": "Q1Y2022", "m": 501, "year": 2022,
                      "j": [TRIX, GG, CHEERIOS],
                      "brand_descr": ["G M TRIX", "POST SELECTS GREAT GRAINS",
                                      "G M CHEERIOS"]})
    r = p.assign(revenue=1.0)
    # one $1M UPC each; brand_code_uc is what the build assigned (merged on
    # the shared route), master_code the UPC's own legacy code (unmerged)
    routes = pd.DataFrame({
        "year": 2022, "upc": [1, 2, 3, 4], "route": "hms_same_year",
        "brand_code_uc": [float(TRIX), float(TRIX), float(GG), float(CHEERIOS)],
        "master_code": [float(TRIX), float(TRIX_NEW), float(GG_OLD), float(CHEERIOS)],
        "dollars": 1e6})
    if shared_route:
        routes["j_route"] = "legacy_master"
    return p, r, routes


TAGS = {"LOST", "CREATED", "shrunk", "grown"}


def _flagged(lines):
    """(j, tag) of the brand-year table rows (not the agreement table's)."""
    cells = [[c.strip() for c in ln.split("|")] for ln in lines
             if ln.startswith("| 2022 |")]
    return {(int(c[2]), c[4]) for c in cells if len(c) > 4 and c[4] in TAGS}


@pytest.fixture(autouse=True)
def _fresh_flags():
    rv.FLAGS.clear()
    yield
    rv.FLAGS.clear()


def test_shared_route_counts_merge_sets_as_j():
    lines = rv.sec_crosswalk(*_frames(shared_route=True))
    assert _flagged(lines) == set()
    assert "None." in lines
    assert "xwalk" not in rv.FLAGS
    assert rv.FLAGS["shared_route"] is True
    assert rv.FLAGS["crosswalk_changes"] == []
    assert rv.FLAGS["legacy_route_mismatches"] == []


def test_name_route_build_reported_as_before():
    # no j_route column: an accepted build; master codes are not merged, so
    # the merges show up exactly as the accepted report would show them
    lines = rv.sec_crosswalk(*_frames(shared_route=False))
    assert _flagged(lines) == {(TRIX, "grown"), (GG, "CREATED")}


def test_shared_route_detects_misrouted_legacy_upc_and_section_seven_change():
    p, r, routes = _frames(shared_route=True)
    routes.loc[routes["upc"] == 3, "brand_code_uc"] = CHEERIOS
    lines = rv.sec_crosswalk(p, r, routes)
    assert (2022, 3) in rv.FLAGS["legacy_route_mismatches"]
    assert (GG, "LOST") in _flagged(lines)
    assert (2022, GG, "LOST") in rv.FLAGS["crosswalk_changes"]
    assert rv.unallowed_crosswalk_changes(rv.FLAGS["crosswalk_changes"],
                                          pd.DataFrame({
                                              "criterion": [], "source": [],
                                              "j": [], "year_lo": [],
                                              "year_hi": [], "reason": []}))


@pytest.mark.parametrize("tag", ["LOST", "CREATED", "shrunk", "grown"])
def test_section_seven_gate_covers_every_change_tag(tag):
    change = [(2022, GG, tag)]
    allow = pd.DataFrame({"criterion": ["D4_7"], "source": ["rms"],
                          "j": [GG], "year_lo": [2022], "year_hi": [2022],
                          "reason": ["reviewed synthetic exception"]})
    assert rv.unallowed_crosswalk_changes(change, allow) == []
    assert rv.unallowed_crosswalk_changes(
        change, allow.assign(criterion="A5")) == change
