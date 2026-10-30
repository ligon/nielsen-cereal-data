"""charter-crosswalk A4 / A5 / A6 as tests (src/crosswalk_checks.py).

Paths (all read-only):
  CROSSWALK_EVAL      the UPC-dollar evaluation dir (src/crosswalk_eval.py);
                      default the Run 1 eval under the vintage dir.
  CROSSWALK_ROUTE     'new' (default) or 'old'.  With 'old' the route
                      tests FAIL: that is today's route (charter A4: "shown
                      to FAIL on today's outputs").
  CROSSWALK_PANEL     market panel parquet(s) for A5 at the panel level,
  CROSSWALK_RMS_PANEL default today's accepted files, on which A5 FAILS
                      (Appendix B); point them at the new vintage after A7.
The *_flags_today tests are always green: they pin what the checks find on
today's route and files, so the checks cannot silently lose their teeth.

Run serially or with -n 4 at most.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import crosswalk_checks as cc  # noqa: E402

MIRROR = Path("/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data")
EVAL = Path(os.environ.get(
    "CROSSWALK_EVAL", MIRROR / "data" / "clean-crosswalk-2026-10-19" / "run1-eval"))
ROUTE = os.environ.get("CROSSWALK_ROUTE", "new")
TODAY_PANEL = MIRROR / "data" / "clean" / "market_panel.parquet"
TODAY_RMS = Path("/global/scratch/fsa/fc_jevons/ligon/Data/NielsenScanner/rms/"
                 "market_panel_rms.parquet")
PANEL = {"hms": Path(os.environ.get("CROSSWALK_PANEL", TODAY_PANEL)),
         "rms": Path(os.environ.get("CROSSWALK_RMS_PANEL", TODAY_RMS))}

needs_eval = pytest.mark.skipif(
    not (EVAL / "hms_upc.parquet").exists(), reason=f"no evaluation in {EVAL}")


def _open(df):
    bad = df[df["allowed"].isna()]
    return {(c, int(j), int(y)) for c, j, y in
            zip(bad["criterion"], bad["j"], bad["year"])}


@pytest.fixture(scope="module", params=["hms", "rms"])
def source(request):
    return request.param


@pytest.fixture(scope="module")
def frame(source):
    return cc.load_eval(EVAL, source)


@needs_eval
def test_a4_retention_and_leakage(source, frame):
    df = cc.failures(source, route=ROUTE, frame=frame)
    a4 = {k for k in _open(df) if k[0] == "A4"}
    assert not a4, f"{source} {ROUTE} route A4 failures: {sorted(a4)}"


@needs_eval
def test_a5_continuity(source, frame):
    df = cc.failures(source, route=ROUTE, frame=frame)
    a5 = {k for k in _open(df) if k[0] == "A5"}
    assert not a5, f"{source} {ROUTE} route A5 failures: {sorted(a5)}"


def test_kashi_allowlist_is_limited_to_authorized_source_years():
    allow = cc.load_allowlist()
    reason = ("genuine decline; renamed UPCs keep the legacy code; "
              "no misrouted UPC found")
    for source, years in {"hms": {2024}, "rms": {2023, 2024}}.items():
        for year in range(2021, 2026):
            assert cc.allowed(allow, "A5", source, 579277, year) == (
                reason if year in years else None)
        assert cc.allowed(allow, "A4", source, 579277, 2024) is None
        assert cc.allowed(allow, "A5", source, 579278, 2024) is None


def test_panel_cli_reports_only_a5_as_evaluated(monkeypatch, capsys):
    # No panel read: exercise the CLI/report boundary with an empty flag table.
    empty = cc.pd.DataFrame(columns=["criterion", "j", "year", "detail", "allowed", "brand"])
    monkeypatch.setattr(cc, "failures", lambda *args, **kwargs: empty)
    monkeypatch.setattr(sys, "argv", ["crosswalk_checks.py", "--panel", "fixture.parquet",
                                     "--source", "hms"])
    with pytest.raises(SystemExit) as result:
        cc.main()
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "A5 pass" in output
    for criterion in ("A4", "A6"):
        assert f"{criterion} not evaluated" in output
        assert f"{criterion} pass" not in output


@needs_eval
def test_a6_coverage_not_below_today(source, frame):
    df = cc.failures(source, route=ROUTE, frame=frame)
    assert not {k for k in _open(df) if k[0] == "A6"}
    r = cc.a6(frame)
    for _, row in r.iterrows():
        assert row[f"cov_{ROUTE}"] >= cc.A6_TODAY[source][row["year"]] - 5e-4


@pytest.mark.parametrize("src", ["hms", "rms"])
def test_a5_panel_file(src):
    """A5 on a panel parquet.  FAILS on today's files (Appendix B); meant to
    pass on the new vintage (set CROSSWALK_PANEL / CROSSWALK_RMS_PANEL)."""
    if not PANEL[src].exists():
        pytest.skip(f"{PANEL[src]} missing")
    if PANEL[src] in (TODAY_PANEL, TODAY_RMS) and not os.environ.get(
            "CROSSWALK_RUN_TODAY"):
        pytest.skip("today's file: known to fail (set CROSSWALK_RUN_TODAY=1 "
                    "to see it; test_a5_flags_today pins the flags)")
    df = cc.failures(src, panel=PANEL[src])
    a5 = {k for k in _open(df) if k[0] == "A5"}
    assert not a5, sorted(a5)


# --- the checks' teeth: what they find on today's route and files ----------

APPENDIX_B = {
    "hms": {557461, 557579, 579277, 580031, 620458, 620481, 620537, 620554,
            623878},
    "rms": {557461, 557579, 579277, 620458, 620554, 623878}}
TABLE_B_A4 = {   # charter Table B / ledger section 0: today's A4 failures
    "hms": {557426, 557461, 557579, 579277, 620458, 620554, 579923},
    "rms": {557426, 557461, 557579, 579277, 620458, 620554, 579923}}


@pytest.mark.parametrize("src", ["hms", "rms"])
def test_a5_flags_today_files_exactly_appendix_b(src):
    path = TODAY_PANEL if src == "hms" else TODAY_RMS
    if not path.exists():
        pytest.skip(f"{path} missing")
    f, _ = cc.a5_panel(path, src)
    assert set(f["j"]) == APPENDIX_B[src]
    gg = f[(f["j"] == 620554) & (f["test"] == "i")]
    want = {"hms": {2012, 2017, 2021}, "rms": {2011, 2016, 2017, 2021}}[src]
    assert set(gg["year"]) == want


@needs_eval
def test_a4_flags_today_route(source, frame):
    r = cc.a4(frame, "old")
    bad = set(r.loc[r["ret_fail"] | r["leak_fail"], "j"])
    assert TABLE_B_A4[source] <= bad
    fmw = r[r["j"] == 579923].set_index("year")["leakage"]
    if source == "rms":                       # charter A4: FMW 13-18% in RMS
        assert fmw.between(0.12, 0.19).all()
    trix = r[r["j"] == 557579]["retention"]
    assert (trix < 0.01).all()


@needs_eval
def test_new_route_keeps_pre2021_cells_except_merges(source, frame):
    """Only the merge sets change pre-2021 brand-years (Q2 (a), Q2b)."""
    old = cc.brand_year_from_eval(frame, "old")
    new = cc.brand_year_from_eval(frame, "new")
    pre = [y for y in old.columns if y < 2021]
    d = (new[pre].fillna(0) - old.reindex(new.index)[pre].fillna(0)).abs()
    changed = set(d.index[(d > 1e-6).any(axis=1)])
    assert changed <= {620554, 557579}
