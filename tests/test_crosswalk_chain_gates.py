"""Small synthetic regressions for the detached crosswalk Slurm gates."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from crosswalk_chain_gates import check_rms_route_alignment, require_current_law


def _route_files(tmp_path, *, new_upc=2, eval_upc=2, new_j=20,
                 eval_j=20, new_dollars=4.0, duplicate_eval=False):
    accepted = pd.DataFrame({"year": [2022, 2022], "upc": [1, 2],
                             "brand_code_uc": [10, 10], "dollars": [3.0, 4.0]})
    new = accepted.copy()
    new.loc[1, "upc"] = new_upc
    new.loc[1, "brand_code_uc"] = new_j
    new.loc[1, "dollars"] = new_dollars
    evaluated = pd.DataFrame({"year": [2022, 2022], "upc": [1, eval_upc],
                              "new_j": [10, eval_j]})
    if duplicate_eval:
        evaluated = pd.concat([evaluated, evaluated.iloc[[1]]],
                              ignore_index=True)
    paths = [tmp_path / f"{name}.parquet" for name in
             ("new", "accepted", "evaluated")]
    for frame, path in zip((new, accepted, evaluated), paths):
        frame.to_parquet(path)
    return paths


def test_route_alignment_requires_exact_index_and_values(tmp_path):
    assert check_rms_route_alignment(*_route_files(tmp_path)) == 2
    for kwargs, message in [
        ({"new_upc": 3}, "accepted RMS UPC-year keys"),
        ({"eval_upc": 3}, "evaluation UPC-year keys"),
        ({"duplicate_eval": True}, "duplicate"),
        ({"new_j": 30}, "evaluated new_j"),
        ({"new_dollars": 5.0}, "dollars differ"),
    ]:
        with pytest.raises(ValueError, match=message):
            check_rms_route_alignment(*_route_files(tmp_path, **kwargs))
    new, accepted, evaluated = _route_files(tmp_path)
    frame = pd.read_parquet(new)
    pd.concat([frame, frame.iloc[[1]]], ignore_index=True).to_parquet(new)
    with pytest.raises(ValueError, match="duplicate RMS"):
        check_rms_route_alignment(new, accepted, evaluated)


def _provenance(source="xw1", vintage="crosswalk-2026-10-19"):
    return {
        "spec": {"source": source},
        "data_provenance": {
            "source": source, "vintage": vintage,
            "files": {"market_panel": {"path": "/vintage/panel.parquet",
                                       "sha256": "data-sha"}}},
        "producer_commit": "code-sha",
        "inference": {"converged": True},
    }


def test_b5_law_must_match_data_code_and_convergence():
    kwargs = {"source": "xw1", "panel_path": "/vintage/panel.parquet",
              "panel_sha": "data-sha", "producer_commit": "code-sha"}
    require_current_law(_provenance(), **kwargs)
    for bad, message in [
        ({"panel_sha": "changed-sha"}, "market panel sha256"),
        ({"producer_commit": "changed-code"}, "producer commit"),
        ({"panel_path": "/other/panel.parquet"}, "market panel path"),
    ]:
        with pytest.raises(ValueError, match=message):
            require_current_law(_provenance(), **(kwargs | bad))
    unconverged = _provenance()
    unconverged["inference"]["converged"] = False
    with pytest.raises(ValueError, match="optimizer convergence"):
        require_current_law(unconverged, **kwargs)
    with pytest.raises(ValueError, match="market panel sha256"):
        require_current_law({}, **kwargs)


@pytest.mark.parametrize("source", ["xw2", "rms-xw2"])
def test_b5_law_accepts_explicit_combined_vintage_only(source):
    vintage = "crosswalk-hhfix-2026-10-23"
    provenance = _provenance(source=source, vintage=vintage)
    kwargs = {"source": source, "panel_path": "/vintage/panel.parquet",
              "panel_sha": "data-sha", "producer_commit": "code-sha"}
    require_current_law(provenance, expected_vintage=vintage, **kwargs)
    with pytest.raises(ValueError, match="data vintage"):
        require_current_law(provenance, **kwargs)
    with pytest.raises(ValueError, match="data vintage"):
        require_current_law(provenance, expected_vintage="wrong-vintage", **kwargs)
