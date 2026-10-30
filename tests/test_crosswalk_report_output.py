"""Reports default to evaluation storage and respect an explicit output path."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import crosswalk_report as report


@pytest.mark.parametrize('explicit', [False, True])
def test_report_output_stays_with_evaluation_or_explicit_path(tmp_path, monkeypatch, explicit):
    evaluation = tmp_path / 'evaluation'
    evaluation.mkdir()
    (evaluation / 'meta.json').write_text(json.dumps(dict(
        commit='synthetic', old_crosswalk_sha256='synthetic',
        upc_routes_sha256='synthetic', rows=dict(hms=0, rms=0))))
    monkeypatch.setattr(report.cc, 'load_eval', lambda *args: None)
    monkeypatch.setattr(report.cc, 'failures', lambda *args, **kwargs:
        pd.DataFrame(columns=['allowed', 'criterion', 'brand', 'year']))
    monkeypatch.setattr(report, 'section_totals', lambda *args: ([], None, None))
    for section in ['checks', 'orientation', 'pre2021', 'd4', 'unassigned',
                    'name_only', 'routes', 'brand_years']:
        monkeypatch.setattr(report, 'section_' + section, lambda *args: [])
    argv = ['crosswalk_report', '--eval', str(evaluation)]
    destination = evaluation / 'crosswalk-before-after.org'
    if explicit:
        destination = tmp_path / 'chosen-report.org'
        argv += ['--out', str(destination)]
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.chdir(tmp_path)
    report.main()
    assert destination.read_text().startswith('#+TITLE:')
    assert not (tmp_path / 'docs').exists()
    if explicit:
        assert not (evaluation / 'crosswalk-before-after.org').exists()


def test_make_report_uses_evaluation_output_directory(tmp_path):
    root = Path(__file__).resolve().parent.parent
    commands = subprocess.check_output(
        ['make', '-n', 'crosswalk-report', 'XW_EVAL=' + str(tmp_path)],
        cwd=root, text=True)
    assert '--out ' + str(tmp_path / 'crosswalk-before-after.org') in commands
    assert '--out docs/' not in commands
