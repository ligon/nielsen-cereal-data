"""Local-code staging for the detached crosswalk Slurm jobs."""
import os
import subprocess
from pathlib import Path


ENV_SCRIPT = Path(__file__).resolve().parent.parent / "slurm/crosswalk-env.sh"
JOBS = ("households", "markets", "rms-d3", "checks", "downstream",
        "hhfix-downstream")


def _run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True,
                          **kwargs).stdout.strip()


def _mirror(path, branch="crosswalk"):
    _run("git", "init", "-q", str(path))
    _run("git", "-C", str(path), "checkout", "-q", "-b", branch)
    (path / "code.txt").write_text("pinned code\n")
    _run("git", "-C", str(path), "add", "code.txt")
    _run("git", "-C", str(path), "-c", "user.name=Test",
         "-c", "user.email=test@example.org", "commit", "-qm", "pin")
    return _run("git", "-C", str(path), "rev-parse", "HEAD")


def _stage(nielsen, cd, root, ni_pin, cd_pin,
           ni_branch="crosswalk", cd_branch="crosswalk"):
    script = f"""
source "$1"
XW_NIELSEN_MIRROR=$2
XW_CD_MIRROR=$3
XW_PIN_NIELSEN=$5
XW_PIN_CD=$6
XW_BRANCH_NIELSEN=$7
XW_BRANCH_CD=$8
xw_stage_job_code "$4"
printf '%s\\n%s\\n%s\\n' "$XW_REPO" "$XW_CD" "$TMPDIR"
"""
    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script, "bash", str(ENV_SCRIPT),
         str(nielsen), str(cd), str(root), ni_pin, cd_pin,
         ni_branch, cd_branch],
        text=True, capture_output=True, env=env)


def test_job_stages_pinned_code_and_reuses_clean_local_clones(tmp_path):
    ni, cd, root = (tmp_path / name for name in ("ni-mirror", "cd-mirror", "job"))
    ni_pin, cd_pin = _mirror(ni), _mirror(cd)
    for _ in range(2):
        result = _stage(ni, cd, root, ni_pin, cd_pin)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[-3:] == [
            str(root / "mirrors/nielsen-cereal-data"),
            str(root / "mirrors/CerealDemand"), str(root / "tmp")]
    assert _run("git", "-C", str(root / "mirrors/nielsen-cereal-data"),
                "rev-parse", "HEAD") == ni_pin
    assert _run("git", "-C", str(root / "mirrors/CerealDemand"),
                "rev-parse", "HEAD") == cd_pin
    bad = _stage(ni, cd, root, "0" * 40, cd_pin)
    assert bad.returncode == 2
    assert "not the reviewed pin" in bad.stderr


def test_staging_supports_distinct_nielsen_and_cerealdemand_branches(tmp_path):
    ni, cd, root = (tmp_path / name for name in ("ni-mirror", "cd-mirror", "job"))
    ni_branch, cd_branch = "crosswalk-hhfix-integration", "crosswalk-current-integration"
    ni_pin, cd_pin = _mirror(ni, ni_branch), _mirror(cd, cd_branch)
    result = _stage(ni, cd, root, ni_pin, cd_pin, ni_branch, cd_branch)
    assert result.returncode == 0, result.stderr


def test_every_job_sources_environment_only_after_pinned_local_bootstrap():
    for name in JOBS:
        script = (ENV_SCRIPT.parent / f"crosswalk-{name}.slurm").read_text()
        source = script.index("source slurm/crosswalk-env.sh")
        for gate in ("BOOT_REPO=/local/job", "rev-parse refs/heads/$XW_BRANCH_NIELSEN",
                     'rev-parse HEAD) == "$XW_PIN_NIELSEN"',
                     "BOOT_STATUS=$(git -C \"$BOOT_REPO\" status --porcelain --untracked-files=all)",
                     'cd "$BOOT_REPO"'):
            assert script.index(gate) < source, (name, gate)
        assert script.index("xw_preflight") > source
        assert 'cd "${SLURM_SUBMIT_DIR' not in script
