"""Market and household parquet readers stay inside the Slurm CPU slice."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import aggregate_markets as am
import aggregate_households as ah


def test_worker_count_honors_slurm_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert am._worker_count() == 4
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    assert am._worker_count() == 24
    assert ah._load_all_data is am._load_all_data
