import json
from pathlib import Path

import pytest

import checkpoint_manager
from checkpoint_manager import CheckpointManager, load_or_create


def _write_checkpoint(path: Path, content: str = "checkpoint") -> None:
    path.write_text(content, encoding="utf-8")


def _make_manager(tmp_path: Path) -> CheckpointManager:
    input_dcp = tmp_path / "input.dcp"
    _write_checkpoint(input_dcp, "input")
    manager = CheckpointManager(str(input_dcp), str(tmp_path / "out"), "clk")
    manager.start_baseline(-1.0, 2.5)
    return manager


def test_baseline(tmp_path: Path) -> None:
    input_dcp = tmp_path / "input.dcp"
    _write_checkpoint(input_dcp, "input")
    manager = CheckpointManager(str(input_dcp), str(tmp_path / "out"), "clk")

    manager.start_baseline(-1.0, 2.5)

    assert manager.baseline_fmax_mhz == pytest.approx(1000.0 / 3.5)
    assert manager.best_fmax_mhz == pytest.approx(manager.baseline_fmax_mhz)
    assert manager.best_checkpoint == str(input_dcp)


def test_record_improvement(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    checkpoint = tmp_path / "iter_001.dcp"
    _write_checkpoint(checkpoint, "iter-001")

    result = manager.record("cell_replacement", ["inst/a", "inst/b"], -0.2, 12, str(checkpoint), batch_size=4)

    assert result["status"] == "improved"
    assert manager.best_checkpoint == str(checkpoint)
    assert manager.stall_count == 0
    assert (tmp_path / "out" / "best.dcp").read_text(encoding="utf-8") == "iter-001"


def test_record_regression_blacklists(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    checkpoint = tmp_path / "iter_001.dcp"
    _write_checkpoint(checkpoint, "iter-001")

    result = manager.record("cell_replacement", ["inst/lut2_bad"], -2.0, 10, str(checkpoint))

    assert result["status"] == "regression"
    assert "inst/lut2_bad" in manager.cells_blacklisted
    assert manager.stall_count == 1
    assert manager.best_checkpoint == str(tmp_path / "input.dcp")


def test_stall_escalation(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    checkpoint = tmp_path / "stall.dcp"
    _write_checkpoint(checkpoint, "stall")

    manager.record("cell_replacement", ["inst/a"], -1.0, 10, str(checkpoint))
    assert manager.should_escalate() is False
    manager.record("cell_replacement", ["inst/b"], -1.0, 10, str(checkpoint))
    assert manager.should_escalate() is False
    manager.record("cell_replacement", ["inst/c"], -1.0, 10, str(checkpoint))
    assert manager.should_escalate() is True
    assert manager.should_escalate() is False


def test_atomic_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _make_manager(tmp_path)
    before = json.loads((tmp_path / "out" / "history.json").read_text(encoding="utf-8"))
    checkpoint = tmp_path / "iter_002.dcp"
    _write_checkpoint(checkpoint, "iter-002")

    calls = {"count": 0}
    original_replace = checkpoint_manager.os.replace

    def fail_first_replace(src: str, dst: str) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated crash")
        return original_replace(src, dst)

    monkeypatch.setattr(checkpoint_manager.os, "replace", fail_first_replace)

    with pytest.raises(RuntimeError, match="simulated crash"):
        manager.record("cell_replacement", ["inst/a"], -0.2, 10, str(checkpoint))

    after = json.loads((tmp_path / "out" / "history.json").read_text(encoding="utf-8"))
    assert after == before


def test_resume(tmp_path: Path) -> None:
    history_dir = tmp_path / "resume"
    history_dir.mkdir()
    state = {
        "input_dcp": str(tmp_path / "input.dcp"),
        "clock_name": "clk",
        "clock_period_ns": 2.5,
        "baseline_wns": -1.0,
        "baseline_fmax_mhz": 285.7142857142857,
        "best_wns": -0.25,
        "best_fmax_mhz": 363.6363636363636,
        "best_checkpoint": "checkpoints/iter_002.dcp",
        "current_iter": 2,
        "stall_count": 1,
        "cells_blacklisted": ["inst/a"],
        "nets_blacklisted": [],
        "iterations": [],
        "started_at_epoch_s": 1000.0,
    }
    (history_dir / "history.json").write_text(json.dumps(state), encoding="utf-8")

    manager = load_or_create(str(history_dir), str(tmp_path / "input.dcp"), "clk")

    assert manager.current_iter == 2
    assert manager.best_fmax_mhz == pytest.approx(363.6363636363636)
    assert manager.best_checkpoint == "checkpoints/iter_002.dcp"


def test_time_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dcp = tmp_path / "input.dcp"
    _write_checkpoint(input_dcp, "input")

    start = 1000.0
    monkeypatch.setattr(checkpoint_manager.time, "time", lambda: start)
    manager = CheckpointManager(str(input_dcp), str(tmp_path / "out"), "clk")
    manager.start_baseline(-1.0, 2.5)

    monkeypatch.setattr(checkpoint_manager.time, "time", lambda: start + 3000)
    assert manager.should_continue() is True

    monkeypatch.setattr(checkpoint_manager.time, "time", lambda: start + 3500)
    assert manager.should_continue() is False


def test_blacklist_not_double_added(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    checkpoint = tmp_path / "iter_001.dcp"
    _write_checkpoint(checkpoint, "iter-001")

    manager.record("cell_replacement", ["inst/dup"], -2.0, 10, str(checkpoint))
    manager.record("cell_replacement", ["inst/dup"], -2.0, 10, str(checkpoint))

    assert manager.cells_blacklisted.count("inst/dup") == 1