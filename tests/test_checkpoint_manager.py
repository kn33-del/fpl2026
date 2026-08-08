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

def test_sub_quantum_flutter_is_not_a_regression(tmp_path: Path) -> None:
    """Run 20260805_160738 (logicnets) froze permanently on this.

    Vivado reports WNS to 1 ps, but the classifier's dead-band was 1e-9 MHz
    -- nine orders of magnitude finer than the instrument. One LSB of
    flutter between the banked best (-0.462) and the reloaded design
    (-0.463) therefore read as a hard regression, triggering a rollback,
    every iteration, forever: seven genuinely different actions each ran for
    51-168 s and every one recorded -0.462 -> -0.463 / "regression". The run
    was structurally incapable of progress from iteration 5 onward.
    """
    manager = _make_manager(tmp_path)
    manager.start_baseline(-0.462, 1.5)
    checkpoint = tmp_path / "iter_001.dcp"
    _write_checkpoint(checkpoint, "iter-001")

    result = manager.record("vivado_phys_opt", ["Explore"], -0.463, 120, str(checkpoint))

    # A single-LSB delta is a tie, not a regression -- crucially it must not
    # trigger the rollback path.
    assert result["status"] != "regression"
    assert manager.should_rollback() is False
    # And a tie must never move the banked best (that is what bounds drift).
    assert manager.best_wns == pytest.approx(-0.462)


def test_bit_identical_wns_is_design_unchanged(tmp_path: Path) -> None:
    """182 of 341 measured iterations across the 20260719-20260806 logs came
    back bit-identical, 5.0 h of Vivado wall-clock. Labelled plain
    "no_improvement", nothing downstream could distinguish an inert lever
    from one that merely hadn't paid off, so the same family kept getting
    re-picked (amd_mini-isp 20260803_161428: 7 straight iterations, 4
    distinct actions, all -0.904)."""
    manager = _make_manager(tmp_path)
    manager.start_baseline(-0.904, 1.5)
    checkpoint = tmp_path / "iter_001.dcp"
    _write_checkpoint(checkpoint, "iter-001")

    result = manager.record("vivado_phys_opt", ["Explore"], -0.904, 48, str(checkpoint))
    assert result["status"] == "design_unchanged"
    # Still a stall, still no best update -- only the label is sharper.
    assert manager.stall_count == 1
    assert manager.best_wns == pytest.approx(-0.904)

    # A real (above-quantum) move that doesn't pay stays "no_improvement"/
    # "regression" -- design_unchanged is reserved for provably-inert.
    second = tmp_path / "b"
    second.mkdir()
    manager2 = _make_manager(second)
    manager2.start_baseline(-0.904, 1.5)
    ckpt2 = tmp_path / "iter_002.dcp"
    _write_checkpoint(ckpt2, "iter-002")
    result2 = manager2.record("vivado_phys_opt", ["Explore"], -0.950, 48, str(ckpt2))
    assert result2["status"] == "regression"


def test_genuine_gain_above_the_noise_floor_still_counts(tmp_path: Path) -> None:
    """The floor must not swallow real wins: a delta comfortably above the
    1 ps quantum still banks."""
    manager = _make_manager(tmp_path)
    manager.start_baseline(-0.500, 1.5)
    checkpoint = tmp_path / "iter_001.dcp"
    _write_checkpoint(checkpoint, "iter-001")

    result = manager.record("place_design_explore", ["directive:Default"], -0.400, 200, str(checkpoint))
    assert result["status"] == "improved"
    assert manager.best_wns == pytest.approx(-0.400)


def test_noise_floor_is_asymmetric(tmp_path: Path) -> None:
    """The floor damps sub-quantum WORSENING only.

    A symmetric floor was tried first; replaying the run history showed it
    also swallowed 13 genuine +1 LSB wins worth 2.7 MHz (including
    rosetta_spam-filter 20260804_184322 iter 5, from the only run that ever
    broke that design's zero streak). The harm in the logs is all on the
    regression side, because that is the side wired to should_rollback().
    """
    # -1 LSB: damped to a tie, and must not arm the rollback.
    worse = _make_manager(tmp_path)
    worse.start_baseline(-0.462, 1.5)
    ck1 = tmp_path / "a.dcp"
    _write_checkpoint(ck1, "a")
    assert worse.record("phys_opt", ["x"], -0.463, 10, str(ck1))["status"] == "no_improvement"
    assert worse.should_rollback() is False

    # +1 LSB: untouched, still banks as a real (if small) win.
    better_dir = tmp_path / "up"
    better_dir.mkdir()
    better = _make_manager(better_dir)
    better.start_baseline(-0.494, 1.5)
    ck2 = better_dir / "b.dcp"
    _write_checkpoint(ck2, "b")
    result = better.record("pblock", ["y"], -0.493, 10, str(ck2))
    assert result["status"] == "marginal"
    assert better.best_wns == pytest.approx(-0.493)
    assert better.stall_count == 0
