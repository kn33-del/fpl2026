#!/usr/bin/env python3
"""Standalone verification script for checkpoint manager behavior."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checkpoint_manager import CheckpointManager


def _write_checkpoint(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> int:
    """Exercise the checkpoint manager and print a pass/fail result."""
    try:
        with tempfile.TemporaryDirectory(prefix="checkpoint_manager_verify_") as temp_dir:
            temp_path = Path(temp_dir)
            input_dcp = temp_path / "input.dcp"
            output_dir = temp_path / "output"
            _write_checkpoint(input_dcp, "input")

            manager = CheckpointManager(str(input_dcp), str(output_dir), "clk")
            manager.start_baseline(-0.8, 2.5)

            recipe_plan = [
                ("cell_replacement", ["inst/a"], -0.4, "iter_001.dcp"),
                ("fanout_rewrite", ["net/a"], -0.9, "iter_002.dcp"),
                ("cell_replacement", ["inst/b"], -0.4, "iter_003.dcp"),
                ("cell_replacement", ["inst/c"], -0.2, "iter_004.dcp"),
                ("fanout_rewrite", ["net/b"], -0.9, "iter_005.dcp"),
            ]

            for recipe, targets, wns_after, checkpoint_name in recipe_plan:
                checkpoint = temp_path / checkpoint_name
                _write_checkpoint(checkpoint, checkpoint_name)
                manager.record(recipe, targets, wns_after, 30, str(checkpoint))

            history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
            print(json.dumps(history, indent=2))
            print(manager.summary())

            if manager.best_fmax_mhz <= manager.baseline_fmax_mhz:
                raise AssertionError("best_fmax_mhz did not improve over baseline")

            print("PASS")
            return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())