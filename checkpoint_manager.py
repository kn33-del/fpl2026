"""Versioned checkpoint tracking for the FPL'26 optimizer."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


class CheckpointManager:
    """Track optimization checkpoints, timing history, and rollback state."""

    def __init__(
        self,
        input_dcp: str,
        output_dir: str,
        clock_name: str,
        hard_limit_seconds: int = 3500,
    ) -> None:
        """Create a checkpoint manager for a single optimization run."""
        self.input_dcp = str(input_dcp)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clock_name = str(clock_name)
        self.hard_limit_seconds = int(hard_limit_seconds)
        self.history_path = self.output_dir / "history.json"
        self.started_at_epoch_s = time.time()

        self.clock_period_ns: float | None = None
        self.baseline_wns: float | None = None
        self.baseline_fmax_mhz: float | None = None
        self.best_wns: float | None = None
        self.best_fmax_mhz: float | None = None
        self.best_checkpoint = self.input_dcp
        self.current_iter = 0
        self.stall_count = 0
        self.cells_blacklisted: list[str] = []
        self.nets_blacklisted: list[str] = []
        self.iterations: list[dict[str, Any]] = []

    def start_baseline(self, wns: float, period_ns: float) -> None:
        """Record the baseline timing measurement for the input DCP."""
        if period_ns - wns <= 0.0:
            raise ValueError("Clock period minus WNS must be positive")

        baseline_fmax_mhz = 1000.0 / (period_ns - wns)
        self.clock_period_ns = float(period_ns)
        self.baseline_wns = float(wns)
        self.baseline_fmax_mhz = baseline_fmax_mhz
        self.best_wns = float(wns)
        self.best_fmax_mhz = baseline_fmax_mhz
        self.best_checkpoint = self.input_dcp
        self.current_iter = 0
        self.stall_count = 0
        self.iterations = []
        self.cells_blacklisted = []
        self.nets_blacklisted = []
        self._persist_history()

    def record(
        self,
        recipe: str,
        targets: list[str],
        wns_after: float,
        vivado_runtime_s: int | float,
        checkpoint_path: str,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Record a Vivado iteration and persist the updated run history."""
        self._require_baseline()

        targets_list = [str(target) for target in targets]
        checkpoint = str(checkpoint_path)
        wns_before = float(self.best_wns)
        fmax_before = float(self.best_fmax_mhz)
        fmax_after = 1000.0 / (self.clock_period_ns - float(wns_after))
        delta_fmax = fmax_after - fmax_before

        if delta_fmax > 0.5:
            status = "improved"
        elif delta_fmax > 0.0:
            status = "marginal"
        elif abs(delta_fmax) <= 1e-9:
            status = "no_improvement"
        else:
            status = "regression"

        iteration = {
            "iter": self.current_iter + 1,
            "recipe": str(recipe),
            "batch_size": batch_size,
            "targets": targets_list,
            "wns_before": wns_before,
            "wns_after": float(wns_after),
            "fmax_before": fmax_before,
            "fmax_after": fmax_after,
            "delta_fmax": delta_fmax,
            "vivado_runtime_s": vivado_runtime_s,
            "status": status,
            "checkpoint": checkpoint,
        }

        next_state = self._snapshot_state()
        next_state["current_iter"] = self.current_iter + 1
        next_state["iterations"] = [*self.iterations, iteration]

        if status in {"improved", "marginal"}:
            next_state["best_wns"] = float(wns_after)
            next_state["best_fmax_mhz"] = fmax_after
            next_state["best_checkpoint"] = checkpoint
            next_state["stall_count"] = 0
        else:
            next_state["stall_count"] = self.stall_count + 1
            blacklist_key = self._blacklist_key_for_recipe(recipe)
            if blacklist_key is not None:
                next_blacklist = list(next_state[blacklist_key])
                for target in targets_list:
                    if self._is_blacklistable_target(target):
                        self._append_unique(next_blacklist, target)
                next_state[blacklist_key] = next_blacklist

        self._persist_state(next_state)

        if status in {"improved", "marginal"}:
            self._copy_best_checkpoint(checkpoint)

        return iteration

    def should_rollback(self) -> bool:
        """Return True when the most recent iteration regressed."""
        return bool(self.iterations and self.iterations[-1].get("status") == "regression")

    def get_best_checkpoint(self) -> str:
        """Return the best checkpoint path seen so far."""
        return self.best_checkpoint

    def should_escalate(self) -> bool:
        """Return True when repeated stalls should trigger a recipe escalation."""
        return self.stall_count >= 3 and self.stall_count % 3 == 0

    def should_continue(self) -> bool:
        """Return True while the wall-clock and iteration budgets remain."""
        if self.current_iter >= 50:
            return False
        elapsed = time.time() - self.started_at_epoch_s
        return elapsed < self.hard_limit_seconds

    def get_blacklist(self) -> dict[str, list[str]]:
        """Return the current cell and net blacklists."""
        return {
            "cells": list(self.cells_blacklisted),
            "nets": list(self.nets_blacklisted),
        }

    def summary(self) -> str:
        """Return a one-paragraph summary of run progress."""
        if self.baseline_fmax_mhz is None or self.best_fmax_mhz is None:
            return (
                f"Iter {self.current_iter}, baseline not initialized, stall_count={self.stall_count}, "
                f"time remaining {self._time_remaining_minutes():.1f} min."
            )

        delta_from_baseline = self.best_fmax_mhz - self.baseline_fmax_mhz
        return (
            f"Iter {self.current_iter}, best Fmax {self.best_fmax_mhz:.2f} MHz "
            f"({delta_from_baseline:+.2f} MHz vs baseline), stall_count={self.stall_count}, "
            f"time remaining {self._time_remaining_minutes():.1f} min."
        )

    def _require_baseline(self) -> None:
        if self.clock_period_ns is None or self.baseline_fmax_mhz is None or self.best_fmax_mhz is None:
            raise RuntimeError("start_baseline() must be called before record()")

    def _is_net_recipe(self, recipe: str) -> bool:
        return self._blacklist_key_for_recipe(recipe) == "nets_blacklisted"

    def _blacklist_key_for_recipe(self, recipe: str) -> str | None:
        lower = recipe.lower()
        if lower in {"rapidwright_fanout", "fanout_split"}:
            return "nets_blacklisted"
        if lower in {"rapidwright_cell_placement"}:
            return "cells_blacklisted"
        return None

    def _is_blacklistable_target(self, target: str) -> bool:
        target = str(target).strip()
        if not target:
            return False
        non_cell_targets = {
            "default",
            "explore",
            "explorewithremap",
            "aggressiveexplore",
            "critical_cell_opt",
            "logic_delay_bound",
            "mixed_path",
            "timing_path",
        }
        return target.lower() not in non_cell_targets

    def _append_unique(self, items: list[str], target: str) -> None:
        if target not in items:
            items.append(target)

    def _copy_best_checkpoint(self, checkpoint_path: str) -> None:
        source = Path(checkpoint_path)
        destination = self.output_dir / "best.dcp"
        if source.resolve(strict=False) == destination.resolve(strict=False):
            return
        shutil.copy2(source, destination)

    def _time_remaining_minutes(self) -> float:
        elapsed_seconds = time.time() - self.started_at_epoch_s
        remaining_seconds = max(0.0, self.hard_limit_seconds - elapsed_seconds)
        return remaining_seconds / 60.0

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "input_dcp": self.input_dcp,
            "clock_name": self.clock_name,
            "clock_period_ns": self.clock_period_ns,
            "baseline_wns": self.baseline_wns,
            "baseline_fmax_mhz": self.baseline_fmax_mhz,
            "best_wns": self.best_wns,
            "best_fmax_mhz": self.best_fmax_mhz,
            "best_checkpoint": self.best_checkpoint,
            "current_iter": self.current_iter,
            "stall_count": self.stall_count,
            "cells_blacklisted": list(self.cells_blacklisted),
            "nets_blacklisted": list(self.nets_blacklisted),
            "iterations": [self._copy_iteration(iteration) for iteration in self.iterations],
            "started_at_epoch_s": self.started_at_epoch_s,
        }

    def _copy_iteration(self, iteration: dict[str, Any]) -> dict[str, Any]:
        copied = dict(iteration)
        copied["targets"] = list(iteration.get("targets", []))
        return copied

    def _persist_history(self) -> None:
        self._persist_state(self._snapshot_state())

    def _persist_state(self, state: dict[str, Any]) -> None:
        self._write_history_atomically(state)
        self._apply_state(state)

    def _apply_state(self, state: dict[str, Any]) -> None:
        self.input_dcp = str(state.get("input_dcp", self.input_dcp))
        self.clock_name = str(state.get("clock_name", self.clock_name))
        self.clock_period_ns = state.get("clock_period_ns")
        self.baseline_wns = state.get("baseline_wns")
        self.baseline_fmax_mhz = state.get("baseline_fmax_mhz")
        self.best_wns = state.get("best_wns")
        self.best_fmax_mhz = state.get("best_fmax_mhz")
        self.best_checkpoint = str(state.get("best_checkpoint", self.best_checkpoint))
        self.current_iter = int(state.get("current_iter", self.current_iter))
        self.stall_count = int(state.get("stall_count", self.stall_count))
        self.cells_blacklisted = list(state.get("cells_blacklisted", []))
        self.nets_blacklisted = list(state.get("nets_blacklisted", []))
        self.iterations = [self._copy_iteration(iteration) for iteration in state.get("iterations", [])]
        self.started_at_epoch_s = float(state.get("started_at_epoch_s", self.started_at_epoch_s))

    def _write_history_atomically(self, state: dict[str, Any]) -> None:
        temp_path = self.history_path.with_name(f"{self.history_path.name}.tmp")
        payload = json.dumps(state, indent=2)
        temp_path.write_text(payload + "\n", encoding="utf-8")
        os.replace(temp_path, self.history_path)


def load_or_create(output_dir: str, input_dcp: str, clock_name: str) -> CheckpointManager:
    """Load a saved manager from history.json or create a fresh one."""
    output_path = Path(output_dir)
    history_path = output_path / "history.json"
    if not history_path.exists():
        return CheckpointManager(input_dcp=input_dcp, output_dir=output_dir, clock_name=clock_name)

    state = json.loads(history_path.read_text(encoding="utf-8"))
    manager = CheckpointManager(
        input_dcp=str(state.get("input_dcp", input_dcp)),
        output_dir=output_dir,
        clock_name=str(state.get("clock_name", clock_name)),
        hard_limit_seconds=int(state.get("hard_limit_seconds", 3500)),
    )
    manager._apply_state(state)
    return manager
