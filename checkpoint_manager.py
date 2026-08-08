"""Versioned checkpoint tracking for the FPL'26 optimizer."""
from __future__ import annotations
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

# Measurement resolution (pipeline audit, 20260806 logs). Vivado reports WNS
# to 3 decimal places in ns, so 1 ps is the instrument's least significant
# bit -- there is no such thing as a meaningful sub-picosecond WNS delta.
# The classifier below used to compare *derived Fmax* against a 1e-9 MHz
# dead-band, a resolution nine orders of magnitude finer than the number it
# was measuring, so a single-LSB flutter between the banked best and the
# reloaded design registered as a hard regression -> rollback -> repeat,
# with no possible escape. Run 20260805_160738 (logicnets) died exactly this
# way: iterations 5-11 each ran a genuinely different action (qor_suggestions,
# pin_swap, phys_opt, pblock, replicate_register, route_explore) for 51-168 s
# apiece and every single one recorded wns -0.462 -> -0.463, one LSB, all
# classified "regression". 7 of 11 iterations and 701 s, structurally
# incapable of progress.
#
# Compare in the measured quantity (WNS ns), not derived MHz, with a floor
# just above the report quantum.
#
# The floor is applied ASYMMETRICALLY, and deliberately so. A symmetric
# version was tried first and replaying the run history against it showed it
# also swallowed 13 genuine +1 LSB wins worth 2.7 MHz -- including
# rosetta_spam-filter 20260804_184322 iter 5, part of the only run that ever
# broke that design's zero-improvement streak. The damage in the logs is
# entirely on the *regression* side, because that is the side wired to
# should_rollback(): a -1 LSB reading resets the design and repeats forever,
# whereas a +1 LSB reading merely banks a checkpoint that is at worst
# equivalent. So:
#   - a sub-floor WORSENING is a tie: no rollback, no best update. This is
#     what breaks the freeze loop -- the design is left free to keep evolving
#     instead of being reset to the same checkpoint every iteration.
#   - a sub-floor IMPROVEMENT is left alone, and still banks as marginal.
WNS_NOISE_FLOOR_NS = 0.0015
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
        self.llm_cost_usd: float = 0.0
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
        delta_wns = float(wns_after) - wns_before
        # BUG FIX: this used to check delta_fmax > 0.5 / > 0.0 before checking
        # abs(delta_fmax) <= 1e-9, so a floating-point-noise delta like
        # +1e-12 was classified as "marginal" (a success that resets
        # stall_count and updates best_wns/best_checkpoint) instead of
        # "no_improvement". Check the near-zero case first.
        #
        # Resolution awareness (see WNS_NOISE_FLOOR_NS): the near-zero test is
        # now done in WNS ns against the instrument's own quantum instead of
        # in derived MHz against 1e-9.
        #
        # "design_unchanged" is split out from "no_improvement" because they
        # are different findings that the old single label conflated: an
        # EXACTLY bit-identical WNS means the action moved nothing measurable
        # at all, whereas no_improvement means it perturbed the design and the
        # perturbation didn't pay. Across the 20260719-20260806 logs, 182 of
        # 341 measured iterations (53%, 5.0 h of Vivado wall-clock) were
        # bit-identical -- and because they carried the same label as a real
        # failed attempt, nothing downstream could tell "this lever is inert
        # on this design" from "this lever might work next time", so the same
        # family kept getting re-picked. Callers treat it as a stronger
        # suppression signal; every existing `status in {...}` check that
        # names "no_improvement" is unaffected because this is a NEW label,
        # and the sites that must treat the two alike are updated explicitly.
        # delta_wns > 0 means WNS got less negative, i.e. better.
        if float(wns_after) == wns_before:
            status = "design_unchanged"
        elif -WNS_NOISE_FLOOR_NS <= delta_wns < 0:
            # Sub-quantum worsening: a tie, not a regression. No rollback.
            status = "no_improvement"
        elif delta_fmax > 0.5:
            status = "improved"
        elif delta_fmax > 0.0:
            status = "marginal"
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
        return self.stall_count >= 3
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
    def add_to_blacklist(self, key: str, targets: list[str]) -> None:
        """Persistently add concrete cell or net targets to a blacklist."""
        if key not in {"cells_blacklisted", "nets_blacklisted"}:
            raise ValueError(f"Unknown blacklist key: {key}")
        next_state = self._snapshot_state()
        next_blacklist = list(next_state[key])
        for target in targets:
            if self._is_blacklistable_target(target):
                self._append_unique(next_blacklist, str(target))
        next_state[key] = next_blacklist
        self._persist_state(next_state)
    def set_llm_cost_usd(self, cost_usd: float) -> None:
        """Update the cumulative LLM (OpenRouter) cost and re-persist history.

        Called by the optimizer whenever it has a fresher cost total, so the
        benchmark_score block in history.json stays current."""
        self.llm_cost_usd = max(0.0, float(cost_usd))
        self._persist_history()
    def benchmark_score(self) -> dict[str, Any]:
        """Contest score for the run so far.

        Benchmark Score = alpha - (0.1*alpha)*beta - (0.1*alpha)*gamma
          alpha = delta Fmax improvement (MHz), best vs baseline
          beta  = OpenRouter cost (USD)
          gamma = wall-clock runtime (s) / 3600
        """
        return self._score_for(self.baseline_fmax_mhz, self.best_fmax_mhz, self.llm_cost_usd)
    def _score_for(
        self,
        baseline_fmax_mhz: float | None,
        best_fmax_mhz: float | None,
        llm_cost_usd: float,
    ) -> dict[str, Any]:
        wall_clock_s = time.time() - self.started_at_epoch_s
        gamma = wall_clock_s / 3600.0
        beta = float(llm_cost_usd)
        alpha = None
        score = None
        if baseline_fmax_mhz is not None and best_fmax_mhz is not None:
            alpha = float(best_fmax_mhz) - float(baseline_fmax_mhz)
            score = alpha - (0.1 * alpha) * beta - (0.1 * alpha) * gamma
        return {
            "formula": "alpha - (0.1*alpha)*beta - (0.1*alpha)*gamma",
            "alpha_delta_fmax_mhz": alpha,
            "beta_openrouter_cost_usd": beta,
            "gamma_runtime_hours": gamma,
            "wall_clock_runtime_s": wall_clock_s,
            "score": score,
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
            "llm_cost_usd": self.llm_cost_usd,
        }
    def _copy_iteration(self, iteration: dict[str, Any]) -> dict[str, Any]:
        copied = dict(iteration)
        copied["targets"] = list(iteration.get("targets", []))
        return copied
    def _persist_history(self) -> None:
        self._persist_state(self._snapshot_state())
    def _persist_state(self, state: dict[str, Any]) -> None:
        # Stamp the contest score from the state being written (not self,
        # which record() has not applied yet at this point).
        state["benchmark_score"] = self._score_for(
            state.get("baseline_fmax_mhz"),
            state.get("best_fmax_mhz"),
            float(state.get("llm_cost_usd", self.llm_cost_usd) or 0.0),
        )
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
        self.llm_cost_usd = float(state.get("llm_cost_usd", self.llm_cost_usd) or 0.0)
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