"""Regression tests for the timing-validity / summary / pblock_full_replace /
failure-memory fixes (2026-07-11).

These exercise DCPOptimizer's orchestration logic without Vivado or
RapidWright: external deps (mcp / openai) are stubbed if not importable, and
tool calls are patched per-test.
"""
import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub mcp/openai only when the real packages are unavailable (e.g. running
# on a machine without the Linux venv). setdefault leaves real installs alone.
try:  # pragma: no cover
    import mcp  # noqa: F401
except ImportError:  # pragma: no cover
    mcp_stub = types.ModuleType("mcp")
    mcp_stub.ClientSession = object
    mcp_stub.StdioServerParameters = object
    mcp_client = types.ModuleType("mcp.client")
    mcp_client_stdio = types.ModuleType("mcp.client.stdio")
    mcp_client_stdio.stdio_client = lambda *a, **k: None
    sys.modules.setdefault("mcp", mcp_stub)
    sys.modules.setdefault("mcp.client", mcp_client)
    sys.modules.setdefault("mcp.client.stdio", mcp_client_stdio)
try:  # pragma: no cover
    import openai  # noqa: F401
except ImportError:  # pragma: no cover
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = type("OpenAI", (), {"__init__": lambda self, *a, **k: None})
    sys.modules.setdefault("openai", openai_stub)

from dcp_optimizer import DCPOptimizer  # noqa: E402
from checkpoint_manager import CheckpointManager  # noqa: E402


@pytest.fixture
def opt(tmp_path):
    return DCPOptimizer(api_key="dummy", run_dir=tmp_path)


# ---------------------------------------------------------------------------
# Fix 1: timing validity / design-state tracking
# ---------------------------------------------------------------------------

def test_design_state_transitions(opt):
    assert opt.design_state == "routed"
    opt._update_design_state("vivado_run_tcl", {"command": "place_design -unplace"}, "ok")
    assert opt.design_state == "unplaced"
    assert opt.last_action_mutated_design is True
    opt._update_design_state("vivado_place_design", {}, "Placement complete.")
    assert opt.design_state == "placed"
    opt._update_design_state("vivado_route_design", {}, "Routing complete.")
    assert opt.design_state == "routed"


def test_failed_place_does_not_advance_state(opt):
    opt._update_design_state("vivado_run_tcl", {"command": "place_design -unplace"}, "ok")
    opt._update_design_state("vivado_place_design", {}, "ERROR: [Place 30-99] failed")
    assert opt.design_state == "unplaced"
    opt._update_design_state("vivado_open_checkpoint", {"dcp_path": "x.dcp"}, "Opened")
    assert opt.design_state == "routed"


def test_wns_refused_when_not_routed(opt):
    opt.design_state = "unplaced"
    assert asyncio.run(opt._get_current_wns()) is None
    payload = json.loads(asyncio.run(opt._run_phys_opt_with_policy({})))
    assert payload["error_type"] == "phys_opt_invalid_design_state"
    payload = json.loads(asyncio.run(opt._maybe_run_pblock_or_phys_opt({})))
    assert payload["error_type"] == "pblock_invalid_design_state"


def test_verify_routed_state(opt):
    async def clean(tool_name, arguments, internal=False):
        if tool_name == "vivado_run_tcl":
            return "STATE_UNPLACED:0"
        return "# of nets with routing errors.......... :           0 :"

    async def unrouted(tool_name, arguments, internal=False):
        if tool_name == "vivado_run_tcl":
            return "STATE_UNPLACED:0"
        return ("# of unrouted nets..................... :           5 :\n"
                "# of nets with routing errors.......... :           5 :")

    async def unplaced(tool_name, arguments, internal=False):
        return "STATE_UNPLACED:42"

    with patch.object(opt, "call_tool", side_effect=clean):
        assert asyncio.run(opt._verify_routed_state())[0] is True
    with patch.object(opt, "call_tool", side_effect=unrouted):
        assert asyncio.run(opt._verify_routed_state())[0] is False
    with patch.object(opt, "call_tool", side_effect=unplaced):
        assert asyncio.run(opt._verify_routed_state())[0] is False


def test_rejected_observation_is_a_failure_with_restore(opt, tmp_path):
    """Run 20260712 regression: a provenance-gate rejection must penalize the
    action and restore the best checkpoint -- not record a passive
    wns_parse_error that leaves the mutant design live and teaches nothing."""
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    opt.checkpoint_manager.start_baseline(wns=-0.978, period_ns=1.5)
    opt.iteration = 1
    opt.clock_period = 1.5
    opt.last_action_key = "rapidwright_optimize_cell_placement"
    opt.last_recipe = "rapidwright_cell_placement"
    opt.last_targets = ["layer0_reg/data_out_reg[41]"]

    restores = []

    async def fake_verify():
        return False, "8 primitive cell(s) are unplaced"

    async def fake_restore(reason):
        restores.append(reason)

    with patch.object(opt, "_verify_routed_state", side_effect=fake_verify), \
         patch.object(opt, "_restore_best_state", side_effect=fake_restore):
        asyncio.run(opt._record_iteration_timing(-1.349, 1.0))

    last = opt.checkpoint_manager.iterations[-1]
    assert last["status"] == "failed"
    assert last["error_type"] == "invalid_design_state"
    assert opt.action_failure_counts.get("rapidwright_optimize_cell_placement") == 1
    assert opt.checkpoint_manager.stall_count == 1
    assert restores, "best checkpoint must be restored after a rejected observation"
    # And the best numbers were never polluted.
    assert opt.checkpoint_manager.best_wns == -0.978


def _run_cell_placement(opt, unplaced_count):
    calls = []

    async def fake_call_tool(tool_name, arguments, internal=False):
        calls.append(tool_name)
        if tool_name == "rapidwright_optimize_cell_placement":
            return json.dumps({"results": [
                {"cell": "a", "status": "success"},
            ], "cells_moved": 1, "moved_cells": ["a"]})
        if tool_name == "vivado_run_tcl":
            return f"STATE_UNPLACED:{unplaced_count}"
        return "ok"

    async def fake_spread(cells):
        return None

    opt.iteration = 1
    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_measure_cluster_spread", side_effect=fake_spread):
        result = asyncio.run(opt.execute_validated_action(
            {"chosen_action": "rapidwright_optimize_cell_placement",
             "action_parameters": {"cell_names": ["a"]}},
            {"worst_path": {}, "delay_class": "net_delay_bound"},
        ))
    return result, calls


def test_grossly_unplaced_candidate_fails_before_routing(opt):
    """A candidate with unplaced primitives far beyond the benign tolerance
    must be rejected as a bare failure JSON, before spending a route."""
    result, calls = _run_cell_placement(opt, unplaced_count=80)
    payload = json.loads(result)
    assert payload["error_type"] == "invalid_design_state"
    assert "vivado_route_design" not in calls, "must fail before spending a route"


def test_benign_unplaced_candidate_still_routes(opt):
    """History(16) regression: this design ALWAYS shows ~6-8 unplaced
    primitives (input artifacts + failed moves). Those candidates must route
    and be measured -- rejecting them vetoed every improving recipe."""
    result, calls = _run_cell_placement(opt, unplaced_count=8)
    assert "vivado_route_design" in calls
    with pytest.raises(json.JSONDecodeError):
        json.loads(result)  # concatenated tool output, not a failure JSON


def test_verify_routed_state_tolerates_benign_unplaced(opt):
    """Clean route status + a handful of unplaced primitives = valid."""
    async def fake_call_tool(tool_name, arguments, internal=False):
        if tool_name == "vivado_run_tcl":
            return "STATE_UNPLACED:6"
        return "# of nets with routing errors.......... :           0 :"
    with patch.object(opt, "call_tool", side_effect=fake_call_tool):
        ok, detail = asyncio.run(opt._verify_routed_state())
    assert ok, detail
    assert "benign" in detail


# ---------------------------------------------------------------------------
# Fix 3: pblock_full_replace must not be intercepted / reordered
# ---------------------------------------------------------------------------

def test_full_replace_command_ordering(opt):
    calls = []

    async def fake_call_tool(tool_name, arguments, internal=False):
        calls.append((tool_name, internal))
        if tool_name == "vivado_create_and_apply_pblock":
            return json.dumps({"success": True, "cells_assigned": 100})
        return "ok"

    async def licensed():
        return True

    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_check_implementation_license", side_effect=licensed):
        opt.iteration = 1
        asyncio.run(opt._execute_pblock_full_replace({"ranges": "SLICE_X40Y100:SLICE_X120Y250"}))

    tool_seq = [name for name, _ in calls]
    # The 20260711 bug: phys_opt_design ran on the just-unplaced design
    # because create_and_apply_pblock was intercepted by call_tool().
    assert not any("phys_opt" in name for name in tool_seq), tool_seq
    apply_calls = [(name, internal) for name, internal in calls
                   if name == "vivado_create_and_apply_pblock"]
    assert apply_calls and apply_calls[0][1] is True
    order = {name: i for i, (name, _) in enumerate(calls)}
    assert order["vivado_create_and_apply_pblock"] < order["vivado_place_design"] < order["vivado_route_design"]


# ---------------------------------------------------------------------------
# Fix 4: hard failures feed failure memory (and cooldowns expire)
# ---------------------------------------------------------------------------

def test_hard_failures_reduce_selection(opt, tmp_path):
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    opt.checkpoint_manager.start_baseline(wns=-0.978, period_ns=1.5)
    opt.last_targets = ["pblock_full_replace_004", "SLICE_X40Y100:SLICE_X120Y250"]
    opt.last_recipe = "pblock_full_replace"
    opt.last_action_key = "pblock_full_replace"

    for iteration in (4, 6, 9):
        opt.iteration = iteration
        opt._record_failed_action({
            "error_type": "vivado_command_failure",
            "command": "pblock_full_replace",
            "message": "boom",
        })

    assert opt.action_failure_counts["pblock_full_replace"] == 3
    assert "pblock_full_replace" in opt._active_exhausted_actions()
    # No permanent deadlock: the cooldown expires.
    opt.iteration = opt.action_structural_cooldown_until_iter["pblock_full_replace"] + 1
    assert "pblock_full_replace" not in opt._active_exhausted_actions()


def test_license_failures_not_penalized(opt, tmp_path):
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    opt.checkpoint_manager.start_baseline(wns=-0.978, period_ns=1.5)
    opt.iteration = 5
    opt.last_recipe = "phys_opt_design"
    opt.last_action_key = "phys_opt_design"
    opt._record_failed_action({
        "error_type": "vivado_license_failure", "command": "phys_opt_design", "message": "no lic"})
    assert opt.action_failure_counts.get("phys_opt_design", 0) == 0
    assert opt.implementation_license_available is False


# ---------------------------------------------------------------------------
# Fix 2: summary/report use validated checkpoint history, not raw ratchets
# ---------------------------------------------------------------------------

def test_summary_and_report_use_validated_best(opt, tmp_path, capsys):
    cm = CheckpointManager(input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    cm.start_baseline(wns=-0.978, period_ns=1.5)
    opt.checkpoint_manager = cm
    # Poison the raw ratchets the way a mid-flight estimated WNS would.
    opt.best_wns = 0.4
    opt.best_fmax_mhz = 900.0
    opt.initial_wns = -0.978
    opt.clock_period = 1.5
    ckpt = tmp_path / "ckpt" / "iter_005.dcp"
    ckpt.write_bytes(b"dcp")
    cm.record(recipe="place_design_explore", targets=["x"], wns_after=-0.493,
              vivado_runtime_s=1.0, checkpoint_path=str(ckpt))

    opt._print_optimization_summary()
    out = capsys.readouterr().out
    assert "-0.493" in out
    assert "900.00" not in out
    assert "iter_005.dcp" in out

    report_path = tmp_path / "usage.json"
    opt.save_token_usage_report(report_path)
    summary = json.loads(report_path.read_text())["summary"]
    assert summary["best_wns"] == -0.493
    assert "iter_005.dcp" in str(summary["best_checkpoint"])
