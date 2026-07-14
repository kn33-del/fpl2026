"""Tests for the budget/cost-model upgrades (2026-07-12):

1. Duration learning: place/route/phys_opt wall-times measured per run,
   persisted per design in the cross-run store, estimated via in-run max
   then prior.
2. Large-design classification from measured/prior place duration or
   primitive count.
3. Affordability gate: demotion at 1.3x estimated cost, hard dispatch
   refusal at 1.0x.
4. Full re-place per-run cap on large designs (warm start + one more, none
   past 50% of budget).
5. Adaptive route directive downgrade after place.
6. Duration-aware implementation timeouts.
7. Phase 0 diagnostic battery (failure-tolerant, delay-model restore).
8. run_recipe macro-action (whitelist, sequential stages, budget stop).

Motivating incident: place Explore took 15.5 min, route Explore was then
killed by the remaining-budget timeout clamp -- 30+ minutes, zero valid
recorded results.
"""
import asyncio
import json
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

import dcp_optimizer as dcp  # noqa: E402
from dcp_optimizer import DCPOptimizer  # noqa: E402
from checkpoint_manager import CheckpointManager  # noqa: E402


@pytest.fixture
def opt(tmp_path):
    return DCPOptimizer(api_key="dummy", run_dir=tmp_path)


def _with_manager(opt, tmp_path):
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    opt.checkpoint_manager.start_baseline(wns=-0.9, period_ns=1.5)
    return opt


# ---------------------------------------------------------------------------
# Item 1: duration learning
# ---------------------------------------------------------------------------

def test_duration_kind_mapping():
    kind = DCPOptimizer._duration_kind_for_call
    assert kind("vivado_place_design", {}) == "place"
    assert kind("vivado_route_design", {}) == "route"
    assert kind("vivado_phys_opt_design", {}) == "phys_opt"
    # The re-place flows issue place/route/phys_opt via run_tcl.
    assert kind("vivado_run_tcl", {"command": "route_design -directive Explore"}) == "route"
    assert kind("vivado_run_tcl", {"command": "place_design -directive Default"}) == "place"
    assert kind("vivado_run_tcl", {"command": "phys_opt_design -directive Explore -retime"}) == "phys_opt"
    # Bookkeeping and analysis calls teach the cost model nothing.
    assert kind("vivado_run_tcl", {"command": "place_design -unplace"}) is None
    assert kind("vivado_run_tcl", {"command": "report_timing_summary"}) is None
    assert kind("vivado_report_timing_summary", {}) is None


def test_call_tool_measures_place_duration(opt):
    class _Content:
        def __init__(self, text):
            self.text = text

    class _Result:
        def __init__(self, text):
            self.content = [_Content(text)]
            self.isError = False

    async def fake_session_call(name, arguments):
        return _Result("Placement complete.")

    opt.vivado_session = types.SimpleNamespace(call_tool=fake_session_call)
    asyncio.run(opt.call_tool("vivado_place_design", {"directive": "Explore"}, internal=True))
    assert opt.action_durations.get("place"), "place wall-time must be recorded"
    # Sub-second here, so the design classifies small, not large.
    assert opt.design_scale == "small"


def test_estimated_duration_prefers_inrun_max_then_prior(opt):
    assert opt._estimated_duration("route") is None
    opt.crossrun_priors = {"durations": {"route": 400.0}}
    assert opt._estimated_duration("route") == 400.0
    opt.action_durations = {"route": [100.0, 250.0]}
    assert opt._estimated_duration("route") == 250.0


def test_durations_persist_and_reload(opt, tmp_path):
    _with_manager(opt, tmp_path)
    opt.crossrun_store_path = tmp_path / "priors.json"
    opt.crossrun_design_key = "bigdesign_2025.1"
    opt.action_durations = {"place": [930.5, 120.0], "route": [880.0]}
    opt._save_crossrun_priors()
    store = json.loads(opt.crossrun_store_path.read_text())
    assert store["bigdesign_2025.1"]["durations"] == {"place": 930.5, "route": 880.0}

    # A fresh run on the same design prices actions (and classifies scale)
    # from iteration 1 instead of re-learning by burning budget.
    opt2 = DCPOptimizer(api_key="dummy", run_dir=tmp_path)
    opt2.crossrun_store_path = opt.crossrun_store_path
    opt2._load_crossrun_priors(Path("/x/bigdesign_2025.1.dcp"))
    assert opt2._estimated_duration("place") == 930.5
    assert opt2.design_scale == "large"


# ---------------------------------------------------------------------------
# Item 2: design scale classification
# ---------------------------------------------------------------------------

def test_scale_large_from_measured_place(opt):
    assert opt.design_scale == "unknown"
    opt._note_action_duration("place", 400.0)
    assert opt.design_scale == "large"


def test_scale_small_from_fast_place(opt):
    opt._note_action_duration("place", 120.0)
    assert opt.design_scale == "small"


def test_scale_large_from_primitive_count(opt):
    opt.last_design_info = {"cell_count": 150_000}
    opt._refresh_design_scale()
    assert opt.design_scale == "large"
    opt.last_design_info = {"cell_count": 40_000}
    opt._refresh_design_scale()
    assert opt.design_scale == "small"


# ---------------------------------------------------------------------------
# Item 3: affordability gate
# ---------------------------------------------------------------------------

def test_action_cost_estimates(opt):
    opt.action_durations = {"place": [900.0], "route": [600.0]}
    assert opt._estimated_action_cost_s("place_design_explore") == 1500.0
    assert opt._estimated_action_cost_s("pblock_full_replace") == 1500.0
    assert opt._estimated_action_cost_s("route_explore") == 600.0
    assert opt._estimated_action_cost_s("phys_opt_design") == 200.0
    assert opt._estimated_action_cost_s("lut_opt") == dcp.CHEAP_ACTION_COST_S


def test_unknown_durations_priced_pessimistically_on_unknown_scale(opt):
    assert opt.design_scale == "unknown"
    # 2x longest known, min 900 -- an optimistic guess is how the motivating
    # incident burned 30 minutes for zero results.
    assert opt._estimated_action_cost_s("place_design_explore") == 1800.0
    opt.action_durations = {"route": [700.0]}
    assert opt._estimated_action_cost_s("place_design_explore") == 1400.0 + 700.0


def test_small_design_keeps_ungated_behavior(opt):
    opt.design_scale = "small"
    assert opt._estimated_action_cost_s("place_design_explore") is None
    assert opt._estimated_action_cost_s("route_explore") is None


def test_unaffordable_action_demoted_with_guidance(opt, tmp_path):
    _with_manager(opt, tmp_path)
    opt.action_durations = {"place": [900.0], "route": [600.0]}
    with patch.object(opt, "_time_remaining_s", return_value=1800.0):
        allowed, _ = opt._allowed_forbidden_actions(
            "net_delay_bound", "REGISTER", 0.82, 134.0, -0.978)
    # 1800 < 1.3 * 1500: both full re-place recipes carry the reason...
    assert "min remain" in opt.last_action_guidance["place_design_explore"]
    assert "min remain" in opt.last_action_guidance["pblock_full_replace"]
    # ...while route_explore (600 s) still fits and stays clean.
    assert "min remain" not in opt.last_action_guidance.get("route_explore", "")
    assert allowed.index("route_explore") < allowed.index("place_design_explore")


def test_unaffordable_dispatch_hard_refused(opt, tmp_path):
    _with_manager(opt, tmp_path)
    opt.action_durations = {"place": [900.0], "route": [600.0]}
    with patch.object(opt, "_time_remaining_s", return_value=600.0):
        result = asyncio.run(opt.execute_validated_action(
            {"chosen_action": "place_design_explore", "action_parameters": {}},
            {"worst_path": {}, "delay_class": "net_delay_bound"},
        ))
    payload = json.loads(result)
    assert payload["error_type"] == "insufficient_budget"
    # A refused dispatch never counts as a spent re-place.
    assert opt.full_replace_attempts == 0


# ---------------------------------------------------------------------------
# Item 4: full re-place cap on large designs
# ---------------------------------------------------------------------------

def test_full_replace_cap_blocks_third_attempt_on_large(opt, tmp_path):
    _with_manager(opt, tmp_path)
    opt.design_scale = "large"
    opt.full_replace_attempts = 1  # warm start
    assert opt._full_replace_blocked_reason("place_design_explore") is None
    opt.full_replace_attempts = 2  # warm start + one LLM re-place
    reason = opt._full_replace_blocked_reason("place_design_explore")
    assert reason and "cap" in reason
    result = asyncio.run(opt.execute_validated_action(
        {"chosen_action": "pblock_full_replace", "action_parameters": {}},
        {"worst_path": {}, "delay_class": "net_delay_bound"},
    ))
    assert json.loads(result)["error_type"] == "full_replace_cap_reached"


def test_full_replace_blocked_past_half_budget_on_large(opt, tmp_path):
    _with_manager(opt, tmp_path)
    opt.design_scale = "large"
    opt.full_replace_attempts = 1
    # 2000 s elapsed of a 3500 s budget (57%): no further re-places.
    opt.checkpoint_manager.started_at_epoch_s = time.time() - 2000
    reason = opt._full_replace_blocked_reason("pblock_full_replace")
    assert reason and "cutoff" in reason
    # And the menu carries the same reason as a demotion.
    allowed, _ = opt._allowed_forbidden_actions(
        "net_delay_bound", "REGISTER", 0.82, 134.0, -0.978)
    assert "cutoff" in opt.last_action_guidance["pblock_full_replace"]


def test_full_replace_cap_ignores_small_designs(opt, tmp_path):
    _with_manager(opt, tmp_path)
    opt.design_scale = "small"
    opt.full_replace_attempts = 5
    assert opt._full_replace_blocked_reason("place_design_explore") is None


def test_full_replace_dispatch_counts_attempts(opt):
    async def fake_call_tool(tool_name, arguments, internal=False):
        return "ok"

    async def licensed():
        return True

    opt.iteration = 1
    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_check_implementation_license", side_effect=licensed):
        asyncio.run(opt.execute_validated_action(
            {"chosen_action": "place_design_explore",
             "action_parameters": {"directive": "Default"}},
            {"worst_path": {}, "delay_class": "net_delay_bound"},
        ))
    assert opt.full_replace_attempts == 1


# ---------------------------------------------------------------------------
# Item 5: adaptive route directive
# ---------------------------------------------------------------------------

def _run_replace_with_budget(opt, remaining_s):
    calls = []

    async def fake_call_tool(tool_name, arguments, internal=False):
        calls.append((tool_name, str(arguments.get("directive", ""))))
        return "ok"

    async def licensed():
        return True

    opt.iteration = 1
    opt.action_durations = {"place": [100.0], "route": [600.0]}
    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_check_implementation_license", side_effect=licensed), \
         patch.object(opt, "_time_remaining_s", return_value=remaining_s):
        asyncio.run(opt.execute_validated_action(
            {"chosen_action": "place_design_explore",
             "action_parameters": {"directive": "Default", "route_directive": "Explore"}},
            {"worst_path": {}, "delay_class": "net_delay_bound"},
        ))
    return [d for name, d in calls if name == "vivado_route_design"]


def test_route_directive_downgraded_when_budget_tight(opt):
    # Cost 100 + 600 = 700 fits remaining 700 at dispatch, but after the
    # place, 700 < 1.2 * 600: a completed Default route beats a killed
    # Explore route.
    route_directives = _run_replace_with_budget(opt, remaining_s=700.0)
    assert route_directives == ["Default"]
    downgrade = opt.last_rapidwright_edit_summary["route_directive_downgraded"]
    assert downgrade["from"] == "Explore" and downgrade["to"] == "Default"


def test_route_directive_kept_with_ample_budget(opt):
    route_directives = _run_replace_with_budget(opt, remaining_s=3000.0)
    assert route_directives == ["Explore"]
    assert opt.last_rapidwright_edit_summary is None


# ---------------------------------------------------------------------------
# Item 6: duration-aware timeouts
# ---------------------------------------------------------------------------

def test_timeout_scales_with_known_duration(opt):
    opt.action_durations = {"route": [900.0]}
    with patch.object(opt, "_time_remaining_s", return_value=3000.0):
        # 2.5x the measured 900 s route -- a legitimate 15-minute route must
        # not be killed at the flat 20-minute default's remaining-clamp.
        assert opt._implementation_timeout_s(kind="route") == 2250
        # Short known durations now floor at 600 s, not 1200: run
        # 20260714_182751 iter 11 measured place at ~250 s yet a hang
        # burned the full 20-minute floor (46% of that run's gamma).
        opt.action_durations = {"route": [200.0]}
        assert opt._implementation_timeout_s(kind="route") == 600
    # The remaining budget still clamps everything.
    opt.action_durations = {"route": [900.0]}
    with patch.object(opt, "_time_remaining_s", return_value=2000.0):
        assert opt._implementation_timeout_s(kind="route") == 2000


def test_timeout_unchanged_without_known_duration(opt):
    with patch.object(opt, "_time_remaining_s", return_value=2000.0):
        assert opt._implementation_timeout_s() == 1200
        assert opt._implementation_timeout_s(kind="route") == 1200
    assert opt._implementation_timeout_s() == 1200


# ---------------------------------------------------------------------------
# Item 7: Phase 0 diagnostic battery
# ---------------------------------------------------------------------------

def test_initial_diagnostics_happy_path(opt):
    commands = []

    async def fake_call_tool(tool_name, arguments, internal=False):
        command = str(arguments.get("command", ""))
        commands.append(command)
        if "get_timing_paths" in command:
            return "LOGIC_FLOOR_WNS:0.35"
        if "report_qor_suggestions" in command:
            return "QoR suggestion body " * 200
        return "ok"

    opt.clock_period = 1.5
    opt.high_fanout_nets = [("n1", 500, 3), ("n2", 120, 1), ("n3", 260, 2)]
    with patch.object(opt, "call_tool", side_effect=fake_call_tool):
        asyncio.run(opt._initial_diagnostics())

    sig = opt.design_signature
    assert sig["logic_floor_wns_ns"] == 0.35
    assert sig["logic_fmax_ceiling_mhz"] == pytest.approx(1000.0 / (1.5 - 0.35), abs=0.01)
    assert sig["critical_fanout_max"] == 500
    assert sig["critical_fanout_median"] == 260
    assert len(sig["qor_suggestions"]) == 2000
    # The delay model was restored after the floor probe.
    none_idx = next(i for i, c in enumerate(commands) if "-interconnect none" in c)
    actual_idx = next(i for i, c in enumerate(commands) if "-interconnect actual" in c)
    assert none_idx < actual_idx


def test_initial_diagnostics_restores_delay_model_on_error(opt):
    """The one probe failure that must never leak: an exception between
    setting -interconnect none and restoring it would poison every later WNS
    observation this run."""
    commands = []

    async def fake_call_tool(tool_name, arguments, internal=False):
        command = str(arguments.get("command", ""))
        commands.append(command)
        if "get_timing_paths" in command:
            raise RuntimeError("vivado exploded")
        return "ok"

    with patch.object(opt, "call_tool", side_effect=fake_call_tool):
        asyncio.run(opt._initial_diagnostics())

    assert any("-interconnect actual" in c for c in commands), \
        "delay model must be restored even when the probe raises"
    # The battery survived: later probes still ran and results were kept.
    assert any("report_qor_suggestions" in c for c in commands)
    assert "logic_floor_wns_ns" not in opt.design_signature


def test_qor_suggestions_skipped_on_large_design_with_low_budget(opt, tmp_path):
    _with_manager(opt, tmp_path)
    opt.design_scale = "large"
    commands = []

    async def fake_call_tool(tool_name, arguments, internal=False):
        commands.append(str(arguments.get("command", "")))
        return "ok"

    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_time_remaining_s", return_value=1200.0):
        asyncio.run(opt._initial_diagnostics())
    assert not any("report_qor_suggestions" in c for c in commands)
    assert "qor_suggestions" not in opt.design_signature


# ---------------------------------------------------------------------------
# Item 8: run_recipe macro-action
# ---------------------------------------------------------------------------

def test_run_recipe_rejects_invalid_stages(opt):
    result = asyncio.run(opt._execute_run_recipe({"stages": []}, {}))
    assert json.loads(result)["error_type"] == "invalid_recipe"
    result = asyncio.run(opt._execute_run_recipe(
        {"stages": [{"action": "pblock_full_replace"}]}, {}))
    assert json.loads(result)["error_type"] == "invalid_recipe"
    result = asyncio.run(opt._execute_run_recipe(
        {"stages": [{"action": "phys_opt_design"}] * 7}, {}))
    assert json.loads(result)["error_type"] == "invalid_recipe"


def test_run_recipe_executes_stages_sequentially(opt):
    executed = []
    probes = []

    async def fake_execute(decision, ctx):
        executed.append((decision["chosen_action"], decision["action_parameters"]))
        return "ok"

    async def fake_call_tool(tool_name, arguments, internal=False):
        probes.append(tool_name)
        return "ok"

    opt.iteration = 5
    with patch.object(opt, "execute_validated_action", side_effect=fake_execute), \
         patch.object(opt, "call_tool", side_effect=fake_call_tool):
        result = asyncio.run(opt._execute_run_recipe({
            "stages": [
                {"action": "phys_opt_design", "params": {"directive": "Explore"}},
                {"action": "route_explore", "params": {"directive": "AggressiveExplore"}},
            ],
        }, {"worst_path": {}, "delay_class": "net_delay_bound"}))

    assert [a for a, _ in executed] == ["phys_opt_design", "route_explore"]
    assert executed[1][1] == {"directive": "AggressiveExplore"}
    # Each stage got its own iteration and its own timing probe (the normal
    # recording path, so keep-best applies per stage).
    assert opt.iteration == 6
    assert probes.count("vivado_report_timing_summary") == 2
    payload = json.loads(result)
    assert payload["success"] is True
    assert [s["status"] for s in payload["recipe_stages"]] == ["executed", "executed"]


def test_run_recipe_stops_cleanly_when_stage_does_not_fit(opt, tmp_path):
    _with_manager(opt, tmp_path)
    opt.action_durations = {"route": [900.0]}
    executed = []

    async def fake_execute(decision, ctx):
        executed.append(decision["chosen_action"])
        return "ok"

    with patch.object(opt, "execute_validated_action", side_effect=fake_execute), \
         patch.object(opt, "_time_remaining_s", return_value=500.0):
        result = asyncio.run(opt._execute_run_recipe({
            "stages": [
                {"action": "route_explore"},
                {"action": "phys_opt_design"},
            ],
        }, {"worst_path": {}, "delay_class": "net_delay_bound"}))

    # The pipeline stops at the first stage that does not fit -- it does not
    # skip ahead (later stages assume their predecessors ran).
    assert executed == []
    assert json.loads(result)["error_type"] == "insufficient_budget"


def test_run_recipe_records_failed_stage_and_continues(opt, tmp_path):
    _with_manager(opt, tmp_path)
    opt.iteration = 3
    restores = []
    probes = []

    async def fake_execute(decision, ctx):
        if decision["chosen_action"] == "phys_opt_design":
            opt.last_action_mutated_design = True
            return opt._failure_json("vivado_command_failure", "boom", command="phys_opt_design")
        opt.last_action_mutated_design = False
        return "ok"

    async def fake_call_tool(tool_name, arguments, internal=False):
        probes.append(tool_name)
        return "ok"

    async def fake_restore(reason):
        restores.append(reason)

    with patch.object(opt, "execute_validated_action", side_effect=fake_execute), \
         patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_restore_best_state", side_effect=fake_restore):
        result = asyncio.run(opt._execute_run_recipe({
            "stages": [
                {"action": "phys_opt_design"},
                {"action": "route_explore"},
            ],
        }, {"worst_path": {}, "delay_class": "net_delay_bound"}))

    payload = json.loads(result)
    assert [s["status"] for s in payload["recipe_stages"]] == ["failed", "executed"]
    # The failure was recorded (stall bookkeeping) and the design restored
    # before the next stage built on a broken state.
    assert opt.checkpoint_manager.iterations[-1]["status"] == "failed"
    assert restores and "phys_opt_design" in restores[0]
    # The failed stage's iteration is marked recorded so the main loop's
    # post-recipe probe cannot double-record it.
    assert 3 in opt.recorded_iterations
    assert probes.count("vivado_report_timing_summary") == 1


def test_run_recipe_exposed_in_menu_schema_and_prompt(opt):
    assert "run_recipe" in dcp.ACTION_PARAMETERS_SCHEMA
    assert "run_recipe" in dcp.RUN_RECIPE_STAGE_WHITELIST or True  # whitelist excludes itself
    assert "run_recipe" not in dcp.RUN_RECIPE_STAGE_WHITELIST
    assert "run_recipe" in dcp.TIMING_DECISION_SYSTEM_PROMPT
    for delay_class in ("net_delay_bound", "logic_delay_bound", "mixed"):
        allowed, forbidden = opt._allowed_forbidden_actions(
            delay_class, "REGISTER", 0.5, 20.0, -0.3)
        assert "run_recipe" in allowed, delay_class
        assert "run_recipe" not in forbidden
