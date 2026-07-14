"""Tests for the priors/learning upgrades (2026-07-11):

1. Gates -> priors: delay-class exclusions become ranked demotions with
   recorded guidance; forbidden_actions holds hard blocks only.
2. Diagnosis -> outcome loop: hypothesis confidence shifts with this run's
   recorded results of its own prescriptions.
3. Search within a recipe: place_design directive sweep; pblock sizing grows
   after a clean-but-regressed attempt instead of repeating.
4. Congestion + clock regions: parsers, evidence, and the pblock demotion.
"""
import asyncio
import json
import sys
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
from dcp_optimizer import DCPOptimizer, PLACE_DIRECTIVE_SWEEP  # noqa: E402
from analysis_layer import (  # noqa: E402
    AnalysisEngine,
    Diagnosis,
    RootCauseHypothesis,
    CONFIDENCE_FLOOR,
)
from checkpoint_manager import CheckpointManager  # noqa: E402


@pytest.fixture
def opt(tmp_path):
    return DCPOptimizer(api_key="dummy", run_dir=tmp_path)


# ---------------------------------------------------------------------------
# Item 1: gates -> priors
# ---------------------------------------------------------------------------

def test_net_delay_bound_discourages_instead_of_forbidding(opt):
    allowed, forbidden = opt._allowed_forbidden_actions(
        "net_delay_bound", "REGISTER", 0.82, 134.0, -0.978)
    # Previously-forbidden logic actions are now allowed, ranked last, with reasons.
    assert "fanout_split" in allowed
    assert "lut_opt" in allowed
    assert forbidden == ["logic_restructure"]
    assert "fanout_split" in opt.last_action_guidance
    assert "net_delay_bound" in opt.last_action_guidance["fanout_split"]
    # Demoted actions rank below the structural ones.
    assert allowed.index("pblock") < allowed.index("lut_opt")


def test_place_design_explore_not_demoted_by_wns_floor(opt):
    # Run 20260711: place_design_explore at WNS -0.92 was the only win.
    allowed, _ = opt._allowed_forbidden_actions(
        "net_delay_bound", "REGISTER", 0.82, 134.0, -0.978)
    assert "place_design_explore" not in opt.last_action_guidance
    # Incremental phys_opt IS demoted with a reason at that WNS.
    assert "phys_opt_design" in opt.last_action_guidance


def test_logic_delay_bound_demotes_placement(opt):
    allowed, forbidden = opt._allowed_forbidden_actions(
        "logic_delay_bound", "REGISTER", 0.2, 5.0, -0.2)
    assert allowed[0] == "lut_opt"
    assert "pblock" in allowed and "pblock" in opt.last_action_guidance
    assert "logic_restructure" in forbidden


# ---------------------------------------------------------------------------
# Item 2: outcome loop
# ---------------------------------------------------------------------------

def _engine_with_history(opt, tmp_path, records):
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    opt.checkpoint_manager.start_baseline(wns=-0.9, period_ns=1.5)
    opt.checkpoint_manager.iterations = records
    return AnalysisEngine(opt)


def test_outcome_adjustment_penalizes_losing_hypothesis(opt, tmp_path):
    records = [
        {"primary_diagnosis": "long_interconnect",
         "llm_chosen_action": "rapidwright_optimize_cell_placement",
         "status": "regression"},
        {"primary_diagnosis": "long_interconnect",
         "llm_chosen_action": "rapidwright_optimize_cell_placement",
         "status": "failed"},
    ]
    engine = _engine_with_history(opt, tmp_path, records)
    hyp = RootCauseHypothesis(
        name="long_interconnect", cluster_id="c0", confidence=0.75,
        supporting_evidence=[], contradicting_evidence=[],
        recommended_actions=["rapidwright_optimize_cell_placement"],
        evidence_requests=[])
    delta, note = engine._outcome_adjustment(hyp)
    assert delta < 0
    assert "2 loss(es)" in note


def test_outcome_adjustment_ignores_other_actions(opt, tmp_path):
    # A loss recorded under a different chosen action doesn't count against
    # the hypothesis -- it only owns results of its own prescriptions.
    records = [
        {"primary_diagnosis": "long_interconnect",
         "llm_chosen_action": "pblock", "status": "regression"},
    ]
    engine = _engine_with_history(opt, tmp_path, records)
    hyp = RootCauseHypothesis(
        name="long_interconnect", cluster_id="c0", confidence=0.75,
        supporting_evidence=[], contradicting_evidence=[],
        recommended_actions=["rapidwright_optimize_cell_placement"],
        evidence_requests=[])
    delta, _ = engine._outcome_adjustment(hyp)
    assert delta == 0.0


def test_outcome_adjustment_reranks(opt, tmp_path):
    records = [
        {"primary_diagnosis": "a", "llm_chosen_action": "x", "status": "regression"},
        {"primary_diagnosis": "a", "llm_chosen_action": "x", "status": "regression"},
    ]
    engine = _engine_with_history(opt, tmp_path, records)
    losing = RootCauseHypothesis(
        name="a", cluster_id="c0", confidence=0.70,
        supporting_evidence=[], contradicting_evidence=[],
        recommended_actions=["x"], evidence_requests=[])
    runner_up = RootCauseHypothesis(
        name="b", cluster_id="c0", confidence=0.65,
        supporting_evidence=[], contradicting_evidence=[],
        recommended_actions=["y"], evidence_requests=[])
    hypotheses = [losing, runner_up]
    engine._apply_outcome_adjustments(hypotheses, [])
    assert hypotheses[0].name == "b"


# ---------------------------------------------------------------------------
# Item 3a: directive sweep
# ---------------------------------------------------------------------------

def test_directive_sweep_advances(opt):
    assert opt._next_place_directive() == PLACE_DIRECTIVE_SWEEP[0]
    opt.place_directive_results["Explore"] = {"status": "improved", "wns_after": -0.493}
    assert opt._next_place_directive() == PLACE_DIRECTIVE_SWEEP[1]


def test_directive_sweep_reuses_best_when_exhausted(opt):
    for i, directive in enumerate(PLACE_DIRECTIVE_SWEEP):
        opt.place_directive_results[directive] = {
            "status": "no_improvement", "wns_after": -1.0 + 0.1 * i}
    opt.place_directive_results["ExtraTimingOpt"] = {"status": "improved", "wns_after": -0.1}
    assert opt._next_place_directive() == "ExtraTimingOpt"


def test_directive_outcome_recorded(opt):
    opt.last_action_key = "place_design_explore"
    opt.last_place_directive = "Explore"
    opt.iteration = 3
    opt._note_recipe_outcome("improved", -0.493)
    assert opt.place_directive_results["Explore"]["status"] == "improved"


def test_place_design_explore_unplaces_first(opt):
    """History(16) forensics: place_design is incremental over an existing
    placement, so without an unplace it is a no-op and the directive sweep
    sweeps nothing. The executor must run `place_design -unplace` before
    place and route, in that order."""
    calls = []

    async def fake_call_tool(tool_name, arguments, internal=False):
        calls.append((tool_name, str(arguments.get("command", "")), str(arguments.get("directive", ""))))
        return "ok"

    async def licensed():
        return True

    opt.iteration = 1
    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_check_implementation_license", side_effect=licensed):
        asyncio.run(opt.execute_validated_action(
            {"chosen_action": "place_design_explore",
             "action_parameters": {"directive": "ExtraTimingOpt"}},
            {"worst_path": {}, "delay_class": "net_delay_bound"},
        ))

    unplace_idx = next(i for i, c in enumerate(calls) if "place_design -unplace" in c[1])
    place_idx = next(i for i, c in enumerate(calls) if c[0] == "vivado_place_design")
    route_idx = next(i for i, c in enumerate(calls) if c[0] == "vivado_route_design")
    assert unplace_idx < place_idx < route_idx
    assert calls[place_idx][2] == "ExtraTimingOpt"
    assert opt.last_place_directive == "ExtraTimingOpt"


def _run_warm_start(opt, initial_wns, classification):
    captured = {}

    async def fake_classify():
        return classification

    async def fake_execute(decision, timing_context):
        captured["decision"] = decision
        return "ok"

    async def fake_call_tool(tool_name, arguments, internal=False):
        return "ok"

    opt.initial_wns = initial_wns
    with patch.object(opt, "_classify_worst_path_delay", side_effect=fake_classify), \
         patch.object(opt, "execute_validated_action", side_effect=fake_execute), \
         patch.object(opt, "call_tool", side_effect=fake_call_tool):
        ran = asyncio.run(opt._maybe_warm_start_replace())
    return ran, captured


def test_warm_start_skips_when_timing_met(opt):
    ran, _ = _run_warm_start(opt, initial_wns=0.05, classification="net_delay_bound")
    assert not ran
    assert opt.iteration == 0


def test_warm_start_skips_when_logic_bound(opt):
    ran, _ = _run_warm_start(opt, initial_wns=-0.9, classification="logic_delay_bound")
    assert not ran


def test_warm_start_runs_replace_on_net_bound_failing_design(opt):
    ran, captured = _run_warm_start(opt, initial_wns=-0.978, classification="net_delay_bound")
    assert ran
    assert opt.iteration == 1
    decision = captured["decision"]
    assert decision["chosen_action"] == "place_design_explore"
    # The recorded winning directive pair from the 501 MHz result.
    assert decision["action_parameters"] == {"directive": "Default", "route_directive": "Default"}
    assert any("warm-start" in str(m.get("content", "")).lower() for m in opt.messages)


def test_exploit_after_win_promotes_refinement(opt, tmp_path):
    """Run 20260712_051231 measured: re-rolls from a winning state went 0/3,
    refinement 5/6. When the last recorded iteration improved (stall_count
    resets to 0), refinement must lead the ranking and fresh re-places must
    carry a demotion reason."""
    cm = CheckpointManager(input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    cm.start_baseline(wns=-0.978, period_ns=1.5)
    ckpt = tmp_path / "ckpt" / "iter_001.dcp"
    ckpt.write_bytes(b"dcp")
    cm.record(recipe="place_design_explore", targets=["directive:Default"],
              wns_after=-0.494, vivado_runtime_s=1.0, checkpoint_path=str(ckpt))
    opt.checkpoint_manager = cm
    assert cm.stall_count == 0

    allowed, _ = opt._allowed_forbidden_actions(
        "net_delay_bound", "REGISTER", 0.74, 112.0, -0.494)
    assert allowed[0] == "pblock"
    assert "phys_opt_design" in allowed[:3]
    assert "place_design_explore" in opt.last_action_guidance
    # Score item C reworded the reason: the guard now protects the unbeaten
    # best through stalls, not just the iteration right after a win.
    assert "unbeaten best" in opt.last_action_guidance["place_design_explore"]


def test_no_refinement_promotion_after_stall(opt, tmp_path):
    cm = CheckpointManager(input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    cm.start_baseline(wns=-0.978, period_ns=1.5)
    ckpt = tmp_path / "ckpt" / "iter_001.dcp"
    ckpt.write_bytes(b"dcp")
    cm.record(recipe="place_design_explore", targets=["x"],
              wns_after=-1.2, vivado_runtime_s=1.0, checkpoint_path=str(ckpt))
    opt.checkpoint_manager = cm
    assert cm.stall_count == 1

    allowed, _ = opt._allowed_forbidden_actions(
        "net_delay_bound", "REGISTER", 0.74, 112.0, -0.978)
    # No fresh win -> normal ranking, re-place family not demoted for that reason.
    assert "IMPROVED" not in opt.last_action_guidance.get("place_design_explore", "")


def test_route_explore_reroutes_without_touching_placement(opt):
    calls = []

    async def fake_call_tool(tool_name, arguments, internal=False):
        calls.append((tool_name, str(arguments.get("directive", "")), internal))
        return "route_design completed successfully"

    async def licensed():
        return True

    opt.iteration = 1
    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_check_implementation_license", side_effect=licensed):
        result = asyncio.run(opt.execute_validated_action(
            {"chosen_action": "route_explore", "action_parameters": {"directive": "AggressiveExplore"}},
            {"worst_path": {}, "delay_class": "net_delay_bound"},
        ))
    assert calls == [("vivado_route_design", "AggressiveExplore", True)]
    assert "completed" in result
    assert not any(name == "vivado_place_design" for name, _, _ in calls)


def test_route_explore_refuses_unrouted_state(opt):
    async def licensed():
        return True

    opt.design_state = "placed"
    with patch.object(opt, "_check_implementation_license", side_effect=licensed):
        result = asyncio.run(opt.execute_validated_action(
            {"chosen_action": "route_explore", "action_parameters": {}},
            {"worst_path": {}, "delay_class": "net_delay_bound"},
        ))
    assert json.loads(result)["error_type"] == "invalid_design_state"


def test_endgame_polish_skips_without_budget(opt, tmp_path):
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    opt.checkpoint_manager.start_baseline(wns=-0.9, period_ns=1.5)
    executed = []

    async def fake_execute(decision, ctx):
        executed.append(decision["chosen_action"])
        return "ok"

    with patch.object(opt, "_time_remaining_s", return_value=120.0), \
         patch.object(opt, "execute_validated_action", side_effect=fake_execute):
        asyncio.run(opt._endgame_polish())
    assert executed == []


def test_endgame_polish_spends_remaining_budget(opt, tmp_path):
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    opt.checkpoint_manager.start_baseline(wns=-0.45, period_ns=1.5)
    executed = []
    restores = []

    async def fake_execute(decision, ctx):
        executed.append(decision["chosen_action"])
        return "ok"

    async def fake_call_tool(tool_name, arguments, internal=False):
        return "ok"

    async def fake_restore(reason):
        restores.append(reason)

    with patch.object(opt, "_time_remaining_s", return_value=2000.0), \
         patch.object(opt, "execute_validated_action", side_effect=fake_execute), \
         patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_restore_best_state", side_effect=fake_restore):
        asyncio.run(opt._endgame_polish())
    # Restored to best before the polish chain (a reconstrain-focus restore may
    # precede it now; what matters is the endgame restore happened and the
    # chain ran).
    assert any("endgame" in reason for reason in restores)
    assert "phys_opt_design" in executed and "route_explore" in executed


def test_crossrun_priors_roundtrip_and_demotion(opt, tmp_path):
    opt.crossrun_store_path = tmp_path / "priors.json"
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    opt.checkpoint_manager.start_baseline(wns=-0.978, period_ns=1.5)
    # Simulate a finished run's ledger: cell placement lost 3x, Default won.
    opt.checkpoint_manager.iterations = [
        {"llm_chosen_action": "rapidwright_optimize_cell_placement", "status": "failed", "targets": []},
        {"llm_chosen_action": "rapidwright_optimize_cell_placement", "status": "regression", "targets": []},
        {"llm_chosen_action": "rapidwright_optimize_cell_placement", "status": "failed", "targets": []},
        {"llm_chosen_action": "place_design_explore", "status": "improved",
         "targets": ["directive:Default"]},
    ]
    opt.crossrun_design_key = "logicnets_jscl_2025.1"
    opt._save_crossrun_priors()
    # Saving twice must not double-count.
    opt._save_crossrun_priors()
    store = json.loads(opt.crossrun_store_path.read_text())
    record = store["logicnets_jscl_2025.1"]["actions"]["rapidwright_optimize_cell_placement"]
    assert record == {"good": 0, "bad": 3}
    assert store["logicnets_jscl_2025.1"]["directives"]["Default"]["good"] == 1

    # A fresh optimizer on the same design loads them and demotes the loser.
    from pathlib import Path
    opt2 = DCPOptimizer(api_key="dummy", run_dir=tmp_path)
    opt2.crossrun_store_path = opt.crossrun_store_path
    opt2._load_crossrun_priors(Path("/x/logicnets_jscl_2025.1.dcp"))
    assert opt2._best_crossrun_directive() == "Default"
    allowed, _ = opt2._allowed_forbidden_actions(
        "net_delay_bound", "REGISTER", 0.82, 112.0, -0.978)
    assert "rapidwright_optimize_cell_placement" in opt2.last_action_guidance
    assert "previous runs" in opt2.last_action_guidance["rapidwright_optimize_cell_placement"]
    assert allowed[-1] == "rapidwright_optimize_cell_placement" or \
        allowed.index("rapidwright_optimize_cell_placement") > allowed.index("pblock")


def test_long_interconnect_recommends_global_replace_first():
    """Cells 100+ tiles apart need a global re-place, not local nudges; the
    hypothesis's recommendation order must lead with the recipe family that
    has recorded wins, so both the LLM ranking and the outcome loop's
    win/loss attribution point at the right actions."""
    from analysis_layer import _rule_long_interconnect, FailureCluster
    cluster = FailureCluster(id="c0", members=[], shared_cells=set(), fanout_hotspots=[])
    hyp = _rule_long_interconnect(cluster, {"cluster_avg_spread": 134.0, "net_pct": 0.82})
    assert hyp.recommended_actions[0] == "place_design_explore"
    assert hyp.recommended_actions[1] == "pblock_full_replace"
    assert "rapidwright_optimize_cell_placement" in hyp.recommended_actions


def test_place_design_explore_in_structural_families():
    """The stuck-detector's structural override must never hide the one
    recipe family with a proven win on this benchmark."""
    assert "place_design_explore" in dcp.RAPIDWRIGHT_STRUCTURAL_ACTIONS
    assert "place_design_explore" in dcp.RAPIDWRIGHT_PLACEMENT_ACTIONS


def test_clockregion_pblock_falls_back_to_site_ranges(opt):
    """History(16) iter 1: a clock-region conversion that misses the cluster
    must auto-fall back to site ranges of the same fabric region instead of
    failing the iteration."""
    opt.current_target_candidates = [{"endpoint": "e", "startpoint": "s", "slack": -0.5}] * 5
    convert_calls = []

    async def fake_call_tool(tool_name, arguments, internal=False):
        if tool_name == "rapidwright_analyze_fabric_for_pblock":
            return json.dumps({"recommended_region": {
                "col_min": 301, "col_max": 321, "row_min": 28, "row_max": 82}})
        if tool_name == "rapidwright_convert_fabric_region_to_pblock":
            convert_calls.append(bool(arguments.get("use_clock_regions")))
            if arguments.get("use_clock_regions"):
                return json.dumps({"pblock_ranges": "CLOCKREGION_X5Y0:CLOCKREGION_X5Y1"})
            return json.dumps({
                "pblock_ranges": "SLICE_X6Y0:SLICE_X142Y599",
                "site_counts": {"LUT": 100000, "FF": 200000},
            })
        return "ok"

    with patch.object(opt, "call_tool", side_effect=fake_call_tool):
        computed, error = asyncio.run(opt._compute_pblock_ranges(
            {"use_clock_regions": True},
            {"cluster_clock_regions": ["X1Y4", "X2Y4"]},
        ))
    assert error is None
    assert convert_calls == [True, False]
    assert computed["ranges"].startswith("SLICE_")


def test_full_replace_aborts_on_resource_validation_errors(opt):
    """History(16) iter 4: the DRC flagged 31370 LUTs into 26880 BEFORE
    placement; the flow must abort there instead of burning the place."""
    calls = []

    async def fake_call_tool(tool_name, arguments, internal=False):
        calls.append(tool_name)
        if tool_name == "vivado_create_and_apply_pblock":
            return json.dumps({
                "success": True, "cells_assigned": 100,
                "resource_validation": {
                    "is_valid": True,
                    "errors": ["LUT as Logic: 31370 assigned, only 26880 available"],
                },
            })
        return "ok"

    async def licensed():
        return True

    opt.iteration = 1
    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_check_implementation_license", side_effect=licensed):
        result = asyncio.run(opt._execute_pblock_full_replace(
            {"ranges": "SLICE_X0Y240:SLICE_X59Y299"}))
    payload = json.loads(result)
    assert payload["error_type"] == "full_replace_region_too_small"
    assert "vivado_place_design" not in calls


# ---------------------------------------------------------------------------
# Item 3b: pblock sizing memory
# ---------------------------------------------------------------------------

def test_pblock_regrows_after_regression(opt):
    opt.current_target_candidates = [{"endpoint": "e", "startpoint": "s", "slack": -0.5}] * 5
    opt.pblock_attempt_history.append({
        "action": "pblock", "iteration": 2, "status": "regression",
        "wns_after": -1.1, "target_lut_count": 800, "target_ff_count": 1600,
    })
    requested = {}

    async def fake_call_tool(tool_name, arguments, internal=False):
        if tool_name == "rapidwright_analyze_fabric_for_pblock":
            requested.update(arguments)
            return json.dumps({"recommended_region": {
                "col_min": 1, "col_max": 10, "row_min": 1, "row_max": 10}})
        if tool_name == "rapidwright_convert_fabric_region_to_pblock":
            return json.dumps({
                "pblock_ranges": "SLICE_X1Y1:SLICE_X10Y10",
                "site_counts": {"LUT": 100000, "FF": 200000},
            })
        return "ok"

    with patch.object(opt, "call_tool", side_effect=fake_call_tool):
        computed, error = asyncio.run(opt._compute_pblock_ranges({}, {}))
    assert error is None
    # 800 LUTs regressed -> next request grows to >= 800 * 1.5 = 1200.
    assert requested["target_lut_count"] >= 1200
    assert opt.last_pblock_sizing["target_lut_count"] >= 1200


def test_pblock_attempt_recorded_on_failure(opt, tmp_path):
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    opt.checkpoint_manager.start_baseline(wns=-0.9, period_ns=1.5)
    opt.last_action_key = "pblock_full_replace"
    opt.last_recipe = "pblock_full_replace"
    opt.last_targets = ["full_design", "SLICE_X40Y100:SLICE_X120Y250"]
    opt.last_pblock_sizing = {"target_lut_count": 30000, "target_ff_count": 60000}
    opt.iteration = 4
    opt._record_failed_action({
        "error_type": "vivado_command_failure",
        "command": "pblock_full_replace", "message": "boom"})
    assert opt.pblock_attempt_history[-1]["status"] == "failed"
    assert opt.pblock_attempt_history[-1]["target_lut_count"] == 30000


# ---------------------------------------------------------------------------
# Item 4: congestion + clock regions
# ---------------------------------------------------------------------------

def test_congestion_parser_strict():
    level, _ = DCPOptimizer._parse_congestion_report(
        "| NORTH | 5 | 12.3 |\n| SOUTH | 3 | 1.0 |")
    assert level == 5
    # The router's authoritative per-direction summary wins.
    level, _ = DCPOptimizer._parse_congestion_report(
        "Direction: North\nCongested clusters found at Level 0\n"
        "Effective congestion level: 0 Aspect Ratio: 1 Sparse Ratio: 0\n"
        "Direction: South\nEffective congestion level: 0 Aspect Ratio: 1")
    assert level == 0
    # Run 20260712 regression: loose text (window sizes, "Max Cong = 68%",
    # prose mentioning levels) must NOT produce a level -- a false positive
    # demotes good actions.
    level, detail = DCPOptimizer._parse_congestion_report(
        "North Dir 1x1 Area, Max Cong = 68.0751%, No Congested Regions.\n"
        "congested window of 32x32 tiles")
    assert level is None and "no congestion level" in detail
    level, _ = DCPOptimizer._parse_congestion_report("no table here")
    assert level is None


def test_clockregion_ranges_cover():
    # Run 20260712 iter 3: cluster in X1Y4/X2Y4, computed range X5Y0:X5Y1.
    assert not DCPOptimizer._clockregion_ranges_cover(
        "CLOCKREGION_X5Y0:CLOCKREGION_X5Y1", {"X1Y4", "X2Y4"})
    assert DCPOptimizer._clockregion_ranges_cover(
        "CLOCKREGION_X1Y3:CLOCKREGION_X2Y4", {"X1Y4"})
    assert DCPOptimizer._clockregion_ranges_cover(
        "CLOCKREGION_X2Y4", {"X2Y4"})
    # Unparsable ranges must not block.
    assert DCPOptimizer._clockregion_ranges_cover(
        "SLICE_X0Y0:SLICE_X10Y10", {"X1Y4"})


def test_congestion_harvested_from_route_log(opt):
    opt.iteration = 3
    opt._update_design_state(
        "vivado_route_design", {},
        "Routing complete.\nEffective congestion level: 4 Aspect Ratio: 1\n"
        "route_design completed successfully")
    assert opt.last_congestion_info["congestion_level"] == 4
    # A later fetch that can't parse report_design_analysis keeps the real
    # measurement instead of downgrading it to unknown.
    opt.iteration = 4

    async def fake_call_tool(tool_name, arguments, internal=False):
        return "unparsable report"
    with patch.object(opt, "call_tool", side_effect=fake_call_tool):
        info = asyncio.run(opt._fetch_congestion_summary())
    assert info["congestion_level"] == 4


def test_clock_region_fetch(opt):
    async def fake_call_tool(tool_name, arguments, internal=False):
        return "CLOCKREGION|cell_a|X0Y1\nCLOCKREGION|cell_b|X1Y2\nnoise"
    with patch.object(opt, "call_tool", side_effect=fake_call_tool):
        regions = asyncio.run(opt._fetch_clock_regions(["cell_a", "cell_b"]))
    assert regions == {"cell_a": "X0Y1", "cell_b": "X1Y2"}


def _make_diagnosis(hyp, congestion_level=None):
    return Diagnosis(
        clusters=[], hypotheses=[hyp], primary_cluster_id="c0",
        primary_hypothesis=hyp, reasoning_trace=[],
        delay_class="net_delay_bound", endpoint_type="REGISTER",
        net_pct=0.82, avg_spread=134.0, congestion_level=congestion_level)


def test_high_congestion_demotes_pblock_family(opt):
    engine = AnalysisEngine(opt)
    hyp = RootCauseHypothesis(
        name="independent_failures", cluster_id="c0", confidence=0.5,
        supporting_evidence=[], contradicting_evidence=[],
        recommended_actions=[], evidence_requests=[])
    diagnosis = _make_diagnosis(hyp, congestion_level=6)
    allowed, forbidden = engine.actions_for(diagnosis, current_wns=-0.9)
    assert "pblock" in allowed and "pblock_full_replace" in allowed
    assert allowed.index("place_design_explore") < allowed.index("pblock")
    assert "congestion" in opt.last_action_guidance["pblock"]
    assert "pblock" not in forbidden


def test_confident_hypothesis_demotes_not_forbids(opt):
    engine = AnalysisEngine(opt)
    hyp = RootCauseHypothesis(
        name="placement_already_compact", cluster_id="c0",
        confidence=CONFIDENCE_FLOOR + 0.05,
        supporting_evidence=["cluster is tightly placed"],
        contradicting_evidence=[],
        recommended_actions=["phys_opt_design", "phys_opt_design_retime", "lut_opt"],
        evidence_requests=[],
        veto_actions=["pblock", "rapidwright_optimize_cell_placement"])
    diagnosis = _make_diagnosis(hyp)
    allowed, forbidden = engine.actions_for(diagnosis, current_wns=-0.2)
    # Vetoed actions stay choosable (demoted with guidance), never forbidden.
    assert "pblock" in allowed and "pblock" not in forbidden
    assert "placement_already_compact" in opt.last_action_guidance["pblock"]
    # Recommendations lead the ranking, and promotion clears their guidance.
    assert allowed[0] == "phys_opt_design"
    assert "lut_opt" not in opt.last_action_guidance


# --- Score items 1 & 2 (2026-07-13): reconstrain focus pass + convergence exit ---

def test_at_logic_ceiling_fires_only_near_ceiling(opt, tmp_path):
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    opt.checkpoint_manager.start_baseline(wns=-0.45, period_ns=1.5)
    opt.design_signature = {"logic_fmax_ceiling_mhz": 520.0}
    # Well below the ceiling -> keep going.
    opt.checkpoint_manager.best_fmax_mhz = 480.0
    assert opt._at_logic_ceiling() is None
    # Within 97% of the ceiling (>= 504.4 MHz) -> converged.
    opt.checkpoint_manager.best_fmax_mhz = 510.0
    reason = opt._at_logic_ceiling()
    assert reason is not None and "ceiling" in reason
    # No ceiling measured -> never fires (avoids a false early stop).
    opt.design_signature = {}
    assert opt._at_logic_ceiling() is None


def test_reconstrain_focus_always_restores_contest_period(opt):
    # The safety invariant: whatever happens inside (even a routing error),
    # the contest clock period must be restored before recording.
    opt._reconstrain_focus_done = False
    opt.clock_period = 1.5
    opt.target_clock = "clk_fpl26contest"
    opt.implementation_license_available = True
    opt.iteration = 3

    periods = []

    async def fake_set_period(p):
        periods.append(round(p, 4))
        return True

    async def fake_get_wns():
        return -9.0  # deeply unmet -> pass should run

    async def fake_phys(_params):
        return "ok"

    async def fake_call_tool(tool_name, arguments, internal=False):
        if tool_name == "vivado_route_design":
            return "ERROR: route blew up"  # force the error branch
        return "ok"

    with patch.object(opt, "_set_clock_period", side_effect=fake_set_period), \
         patch.object(opt, "_get_current_wns", side_effect=fake_get_wns), \
         patch.object(opt, "_run_phys_opt_with_policy", side_effect=fake_phys), \
         patch.object(opt, "_time_remaining_s", return_value=3000.0), \
         patch.object(opt, "call_tool", side_effect=fake_call_tool):
        ran = asyncio.run(opt._reconstrain_focus_pass())

    assert ran is True
    # Relaxed to 0.95 * achieved delay (1.5 - (-9.0) = 10.5 -> 9.975), then
    # restored to the contest period 1.5 last.
    assert periods[0] == pytest.approx(9.975, abs=1e-3)
    assert periods[-1] == 1.5


def test_menu_collapse_stops_only_on_all_losers(opt, tmp_path):
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    opt.checkpoint_manager.start_baseline(wns=-0.978, period_ns=1.5)
    opt.consecutive_no_improvement = 3
    opt.crossrun_priors = {"actions": {
        "pblock_full_replace": {"good": 0, "bad": 3},
        "rapidwright_optimize_cell_placement": {"good": 0, "bad": 4},
    }}
    # place_design_explore: 1 win but 2 regressions THIS run -> fresh loser.
    opt.checkpoint_manager.iterations = [
        {"llm_chosen_action": "place_design_explore", "status": "improved"},
        {"llm_chosen_action": "place_design_explore", "status": "regression"},
        {"llm_chosen_action": "place_design_explore", "status": "regression"},
    ]
    opt.last_timing_context = {"allowed_actions": [
        "pblock_full_replace", "place_design_explore",
        "rapidwright_optimize_cell_placement"]}
    reason = opt._menu_collapse_reason()
    assert reason is not None and "menu collapsed" in reason
    # One healthy action on the menu -> no collapse.
    opt.last_timing_context["allowed_actions"].append("phys_opt_design")
    assert opt._menu_collapse_reason() is None
    # Same losers but not stalled long enough -> no collapse.
    opt.last_timing_context["allowed_actions"].pop()
    opt.consecutive_no_improvement = 2
    assert opt._menu_collapse_reason() is None


def test_protecting_best_persists_through_stalls(opt, tmp_path):
    cm = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    cm.start_baseline(wns=-0.978, period_ns=1.5)
    opt.checkpoint_manager = cm
    assert opt._sitting_on_fresh_win() is False  # nothing recorded yet
    ck = tmp_path / "iter.dcp"
    ck.write_text("x")
    cm.record("place_design_explore", ["directive:Default"], -0.494, 10, str(ck))
    assert cm.stall_count == 0
    assert opt._sitting_on_fresh_win() is True   # fresh win
    cm.record("route_explore", ["directive:Explore"], -0.494, 10, str(ck))
    assert cm.stall_count == 1
    # Score item C: the unbeaten best is still at stake after a stall.
    assert opt._sitting_on_fresh_win() is True


def test_llm_failure_memory_drops_fingerprint_blobs(opt):
    opt.iteration = 8
    opt.action_failure_memory = {"pblock": {
        "consecutive_no_action_failures": 2,
        "failed_targets": ["a", "b", "c", "d", "e"],
        "target_fingerprint": json.dumps({"candidates": [{"slack": -0.4}] * 5}),
        "last_failed_iter": 7,
    }}
    compact = opt._llm_action_failure_memory()
    assert "target_fingerprint" not in compact["pblock"]
    assert compact["pblock"]["failed_targets"] == ["a", "b", "c"]
    assert compact["pblock"]["last_failed_iter"] == 7
    assert "failed_on_current_targets" in compact["pblock"]


def test_reconstrain_focus_noop_when_not_deeply_unmet(opt):
    opt._reconstrain_focus_done = False
    opt.clock_period = 1.5
    opt.target_clock = "clk_fpl26contest"
    opt.implementation_license_available = True

    async def fake_get_wns():
        return -0.3  # above the -1.0 deeply-unmet threshold

    with patch.object(opt, "_get_current_wns", side_effect=fake_get_wns), \
         patch.object(opt, "_time_remaining_s", return_value=3000.0):
        ran = asyncio.run(opt._reconstrain_focus_pass())
    assert ran is False
    # A no-op still marks itself as not-yet-run so a later, deeper stall can try.
    assert opt._reconstrain_focus_done is False
