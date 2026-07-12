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

def test_congestion_parser_table_and_window():
    level, _ = DCPOptimizer._parse_congestion_report(
        "| NORTH | 5 | 12.3 |\n| SOUTH | 3 | 1.0 |")
    assert level == 5
    level, _ = DCPOptimizer._parse_congestion_report("congested window of 32x32 tiles")
    assert level == 5
    level, detail = DCPOptimizer._parse_congestion_report("no table here")
    assert level is None and "no congestion level" in detail


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
