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
from dcp_optimizer import (  # noqa: E402
    DCPOptimizer, PLACE_DIRECTIVE_SWEEP, PHYS_OPT_DIRECTIVE_SWEEP,
    PHYS_OPT_SECONDARY_DIRECTIVE,
)
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
    # fanout_split is the exception: the 2026-07-28 pipeline audit found it
    # 0-for-11 across every run, so it's removed from the menu outright
    # rather than merely demoted (see _allowed_forbidden_actions).
    assert "fanout_split" not in allowed
    assert "lut_opt" in allowed
    assert forbidden == ["logic_restructure"]
    # The net_delay_bound demotion reason is still recorded even though the
    # action is no longer offered -- harmless, and useful forensic context.
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


def test_low_yield_recipe_demoted_below_this_run_winner(opt, tmp_path):
    # Fix 3 (pipeline audit, 20260802-20260804 sweep): vivado_phys_opt's
    # family was attempted 46% of the time across that sweep yet only 20%
    # improved, while place_design_explore hit 68% on a fifth as many
    # tries -- but the cross-run "0 wins" kill switch never catches
    # phys_opt_design since it does win sometimes. This-run tallies should
    # demote it below a clearly-outperforming alternative once both have
    # enough attempts to be signal.
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="dummy.dcp", output_dir=tmp_path, clock_name="clk")
    # Keep _sitting_on_fresh_win() False (best_fmax_mhz stays None with a
    # nonzero stall_count) so its own place_design_explore demotion doesn't
    # confound what this test is isolating.
    opt.checkpoint_manager.stall_count = 1
    for status in ("no_improvement", "no_improvement", "failed"):
        opt.checkpoint_manager.iterations.append(
            {"llm_chosen_action": "phys_opt_design", "status": status})
    for status in ("improved", "improved"):
        opt.checkpoint_manager.iterations.append(
            {"llm_chosen_action": "place_design_explore", "status": status})

    allowed, _ = opt._allowed_forbidden_actions(
        "net_delay_bound", "REGISTER", 0.82, 5.0, -0.2)

    assert allowed.index("place_design_explore") < allowed.index("phys_opt_design")
    assert "this run" in opt.last_action_guidance["phys_opt_design"]
    assert "place_design_explore" in opt.last_action_guidance["phys_opt_design"]


def test_low_yield_recipe_not_demoted_without_a_proven_alternative(opt, tmp_path):
    # The same laggard record should NOT be demoted if nothing currently
    # allowed has actually earned a better rate this run yet -- otherwise
    # the LLM would be steered away from every real lever on a hard design
    # where nothing has worked well so far.
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="dummy.dcp", output_dir=tmp_path, clock_name="clk")
    for status in ("no_improvement", "no_improvement", "failed"):
        opt.checkpoint_manager.iterations.append(
            {"llm_chosen_action": "phys_opt_design", "status": status})

    allowed, _ = opt._allowed_forbidden_actions(
        "net_delay_bound", "REGISTER", 0.82, 5.0, -0.2)

    assert "phys_opt_design" not in opt.last_action_guidance


def test_lut_opt_defaults_pins_from_worst_path_candidates(opt):
    # Pipeline audit (20260802-20260804 sweep): hierarchical_input_pins is a
    # REQUIRED, design-specific parameter the LLM has no way to invent, and
    # lut_opt was offered 105 times / chosen 0 across that sweep. Falling
    # back to the current worst-path candidates' own endpoint pins gives the
    # LLM a usable default instead of a guaranteed missing_action_parameters
    # failure whenever it omits the field.
    calls = []

    async def fake_call_tool(tool_name, params, internal=False):
        calls.append((tool_name, dict(params)))
        return "ok"

    opt.current_target_candidates = [
        {"endpoint": "top/sub/inst1/D", "startpoint": "top/sub/inst0/Q", "slack": -0.5},
        {"endpoint": "top/sub/inst2/D", "startpoint": "top/sub/inst1/Q", "slack": -0.4},
        {"endpoint": "top/sub/inst1/D", "startpoint": "top/sub/inst3/Q", "slack": -0.3},
    ]
    with patch.object(opt, "call_tool", side_effect=fake_call_tool):
        result = asyncio.run(opt.execute_validated_action(
            {"chosen_action": "lut_opt", "action_parameters": {}},
            {"delay_class": "logic_delay_bound"},
        ))
    assert calls[0][0] == "rapidwright_optimize_lut_input_cone"
    # Deduplicated, in candidate order.
    assert calls[0][1]["hierarchical_input_pins"] == ["top/sub/inst1/D", "top/sub/inst2/D"]
    assert result == "ok"


def test_lut_opt_fails_when_no_candidates_to_derive_from(opt):
    opt.current_target_candidates = []
    result = asyncio.run(opt.execute_validated_action(
        {"chosen_action": "lut_opt", "action_parameters": {}},
        {"delay_class": "logic_delay_bound"},
    ))
    payload = json.loads(result)
    assert payload["success"] is False
    assert payload["error_type"] == "missing_action_parameters"


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
# phys_opt directive sweep (same mechanism as place, for the fix where
# amd_mini run 20260802_135418 picked "Explore" as the phys_opt directive
# three separate iterations in a row on a fanout-diagnosed path)
# ---------------------------------------------------------------------------

def test_phys_opt_directive_sweep_advances(opt):
    assert opt._next_phys_opt_directive() == PHYS_OPT_DIRECTIVE_SWEEP[0]
    opt.phys_opt_directive_results[PHYS_OPT_DIRECTIVE_SWEEP[0]] = {
        "status": "no_improvement", "wns_after": -0.9}
    assert opt._next_phys_opt_directive() == PHYS_OPT_DIRECTIVE_SWEEP[1]


def test_phys_opt_directive_sweep_reuses_best_when_exhausted(opt):
    for i, directive in enumerate(PHYS_OPT_DIRECTIVE_SWEEP):
        opt.phys_opt_directive_results[directive] = {
            "status": "no_improvement", "wns_after": -1.0 + 0.1 * i}
    opt.phys_opt_directive_results["AggressiveFanoutOpt"] = {
        "status": "improved", "wns_after": -0.1}
    assert opt._next_phys_opt_directive() == "AggressiveFanoutOpt"


def test_phys_opt_directive_outcome_recorded_across_action_family(opt):
    # phys_opt_design, phys_opt_design_retime, and phys_opt_design_pin_swap
    # all resolve to the same underlying phys_opt_design Tcl call, so a
    # directive tried under one should register for all three.
    opt.last_action_key = "phys_opt_design_retime"
    opt.last_phys_opt_directive = "Explore"
    opt.iteration = 2
    opt._note_recipe_outcome("no_improvement", -0.904)
    assert opt.phys_opt_directive_results["Explore"]["status"] == "no_improvement"
    assert "Explore" not in [
        d for d in PHYS_OPT_DIRECTIVE_SWEEP if d not in opt.phys_opt_directive_results
    ]


def test_recent_stall_action_families_last_n(opt):
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir="ckpt", clock_name="clk")
    opt.checkpoint_manager.iterations = [
        {"llm_chosen_action": "place_design_explore", "status": "improved"},
        {"llm_chosen_action": "pblock", "status": "no_improvement"},
        {"llm_chosen_action": "route_explore", "status": "no_improvement"},
        {"llm_chosen_action": "pblock", "status": "no_improvement"},
    ]
    assert opt._recent_stall_action_families(3) == {"pblock", "route_explore"}


def test_structural_override_widens_menu_with_genuinely_untried_phys_opt(opt):
    # Reproduces run 20260803_141612 (finn_radioml) iters 2-4: pblock
    # (no_improvement), route_explore (no_improvement), phys_opt_design_pin_swap
    # (failed -- the -critical_pin_opt/-directive Tcl bug). The stuck detector
    # then fired at iter 5. Old behavior forced MORE placement, walling off
    # plain phys_opt_design/phys_opt_design_retime -- never attempted -- for
    # the rest of the override window; phys_opt_design only won (+7.4 MHz)
    # later via the LLM-independent endgame_polish fallback. pin_swap
    # specifically stays excluded (it WAS just tried), but the untried
    # variants should be added back onto the menu.
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir="ckpt", clock_name="clk")
    opt.checkpoint_manager.iterations = [
        {"llm_chosen_action": "pblock", "status": "no_improvement"},
        {"llm_chosen_action": "route_explore", "status": "no_improvement"},
        {"llm_chosen_action": "phys_opt_design_pin_swap", "status": "failed"},
    ]
    allowed = [
        "pblock", "pblock_full_replace", "place_design_explore",
        "phys_opt_design", "phys_opt_design_retime", "phys_opt_design_pin_swap",
    ]
    structural_allowed = ["pblock", "pblock_full_replace", "place_design_explore"]
    widened = opt._widen_override_with_untried_phys_opt(structural_allowed, allowed)
    assert set(widened) == set(structural_allowed) | {"phys_opt_design", "phys_opt_design_retime"}
    assert "phys_opt_design_pin_swap" not in widened


def test_structural_override_does_not_widen_when_a_stall_was_phys_opt(opt):
    # If a recent stall was itself a phys_opt attempt, it's not an untried
    # family -- don't widen (matches _recent_stall_action_families excluding
    # it from "untried" via the `action not in recent_stall_actions` filter).
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir="ckpt", clock_name="clk")
    opt.checkpoint_manager.iterations = [
        {"llm_chosen_action": "pblock", "status": "no_improvement"},
        {"llm_chosen_action": "phys_opt_design", "status": "no_improvement"},
        {"llm_chosen_action": "pblock", "status": "no_improvement"},
    ]
    allowed = ["pblock", "pblock_full_replace", "phys_opt_design"]
    structural_allowed = ["pblock", "pblock_full_replace"]
    widened = opt._widen_override_with_untried_phys_opt(structural_allowed, allowed)
    assert widened == structural_allowed


def test_structural_override_lookback_covers_the_whole_stall_streak(opt):
    # Fix (pipeline audit, rosetta_optical-flow, 20260804 sweep): the old
    # fixed 3-iteration lookback let an action tried at the START of a
    # longer stall streak "age out" and get re-offered as genuinely untried
    # once the streak grew past 3 -- optical-flow's override re-picked
    # replicate_register at iter 6 even though it had already failed at
    # iters 1-2, because by then those were outside the last-3 window.
    # Across a 5-iteration streak, an action tried at iterations 1-2 must
    # stay excluded; one that's never appeared anywhere in the streak is
    # still offered.
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir="ckpt", clock_name="clk")
    opt.checkpoint_manager.iterations = [
        {"llm_chosen_action": "replicate_register", "status": "no_improvement"},
        {"llm_chosen_action": "replicate_register", "status": "no_improvement"},
        {"llm_chosen_action": "phys_opt_design", "status": "no_improvement"},
        {"llm_chosen_action": "phys_opt_design", "status": "no_improvement"},
        {"llm_chosen_action": "phys_opt_design_pin_swap", "status": "no_improvement"},
    ]
    opt.consecutive_no_improvement = 5
    allowed = [
        "pblock", "place_design_explore", "phys_opt_design",
        "phys_opt_design_retime", "phys_opt_design_pin_swap", "replicate_register",
    ]
    structural_allowed = ["pblock", "place_design_explore"]
    widened = opt._widen_override_with_untried_phys_opt(structural_allowed, allowed)
    # Tried 3+ iterations ago -- a fixed 3-window would have missed this.
    assert "replicate_register" not in widened
    # Tried within the last 3 either way -- excluded regardless of window.
    assert "phys_opt_design_pin_swap" not in widened
    # Genuinely never tried anywhere in the streak -- still offered.
    assert "phys_opt_design_retime" in widened


def test_replicate_register_actually_sends_critical_cell_opt(opt):
    # Pipeline audit (20260804 sweep): _run_phys_opt_with_policy only ever
    # read directive/critical_pin_opt from its arguments, so
    # replicate_register's {"critical_cell_opt": True} was silently dropped
    # on EVERY call in the project's history -- the action ran a plain
    # directive pass indistinguishable from phys_opt_design, which is why it
    # went 0-for-everything on the excessive_fanout-stalled designs. The
    # final Tcl must now carry -critical_cell_opt and (per the Vivado_Tcl
    # 4-167 specific-options rule) no -directive.
    commands = []

    async def fake_call_tool(tool_name, params, internal=False):
        if tool_name == "vivado_phys_opt_design" and not internal:
            return await opt._run_phys_opt_with_policy(params)
        if tool_name == "vivado_run_tcl":
            commands.append(params["command"])
        return "ok"

    async def fake_wns():
        return -0.2

    opt.design_state = "routed"
    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_get_current_wns", side_effect=fake_wns):
        asyncio.run(opt.execute_validated_action(
            {"chosen_action": "replicate_register", "action_parameters": {}},
            {"worst_path": {"end_cell": "some/cell"}, "delay_class": "mixed"},
        ))
    assert commands, "no phys_opt Tcl was ever issued"
    assert "-critical_cell_opt" in commands[0]
    assert "-directive" not in commands[0]


def test_replicate_register_never_sends_force_replication(opt):
    # REVERSAL of the original targeted-replication fix: run 20260806_193354
    # (rosetta_optical-flow iter 1) proved -force_replication_on_nets is
    # rejected outright in post-route physical synthesis ("ERROR: [Vivado_Tcl
    # 4-265] ... not supported yet for post-route physical synthesis"), and
    # this pipeline only ever runs phys_opt post-route. Even with an
    # excessive_fanout diagnosis naming hotspot cells, the command must NOT
    # carry the option -- it would be a guaranteed hard failure.
    hyp = RootCauseHypothesis(
        name="excessive_fanout", cluster_id="c0", confidence=0.8,
        supporting_evidence=[], contradicting_evidence=[],
        recommended_actions=["replicate_register"], evidence_requests=[])
    from analysis_layer import FailureCluster
    cluster = FailureCluster(
        id="c0", members=[], shared_cells={"top/hot_reg"},
        fanout_hotspots=["top/hot_reg"])
    diagnosis = _make_diagnosis(hyp)
    diagnosis.clusters = [cluster]
    opt.last_diagnosis = diagnosis

    commands = []

    async def fake_call_tool(tool_name, params, internal=False):
        if tool_name == "vivado_phys_opt_design" and not internal:
            return await opt._run_phys_opt_with_policy(params)
        if tool_name == "vivado_run_tcl":
            commands.append(params["command"])
        return "ok"

    async def fake_wns():
        return -0.2

    opt.design_state = "routed"
    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_get_current_wns", side_effect=fake_wns):
        asyncio.run(opt.execute_validated_action(
            {"chosen_action": "replicate_register", "action_parameters": {}},
            {"worst_path": {"end_cell": "some/cell"}, "delay_class": "mixed"},
        ))
    assert commands
    assert "-force_replication_on_nets" not in commands[0]
    assert "-critical_cell_opt" in commands[0]


def test_coverage_ledger_tracks_offered_chosen_and_never_attempted(opt):
    # First-contact/hidden-benchmark feature: "never attempted on this
    # design" becomes a visible fact in the timing context. Everything found
    # in the 20260804-06 audits (optical-flow/spam-filter never getting a
    # full re-place) was invisible precisely because withheld actions leave
    # no trace in any log.
    ledger = opt._coverage_context(["pblock", "phys_opt_design"])
    assert ledger["pblock"]["offered"] == 1
    assert ledger["phys_opt_design"]["offered"] == 1

    # Dispatch marks "chosen" even when the action is later refused.
    with patch.object(opt, "_time_remaining_s", return_value=10.0), \
         patch.object(opt, "_estimated_action_cost_s", return_value=6000.0):
        asyncio.run(opt.execute_validated_action(
            {"chosen_action": "phys_opt_design", "action_parameters": {}}, {},
        ))
    assert opt.action_coverage["phys_opt_design"]["chosen"] == 1
    assert opt.action_coverage["pblock"]["chosen"] == 0

    # Second menu: offered increments again; never-attempted derives from
    # chosen == 0, exactly what _build_timing_context ships.
    opt._coverage_context(["pblock", "phys_opt_design"])
    never = [a for a in ["pblock", "phys_opt_design"]
             if opt.action_coverage.get(a, {}).get("chosen", 0) == 0]
    assert never == ["pblock"]


def test_check_hold_safety_probe_and_single_fix(opt):
    # Hidden-benchmark insurance: setup WNS is the score, but a submitted DCP
    # with hold violations risks failing organizer validation. One probe per
    # banked win; on the first violation, one hold-fix pass for the whole run.
    calls = []

    async def clean(tool_name, params, internal=False):
        calls.append(params.get("command", ""))
        return "WHS_OK"

    with patch.object(opt, "call_tool", side_effect=clean):
        assert asyncio.run(opt._check_hold_safety()) is True
    assert len(calls) == 1  # no fix pass when clean

    # Violated -> fix -> clean. Fix pass is spent for the rest of the run.
    opt.hold_fix_attempted = False
    seq = iter(["WHS_VIOLATED:-0.031", "phys_opt done", "WHS_OK"])

    async def viol_then_fixed(tool_name, params, internal=False):
        return next(seq)

    with patch.object(opt, "call_tool", side_effect=viol_then_fixed):
        assert asyncio.run(opt._check_hold_safety()) is True
    assert opt.hold_fix_attempted is True

    # Later violation with the fix already spent: fail fast, no second pass.
    async def viol(tool_name, params, internal=False):
        return "WHS_VIOLATED:-0.010"

    with patch.object(opt, "call_tool", side_effect=viol):
        assert asyncio.run(opt._check_hold_safety()) is False


def test_pblock_cluster_shrink_retries_without_hard_block_candidates(opt):
    # rosetta_3d-rendering (every run of the 20260802-20260806 sweep): both
    # pblock attempts per run died terminally with "none of the 3 candidate
    # regions had enough BRAM capacity; reduce the number of BRAM cells being
    # clustered instead". The fix does what the message says: drop the
    # BRAM-touching candidate paths, zero the BRAM demand, retry the fabric
    # search with the LUT/FF-dominated subset.
    opt.current_target_candidates = [
        {"startpoint": "u0/ram_reg_bank/a/Q", "endpoint": "u0/proc/lut_a/D", "slack": -2.1},
        {"startpoint": "u0/proc/lut_b/Q", "endpoint": "u0/proc/lut_c/D", "slack": -2.0},
        {"startpoint": "u0/proc/lut_c/Q", "endpoint": "u0/proc/lut_d/D", "slack": -1.9},
    ]
    analysis_calls = []

    async def fake_call_tool(tool_name, params, internal=False):
        if tool_name == "rapidwright_analyze_fabric_for_pblock":
            analysis_calls.append(dict(params))
            return json.dumps({"recommended_region": {
                "col_min": 10, "col_max": 30, "row_min": 5, "row_max": 45}})
        if tool_name == "rapidwright_convert_fabric_region_to_pblock":
            return json.dumps({
                "pblock_ranges": "SLICE_X10Y5:SLICE_X30Y45",
                # 30 RAMB sites available -- first pass demands 40 (short),
                # second pass (post-shrink) demands 0 and passes.
                "site_counts": {"SLICE": 500, "RAMB18": 20, "RAMB36": 10, "DSP48E2": 8},
            })
        return "ok"

    # First-pass hard-block demand: pretend the DRC floor learned 40 BRAM.
    opt.pblock_hard_block_demand = {"bram": 40, "dsp": 0}
    with patch.object(opt, "call_tool", side_effect=fake_call_tool):
        computed, error = asyncio.run(opt._compute_pblock_ranges({}, {}))

    assert error is None, error
    assert computed is not None and computed.get("ranges")
    # Two fabric searches: original cluster, then the shrunken one.
    assert len(analysis_calls) == 2
    assert analysis_calls[0]["target_bram_count"] == 40
    assert analysis_calls[1]["target_bram_count"] == 0
    # The BRAM-touching candidate's cells are gone from the retry's anchors.
    assert not any("ram_reg" in c for c in analysis_calls[1]["target_cell_names"])
    assert opt.last_rapidwright_edit_summary["cluster_shrink"]["dropped_hard_block_candidates"] == 1


def test_full_replace_grows_region_on_small_validation_shortage(opt):
    # Run 20260806_193354 (rosetta_optical-flow iter 7): the design's
    # first-ever pblock_full_replace failed pre-placement validation by a
    # shortage of 2 FIFO sites ("FIFO: requires 122, only 120 available")
    # and the recipe was abandoned. On a validation shortage with known
    # region geometry, the flow must delete the too-small pblock, grow the
    # window, re-convert, and re-apply -- succeeding when the grown region
    # fits -- instead of returning full_replace_region_too_small on the
    # first miss.
    apply_attempts = []
    convert_calls = []

    async def fake_call_tool(tool_name, params, internal=False):
        if tool_name == "vivado_report_utilization_for_pblock":
            return "1.5x Multiplier\nLUTs: 1000\nFFs: 2000\nDSPs: 0\nBRAMs: 60"
        if tool_name == "rapidwright_analyze_fabric_for_pblock":
            return json.dumps({"recommended_region": {
                "col_min": 100, "col_max": 120, "row_min": 50, "row_max": 90}})
        if tool_name == "rapidwright_convert_fabric_region_to_pblock":
            convert_calls.append(dict(params))
            return json.dumps({"pblock_ranges": f"SLICE_X{params['col_min']}Y{params['row_min']}:SLICE_X{params['col_max']}Y{params['row_max']}"})
        if tool_name == "vivado_create_and_apply_pblock":
            apply_attempts.append(dict(params))
            if len(apply_attempts) == 1:
                return json.dumps({
                    "cells_assigned": 5000, "cells_matched": 5000,
                    "resource_validation": {"errors": [
                        "FIFO: requires 122, only 120 available (shortage: 2)"]},
                })
            return json.dumps({"cells_assigned": 5000, "cells_matched": 5000,
                               "resource_validation": {"errors": []}})
        return "ok"

    async def licensed():
        return True

    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_check_implementation_license", side_effect=licensed):
        result = asyncio.run(opt._execute_pblock_full_replace({}))

    # One failed apply, one successful re-apply of a strictly larger window.
    assert len(apply_attempts) == 2
    assert len(convert_calls) == 2
    grown = convert_calls[1]
    assert grown["col_min"] < 100 and grown["col_max"] > 120
    assert grown["row_min"] < 50 and grown["row_max"] > 90
    assert "full_replace_region_too_small" not in result
    assert opt.last_rapidwright_edit_summary.get("region_grow_attempts") == 1


def test_fresh_win_requires_material_gain(opt, tmp_path):
    # Fix (rosetta_optical-flow, 20260804 sweep): a +0.42 MHz marginal at
    # iter 1 (0.13% over baseline) used to count as a "win worth protecting"
    # and kept place_design_explore/pblock_full_replace demoted for the
    # entire rest of the run. Sub-materiality gains must not engage the
    # exploit-after-win demotion.
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir=str(tmp_path / "ckpt"), clock_name="clk")
    opt.checkpoint_manager.start_baseline(wns=-1.078, period_ns=2.0)
    opt.checkpoint_manager.iterations = [
        {"llm_chosen_action": "replicate_register", "status": "marginal"}]
    baseline = opt.checkpoint_manager.baseline_fmax_mhz

    # optical-flow's actual situation: +0.13% -- below the floor.
    opt.checkpoint_manager.best_fmax_mhz = baseline * 1.0013
    opt.checkpoint_manager.stall_count = 3
    assert opt._sitting_on_fresh_win() is False

    # A real win -- comfortably above the floor -- still protected.
    opt.checkpoint_manager.best_fmax_mhz = baseline * 1.05
    assert opt._sitting_on_fresh_win() is True


def test_phys_opt_exhausted_this_streak(opt):
    # Fix (pipeline audit, rosetta_optical-flow, 20260804 sweep): the
    # structural override's decay timer used to release the menu back to
    # phys_opt on a fixed schedule even when every phys_opt variant had
    # already been tried this streak -- confirmed on optical-flow, where the
    # decay handed the menu back to an exhausted phys_opt_design instead of
    # ever forcing place_design_explore/pblock. This helper is what lets the
    # override hold through the decay window in that case.
    opt.checkpoint_manager = CheckpointManager(
        input_dcp="in.dcp", output_dir="ckpt", clock_name="clk")
    opt.checkpoint_manager.iterations = [
        {"llm_chosen_action": "replicate_register", "status": "no_improvement"},
        {"llm_chosen_action": "phys_opt_design", "status": "no_improvement"},
        {"llm_chosen_action": "phys_opt_design_retime", "status": "no_improvement"},
        {"llm_chosen_action": "phys_opt_design_pin_swap", "status": "no_improvement"},
    ]
    opt.consecutive_no_improvement = 4
    allowed = [
        "pblock", "place_design_explore", "phys_opt_design",
        "phys_opt_design_retime", "phys_opt_design_pin_swap", "replicate_register",
    ]
    # All four variants tried -- nothing left for the widen step to add.
    assert opt._phys_opt_exhausted_this_streak(allowed, high_spread=False) is True

    # One variant genuinely untried -- not exhausted yet.
    opt.checkpoint_manager.iterations.pop()  # phys_opt_design_pin_swap never tried
    assert opt._phys_opt_exhausted_this_streak(allowed, high_spread=False) is False

    # No structural fallback on offer at all -- refuse to force a dead end.
    no_structural_allowed = ["phys_opt_design", "phys_opt_design_retime", "phys_opt_design_pin_swap"]
    assert opt._phys_opt_exhausted_this_streak(no_structural_allowed, high_spread=False) is False


def test_critical_pin_opt_drops_directive_flag(opt):
    # ERROR: [Vivado_Tcl 4-167] "Cannot specify '-critical_pin_opt' when
    # '-directive' is specified" -- seen identically across corescore x2,
    # finn_radioml, and amd_mini every time phys_opt_design_pin_swap ran,
    # regardless of directive. -directive must be omitted whenever
    # critical_pin_opt is requested.
    commands = []

    async def fake_call_tool(tool_name, params, internal=False):
        commands.append(params["command"])
        return "ok"

    with patch.object(opt, "call_tool", side_effect=fake_call_tool):
        asyncio.run(opt._run_phys_opt_tcl(
            directive="AggressiveFanoutOpt", retime=True, critical_pin_opt=True))
    assert len(commands) == 1
    assert "-directive" not in commands[0]
    assert "-critical_pin_opt" in commands[0]
    assert "-retime" in commands[0]


def test_normal_phys_opt_still_uses_directive(opt):
    # Confirms the fix doesn't remove -directive for the normal (non-pin-swap)
    # case, which Vivado does accept combined with -directive/-retime.
    commands = []

    async def fake_call_tool(tool_name, params, internal=False):
        commands.append(params["command"])
        return "ok"

    with patch.object(opt, "call_tool", side_effect=fake_call_tool):
        asyncio.run(opt._run_phys_opt_tcl(
            directive="Explore", retime=True, critical_pin_opt=False))
    assert commands[0] == "phys_opt_design -directive Explore -retime"


def test_phys_opt_dispatch_defaults_to_untried_directive(opt):
    calls = []

    async def fake_call_tool(tool_name, params, internal=False):
        calls.append((tool_name, dict(params)))
        return "ok"

    async def fake_license():
        return True

    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_check_implementation_license", side_effect=fake_license):
        asyncio.run(opt.execute_validated_action(
            {"chosen_action": "phys_opt_design", "action_parameters": {}},
            {"delay_class": "logic_delay_bound"},
        ))
    assert calls[0][1]["directive"] == PHYS_OPT_DIRECTIVE_SWEEP[0]
    assert opt.last_phys_opt_directive == PHYS_OPT_DIRECTIVE_SWEEP[0]


def test_inert_action_exhausts_in_two_attempts_not_three(opt):
    # Resolution-awareness fix (20260806 log audit): an action whose WNS came
    # back bit-identical provably moved nothing, which is harder evidence
    # than "it moved something that didn't pay". It is weighted double in the
    # failure memory so an inert lever is suppressed after TWO attempts,
    # rather than being re-picked until the ordinary 3-strike threshold --
    # the dominant waste mode in the logs (182/341 measured iterations,
    # 5.0 h of Vivado time).
    opt.current_target_candidates = [{"endpoint": "e", "startpoint": "s", "slack": -0.5}]

    opt._remember_no_action_failure("route_explore", [], weight=2)
    assert "route_explore" not in opt._active_exhausted_actions()  # 2 < 3
    opt._remember_no_action_failure("route_explore", [], weight=2)
    assert "route_explore" in opt._active_exhausted_actions()      # 4 >= 3

    # An ordinary (weight-1) failure still needs the full three strikes.
    opt2_targets = [{"endpoint": "e2", "startpoint": "s2", "slack": -0.5}]
    opt.current_target_candidates = opt2_targets
    for _ in range(dcp.ACTION_FAILURE_EXHAUSTION_THRESHOLD - 1):
        opt._remember_no_action_failure("pblock", [])
    assert "pblock" not in opt._active_exhausted_actions()
    opt._remember_no_action_failure("pblock", [])
    assert "pblock" in opt._active_exhausted_actions()


def test_repeated_budget_refusal_exhausts_the_action(opt):
    # Fix 2 (pipeline audit, 20260802-20260804 sweep): insufficient_budget
    # refusals used to leave no memory at all -- the demotion in
    # _allowed_forbidden_actions is soft, so a refused action could be
    # re-proposed and re-refused indefinitely. ispd16_example2 burned its
    # last 15 of 19 iterations exactly this way. A refusal should now feed
    # the same cooldown machinery other repeat-failing actions use, so the
    # action actually drops out of allowed_actions after
    # ACTION_FAILURE_EXHAUSTION_THRESHOLD refusals on the same target set.
    opt.current_target_candidates = [{"endpoint": "e", "startpoint": "s", "slack": -0.5}]

    async def fake_call_tool(tool_name, params, internal=False):
        return "ok"

    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_time_remaining_s", return_value=60.0), \
         patch.object(opt, "_estimated_action_cost_s", return_value=6000.0):
        for _ in range(dcp.ACTION_FAILURE_EXHAUSTION_THRESHOLD):
            result = asyncio.run(opt.execute_validated_action(
                {"chosen_action": "route_explore", "action_parameters": {}}, {},
            ))
            assert json.loads(result)["error_type"] == "insufficient_budget"

    assert "route_explore" in opt._active_exhausted_actions()


def test_repeated_phys_opt_wns_refusal_exhausts_the_action(opt):
    # Same mechanism, for the other self-refusal type (13/47 failures in the
    # same sweep): phys_opt_below_useful_wns.
    opt.current_target_candidates = [{"endpoint": "e", "startpoint": "s", "slack": -8.0}]
    opt.path_delay_breakdown = {"logic_pct": 0.1}
    opt.consecutive_no_improvement = 0

    async def fake_wns():
        return -8.0

    # Deliberately NOT patching call_tool: the guard fires and returns
    # before execute_validated_action's phys_opt branch ever reaches its
    # own self.call_tool("vivado_phys_opt_design", ...) -- patching it here
    # would bypass call_tool's own interception into _run_phys_opt_with_policy
    # (the guard under test) entirely.
    with patch.object(opt, "_get_current_wns", side_effect=fake_wns), \
         patch.object(opt, "_active_exhausted_actions", side_effect=lambda: []):
        for _ in range(dcp.ACTION_FAILURE_EXHAUSTION_THRESHOLD):
            result = asyncio.run(opt.execute_validated_action(
                {"chosen_action": "phys_opt_design", "action_parameters": {}}, {},
            ))
            assert json.loads(result)["error_type"] == "phys_opt_below_useful_wns"

    assert "phys_opt_design" in opt._active_exhausted_actions()


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
    # The 2026-07-28 pipeline audit found rapidwright_optimize_cell_placement
    # 0-for-19 across every run since 07-19, so it's removed from the normal
    # menu outright now (see _allowed_forbidden_actions) rather than merely
    # demoted to the end -- the crossrun-priors guidance above still fires
    # first and is preserved for forensic visibility.
    assert "rapidwright_optimize_cell_placement" not in allowed


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


def test_phys_opt_secondary_directive_is_valid():
    # "ExploreWithRemap" (the old value) isn't a real Vivado phys_opt_design
    # directive -- run 20260803_153128's vivado.jou showed it always
    # immediately followed by "-directive Default", the error-fallback
    # firing every time. Guard against reintroducing an invalid name here.
    valid_directives = {
        "Default", "Explore", "ExploreWithHoldFix", "ExploreWithAggressiveHoldFix",
        "AggressiveExplore", "AlternateReplication", "AggressiveFanoutOpt",
        "AlternateFlowWithRetiming", "AddRetime", "RuntimeOptimized", "RQS",
    }
    assert PHYS_OPT_SECONDARY_DIRECTIVE in valid_directives
    assert PHYS_OPT_SECONDARY_DIRECTIVE != "ExploreWithRemap"


def test_post_route_physsynth_crash_detection(opt):
    # Confirmed signature (fir_systolic_transposed_routed_2025.1, runs
    # 20260801_195142 and 20260803_153128): routing completes cleanly, then
    # Vivado's own post-route physical-synthesis re-optimization pass
    # (Phase 15/15.1) throws this specific exception. A generic Vivado
    # error ("ERROR: [Route ...]" from some other cause) must NOT match --
    # only this pipeline's own crash signature should trigger a retry.
    crash_log = (
        "Phase 15 Physical Synthesis in Router\n"
        "Phase 15.1 Physical Synthesis Initialization\n"
        "ERROR: [Route 35-9] Router encountered a fatal exception of type "
        "'13HDPLException' - 'Error in placer init in PSFlow '.\n"
    )
    assert opt._is_post_route_physsynth_crash(crash_log)
    other_error = "ERROR: [Route 35-4] Router failed to route some nets.\n"
    assert not opt._is_post_route_physsynth_crash(other_error)


def test_pblock_route_crash_retries_with_default_directive(opt):
    # Fix (evaluation follow-up, fir_systolic_transposed): the old behavior
    # took pblock off the table for the rest of the run over a Vivado-side
    # crash unrelated to the design. A single retry with -directive Default
    # (which doesn't invoke the crashing pass) should recover the win
    # instead of returning a terminal pblock_route_failed.
    crash_text = (
        "Phase 15.1 Physical Synthesis Initialization\n"
        "ERROR: [Route 35-9] Router encountered a fatal exception of type "
        "'13HDPLException' - 'Error in placer init in PSFlow '.\n"
    )
    calls = []

    async def fake_call_tool(tool_name, params, internal=False):
        calls.append((tool_name, dict(params)))
        if tool_name == "vivado_route_design" and params.get("directive") == "Explore":
            return crash_text
        if tool_name == "vivado_create_and_apply_pblock":
            return json.dumps({"cells_assigned": 5, "cells_matched": 5})
        return "ok"

    opt.current_target_candidates = []
    with patch.object(opt, "call_tool", side_effect=fake_call_tool), \
         patch.object(opt, "_compute_pblock_ranges", side_effect=lambda params, ctx: (
             {**params, "ranges": "SLICE_X0Y0:SLICE_X10Y10", "pblock_name": "pb", "apply_to": "current_design", "is_soft": False}, None)):
        result = asyncio.run(opt.execute_validated_action(
            {"chosen_action": "pblock", "action_parameters": {}},
            {"delay_class": "net_delay_bound", "cluster_clock_regions": []},
        ))
    route_calls = [c for c in calls if c[0] == "vivado_route_design"]
    assert [c[1]["directive"] for c in route_calls] == ["Explore", "Default"]
    # Recovered via the Default retry -- not the terminal failure path.
    assert "pblock_route_failed" not in result


def test_timeout_floor_drops_when_duration_known(opt):
    # Run 20260714_182751 iter 11: place measured ~250 s but the old
    # max(1200, 2.5x) floor let a hang burn 20 minutes.
    with patch.object(opt, "_time_remaining_s", return_value=3000.0), \
         patch.object(opt, "_estimated_duration", return_value=250.0):
        assert opt._implementation_timeout_s(kind="place") == 625
    with patch.object(opt, "_time_remaining_s", return_value=3000.0), \
         patch.object(opt, "_estimated_duration", return_value=100.0):
        assert opt._implementation_timeout_s(kind="place") == 600
    # Unmeasured kind keeps the conservative default.
    with patch.object(opt, "_time_remaining_s", return_value=3000.0), \
         patch.object(opt, "_estimated_duration", return_value=None):
        assert opt._implementation_timeout_s(kind="place") == 1200


def test_invalid_place_directive_rejected_at_validation(opt):
    context = {
        "allowed_actions": ["place_design_explore", "pblock_full_replace"],
        "forbidden_actions": [],
        "delay_class": "net_delay_bound",
    }
    bad = {
        "chosen_action": "place_design_explore",
        "delay_class_acknowledged": "net_delay_bound",
        "action_parameters": {"directive": "AggressiveExplore"},  # phys_opt-only
    }
    ok_flag, reason = opt.validate_llm_action(bad, context)
    assert ok_flag is False and "invalid_place_directive" in reason
    good = {
        "chosen_action": "place_design_explore",
        "delay_class_acknowledged": "net_delay_bound",
        "action_parameters": {"directive": "ExtraNetDelay_high"},
    }
    assert opt.validate_llm_action(good, context) == (True, "ok")
    # pblock_full_replace's place_directive is validated too.
    bad_full = {
        "chosen_action": "pblock_full_replace",
        "delay_class_acknowledged": "net_delay_bound",
        "action_parameters": {"place_directive": "NoSuchDirective"},
    }
    ok_flag, reason = opt.validate_llm_action(bad_full, context)
    assert ok_flag is False and "invalid_place_directive" in reason


def test_discouraged_action_below_min_confidence_rejected(opt):
    # Run 20260803_150309 (amd_mini): every discouraged-action pick that run
    # (iters 5, 6, 10, 11) self-rated confidence exactly 3/5 -- the old
    # threshold (reject < 3) never fired. Confidence 3 must now be rejected;
    # only 4+ overrides guidance.
    context = {
        "allowed_actions": ["pblock", "place_design_explore"],
        "forbidden_actions": [],
        "delay_class": "logic_delay_bound",
        "action_guidance": {"pblock": "0 wins / 3 losses across previous runs on this design"},
    }
    decision = {
        "chosen_action": "pblock",
        "delay_class_acknowledged": "logic_delay_bound",
        "action_parameters": {},
        "confidence": 3,
    }
    ok_flag, reason = opt.validate_llm_action(decision, context)
    assert ok_flag is False and "discouraged_action_low_confidence" in reason
    decision["confidence"] = 4
    assert opt.validate_llm_action(decision, context) == (True, "ok")
    # An action NOT in action_guidance is never gated on confidence.
    decision2 = {
        "chosen_action": "place_design_explore",
        "delay_class_acknowledged": "logic_delay_bound",
        "action_parameters": {},
        "confidence": 1,
    }
    assert opt.validate_llm_action(decision2, context) == (True, "ok")


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
