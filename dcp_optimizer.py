#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
FPGA Design Optimization Agent

An autonomous AI agent that analyzes FPGA designs and applies optimizations
using RapidWright and Vivado via MCP servers.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

from analysis_layer import AnalysisEngine, Diagnosis
from checkpoint_manager import CheckpointManager, load_or_create
from eco_router import ECORouter
from shield_generator import ShieldGenerator, escape_tcl_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL = "x-ai/grok-4.3"

# Optimization policy constants
TIER2_TOP_PATHS_DEFAULT = 20
CLOCK_TIGHTEN_FRACTION = 0.05
MIN_CLOCK_TIGHTEN_STEP_NS = 0.01
MAX_BISECT_ITERS = 6
DELAY_CLASSIFICATION_THRESHOLD = 0.60
CONSTRAINT_AUDIT_WNS_TRIGGER_NS = -1.0
CONSTRAINT_AUDIT_TNS_TRIGGER_NS = -10000.0
CONSTRAINT_AUDIT_ENDPOINT_TRIGGER = 5000
CONSTRAINT_AUDIT_TOP_PATHS = 50
CONSTRAINT_AUDIT_COMMON_PREFIX_FRACTION = 0.30
TIME_BUDGET_RESERVE_S = 300
PHYS_OPT_PRIMARY_DIRECTIVE = "AggressiveExplore"
# "ExploreWithRemap" is not a real Vivado phys_opt_design directive (absent
# from the full confirmed-valid set in VivadoMCP/vivado_mcp_server.py's own
# schema: Default, Explore, ExploreWithHoldFix, ExploreWithAggressiveHoldFix,
# AggressiveExplore, AlternateReplication, AggressiveFanoutOpt,
# AlternateFlowWithRetiming, AddRetime, RuntimeOptimized, RQS). Confirmed via
# run 20260803_153128's vivado.jou: every "-directive ExploreWithRemap" call
# is immediately followed by "-directive Default" -- the error-fallback in
# _run_phys_opt_with_policy firing every single time, silently wasting one
# Vivado round-trip on every phys_opt attempt that didn't already improve
# WNS, across the entire project's run history.
PHYS_OPT_SECONDARY_DIRECTIVE = "ExploreWithHoldFix"
# Directive sweep for phys_opt_design/phys_opt_design_retime/phys_opt_design_pin_swap,
# mirroring PLACE_DIRECTIVE_SWEEP: without it, the LLM has no reason to move
# off one directive string once it's typed it, and nothing tracks that it's
# already been tried. Run 20260802_135418 (amd_mini) picked "Explore" as the
# phys_opt directive three separate times across three separate iterations
# (2, 3, 8) for a logic/fanout-bound path -- never AggressiveFanoutOpt, the
# one directive in Vivado's own list (VivadoMCP/vivado_mcp_server.py) whose
# description is literally "driver replication for fanout-related
# optimizations." Directive names below are the confirmed-valid set from
# that MCP tool's schema; Default (too weak / the automatic error-fallback
# already) and RQS (coupled to the qor_suggestions flow only) are excluded.
PHYS_OPT_DIRECTIVE_SWEEP = [
    "AggressiveExplore",
    "Explore",
    "AggressiveFanoutOpt",
    "AlternateReplication",
    "ExploreWithHoldFix",
    "AlternateFlowWithRetiming",
    "ExploreWithAggressiveHoldFix",
    "AddRetime",
    "RuntimeOptimized",
]
DECISION_NET_DELAY_BOUND_THRESHOLD = 0.70
DECISION_LOGIC_DELAY_BOUND_THRESHOLD = 0.70
DECISION_SPREAD_NET_THRESHOLD = 0.60
DECISION_SPREAD_TILE_THRESHOLD = 30
# --- BRAM/DSP bottleneck gate (2026-07-28) ---
# The old hard-block sizing trigger inflated target_bram_count/
# target_dsp_count the moment ANY current candidate touched a BRAM/DSP cell
# name -- one stray candidate among 20 was enough. Same dominance-fraction
# bar as analysis_layer.FANOUT_DOMINANCE_FRACTION: a cause needs to show up
# in a meaningful share of the candidates before it's treated as the actual
# bottleneck, not just present.
BRAM_DSP_BOTTLENECK_FRACTION_THRESHOLD = 0.30
PHYS_OPT_MIN_USEFUL_WNS_NS = -0.5
WNS_SANITY_ABS_LIMIT_NS = 50.0
WNS_SANITY_POSITIVE_CLOCK_FRACTION = 0.10
STUCK_ITERATION_THRESHOLD = 3
STRUCTURAL_OVERRIDE_MAX_ITERS = 6
WNS_IMPROVEMENT_EPSILON_NS = 1e-4
ACTION_FAILURE_EXHAUSTION_THRESHOLD = 3
# action_guidance is rebuttable by design (the LLM can still choose a
# discouraged action if it argues why), but nothing checked whether the
# argument was actually convincing. Run 20260803_131720 (corescore) chose
# rapidwright_optimize_cell_placement at iters 13 AND 14 despite guidance
# reading "0 wins / 4 losses across previous runs on this design" both
# times, self-rating confidence 2/5 both times, and repeating a
# near-identical rebuttal right after iter 13's own fresh failure. A
# low-confidence pick of a discouraged action is now treated as invalid and
# re-prompted instead of executed as-is (see validate_llm_action).
#
# Calibration note (run 20260803_150309, amd_mini): threshold 3 never fired
# in practice -- every discouraged-action pick that run (iters 5, 6, 10, 11)
# self-rated confidence exactly 3/5, landing right on the boundary of a
# "< 3" check. The model appears to hedge at exactly 3 when overriding
# guidance rather than genuinely varying its confidence, so the bar is
# raised to require 4+ to actually have teeth.
DISCOURAGED_ACTION_MIN_CONFIDENCE = 4
# Fix #6: how many iterations a cell stays blacklisted before it becomes
# eligible again. Without this, cells_blacklisted only grows, and long runs
# eventually exhaust the critical-path candidate pool entirely.
BLACKLIST_TTL_ITERS = 15
# Fix #7: how long "pblock" (and its family) is withheld from allowed_actions
# once a recommended region is found to overlap an already-applied pblock.
# Deliberately much longer than ACTION_FAILURE_COOLDOWN_ITERS, since this is
# a geometric fact about the current pblock layout, not a transient failure
# that's likely to succeed on the next retry.
PBLOCK_OVERLAP_COOLDOWN_ITERS = 20
ACTION_FAILURE_COOLDOWN_ITERS = 5
# This-run recipe-yield demotion (pipeline audit, 20260802-20260804 sweep):
# thresholds for the live win-rate prior in _allowed_forbidden_actions.
# RECIPE_YIELD_MIN_ATTEMPTS/RECIPE_YIELD_LOW_RATE gate the laggard (enough
# attempts this run to be signal, not noise, and a rate clearly below
# chance); RECIPE_YIELD_ALT_MIN_ATTEMPTS/RECIPE_YIELD_ALT_MIN_RATE gate the
# alternative it's compared against (has to actually be working this run,
# not just untried). Demotion only, same as every other prior here -- see
# DISCOURAGED_ACTION_MIN_CONFIDENCE for how a demoted pick still gets used
# when the LLM's rebuttal is unconvincing.
RECIPE_YIELD_MIN_ATTEMPTS = 3
RECIPE_YIELD_LOW_RATE = 0.25
RECIPE_YIELD_ALT_MIN_ATTEMPTS = 2
RECIPE_YIELD_ALT_MIN_RATE = 0.5
# Exploit-after-win materiality floor (see _sitting_on_fresh_win): the
# banked best must exceed baseline by this fraction before the "protect the
# win, demote fresh re-rolls" prior engages. 0.5% -- rosetta_optical-flow's
# +0.13% marginal is comfortably below it; the +24-32% wins the prior was
# actually measured on are far above.
FRESH_WIN_MATERIALITY_FRACTION = 0.005
# pblock_full_replace region-grow retry (run 20260806_193354,
# rosetta_optical-flow iter 7): the design's first-ever full_replace failed
# pre-placement validation by a shortage of 2 FIFO sites out of 122 and the
# recipe was abandoned outright. On a validation shortage with known region
# geometry, grow the window by this fraction per side and re-apply, up to
# this many times, before declaring the region too small.
FULL_REPLACE_REGION_GROW_RETRIES = 2
FULL_REPLACE_REGION_GROW_FRACTION = 0.15
# Fix #10 (score-aware stall stop): the contest score is
# alpha - 0.1*alpha*cost - 0.1*alpha*(runtime/3600), so every iteration past
# the last improvement is pure negative value. Run history showed runs
# burning 30+ consecutive stalled iterations (~$0.5 and ~15 min of runtime)
# after the final improvement at iteration 2. Five consecutive stalls is
# enough to have tried every strategy family at least once given the
# deadlock fixes (#11/#12) that keep the action menu from collapsing.
ABSOLUTE_STALL_HARD_LIMIT = 5
# Fix #13 (context cap): how many of the most recent conversation messages
# (after the pinned system prompt + initial-analysis message) are kept
# verbatim when pruning the LLM conversation each iteration. Older turns are
# collapsed into a compact per-iteration summary derived from
# checkpoint_manager history. Without this, runs were sending 2.4M-7M prompt
# tokens (65k+/call, zero cache hits) of stale per-iteration JSON.
CONTEXT_KEEP_RECENT_MESSAGES = 6
# --- Cluster-aware cell placement guard (fix #4) ---
# rapidwright_optimize_cell_placement moves each requested cell independently.
# If cells on the same critical path are tightly coupled, moving one without
# the others can increase the spread between them even though each individual
# move looked locally reasonable. This threshold gates the move: if the
# post-move spread across the targeted cluster is worse than before the move
# by more than this fraction, the move is rejected before it is ever routed
# (saving a full route cycle and avoiding a checkpoint that we know regressed).
CLUSTER_SPREAD_REGRESSION_FRACTION = 0.15
PBLOCK_OVERLAP_MAX_RETRIES = 3
PBLOCK_OVERLAP_GROW_FACTOR = 1.5
CELL_PLACEMENT_FANOUT_GUARD_ENABLED = True
ACTION_STRUCTURAL_FAILURE_WINDOW_ITERS = 10
ACTION_STRUCTURAL_FAILURE_THRESHOLD = 3
ACTION_STRUCTURAL_FAILURE_COOLDOWN_ITERS = 15
# --- Pblock region validation (fix #2) ---
# Reject RapidWright-recommended pblock regions that would be packed too
# densely (congestion risk) or that overlap a pblock already applied earlier
# in this run (Vivado handles overlapping pblocks poorly and it creates
# ambiguous, hard-to-solve placement scenarios).
PBLOCK_MAX_UTILIZATION_FRACTION = 0.85
RAPIDWRIGHT_STRUCTURAL_ACTIONS = [
    "rapidwright_optimize_cell_placement",
    # "rapidwright_analyze_net_detour",
    # "rapidwright_analyze_fabric_for_pblock",
    # "rapidwright_convert_fabric_region_to_pblock",
    "pblock",
    # Fix #9: whole-design pblock re-place. This is the recipe that test mode
    # uses on LogicNets (403 -> 521 MHz) and it was previously unreachable in
    # agent mode: the "pblock" action only ever builds a tiny clustering
    # pblock sized per candidate path (~40 LUTs each). pblock_full_replace
    # sizes a region for the ENTIRE design (1.5x utilization), unplaces
    # everything, applies the pblock to the whole design, and re-places +
    # re-routes inside it.
    "pblock_full_replace",
    # History(16) forensics: the old run's 501 MHz turned out to come from an
    # ACCIDENTAL whole-design unplace + place + route (the pblock_full_replace
    # interception bug's side effect), credited to the place_design_explore
    # iteration that merely measured its leftover. A full re-place is the one
    # recipe with a proven win on this benchmark, so its action belongs in
    # the structural family -- previously the stuck-detector's structural
    # override REMOVED place_design_explore from the menu mid-stall because
    # it wasn't listed here.
    "place_design_explore",
]
PBLOCK_ACTION_FAMILY = {
    "pblock",
    "rapidwright_analyze_fabric_for_pblock",
    "rapidwright_convert_fabric_region_to_pblock",
}
# Used specifically when spread analysis flags high dispersion + net-delay-bound
# paths (see the avg_spread/net_pct reorder below). Deliberately different order
# from RAPIDWRIGHT_STRUCTURAL_ACTIONS: local cell nudging (rapidwright_optimize_
# cell_placement) is a weak, greedy fix for widely-spread critical paths and has
# been observed to actively regress WNS in that regime, while the pblock track
# addresses the actual dispersion. Put the pblock track first so it's genuinely
# prioritized instead of just re-deriving the same default order.
RAPIDWRIGHT_PLACEMENT_ACTIONS = [
    # Fix #9: for widely-spread net-delay-bound designs, the whole-design
    # re-place is the empirically strongest known recipe, so it goes first.
    "pblock_full_replace",
    # The plain unplace-first re-place (see history(16) forensics above) --
    # the actual source of the 501 MHz result -- ranks right behind it.
    "place_design_explore",
    # "rapidwright_analyze_fabric_for_pblock",
    # "rapidwright_convert_fabric_region_to_pblock",
    "pblock",
    # "rapidwright_analyze_net_detour",
    "rapidwright_optimize_cell_placement",
]
VIVADO_INCREMENTAL_IMPLEMENTATION_ACTIONS = {
    "phys_opt_design",
    "phys_opt_design_retime",
    "phys_opt_design_pin_swap",
    "replicate_register",
    "place_design_explore",
    "route_explore",
}
# Endgame budget-spender: when the stall limit fires but this much wall-clock
# remains, polish the best checkpoint with cheap incremental passes instead of
# exiting early -- every unspent minute is score left on the table only if
# the passes regress, and regressions roll back.
ENDGAME_MIN_REMAINING_S = 600
# --- Convergence early-exit (score item 2) ---
# The score is alpha - 0.1*alpha*beta - 0.1*alpha*gamma, so every iteration
# past genuine convergence is pure negative value (more gamma runtime, more
# beta cost). Two convergence signals stop the run early:
#  (a) physical ceiling: best Fmax is within LOGIC_CEILING_STOP_FRACTION of the
#      zero-interconnect logic Fmax ceiling measured in phase-0 diagnostics --
#      no placement/routing can beat pure logic depth, so there is nothing left
#      to find.
#  (b) menu exhaustion: after CONVERGENCE_MIN_STALLS stalls, every base action
#      family is simultaneously in cooldown/exhausted on the current target --
#      the LLM has no genuinely-new move, and waiting for cooldowns to lapse
#      only burns budget re-trying things that already failed.
LOGIC_CEILING_STOP_FRACTION = 0.97
CONVERGENCE_MIN_STALLS = 3
# --- Clock re-constraint focus pass (score item 1) ---
# On a deeply-unmet design (WNS far below zero against the contest clock), the
# placer/router/phys_opt see every path as violating and spread timing-driven
# effort uselessly thin. Re-constraining the clock to a barely-unmet period
# (RECONSTRAIN_RELAX x the achieved delay) gives the tools an honest target so
# they concentrate on the genuinely-critical set. The relaxed clock is ONLY an
# internal guide: the contest period is always restored before any measurement
# or publish, and keep-best/rollback gates the outcome, so the downside is one
# phys_opt+route cycle.
RECONSTRAIN_MIN_UNMET_WNS_NS = -1.0
RECONSTRAIN_RELAX = 0.95
# Actions that only make small incremental improvements to an existing
# placement, and therefore genuinely cannot fix a deeply negative WNS.
# Deliberately EXCLUDES place_design_explore: a full re-place is not
# incremental, and applying the phys_opt WNS floor to it was a category
# error -- run 20260711's only improvement (-0.92 -> -0.493 ns) came from
# place_design_explore at a WNS the old gate said was "too negative".
PHYS_OPT_INCREMENTAL_ACTIONS = {
    "phys_opt_design",
    "phys_opt_design_retime",
    "phys_opt_design_pin_swap",
    "replicate_register",
}
# Directive sweep for place_design_explore (item 3a: search within a recipe).
# Once one directive has been tried on this design, the next attempt should
# try a different one instead of repeating -- this is the sweep order an
# implementation engineer would run, timing-focused first, then the
# spread-logic variants that help congestion-bound designs.
PLACE_DIRECTIVE_SWEEP = [
    "Explore",
    "ExtraTimingOpt",
    # BUG FIX (run 20260714_182751 iter 9): "AggressiveExplore" was in this
    # sweep but it is a phys_opt/route directive -- place_design rejects it
    # ("not a recognized directive"), so the slot was a guaranteed cheap
    # failure that also poisoned the sweep memory. ExtraPostPlacementOpt is
    # the timing-focused place directive it was standing in for.
    "ExtraPostPlacementOpt",
    "ExtraNetDelay_high",
    "AltSpreadLogic_high",
]
# The full set of directives place_design actually accepts on UltraScale+
# (2025.1). Anything else from the LLM is rejected at validation time and
# reprompted instead of burning an unplace + failed place at dispatch.
VALID_PLACE_DIRECTIVES = {
    "Default",
    "Explore",
    "WLDrivenBlockPlacement",
    "EarlyBlockPlacement",
    "ExtraNetDelay_high",
    "ExtraNetDelay_low",
    "AltSpreadLogic_high",
    "AltSpreadLogic_medium",
    "AltSpreadLogic_low",
    "ExtraPostPlacementOpt",
    "ExtraTimingOpt",
    "SSI_SpreadLogic_high",
    "SSI_SpreadLogic_low",
    "SSI_SpreadSLLs",
    "SSI_BalanceSLLs",
    "SSI_BalanceSLRs",
    "SSI_HighUtilSLRs",
    "RuntimeOptimized",
    "Quick",
}
# Item 3b: when the most recent pblock attempt executed cleanly but REGRESSED
# timing (or changed nothing), the region was likely too tight; grow the next
# auto-sized request instead of retrying the identical size.
PBLOCK_REGRESSION_GROW_FACTOR = 1.5
# Item 4: router congestion level at/above which packing cells tighter (any
# pblock action) is more likely to hurt than help. Level N means a 2^N x 2^N
# congested tile window; AMD guidance treats level 5 (32x32) as problematic.
CONGESTION_LEVEL_HIGH = 5
# Run history(16) lesson: this OOC design consistently shows ~6 unplaced
# primitive cells after ANY fresh place (even Vivado's own place_design) or
# RapidWright checkpoint round-trip, while route status still reports 0
# failed nets / fully routed -- they have no routable nets and cannot be on
# a timing path. The previous run's best checkpoint (501 MHz) came from this
# exact flow. Treating any unplaced primitive as fatal therefore rejected
# every improving recipe. Unplaced primitives are benign as long as the
# count stays small AND route status is clean; a genuinely broken state
# (failed placer, mid-flight unplace) leaves hundreds-to-thousands unplaced
# and unrouted nets, both far past this tolerance.
UNPLACED_CELLS_BENIGN_MAX = 25
# Score-aware stall accounting: validation failures that abort in seconds
# (range checks, candidate rejections) must not burn the expensive-stall
# budget that exists to stop wasted place/route cycles -- run history(16)
# ended after 5 failures that consumed almost no Vivado time, abandoning the
# whole remaining time budget. Failures cheaper than this threshold count
# against a separate, much larger cap instead.
CHEAP_FAILURE_RUNTIME_S = 60
STALL_LIMIT_CHEAP_FAILURES = 15
# --- Failure-message capture (pipeline audit 2026-07-28) ---
# Vivado's own console output for a failed place/route/pblock command often
# runs several KB of phase-by-phase progress before its actual error line, so
# a plain head-slice (the old [:2000]/[:4000] behavior) reliably cut every
# long failure message off mid-log, before the real explanation -- confirmed
# empirically: every pblock_route_failed record in the 07-19..07-28 run
# window was truncated at the exact same offset, always mid "Phase 4.1
# Initial Net Routing Pass". Keep the TAIL instead: Vivado's actual failure
# line is almost always the last thing it printed before giving up.
FAILURE_MESSAGE_CAPTURE_CHARS = 6000
# --- Cost model (2026-07-12) ---
# Motivating incident: on a large design, place_design -directive Explore took
# 15.5 min and the following route Explore was killed by the remaining-budget
# timeout clamp at ~15 min -- 30+ minutes spent, ZERO valid recorded results.
# Root cause: the optimizer had no idea what actions cost. Durations are now
# measured per kind (place/route/phys_opt) in self.action_durations, persisted
# per design in the cross-run store ("durations"), and every dispatch is
# priced before it runs.
# A design is "large" when a full place is measured (or known from a prior
# run) to exceed this, or the primitive count does:
LARGE_DESIGN_PLACE_DURATION_S = 300
LARGE_DESIGN_PRIMITIVE_COUNT = 100_000
# --- Device resource capacity (2026-08-01 large-design audit) ---
# Contest device is fixed (AMD UltraScale+ xcvu3p, see
# TIMING_DECISION_SYSTEM_PROMPT) -- these are its published LUT/FF/DSP/BRAM
# totals, used to turn a design's raw cell counts into a utilization
# fraction. A design near capacity on ANY resource leaves the placer little
# free space to legalize into during a full re-place -- ispd16_example2
# (298k/394k = 76% LUT utilization) timed out identically on place_design in
# three separate runs (07-20, 07-21, 08-01), never once completing. Unlike
# the cross-run kill switch in _maybe_warm_start_replace, utilization is a
# forward-looking signal: it flags the risk on the FIRST run of a design
# never seen before, not only after it has already failed here.
XCVU3P_LUT_CAPACITY = 394_080
XCVU3P_FF_CAPACITY = 788_160
XCVU3P_DSP_CAPACITY = 2_280
XCVU3P_BRAM_CAPACITY = 720
HIGH_UTILIZATION_FRACTION = 0.70
# An action is demoted when the remaining budget is below 1.3x its estimated
# cost, and dispatch is hard-refused below 1.0x -- a cheap recorded failure
# always beats a half-finished expensive action.
ACTION_COST_DEMOTE_FACTOR = 1.3
# Pricing for an action whose duration has never been measured on a
# large/unknown-scale design: 2x the longest known duration, at least this.
# Small designs keep the pre-cost-model behavior (no gate) instead.
UNKNOWN_EXPENSIVE_ACTION_MIN_S = 900
CHEAP_ACTION_COST_S = 120
# On a large design, full re-places (place_design_explore +
# pblock_full_replace, warm start included) are capped per run: warm start +
# at most one more, and none once elapsed time passes this budget fraction.
FULL_REPLACE_LARGE_DESIGN_CAP = 2
FULL_REPLACE_BUDGET_FRACTION_CUTOFF = 0.5
# Warm-start kill switch (2026-08-01 ispd16_example2 incident): the
# deterministic warm start's whole-design re-place is a bet with a bounded
# downside ONLY when failure is cheap (rolled back in seconds). On this
# design, place_design_explore hung/timed out on this exact warm-start call
# twice across prior runs (07-20, 07-21) with zero recorded wins, then did so
# again before this check existed, burning ~20 minutes of a ~58-minute budget
# on a place_design call that has never once completed here. Lower than the
# main loop's cross-run demotion bar (bad >= 3 in _allowed_forbidden_actions)
# on purpose: an uncompleted full re-place is a much worse failure mode than
# a completed-but-losing one, so it should be believed sooner.
WARM_START_SKIP_AFTER_LOSSES = 2
# After place completes inside a re-place recipe, the requested route
# directive is downgraded to Default when the remaining budget is below this
# multiple of the estimated route duration -- a completed Default route
# always beats a killed Explore route.
ROUTE_DOWNGRADE_FACTOR = 1.2
# Phase 0 diagnostic battery (one-time, pre-LLM): total wall-clock cap, and
# the minimum remaining budget for the report_qor_suggestions probe on a
# large design (the report itself can take minutes there).
INITIAL_DIAGNOSTICS_BUDGET_S = 240
QOR_SUGGESTIONS_MIN_REMAINING_S = 45 * 60
# run_recipe macro-action (recipe architecture, tranche 1): stage whitelist
# and cap. Excludes run_recipe itself (no recursion); stages still pass
# through execute_validated_action, so the budget gate and the full re-place
# cap apply to each stage individually.
RUN_RECIPE_MAX_STAGES = 6
RUN_RECIPE_STAGE_WHITELIST = {
    "place_design_explore",
    "route_explore",
    "phys_opt_design",
    "phys_opt_design_retime",
    "phys_opt_design_pin_swap",
    "pblock",
    # fanout_split removed (pipeline audit 2026-07-28): _execute_run_recipe
    # dispatches stages straight to execute_validated_action without
    # re-checking allowed_actions, so leaving it whitelisted here let the
    # LLM reach the same 0-for-11 action through run_recipe even after it
    # was removed from the normal menu in _allowed_forbidden_actions.
}
DEFAULT_PBLOCK_TARGET_LUT_COUNT = 20000
DEFAULT_PBLOCK_TARGET_FF_COUNT = 40000
DEFAULT_PBLOCK_NAME_PREFIX = "pblock_net_delay"
# lut_opt auto-derivation cap (pipeline audit, 20260802-20260804 sweep):
# hierarchical_input_pins is a REQUIRED, design-specific parameter
# (real hierarchy names like "module/submodule/inst/pin") the LLM has no
# principled way to invent, unlike every other action's parameters, which
# are either optional or auto-derived when omitted. lut_opt was offered
# 105 times across that sweep's 172 iterations and chosen 0 -- consistent
# with the LLM consistently avoiding a REQUIRED field it cannot fill in
# confidently. Default to the current worst-path candidates' own endpoint
# pins (the same names already surfaced in worst_path/current_target_
# candidates) instead of failing outright on missing_action_parameters.
LUT_OPT_DEFAULT_MAX_PINS = 5
# Fix #14: documents the valid `action_parameters` keys per action, shipped
# to the LLM inside every timing context. Previously the decision prompt
# showed `"action_parameters": { ... }` with no key documentation anywhere,
# and run history confirmed the model sent `{}` on every single call -- it
# had no way to know what it could control.
ACTION_PARAMETERS_SCHEMA = {
    "pblock": {
        "target_lut_count": "int, LUTs the clustering pblock region must fit (default: sized per candidate path)",
        "target_ff_count": "int, FFs the region must fit",
        "use_clock_regions": (
            "bool, emit CLOCKREGION ranges instead of site ranges (default false; "
            "DISCOURAGED -- twice produced regions far from the critical cluster; "
            "if the result misses the cluster it is auto-converted back to site ranges)"
        ),
    },
    "pblock_full_replace": {
        "target_lut_count": "int, override whole-design LUT budget (default: 1.5x design utilization)",
        "target_ff_count": "int, override whole-design FF budget",
        "ranges": "str, explicit pblock range (e.g. 'SLICE_X55Y60:SLICE_X111Y254') skipping fabric analysis",
        "place_directive": "str, place_design directive (default 'Default')",
        "route_directive": "str, route_design directive (default 'Default')",
    },
    "rapidwright_optimize_cell_placement": {
        "cell_names": "list[str], explicit cells to move (default: derived from critical paths)",
        "max_candidates": "int, cap on cells moved this iteration",
        "num_paths": "int, critical paths to extract when deriving cells",
        "detour_threshold": "float, min routed/manhattan detour ratio to flag a cell",
    },
    "fanout_split": {
        "split_factor": "int, driver replication factor (default fanout/100, clamped 2-8)",
    },
    "lut_opt": {
        "hierarchical_input_pins": (
            "list[str], input pins of the LUT cone to collapse; if omitted, defaults to the "
            f"current worst-path candidates' own endpoint pins (up to {LUT_OPT_DEFAULT_MAX_PINS})"
        ),
    },
    "phys_opt_design": {
        "directive": (
            "str, e.g. 'Explore', 'AggressiveExplore', 'AggressiveFanoutOpt', "
            "'RuntimeOptimized'; if omitted, the next untried entry of "
            f"{PHYS_OPT_DIRECTIVE_SWEEP} is chosen automatically (see phys_opt_directives_tried)"
        ),
    },
    "phys_opt_design_retime": {
        "directive": (
            "str, phys_opt directive used with retiming enabled; if omitted, the next "
            "untried entry of phys_opt_directives_untried is chosen automatically"
        ),
    },
    "phys_opt_design_pin_swap": {
        "directive": (
            "str, phys_opt directive used with LUT pin-swapping enabled (-critical_pin_opt: "
            "remaps logical to physical pins within a SLICE to reduce routing congestion on "
            "critical nets, without moving any cells); if omitted, the next untried entry of "
            "phys_opt_directives_untried is chosen automatically"
        ),
    },
    "place_design_explore": {
        "directive": (
            "str, place_design directive; if omitted, the next untried entry of "
            f"{PLACE_DIRECTIVE_SWEEP} is chosen automatically (see place_directives_tried)"
        ),
        "route_directive": "str, route_design directive (default 'Explore')",
    },
    "route_explore": {
        "directive": (
            "str, route_design directive for a re-route of the CURRENT placement "
            "(placement untouched -- lowest-variance refinement). Options: 'Explore' "
            "(default), 'AggressiveExplore', 'NoTimingRelaxation', 'MoreGlobalIterations', "
            "'HigherDelayCost'"
        ),
    },
    "qor_suggestions": {
        # No tunable parameters: this runs Vivado's own ML QoR advisor
        # (report_qor_suggestions) on the current placed+routed design, loads
        # the resulting RQS strategy, and applies it via phys_opt + route.
        # It degrades gracefully to a plain phys_opt+route if the RQS
        # directive is unsupported on the design.
    },
    "replicate_register": {},
    "run_recipe": {
        "stages": (
            "list[{\"action\": str, \"params\": dict}], max 6 stages executed "
            "sequentially with a per-stage budget check (the recipe stops cleanly "
            "when the next stage no longer fits) and per-stage keep-best/rollback. "
            "Allowed stage actions: place_design_explore, route_explore, "
            "phys_opt_design, phys_opt_design_retime, phys_opt_design_pin_swap, pblock"
        ),
    },
}
# Fix #5 (pblock sizing): the old fallback chain used the WHOLE design's
# lut_count/ff_count (from rapidwright_read_checkpoint) or a hardcoded 20000
# whenever the LLM didn't pass an explicit target_lut_count. pblock is meant
# to cluster a small batch of critical-path cells into a tight region, not
# fit the entire design into a sub-region -- that's what produced
# "353% utilization" rejections on every single call. These constants give a
# much smaller, per-candidate-path estimate instead, and PBLOCK_SIZE_SHRINK_*
# below let the sizing self-correct if even that estimate is still too big
# for the recommended region, rather than failing outright on the first try.
PBLOCK_LUTS_PER_CANDIDATE_PATH = 40
PBLOCK_FFS_PER_CANDIDATE_PATH = 80
PBLOCK_MIN_TARGET_LUT_COUNT = 200
PBLOCK_MIN_TARGET_FF_COUNT = 200
PBLOCK_SIZE_SHRINK_FACTOR = 0.5
PBLOCK_SIZE_SHRINK_MAX_RETRIES = 3

class WNSParseError(ValueError):
    """Raised when Vivado WNS output cannot be parsed into a plausible slack."""


class VivadoToolCallError(RuntimeError):
    """Raised when an MCP tool call returns isError=True (a real Vivado/RapidWright
    failure, e.g. a pipe desync + restart) rather than normal command output.

    This must be allowed to propagate up to the main optimize() loop so it can
    reopen the last-known-good checkpoint instead of quietly falling through to
    WNS/timing parsers that would otherwise misinterpret the error text as data.
    """

    def __init__(self, tool_name: str, error_text: str):
        self.tool_name = tool_name
        self.error_text = error_text
        super().__init__(f"{tool_name} returned an MCP error: {error_text[:500]}")

TIMING_DECISION_SYSTEM_PROMPT = """
You are an FPGA timing optimization agent operating on AMD UltraScale+ xcvu3p designs.
You receive a structured timing state and must choose a physical optimization action.

HOW THE ACTION MENU WORKS:
- allowed_actions is a RANKED list: earlier actions carry a stronger prior for the
  current timing evidence. It is a prior, not a law -- empirical results in this run
  (action_failure_memory, last_action_failure, place_directives_tried,
  pblock_attempt_history) outrank the static ranking.
- action_guidance maps some allowed actions to a reason they are DISCOURAGED right now.
  You MAY choose a discouraged action, but only if you explicitly rebut its guidance
  reason with evidence from the timing state (e.g. every preferred action has already
  failed or regressed on these targets this run). Put the rebuttal in
  why_this_fits_delay_class.
- forbidden_actions are hard blocks (unimplemented or license-blocked). Never choose them.

PRIORS (the reasoning behind the ranking -- use them, and notice when evidence contradicts them):
- net_delay_bound (net_pct > 0.70): routing-bound paths usually need cell movement or
  placement constraints (pblock, pblock_full_replace, place_design_explore), not logic
  restructuring. lut_opt is discouraged.
- logic_delay_bound (logic_pct > 0.70): placement changes do not reduce logic depth;
  prefer lut_opt, phys_opt_design_retime. Pblock/placement actions are discouraged.
- mixed: phys_opt_design with -retime is a sensible first probe.
- BRAM_CONTROL/DSP_CONTROL endpoints: routing to a hard-block control pin needs physical
  proximity (pblock, replicate_register, place_design_explore); net splitting rarely helps.
- avg_tile_spread > 30 on a net-bound path: address placement before any routing-only fix.
- ONLY if measured congestion_level >= 5: packing cells tighter (any pblock action) makes
  congestion worse; prefer spread-oriented placement or fanout reduction. If
  congestion_level is 0 or unknown, do NOT pick spreading directives (AltSpreadLogic,
  ExtraNetDelay) -- spreading an already-spread-out design makes net delay WORSE
  (measured: -92 MHz).

LEARN FROM THIS RUN'S RESULTS:
- While this run holds a best above baseline, you are protecting that win: polish it
  with incremental moves (phys_opt_design, route_explore, pblock + incremental
  re-route) before any fresh whole-design re-place -- a re-roll discards the winning
  placement and measured 50-120 MHz losses far more often than wins, while
  refinement measured 5/6 positive.
- qor_suggestions runs Vivado's own QoR advisor (report_qor_suggestions -> RQS
  phys_opt -> re-route) on the current placed+routed design. It is a bounded,
  keep-best-gated refinement: worth ONE attempt per run once plain
  phys_opt/route_explore refinement has stalled and before resorting to re-rolls.
- phys_opt_design_pin_swap runs phys_opt_design with LUT pin-swapping enabled
  (-critical_pin_opt): it remaps logical-to-physical pin assignments within a SLICE
  to reduce routing congestion on critical nets, without moving any cell or changing
  placement. Zero placement risk, so it is a safe refinement to try alongside
  phys_opt_design/phys_opt_design_retime once those have stalled -- particularly on
  net-delay-bound paths where the bottleneck is routing, not logic depth.
- An action that regressed or failed on the same targets is worth less than its ranking.
- When choosing a place directive, pick from place_directives_untried (in the timing
  state) rather than inventing one; place_directives_tried shows measured results.
- Same for phys_opt_design/phys_opt_design_retime/phys_opt_design_pin_swap: pick from
  phys_opt_directives_untried rather than repeating one already in phys_opt_directives_tried.
  AggressiveFanoutOpt in particular replicates drivers to fix a shared high-fanout
  net -- try it before concluding a fanout-diagnosed path has no phys_opt lever left.
- Do not repeat an action+parameters combination that already failed; change what the
  failure evidence says was wrong (region size, directive, targets), or change action.

BUDGET AWARENESS:
- The timing state carries design_scale, time_remaining_s, and this design's measured
  action durations. An action whose estimated cost exceeds the remaining budget is
  refused at dispatch (insufficient_budget) -- treat "costs ~N min" guidance as real,
  and prefer refinements that fit over expensive re-rolls that will be refused.
- run_recipe executes a full pipeline in one decision -- prefer it over issuing the
  same stages one iteration at a time. Stages are whitelisted (place_design_explore,
  route_explore, phys_opt_design, phys_opt_design_retime, phys_opt_design_pin_swap,
  pblock), max 6; each stage is measured and kept-or-rolled-back individually, and
  the recipe stops cleanly when the next stage no longer fits the remaining budget.
"""


def parse_timing_summary_static(timing_report: str) -> dict:
    """
    Parse timing summary report to extract WNS, TNS, and failing endpoints.
    Returns dict with keys: wns, tns, failing_endpoints
    
    Parses the Design Timing Summary table:
        WNS(ns)      TNS(ns)  TNS Failing Endpoints  ...
        -------      -------  ---------------------  ...
         -0.099       -1.449                     42  ...
    
    This is a shared utility function used by both FPGAOptimizer and FPGAOptimizerTest.
    """
    result = {
        "wns": None,
        "tns": None,
        "failing_endpoints": None
    }
    
    lines = timing_report.split('\n')
    
    # Find the line with "WNS(ns)" header
    header_idx = -1
    for i, line in enumerate(lines):
        if 'WNS(ns)' in line and 'TNS(ns)' in line:
            header_idx = i
            break
    
    if header_idx == -1:
        return result
    
    # The data line should be 2 lines after the header (skipping the dashes line)
    # Format: whitespace + values separated by whitespace
    data_idx = header_idx + 2
    if data_idx >= len(lines):
        return result
    
    data_line = lines[data_idx].strip()
    if not data_line:
        return result
    
    # Split by whitespace and extract first 3 values: WNS, TNS, TNS Failing Endpoints
    parts = data_line.split()
    if len(parts) >= 3:
        try:
            result["wns"] = float(parts[0])
            result["tns"] = float(parts[1])
            result["failing_endpoints"] = int(parts[2])
        except (ValueError, IndexError):
            # If parsing fails, leave as None
            pass
    
    return result


def validate_wns_sanity_static(
    wns: float,
    clock_period_ns: Optional[float] = None,
    source: str = "WNS",
) -> float:
    """Reject WNS values that are implausible before history/regression logic sees them."""
    if abs(wns) > WNS_SANITY_ABS_LIMIT_NS:
        raise WNSParseError(
            f"{source} parsed implausible WNS {wns} ns; abs limit is {WNS_SANITY_ABS_LIMIT_NS} ns"
        )
    if clock_period_ns is not None and clock_period_ns > 0:
        max_positive = max(clock_period_ns * WNS_SANITY_POSITIVE_CLOCK_FRACTION, 0.001)
        if wns > max_positive:
            raise WNSParseError(
                f"{source} parsed implausible positive WNS {wns} ns; "
                f"max allowed is {max_positive:.6f} ns for clock period {clock_period_ns:.6f} ns"
            )
    return wns


def parse_wns_value_static(
    output: str,
    clock_period_ns: Optional[float] = None,
    source: str = "vivado_get_wns",
) -> float:
    """Parse WNS only from sentinel-delimited or single-line numeric Tcl output."""
    text = str(output or "").strip()
    sentinel = re.search(
        r"WNS_VALUE_BEGIN\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*WNS_VALUE_END",
        text,
        re.MULTILINE,
    )
    if sentinel:
        value = float(sentinel.group(1))
        return validate_wns_sanity_static(value, clock_period_ns, source)

    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
        value = float(text)
        return validate_wns_sanity_static(value, clock_period_ns, source)

    raise WNSParseError(f"{source} did not contain a single valid WNS value: {text[:200]}")


def parse_timing_summary_wns_static(
    timing_report: str,
    clock_period_ns: Optional[float] = None,
) -> Optional[float]:
    """Extract WNS from the timing summary table and sanity-check it."""
    wns = parse_timing_summary_static(timing_report).get("wns")
    if wns is None:
        return None
    return validate_wns_sanity_static(float(wns), clock_period_ns, "report_timing_summary")


def load_system_prompt() -> str:
    """Load system prompt from SYSTEM_PROMPT.TXT file."""
    script_dir = Path(__file__).parent.resolve()
    prompt_file = script_dir / "SYSTEM_PROMPT.TXT"
    
    try:
        with open(prompt_file, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"System prompt file not found: {prompt_file}")
        raise
    except Exception as e:
        logger.error(f"Failed to load system prompt: {e}")
        raise


def convert_mcp_tool_to_openai(tool, server_prefix: str) -> dict:
    """Convert MCP tool definition to OpenAI-compatible format with server prefix."""
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": f"{server_prefix}_{tool.name}",
            "description": tool.description or "",
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }
    }


class VivadoMCPAdapter:
    """Small adapter used by local orchestration helpers to call Vivado MCP Tcl."""

    def __init__(self, optimizer: "DCPOptimizer"):
        self.optimizer = optimizer

    async def run_tcl_command(self, tcl: str, timeout: int = 300) -> str:
        return await self.optimizer.call_tool(
            "vivado_run_tcl",
            {"command": tcl, "timeout": timeout},
            internal=True,
        )


class DCPOptimizerBase:
    """Base class with shared functionality for FPGA optimization."""
    
    def __init__(self, debug: bool = False, run_dir: Optional[Path] = None):
        self.debug = debug
        
        # Create run directory if not provided
        if run_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path.cwd() / f"dcp_optimizer_run-{timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created run directory: {self.run_dir}")
        else:
            self.run_dir = run_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.exit_stack = AsyncExitStack()
        self.rapidwright_session: Optional[ClientSession] = None
        self.vivado_session: Optional[ClientSession] = None
        
        # Use run directory for all temporary files
        self.temp_dir = self.run_dir
        logger.info(f"Working directory: {self.temp_dir}")
        
        # Timing tracking
        self.initial_wns = None
        self.initial_tns = None
        self.initial_failing_endpoints = None
        self.high_fanout_nets = []
        self.clock_period = None
        self.target_clock = None  # Set to clock name (e.g. "clk_fpl26contest") for clock-specific Fmax
        
        # Log file handles
        self._rw_log_file = None
        self._v_log_file = None

        # Structural-action failure/cooldown tracking (fix #7)
        self.action_structural_failure_iters: dict[str, list[int]] = {}
        self.action_structural_cooldown_until_iter: dict[str, int] = {}
    
    async def start_servers(self, log_prefix: str = ""):
        """Start and connect to both MCP servers."""
        script_dir = Path(__file__).parent.resolve()
        
        # Create log files in run directory
        rapidwright_log = self.run_dir / "rapidwright.log"
        rapidwright_mcp_log = self.run_dir / "rapidwright-mcp.log"
        vivado_log = self.run_dir / "vivado.log"
        vivado_journal = self.run_dir / "vivado.jou"
        vivado_mcp_log = self.run_dir / "vivado-mcp.log"
        
        # Open log files (if not in debug mode, redirect stderr to log)
        if self.debug:
            self._rw_log_file = None
            self._v_log_file = None
            logger.info("Debug mode: MCP server output will be shown in console")
            if log_prefix:
                print(f"{log_prefix} Debug mode: MCP server output will be shown in console")
        else:
            self._rw_log_file = open(rapidwright_mcp_log, 'w')
            self._v_log_file = open(vivado_mcp_log, 'w')
            logger.info(f"RapidWright Java output: {rapidwright_log}")
            logger.info(f"RapidWright MCP output: {rapidwright_mcp_log}")
            logger.info(f"Vivado output: {vivado_log}")
            logger.info(f"Vivado journal: {vivado_journal}")
            logger.info(f"Vivado MCP output: {vivado_mcp_log}")
            print(f"Log files in {self.run_dir.name}/: {rapidwright_log.name}, {rapidwright_mcp_log.name}, {vivado_log.name}, {vivado_journal.name}, {vivado_mcp_log.name}")
        
        # RapidWright MCP server config
        rapidwright_args = [str(script_dir / "RapidWrightMCP" / "server.py")]
        if not self.debug:
            rapidwright_args.extend([
                "--java-log", str(rapidwright_log),
                "--mcp-log", str(rapidwright_mcp_log)
            ])
        
        env = {**os.environ}
        rapidwright_submodule = script_dir / "RapidWright"
        if rapidwright_submodule.is_dir() and "RAPIDWRIGHT_PATH" not in env:
            env["RAPIDWRIGHT_PATH"] = str(rapidwright_submodule)
            env["CLASSPATH"] = f"{rapidwright_submodule}/bin:{rapidwright_submodule}/jars/*"
        
        rapidwright_config = {
            "command": sys.executable,
            "args": rapidwright_args,
            "cwd": str(self.run_dir),
            "env": env
        }
        
        # Vivado MCP server config
        vivado_args = [str(script_dir / "VivadoMCP" / "vivado_mcp_server.py")]
        if not self.debug:
            vivado_args.extend([
                "--vivado-log", str(vivado_log),
                "--vivado-journal", str(vivado_journal)
            ])
        
        vivado_config = {
            "command": sys.executable,
            "args": vivado_args,
            "cwd": str(self.run_dir),
            "env": {**os.environ}
        }
        
        # Start RapidWright MCP
        logger.info("Starting RapidWright MCP server...")
        if log_prefix:
            print(f"{log_prefix} Starting RapidWright MCP server...")
        start_time = time.time()
        
        rw_params = StdioServerParameters(**rapidwright_config)
        rw_transport = await self.exit_stack.enter_async_context(
            stdio_client(rw_params, errlog=self._rw_log_file)
        )
        rw_read, rw_write = rw_transport
        self.rapidwright_session = await self.exit_stack.enter_async_context(
            ClientSession(rw_read, rw_write)
        )
        await self.rapidwright_session.initialize()
        
        elapsed = time.time() - start_time
        logger.info(f"RapidWright MCP server started in {elapsed:.2f}s")
        if log_prefix:
            print(f"{log_prefix} RapidWright MCP server started in {elapsed:.2f}s")
        
        # Start Vivado MCP
        logger.info("Starting Vivado MCP server...")
        if log_prefix:
            print(f"{log_prefix} Starting Vivado MCP server...")
        start_time = time.time()
        
        vivado_params = StdioServerParameters(**vivado_config)
        vivado_transport = await self.exit_stack.enter_async_context(
            stdio_client(vivado_params, errlog=self._v_log_file)
        )
        v_read, v_write = vivado_transport
        self.vivado_session = await self.exit_stack.enter_async_context(
            ClientSession(v_read, v_write)
        )
        await self.vivado_session.initialize()
        
        elapsed = time.time() - start_time
        logger.info(f"Vivado MCP server started in {elapsed:.2f}s")
        if log_prefix:
            print(f"{log_prefix} Vivado MCP server started in {elapsed:.2f}s")
        
        logger.info("Both MCP servers connected")
        if log_prefix:
            print(f"{log_prefix} Both MCP servers connected successfully")
    
    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()
        
        if self._rw_log_file:
            self._rw_log_file.close()
        if self._v_log_file:
            self._v_log_file.close()
        
        logger.info(f"Run directory preserved at: {self.run_dir}")
    
    def calculate_fmax(self, wns: Optional[float], clock_period: Optional[float]) -> Optional[float]:
        """
        Calculate achievable fmax in MHz based on WNS and clock period.
        
        fmax = 1 / (clock_period - WNS) when WNS < 0 (timing violation)
        fmax = 1 / clock_period when WNS >= 0 (timing met)
        
        Returns fmax in MHz, or None if cannot be calculated.
        """
        if clock_period is None or clock_period <= 0:
            return None
        if wns is None:
            return None
        
        achievable_period_ns = clock_period - wns
        if achievable_period_ns <= 0:
            return None
        
        return 1000.0 / achievable_period_ns
    
    async def get_clock_period(self, call_tool_fn) -> Optional[float]:
        """
        Query the clock period of the target clock from Vivado in nanoseconds.
        
        First checks for the contest clock 'clk_fpl26contest'. If found, uses its
        period and sets self.target_clock. Otherwise falls back to the endpoint clock
        of the worst setup timing path.
        
        Args:
            call_tool_fn: Function to call Vivado tools, should accept (tool_name, arguments)
        
        Returns the period of the target clock, or None if no clocks found.
        """
        tcl_cmd = (
            "set contest_clk [get_clocks -quiet clk_fpl26contest]; "
            "if {$contest_clk ne {}} { "
            "  puts \"CLOCK:clk_fpl26contest\"; "
            "  puts [get_property PERIOD $contest_clk]; "
            "} else { "
            "  set tp [get_timing_paths -max_paths 1 -setup]; "
            "  if {$tp ne {}} { "
            "    set clk [get_property ENDPOINT_CLOCK $tp]; "
            "    if {$clk ne {}} { "
            "      puts \"CLOCK:$clk\"; "
            "      puts [get_property PERIOD [get_clocks $clk]]; "
            "    } "
            "  } "
            "}"
        )
        try:
            result = await call_tool_fn("run_tcl", {"command": tcl_cmd})
            
            clock_name = None
            for token in result.strip().split():
                if token.startswith('CLOCK:'):
                    clock_name = token[len('CLOCK:'):]
                    continue
                if token.startswith('ERROR') or token.startswith('WARNING'):
                    continue
                try:
                    period = float(token)
                    if period > 0:
                        if clock_name:
                            self.target_clock = clock_name
                            logger.info(f"Target clock: {clock_name}, period: {period:.3f} ns")
                        else:
                            logger.info(f"Critical clock period: {period:.3f} ns")
                        return period
                except ValueError:
                    continue
        except Exception as e:
            logger.warning(f"Failed to get clock period: {e}")
        
        logger.warning("Could not determine clock period from Vivado")
        return None
    
    async def get_wns_for_target_clock(self, call_tool_fn) -> Optional[float]:
        """
        Get WNS specifically for the target clock domain.
        
        When target_clock is set (e.g. 'clk_fpl26contest'), queries WNS filtered
        to that clock's timing paths. Falls back to overall WNS if no target clock.
        
        Args:
            call_tool_fn: Function to call Vivado tools, should accept (tool_name, arguments)
        
        Returns WNS in nanoseconds, or None if query fails.
        """
        if self.target_clock:
            tcl_cmd = (
                f"set clk_obj [get_clocks -quiet {{{self.target_clock}}}]; "
                f"if {{$clk_obj ne {{}}}} {{ "
                f"  set tp [get_timing_paths -max_paths 1 -setup -to $clk_obj]; "
                f"  if {{[llength $tp] > 0}} {{set wns_value [get_property SLACK $tp]}} else {{set wns_value 0.0}}; "
                f"}} else {{ "
                f"  set tp [get_timing_paths -max_paths 1 -slack_lesser_than 999]; "
                f"  if {{[llength $tp] > 0}} {{set wns_value [get_property SLACK $tp]}} else {{set wns_value 0.0}} "
                f"}}; "
                f"puts {{WNS_VALUE_BEGIN}}; puts $wns_value; puts {{WNS_VALUE_END}}"
            )
        else:
            tcl_cmd = (
                "set tp [get_timing_paths -max_paths 1 -slack_lesser_than 999]; "
                "if {[llength $tp] > 0} {set wns_value [get_property SLACK $tp]} else {set wns_value 0.0}; "
                "puts {WNS_VALUE_BEGIN}; puts $wns_value; puts {WNS_VALUE_END}"
            )
        
        try:
            result = await call_tool_fn("run_tcl", {"command": tcl_cmd})
            wns = parse_wns_value_static(result, self.clock_period, "target_clock_wns")
            clock_info = f" (clock: {self.target_clock})" if self.target_clock else ""
            logger.info(f"WNS{clock_info}: {wns:.3f} ns")
            return wns
        except WNSParseError as e:
            logger.error(f"Failed to parse WNS for target clock: {e}")
        except VivadoToolCallError:
            # Real Vivado-side failure, not a parsing issue -- propagate so the
            # caller's recovery handler (reopen last-good checkpoint) can run,
            # instead of silently returning None here.
            raise
        except Exception as e:
            logger.warning(f"Failed to get WNS for target clock: {e}")
        
        return None
    
    def parse_high_fanout_nets(self, report: str) -> list[tuple[str, int, int]]:
        """
        Parse high fanout nets report and return list of (net_name, fanout, path_count).
        """
        nets = []
        lines = report.split('\n')
        in_net_section = False
        
        for line in lines:
            if 'Paths' in line and 'Fanout' in line and 'Parent Net Name' in line:
                in_net_section = True
                continue
            
            if in_net_section:
                if line.startswith('---') or not line.strip():
                    continue
                if line.startswith('==='):
                    break
                
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        path_count = int(parts[0])
                        fanout = int(parts[1])
                        net_name = parts[2]
                        
                        if (net_name and 
                            '/' in net_name and
                            not net_name.startswith('get_') and
                            not net_name.startswith('ERROR') and
                            not net_name.startswith('WARNING')):
                            nets.append((net_name, fanout, path_count))
                    except ValueError:
                        continue
        
        return nets

    def _format_fmax_results(
        self,
        clock_period: Optional[float],
        initial_wns: Optional[float],
        result_wns: Optional[float],
        result_label: str = "Final",
    ) -> list[str]:
        """Format Fmax/WNS results block as a list of lines.
        
        """
        initial_fmax = self.calculate_fmax(initial_wns, clock_period)
        result_fmax = self.calculate_fmax(result_wns, clock_period)
        result_fmax_label = f"{result_label} Fmax:"
        result_wns_label = f"{result_label} WNS:"
        
        lines: list[str] = []
        if initial_fmax is not None and result_fmax is not None:
            target_fmax = 1000.0 / clock_period
            fmax_change = result_fmax - initial_fmax
            lines.append(f"  {'Target Fmax:':<21s}{target_fmax:8.2f} MHz  (clock period: {clock_period:.3f} ns)")
            lines.append(f"  {'Initial Fmax:':<21s}{initial_fmax:8.2f} MHz  (WNS: {initial_wns:.3f} ns)")
            lines.append(f"  {result_fmax_label:<21s}{result_fmax:8.2f} MHz  (WNS: {result_wns:.3f} ns)")
            lines.append(f"  {'Fmax Improvement:':<21s}{fmax_change:+8.2f} MHz  (WNS: {result_wns - initial_wns:+.3f} ns)")
        else:
            if clock_period is not None:
                target_fmax = 1000.0 / clock_period
                lines.append(f"  {'Clock period:':<21s}{clock_period:8.3f} ns (target: {target_fmax:.2f} MHz)")
            if initial_wns is not None:
                fmax_str = f"  (fmax: {initial_fmax:.2f} MHz)" if initial_fmax else ""
                lines.append(f"  {'Initial WNS:':<21s}{initial_wns:8.3f} ns{fmax_str}")
            if result_wns is not None:
                fmax_str = f"  (fmax: {result_fmax:.2f} MHz)" if result_fmax else ""
                lines.append(f"  {result_wns_label:<21s}{result_wns:8.3f} ns{fmax_str}")
            if initial_wns is not None and result_wns is not None:
                lines.append(f"  {'WNS Improvement:':<21s}{result_wns - initial_wns:+8.3f} ns")
        
        return lines
    
    
    def print_wns_change(
        self,
        initial_wns: Optional[float],
        final_wns: Optional[float],
        clock_period: Optional[float]
    ):
        """Print Fmax/WNS change comparison with improvement/regression status."""
        if final_wns is None or initial_wns is None:
            return
        
        initial_fmax = self.calculate_fmax(initial_wns, clock_period)
        final_fmax = self.calculate_fmax(final_wns, clock_period)
        
        if initial_fmax is not None and final_fmax is not None:
            fmax_improvement = final_fmax - initial_fmax
            pct = (fmax_improvement / initial_fmax) * 100 if initial_fmax else 0
            print(f"\n*** Fmax: {initial_fmax:.2f} -> {final_fmax:.2f} MHz ({fmax_improvement:+.2f} MHz, {pct:+.1f}%) ***")
            print(f"*** WNS:  {initial_wns:.3f} -> {final_wns:.3f} ns ***")
            if fmax_improvement > 0:
                print(f"IMPROVEMENT: Fmax improved by {fmax_improvement:.2f} MHz")
            elif fmax_improvement < 0:
                print(f"REGRESSION: Fmax got worse by {-fmax_improvement:.2f} MHz")
            else:
                print("NO CHANGE: Fmax is the same")
        else:
            wns_improvement = final_wns - initial_wns
            print(f"\n*** WNS: {initial_wns:.3f} -> {final_wns:.3f} ns ({wns_improvement:+.3f} ns) ***")
            if wns_improvement > 0:
                print(f"IMPROVEMENT: WNS improved by {wns_improvement:.3f} ns")
            elif wns_improvement < 0:
                print(f"REGRESSION: WNS got worse by {-wns_improvement:.3f} ns")
            else:
                print("NO CHANGE")
    
    def print_fmax_status(self, label: str, wns: Optional[float]):
        """Print Fmax (primary) and WNS (secondary) for a given measurement point."""
        if wns is None:
            print(f"*** {label}: WNS unknown ***")
            return
        fmax = self.calculate_fmax(wns, self.clock_period)
        clock_info = f" (clock: {self.target_clock})" if self.target_clock else ""
        if fmax is not None:
            print(f"*** {label} Fmax{clock_info}: {fmax:.2f} MHz (WNS: {wns:.3f} ns) ***")
        else:
            print(f"*** {label} WNS{clock_info}: {wns:.3f} ns ***")
    
    def print_test_summary(
        self,
        title: str,
        elapsed_seconds: float,
        initial_wns: Optional[float],
        final_wns: Optional[float],
        clock_period: Optional[float],
        extra_info: str = ""
    ):
        """Print formatted test summary."""
        print("\n" + "="*70)
        print(title)
        print("="*70)
        print(f"Total runtime: {elapsed_seconds:.2f} seconds ({elapsed_seconds/60:.2f} minutes)")
        
        result_lines = self._format_fmax_results(clock_period, initial_wns, final_wns)
        if result_lines:
            print(f"\nFmax Results:")
            print("\n".join(result_lines))
        
        if extra_info:
            print(f"\n{extra_info}")
        print("="*70)


class DCPOptimizer(DCPOptimizerBase):
    """FPGA Design Optimization Agent using RapidWright and Vivado MCPs."""
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        debug: bool = False,
        run_dir: Optional[Path] = None,
        hard_limit_seconds: Optional[int] = None,
    ):
        super().__init__(debug=debug, run_dir=run_dir)

        self.api_key = api_key
        self.model = model
        # 2026-08-01: override for exploratory/offline runs (e.g. "let a
        # large design run overnight to see if a full re-place ever
        # converges"). None keeps CheckpointManager's contest-safe 3500 s
        # default -- this must never silently change what a real contest
        # submission run does. See --budget-seconds in main().
        self.hard_limit_seconds_override = hard_limit_seconds
        self.tools: list[dict] = []
        self.messages: list[dict] = []
        
        self.openai = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Track optimization progress
        self.iteration = 0
        self.best_wns = float('-inf')
        self.no_improvement_count = 0
        self.llm_call_count = 0
        
        # Track token usage and costs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.api_call_details = []
        
        # Track all tool calls with timing and WNS
        self.tool_call_details = []
        
        # Track total runtime
        self.start_time = None
        self.end_time = None

        # Integrated run-safety helpers. These are initialized after baseline
        # timing is known so the checkpoint manager has a real clock/wns basis.
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.vivado_adapter: Optional[VivadoMCPAdapter] = None
        self.eco_router: Optional[ECORouter] = None
        self.shield_generator: Optional[ShieldGenerator] = None
        self.active_checkpoint: Optional[str] = None
        self.pending_candidate_checkpoint: Optional[str] = None
        self.last_recipe: str = "initial"
        self.last_targets: list[str] = []
        self.last_batch_size: Optional[int] = None
        self.last_vivado_runtime_s: float = 0.0
        # Cumulative vivado_*/rapidwright_* tool time since the last history
        # record. history.json's per-iteration vivado_runtime_s used to be
        # just the LAST tool call's elapsed time (~0.5 s for a full
        # place+route iteration that took minutes), which fed the
        # time-awareness features garbage.
        self.iteration_tool_elapsed_s: float = 0.0
        self.last_route_result = None
        self.recorded_iterations: set[int] = set()
        self.last_design_info: dict = {}
        self.target_tier: str = "unknown"
        self.current_target_candidates: list[dict] = []
        self.path_delay_classification: str = "unknown"
        self.path_delay_breakdown: dict = {}
        self.best_fmax_mhz: Optional[float] = None
        self.current_period_ns: Optional[float] = None
        self.last_passing_period_ns: Optional[float] = None
        self.first_failing_period_ns: Optional[float] = None
        self.constraint_audit_done = False
        self.bisection_active = False
        # Score item 1: the honest-target reconstrain focus pass runs at most
        # once per run (see _reconstrain_focus_pass).
        self._reconstrain_focus_done = False
        # Measured hard-block demand from a failed pblock's own DRC (run
        # 20260714_182751: candidate-based estimate said ~14 BRAMs, the
        # server-side cell expansion actually needed 160). Once a pblock
        # aborts on resource validation, the validation's `required` counts
        # become the demand floor for every later sizing this run.
        self.pblock_hard_block_demand: dict[str, int] = {}
        self.last_spread_info: dict = {}
        self.last_timing_context: dict = {}
        self.last_llm_decision: dict = {}
        self.last_decision_trace: dict = {}
        self.last_rapidwright_edit_summary: Optional[dict] = None
        self.last_recorded_wns: Optional[float] = None
        self.consecutive_no_improvement = 0
        self.structural_override_active = False
        self.action_failure_counts: dict[str, int] = {}
        self.action_failure_memory: dict[str, dict] = {}
        self.last_no_action_failure_key: Optional[tuple] = None
        # Fix #6 (blacklist expiry): checkpoint_manager.cells_blacklisted is a
        # plain append-only list with no decay. On long runs, every cell that
        # ever failed/regressed once stays permanently blacklisted, so the
        # critical-path candidate pool (which tends to be dominated by a small
        # recurring set of cells) shrinks every iteration until placement
        # candidate search comes back empty (no_action_target). This dict
        # tracks *when* each cell was blacklisted so we can expire entries
        # after BLACKLIST_TTL_ITERS instead of banning them forever.
        self.cell_blacklist_added_iter: dict[str, int] = {}
        # Fix #7 (pblock overlap cooldown): a pblock overlapping an
        # already-applied region is a deterministic geometric fact, not a
        # flaky failure -- it will fail again every time until the applied
        # pblock layout itself changes. The generic no-action cooldown
        # (ACTION_FAILURE_COOLDOWN_ITERS=5) is keyed to target_fingerprint,
        # which drifts as candidate lists shift slightly (e.g. due to
        # blacklist churn), silently resetting the strike counter and letting
        # "pblock" get re-offered and re-fail every few iterations for the
        # rest of the run. This cooldown is tracked independently of
        # fingerprint so it can't be reset by that drift.
        self.pblock_region_cooldown_until_iter: int = -1
        self.phys_opt_retime_supported: Optional[bool] = None
        self.implementation_license_available: Optional[bool] = None
        # Fix #1 (action-key bookkeeping): last_recipe is a human-readable
        # history label and gets renamed by _remember_recipe() for display
        # purposes (e.g. "rapidwright_optimize_cell_placement" -> displayed
        # as "rapidwright_cell_placement"). Every piece of code that decides
        # whether an action is suppressed/cooling-down must instead key off
        # last_action_key, which is set exactly once per dispatched action in
        # execute_validated_action() and is NEVER touched by _remember_recipe.
        self.last_action_key: str = "initial"
        # Fix #2 (pblock validation): remember every pblock region that has
        # been successfully applied this run so future regions can be
        # checked for overlap before being applied.
        self.applied_pblock_regions: list[dict] = []
        # Analysis Layer (Stage 2): replaces the inside of
        # _build_timing_context() with a normalize -> cluster -> diagnose ->
        # gather_evidence -> rank_hypotheses pipeline. actions_for() still
        # calls _allowed_forbidden_actions first as the base allowed/forbidden
        # (unchanged), then applies at most one confidence-scored hypothesis
        # (>= 0.75 confidence) on top -- see analysis_layer.py's module
        # docstring for exactly which hypotheses are real vs. still deferred.
        # last_diagnosis is kept around (like last_spread_info) for
        # logging/inspection.
        self.analysis_engine = AnalysisEngine(self)
        self.last_diagnosis: Optional[Diagnosis] = None
        # Fix #10: the contest scores whatever design sits at the output DCP
        # path when the 1-hour budget expires ("Teams should update the best
        # solution ... as they go"). Keep the path around so every improvement
        # and every stop path can publish the current best checkpoint there.
        self.output_dcp_path: Optional[Path] = None
        # Fix #14: the full error text of the most recent failed action, fed
        # back to the LLM in the next timing context so it can actually react
        # (e.g. change pblock target counts) instead of guessing blind.
        self.last_action_failure: Optional[dict] = None
        # Timing validity (design-state provenance): WNS is only meaningful on
        # a fully placed AND routed design. Contest inputs are implemented
        # DCPs, so the state starts "routed"; _update_design_state() tracks
        # every place/route/unplace/open_checkpoint after that. Any WNS
        # observed while state != "routed" (e.g. mid-pblock_full_replace,
        # after `place_design -unplace`) is estimated/garbage and must never
        # reach best_wns, iteration history, or the LLM.
        self.design_state: str = "routed"
        # Whether the currently executing action has issued any command that
        # mutates the live design (place/route/phys_opt/pblock/unplace or a
        # RapidWright edit). If such an action then FAILS, the live session is
        # in an undefined state that no longer matches best_checkpoint, so the
        # main loop must restore before the next iteration.
        self.last_action_mutated_design: bool = False
        # Priors model (item 1): reasons why currently-allowed actions are
        # discouraged. Rebuilt by _allowed_forbidden_actions each iteration,
        # shipped to the LLM as action_guidance.
        self.last_action_guidance: dict[str, str] = {}
        # Directive sweep memory (item 3a): directive -> outcome for every
        # place_design_explore attempt this run, so the next attempt tries a
        # different directive instead of repeating one that already ran.
        self.place_directive_results: dict[str, dict] = {}
        self.last_place_directive: Optional[str] = None
        # Same directive-sweep memory as place_directive_results, but for the
        # phys_opt_design/_retime/_pin_swap family (see PHYS_OPT_DIRECTIVE_SWEEP).
        # Shared across all three actions since they all resolve to the same
        # underlying `phys_opt_design -directive X [-retime] [-critical_pin_opt]`
        # Tcl call -- the directive is the actual lever, the flags are modifiers.
        self.phys_opt_directive_results: dict[str, dict] = {}
        self.last_phys_opt_directive: Optional[str] = None
        # Pblock search memory (item 3b): one entry per pblock/full_replace
        # attempt (ranges, target sizes, outcome), so sizing can adapt --
        # regression at size S => try a looser region, not the same one.
        self.pblock_attempt_history: list[dict] = []
        self.last_pblock_sizing: Optional[dict] = None
        # Congestion evidence cache (item 4), refreshed at most once per
        # iteration: {"iteration": int, "congestion_level": Optional[int],
        # "detail": str}.
        self.last_congestion_info: dict = {}
        # History(16) forensics: the input DCP itself carries ~6 benign
        # unplaced primitives (no routable nets; route status is clean).
        # Probed once at run start so the provenance gate can tell "the
        # design's normal artifacts" apart from "an action broke placement".
        self.baseline_unplaced_cells: Optional[int] = None
        # Consecutive action failures that aborted cheaply (validation
        # rejections costing < CHEAP_FAILURE_RUNTIME_S). These get a much
        # larger stall budget than expensive place/route stalls.
        self.cheap_failure_streak: int = 0
        # Cross-run persistent priors: per-design action/directive win-loss
        # records accumulated across runs (~/.fpl26_action_priors.json). Used
        # to demote proven losers on the SAME design from iteration 1 and to
        # pick the warm start's directive from measured history instead of a
        # hardcoded default.
        self.crossrun_store_path: Path = Path.home() / ".fpl26_action_priors.json"
        self.crossrun_design_key: Optional[str] = None
        self.crossrun_priors: dict = {}
        self._crossrun_saved: bool = False
        # Cost model (2026-07-12): wall-times measured this run, per action
        # kind ("place"/"route"/"phys_opt"). max() of these -- else the
        # cross-run "durations" prior -- prices every dispatch; see
        # _estimated_action_cost_s and the insufficient_budget gate.
        self.action_durations: dict[str, list[float]] = {}
        # "small" | "large" | "unknown". Drives the full re-place cap and the
        # pessimistic pricing of never-measured actions; refined as evidence
        # arrives (cross-run priors at load, primitive count after
        # rapidwright_read_checkpoint, measured place after the warm start).
        self.design_scale: str = "unknown"
        # Design resource counts vs. known xcvu3p device capacity, keyed
        # "LUT"/"FF"/"DSP"/"BRAM" -> fraction (2026-08-01 large-design
        # audit). Populated by _compute_resource_utilization during Phase 0
        # diagnostics; empty until then. See HIGH_UTILIZATION_FRACTION.
        self.resource_utilization: dict = {}
        # Full re-places dispatched this run (place_design_explore +
        # pblock_full_replace, warm start and run_recipe stages included).
        # Capped on large designs -- see _full_replace_blocked_reason.
        self.full_replace_attempts: int = 0
        # Phase 0 diagnostic battery results (logic floor WNS, fanout
        # profile, QoR suggestions), probed once before the first LLM turn.
        self.design_signature: dict = {}

    async def start_servers(self):
        """Start and connect to both MCP servers."""
        await super().start_servers()
        await self._collect_tools()
        logger.info(f"Connected to servers with {len(self.tools)} tools available")
    
    async def _collect_tools(self):
        """Collect and convert tools from both MCP servers."""
        self.tools = []
        
        rw_response = await self.rapidwright_session.list_tools()
        for tool in rw_response.tools:
            self.tools.append(convert_mcp_tool_to_openai(tool, "rapidwright"))
        
        v_response = await self.vivado_session.list_tools()
        for tool in v_response.tools:
            self.tools.append(convert_mcp_tool_to_openai(tool, "vivado"))
    
    async def call_tool(self, tool_name: str, arguments: dict, internal: bool = False) -> str:
        """Execute a tool call on the appropriate MCP server."""
        # Parse server prefix from tool name
        if tool_name.startswith("rapidwright_"):
            session = self.rapidwright_session
            actual_name = tool_name[len("rapidwright_"):]
        elif tool_name.startswith("vivado_"):
            session = self.vivado_session
            actual_name = tool_name[len("vivado_"):]
        else:
            return json.dumps({"error": f"Unknown tool prefix in: {tool_name}"})
        
        # Track timing for this tool call
        start_time = time.time()
        wns_measured = None
        error_occurred = False
        
        try:
            logger.info(f"Calling {tool_name} with args: {json.dumps(arguments)[:200]}...")
            if (
                not internal
                and tool_name == "vivado_route_design"
                and self.eco_router is not None
                and self.pending_candidate_checkpoint is not None
            ):
                result_text = await self._route_candidate_with_eco(arguments)
            elif not internal and tool_name == "vivado_phys_opt_design":
                result_text = await self._run_phys_opt_with_policy(arguments)
            elif not internal and tool_name == "vivado_create_and_apply_pblock":
                result_text = await self._maybe_run_pblock_or_phys_opt(arguments)
            else:
                result = await session.call_tool(actual_name, arguments)

                # Extract text content from result
                if result.content:
                    text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                    result_text = "\n".join(text_parts)
                else:
                    result_text = "(no output)"

                # A real server-side failure (e.g. Vivado pipe desync + restart)
                # comes back as an MCP error result, not normal command output.
                # Without this check, that error text gets fed straight into the
                # WNS/timing parsers below as if it were real Vivado output.
                if getattr(result, "isError", False):
                    logger.error(f"{tool_name} returned an MCP error, not Vivado output: {result_text[:500]}")
                    raise VivadoToolCallError(tool_name, result_text)

            # Vivado hang/crash auto-recovery (pipeline audit 2026-07-28): a
            # timed-out or crashed Vivado command comes back as a normal
            # (non-isError) text result telling the caller to call
            # restart_vivado -- the server catches pexpect.TIMEOUT/EOF
            # internally and never restarts itself. Nothing ever called it
            # automatically, so one hang/crash silently failed every
            # remaining Vivado call for the rest of the run (boom_soc and
            # ispd16_example2's 07-19/07-20 runs each burned their whole
            # remaining budget this way after a single place/route timeout).
            # The server's own error text always names "restart_vivado" as
            # the fix, on every failure mode that needs it (timeout, crash,
            # pty desync) -- restart right here, at the single choke point
            # every vivado_ call passes through, then raise the existing
            # VivadoToolCallError so optimize()'s established recovery path
            # (reopen best checkpoint, resync RapidWright, continue the run)
            # picks up from a live session instead of a dead one.
            if (
                tool_name.startswith("vivado_")
                and tool_name != "vivado_restart_vivado"
                and "restart_vivado" in result_text
            ):
                logger.error(
                    "Vivado hang/crash detected in %s output; restarting Vivado "
                    "and resyncing instead of losing the rest of the run's budget.",
                    tool_name,
                )
                try:
                    restart_result = await self.call_tool("vivado_restart_vivado", {}, internal=True)
                    logger.info("restart_vivado result: %s", restart_result[:200])
                except Exception as restart_exc:
                    logger.error("restart_vivado itself failed: %s", restart_exc)
                raise VivadoToolCallError(tool_name, result_text)

            self._update_design_state(tool_name, arguments, result_text)

            # Track WNS from timing reports and get_wns calls -- but only when
            # the design is in a state where timing is real. On an unplaced or
            # partially implemented design get_timing_paths still returns
            # estimated slacks (or 0.0 when no paths exist at all), and letting
            # those ratchet best_wns / flow into iteration recording poisons
            # the run summary and the bisection trigger.
            if (
                tool_name in ("vivado_report_timing_summary", "vivado_get_wns")
                and self.design_state != "routed"
            ):
                logger.warning(
                    "Ignoring WNS from %s: design state is '%s', timing is only valid on a fully placed+routed design.",
                    tool_name, self.design_state,
                )
            elif tool_name == "vivado_report_timing_summary":
                # If target clock is set, get clock-specific WNS instead of overall
                if self.target_clock:
                    try:
                        clock_wns = await super().get_wns_for_target_clock(self._call_vivado_tool)
                        if clock_wns is not None:
                            current_wns = clock_wns
                            wns_measured = current_wns
                            current_fmax = self.calculate_fmax(current_wns, self.clock_period)
                            fmax_str = f", fmax: {current_fmax:.2f} MHz" if current_fmax is not None else ""
                            if current_wns > self.best_wns:
                                logger.info(f"New best WNS (clock: {self.target_clock}): {current_wns:.3f} ns{fmax_str} (improved from {self.best_wns:.3f} ns)")
                                self.best_wns = current_wns
                            else:
                                logger.info(f"Current WNS (clock: {self.target_clock}): {current_wns:.3f} ns{fmax_str} (best is still {self.best_wns:.3f} ns)")
                    except VivadoToolCallError:
                        # Real Vivado-side failure, not a parsing issue -- let it
                        # bubble all the way up to optimize()'s recovery handler
                        # rather than silently falling back to overall WNS parsing.
                        raise
                    except Exception as e:
                        logger.warning(f"Failed to get clock-specific WNS, falling back to overall: {e}")
                        self.target_clock = None  # Fall through to overall WNS parsing
                
                if not self.target_clock or wns_measured is None:
                    try:
                        current_wns = parse_timing_summary_wns_static(result_text, self.clock_period)
                    except WNSParseError as e:
                        logger.error("Ignoring invalid WNS from timing summary: %s", e)
                        if not internal:
                            await self._record_wns_parse_error("report_timing_summary", str(e), result_text)
                        current_wns = None
                    if current_wns is not None:
                        wns_measured = current_wns
                        current_fmax = self.calculate_fmax(current_wns, self.clock_period)
                        fmax_str = f", fmax: {current_fmax:.2f} MHz" if current_fmax is not None else ""
                        if current_wns > self.best_wns:
                            logger.info(f"New best WNS: {current_wns:.3f} ns{fmax_str} (improved from {self.best_wns:.3f} ns)")
                            self.best_wns = current_wns
                        else:
                            logger.info(f"Current WNS: {current_wns:.3f} ns{fmax_str} (best is still {self.best_wns:.3f} ns)")
            
            # Also track WNS from get_wns tool (returns just the numeric WNS value)
            elif tool_name == "vivado_get_wns":
                try:
                    current_wns = parse_wns_value_static(result_text, self.clock_period, "vivado_get_wns")
                    wns_measured = current_wns
                    current_fmax = self.calculate_fmax(current_wns, self.clock_period)
                    fmax_str = f", fmax: {current_fmax:.2f} MHz" if current_fmax is not None else ""
                    if current_wns > self.best_wns:
                        logger.info(f"New best WNS (from get_wns): {current_wns:.3f} ns{fmax_str} (improved from {self.best_wns:.3f} ns)")
                        self.best_wns = current_wns
                    else:
                        logger.info(f"Current WNS (from get_wns): {current_wns:.3f} ns{fmax_str} (best is still {self.best_wns:.3f} ns)")
                except (WNSParseError, ValueError, AttributeError) as e:
                    logger.error(f"Could not parse WNS from get_wns output: {result_text[:100]} ({e})")
                    if not internal:
                        await self._record_wns_parse_error("vivado_get_wns", str(e), result_text)
            
            elapsed_time = time.time() - start_time
            self.last_vivado_runtime_s = elapsed_time if tool_name.startswith("vivado_") else self.last_vivado_runtime_s
            if tool_name.startswith(("vivado_", "rapidwright_")):
                self.iteration_tool_elapsed_s += elapsed_time

            # Cost model: learn what place/route/phys_opt actually cost on
            # THIS design (including the re-place flows issued via run_tcl).
            duration_kind = self._duration_kind_for_call(tool_name, arguments)
            if duration_kind:
                self._note_action_duration(duration_kind, elapsed_time)

            if not internal:
                await self._after_tool_success(tool_name, arguments, result_text, wns_measured, elapsed_time)
            
            # Record tool call details
            self.tool_call_details.append({
                "tool_name": tool_name,
                "iteration": self.iteration,
                "elapsed_time": elapsed_time,
                "wns": wns_measured,
                "error": False
            })
            
            return result_text

        except VivadoToolCallError as e:
            # Real Vivado/RapidWright-side failure (e.g. pipe desync + restart).
            # This must propagate all the way up to optimize()'s main loop so it
            # can reopen the last-known-good checkpoint, instead of being
            # swallowed here and turned into a plain string that downstream
            # WNS/timing parsers would misread as real data.
            error_occurred = True
            elapsed_time = time.time() - start_time
            # A place/route killed by its timeout still teaches the cost
            # model: the wall-time is a LOWER bound on the true duration,
            # which is exactly what the budget gate needs to refuse the next
            # dispatch that cannot fit.
            duration_kind = self._duration_kind_for_call(tool_name, arguments)
            if duration_kind:
                self._note_action_duration(duration_kind, elapsed_time)
            self.tool_call_details.append({
                "tool_name": tool_name,
                "iteration": self.iteration,
                "elapsed_time": elapsed_time,
                "wns": None,
                "error": True,
                "error_message": str(e)
            })
            logger.error(f"Tool call failed (propagating for recovery): {e}")
            raise

        except Exception as e:
            error_occurred = True
            elapsed_time = time.time() - start_time
            
            # Record failed tool call
            self.tool_call_details.append({
                "tool_name": tool_name,
                "iteration": self.iteration,
                "elapsed_time": elapsed_time,
                "wns": None,
                "error": True,
                "error_message": str(e)
            })
            
            logger.error(f"Tool call failed: {e}")
            return json.dumps({"error": str(e)})
    
    async def _call_vivado_tool(self, tool_name: str, arguments: dict) -> str:
        """Helper to call Vivado tools (for use with base class methods)."""
        return await self.call_tool(f"vivado_{tool_name}", arguments)

    def _initialize_run_helpers(self, input_dcp: Path) -> None:
        """Create checkpoint, shield, and ECO-routing helpers after baseline analysis."""
        if self.initial_wns is None or self.clock_period is None:
            logger.warning("Checkpoint manager not initialized: missing initial WNS or clock period")
            return

        clock_name = self.target_clock or "unknown_clock"
        checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_manager = load_or_create(
            output_dir=str(checkpoint_dir),
            input_dcp=str(input_dcp.resolve()),
            clock_name=clock_name,
        )
        if self.hard_limit_seconds_override is not None:
            logger.warning(
                "Overriding contest budget: hard_limit_seconds=%d (exploratory run, not "
                "contest-compliant).",
                self.hard_limit_seconds_override,
            )
            self.checkpoint_manager.hard_limit_seconds = int(self.hard_limit_seconds_override)
        self.checkpoint_manager.start_baseline(self.initial_wns, self.clock_period)

        self.vivado_adapter = VivadoMCPAdapter(self)
        self.eco_router = ECORouter(self.vivado_adapter)
        self.shield_generator = ShieldGenerator(self.vivado_adapter, self.checkpoint_manager)
        self.active_checkpoint = str(input_dcp.resolve())

        logger.info("CheckpointManager, ShieldGenerator, and ECORouter initialized")
        print(f"Run safety enabled: checkpoint history at {checkpoint_dir / 'history.json'}")

    def _remember_recipe(self, tool_name: str, arguments: dict) -> None:
        """Track the most recent optimization operation for checkpoint records.

        NOTE: this method sets self.last_recipe, which is a human-readable
        history/display label ONLY. It intentionally renames some actions
        (e.g. "rapidwright_optimize_cell_placement" -> "rapidwright_cell_placement")
        for nicer history entries. Because of that renaming, nothing that
        gates action selection (cooldowns, exhaustion, blacklisting) may key
        off self.last_recipe -- use self.last_action_key for that instead.
        """
        if tool_name == "rapidwright_optimize_cell_placement":
            targets = [str(name) for name in arguments.get("cell_names", [])]
            self.last_recipe = "rapidwright_cell_placement"
            self.last_targets = targets
            self.last_batch_size = len(targets)
        elif tool_name == "rapidwright_optimize_fanout":
            target = str(arguments.get("net_name", ""))
            self.last_recipe = "rapidwright_fanout"
            self.last_targets = [target] if target else []
            self.last_batch_size = 1 if target else 0
        elif tool_name == "rapidwright_optimize_lut_input_cone":
            targets = [str(name) for name in arguments.get("hierarchical_input_pins", [])]
            self.last_recipe = "rapidwright_lut_input_cone"
            self.last_targets = targets
            self.last_batch_size = len(targets)
        elif tool_name == "vivado_create_and_apply_pblock":
            targets = [
                str(arguments.get("pblock_name", "pblock")),
                str(arguments.get("ranges", "")),
            ]
            self.last_recipe = "vivado_pblock"
            self.last_targets = [target for target in targets if target]
            self.last_batch_size = 1
        elif tool_name == "vivado_phys_opt_design":
            directive = arguments.get("directive")
            enabled = [key for key, value in arguments.items() if value is True]
            self.last_recipe = "vivado_phys_opt"
            self.last_targets = [str(directive)] if directive else enabled
            self.last_batch_size = len(self.last_targets) or 1

    async def _get_current_wns(self) -> Optional[float]:
        if self.design_state != "routed":
            logger.warning(
                "Refusing WNS query: design state is '%s' (WNS is only valid on a fully placed+routed design).",
                self.design_state,
            )
            return None
        args = {"timeout": 60}
        if self.target_clock:
            args["clock"] = self.target_clock
        result = await self.call_tool("vivado_get_wns", args, internal=True)
        try:
            return parse_wns_value_static(result, self.clock_period, "vivado_get_wns")
        except (WNSParseError, ValueError, IndexError) as e:
            logger.error(f"Could not parse current WNS from Vivado: {result[:200]} ({e})")
            return None

    async def _refresh_target_candidates(self, wns: Optional[float]) -> list[dict]:
        """Refresh tiered timing targets; never returns empty if Vivado has paths."""
        if wns is not None and wns < 0:
            self.target_tier = "tier1_violating_paths"
            tcl_filter = " -slack_lesser_than 0"
        else:
            self.target_tier = "tier2_worst_slack_paths"
            tcl_filter = ""

        # NOTE (bug fix): the previous version of this command nested the
        # foreach/puts loop *inside* the "if {[llength $paths] == 0}" branch,
        # so path data was only ever emitted in the empty-fallback case, and
        # the "if" body itself was missing a closing brace (9 open braces vs
        # 8 close braces), which is a Tcl parse error on every call. Fixed by
        # (a) balancing the fallback's own braces, and (b) moving the
        # foreach/puts loop outside the if so it always runs against
        # whichever $paths ended up populated, filtered or fallback.
        tcl_cmd = (
            f"set paths [get_timing_paths -max_paths {TIER2_TOP_PATHS_DEFAULT} -setup{tcl_filter}]; "
            "if {[llength $paths] == 0} {"
            f"set paths [get_timing_paths -max_paths {TIER2_TOP_PATHS_DEFAULT} -setup]"
            "}; "
            "foreach p $paths { "
            "  set slack [get_property SLACK $p]; "
            "  set start [get_property STARTPOINT_PIN $p]; "
            "  set end [get_property ENDPOINT_PIN $p]; "
            "  if {$start eq {}} {set start [get_property STARTPOINT_CELL $p]}; "
            "  if {$end eq {}} {set end [get_property ENDPOINT_CELL $p]}; "
            "  puts \"PATH|$slack|$start|$end\"; "
            "}"
        )
        raw = await self.call_tool("vivado_run_tcl", {"command": tcl_cmd, "timeout": 120}, internal=True)
        candidates: list[dict] = []
        for line in raw.splitlines():
            if not line.startswith("PATH|"):
                continue
            parts = line.split("|", 3)
            if len(parts) != 4:
                continue
            try:
                slack = float(parts[1])
            except ValueError:
                slack = None
            candidates.append({
                "slack": slack,
                "startpoint": parts[2].strip(),
                "endpoint": parts[3].strip(),
            })

        self.current_target_candidates = candidates
        logger.info(
            "Target tier active: %s, candidates=%d",
            self.target_tier,
            len(self.current_target_candidates),
        )
        return candidates

    async def _classify_worst_path_delay(self) -> str:
        raw = await self.call_tool(
            "vivado_run_tcl",
            {
                "command": "report_timing -return_string -max_paths 1 -delay_type max",
                "timeout": 300,
            },
            internal=True,
        )
        logic_match = re.search(r"logic\s+([\d.]+)ns\s*\(([\d.]+)%\)", raw, re.IGNORECASE)
        net_match = re.search(r"(?:route|net)\s+([\d.]+)ns\s*\(([\d.]+)%\)", raw, re.IGNORECASE)
        logic_pct = None
        net_pct = None
        if logic_match:
            logic_pct = float(logic_match.group(2)) / 100.0
        if net_match:
            net_pct = float(net_match.group(2)) / 100.0

        classification = "mixed"
        if logic_pct is not None and logic_pct > DECISION_LOGIC_DELAY_BOUND_THRESHOLD:
            classification = "logic_delay_bound"
        elif net_pct is not None and net_pct > DECISION_NET_DELAY_BOUND_THRESHOLD:
            classification = "net_delay_bound"
        elif logic_pct is None and net_pct is None:
            classification = "unknown"

        self.path_delay_classification = classification
        self.path_delay_breakdown = {
            "logic_pct": logic_pct,
            "net_pct": net_pct,
            "logic_delay_ns": float(logic_match.group(1)) if logic_match else None,
            "net_delay_ns": float(net_match.group(1)) if net_match else None,
        }

        if classification == "logic_delay_bound":
            logger.info("logic-delay-bound path, skipping pblock")
        else:
            logger.info(
                "Worst path delay classification: %s (logic=%s, net=%s)",
                classification,
                logic_pct,
                net_pct,
            )
        return classification

    async def _run_constraint_audit(self) -> None:
        if self.constraint_audit_done:
            return
        self.constraint_audit_done = True
        if self.initial_wns is None or self.initial_wns > CONSTRAINT_AUDIT_WNS_TRIGGER_NS:
            return
        if (
            self.initial_tns is None
            or self.initial_tns > CONSTRAINT_AUDIT_TNS_TRIGGER_NS
            or self.initial_failing_endpoints is None
            or self.initial_failing_endpoints < CONSTRAINT_AUDIT_ENDPOINT_TRIGGER
        ):
            return

        tcl_cmd = (
            f"set paths [get_timing_paths -max_paths {CONSTRAINT_AUDIT_TOP_PATHS} -setup]; "
            "foreach p $paths { "
            "  set start [get_property STARTPOINT_PIN $p]; "
            "  set end [get_property ENDPOINT_PIN $p]; "
            "  puts \"AUDIT|$start|$end\"; "
            "}"
        )
        raw = await self.call_tool("vivado_run_tcl", {"command": tcl_cmd, "timeout": 300}, internal=True)
        prefixes: dict[str, int] = {}
        total = 0
        for line in raw.splitlines():
            if not line.startswith("AUDIT|"):
                continue
            total += 1
            source = line.split("|", 2)[1].strip()
            prefix = self._timing_prefix(source)
            if prefix:
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
        if not prefixes or total == 0:
            return
        prefix, count = max(prefixes.items(), key=lambda item: item[1])
        if count / total >= CONSTRAINT_AUDIT_COMMON_PREFIX_FRACTION:
            logger.warning(
                "Possible missing multicycle/false-path constraint on %s — %d of top %d worst paths share this source. "
                "Consider adding a multicycle_path or false_path constraint before physical optimization.",
                prefix,
                count,
                total,
            )

    def _timing_prefix(self, name: str) -> str:
        cleaned = name.strip("{} ")
        if not cleaned:
            return ""
        cell = cleaned.rsplit("/", 1)[0] if "/" in cleaned else cleaned
        parts = [part for part in cell.split("/") if part]
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return parts[0] if parts else cell

    async def _run_phys_opt_with_policy(self, arguments: dict) -> str:
        if self.design_state != "routed":
            # phys_opt_design on an unplaced/unrouted design either errors out
            # or "optimizes" against estimated delays -- both useless. This
            # guard is what stops the pblock_full_replace interception bug
            # class (phys_opt after `place_design -unplace`) from ever
            # recurring, whatever path leads here.
            return self._failure_json(
                "phys_opt_invalid_design_state",
                f"phys_opt_design requires a fully placed and routed design; current state is '{self.design_state}'.",
                command="phys_opt_design",
            )
        before_guard_wns = await self._get_current_wns()
        if before_guard_wns is not None and before_guard_wns < PHYS_OPT_MIN_USEFUL_WNS_NS:
            # Escape hatches (run 20260713 vexriscv lesson: this guard refused
            # retime 5x in one run while the diagnosis said the path was
            # logic-dominated and every structural action had already failed
            # -- the run then ground out 10 straight stalls with no logic-side
            # lever left). The "structural first" premise only holds while
            # structural actions are actually viable AND the delay is
            # net-dominated; otherwise phys_opt/retime IS the remaining move.
            logic_pct = (self.path_delay_breakdown or {}).get("logic_pct")
            logic_heavy = isinstance(logic_pct, (int, float)) and logic_pct >= 0.45
            stuck = self.consecutive_no_improvement >= STUCK_ITERATION_THRESHOLD
            active_exhausted = set(self._active_exhausted_actions())
            structural_remaining = any(
                action not in active_exhausted for action in RAPIDWRIGHT_STRUCTURAL_ACTIONS
            )
            if logic_heavy or stuck or not structural_remaining:
                logger.info(
                    "phys_opt WNS guard bypassed at %.3f ns (logic_pct=%s, "
                    "consecutive_no_improvement=%d, structural_remaining=%s): "
                    "phys_opt/retime is the appropriate remaining lever.",
                    before_guard_wns, logic_pct,
                    self.consecutive_no_improvement, structural_remaining,
                )
            else:
                message = (
                    f"Skipping phys_opt_design because current WNS {before_guard_wns:.3f} ns is below "
                    f"{PHYS_OPT_MIN_USEFUL_WNS_NS:.3f} ns; use structural placement actions first."
                )
                logger.info(message)
                # Pipeline audit (20260802-20260804 sweep): this guard alone
                # fired 13 times across 172 iterations, and the demotion in
                # _allowed_forbidden_actions (same WNS check) is soft -- the
                # LLM kept re-choosing a phys_opt variant it had already been
                # refused for. Feed the refusal into the same cooldown
                # machinery that already suppresses repeat no-action
                # failures (_active_exhausted_actions), so after
                # ACTION_FAILURE_EXHAUSTION_THRESHOLD refusals on this same
                # target set the action is actually removed from
                # allowed_actions for ACTION_FAILURE_COOLDOWN_ITERS instead
                # of merely demoted.
                refused_action = self.last_action_key if self.last_action_key in PHYS_OPT_INCREMENTAL_ACTIONS else "phys_opt_design"
                self._remember_no_action_failure(refused_action, [])
                return self._failure_json("phys_opt_below_useful_wns", message, command="phys_opt_design")
        if not await self._check_implementation_license():
            return self._failure_json(
                "vivado_license_failure",
                "Vivado Implementation license is unavailable; phys_opt_design disabled.",
                command="phys_opt_design",
            )
        before_wns = before_guard_wns
        directive = arguments.get("directive") or PHYS_OPT_PRIMARY_DIRECTIVE
        # LUT pin-swapping (-critical_pin_opt): confirmed NOT combinable with
        # -directive in this Vivado 2025.1 install -- ERROR: [Vivado_Tcl 4-167]
        # "Cannot specify '-critical_pin_opt' when '-directive' is specified",
        # seen identically across at least 4 separate runs (corescore x2,
        # finn_radioml, amd_mini) any time phys_opt_design_pin_swap ran,
        # regardless of directive value. The prior comment here claiming the
        # two were combinable was wrong; _run_phys_opt_tcl now drops
        # -directive entirely when critical_pin_opt is set.
        critical_pin_opt = bool(arguments.get("critical_pin_opt", False))
        # See _run_phys_opt_tcl: these two were previously read by NOTHING in
        # this interception layer, so replicate_register's critical_cell_opt
        # (and any targeted replication request) was silently discarded on
        # every call in the project's history.
        critical_cell_opt = bool(arguments.get("critical_cell_opt", False))
        force_replication_on_nets = arguments.get("force_replication_on_nets") or None
        specific_opts = dict(
            critical_pin_opt=critical_pin_opt,
            critical_cell_opt=critical_cell_opt,
            force_replication_on_nets=force_replication_on_nets,
        )
        primary = await self._run_phys_opt_tcl(directive=directive, retime=True, **specific_opts)
        if self._action_failure(primary, default_command="phys_opt_design").get("error_type") == "vivado_license_failure":
            return primary
        if self._vivado_output_has_error(primary) and directive != "Default":
            logger.warning("phys_opt directive %s unsupported or failed; falling back to Default", directive)
            primary = await self._run_phys_opt_tcl(directive="Default", retime=True, **specific_opts)

        after_wns = await self._get_current_wns()
        output = [primary]
        if before_wns is not None and after_wns is not None and after_wns <= before_wns:
            secondary = await self._run_phys_opt_tcl(directive=PHYS_OPT_SECONDARY_DIRECTIVE, retime=True, **specific_opts)
            if self._action_failure(secondary, default_command="phys_opt_design").get("error_type") == "vivado_license_failure":
                output.append(secondary)
                return "\n\n".join(output)
            if self._vivado_output_has_error(secondary):
                logger.warning("phys_opt directive %s unsupported or failed; falling back to Default", PHYS_OPT_SECONDARY_DIRECTIVE)
                secondary = await self._run_phys_opt_tcl(directive="Default", retime=True, **specific_opts)
            output.append(secondary)
        return "\n\n".join(output)

    async def _maybe_run_pblock_or_phys_opt(self, arguments: dict) -> str:
        if self.design_state != "routed":
            # Classifying the worst path (and possibly running phys_opt) only
            # makes sense on real routed timing. Reaching here in any other
            # state means an orchestration bug upstream -- fail loudly instead
            # of running phys_opt/pblock math against estimated delays.
            return self._failure_json(
                "pblock_invalid_design_state",
                f"pblock/phys_opt decision requires a fully placed and routed design; current state is '{self.design_state}'.",
                command="vivado_create_and_apply_pblock",
            )
        classification = await self._classify_worst_path_delay()
        if classification == "logic_delay_bound":
            logger.info("logic-delay-bound path, skipping pblock")
            self.last_recipe = "vivado_phys_opt_logic_delay"
            self.last_targets = ["logic_delay_bound"]
            self.last_batch_size = 1
            phys = await self._run_phys_opt_with_policy({})
            if self._action_failure(phys, default_command="phys_opt_design"):
                return phys
            return "logic-delay-bound path, skipping pblock\n\n" + phys

        if classification == "mixed":
            before = await self._get_current_wns()
            phys = await self._run_phys_opt_with_policy({})
            if self._action_failure(phys, default_command="phys_opt_design"):
                return phys
            after = await self._get_current_wns()
            if before is not None and after is not None and after > before:
                self.last_recipe = "vivado_phys_opt_mixed_path"
                self.last_targets = ["mixed_path"]
                self.last_batch_size = 1
                return "mixed path: phys_opt improved timing, pblock deferred\n\n" + phys
            logger.info("mixed path: phys_opt did not improve timing; proceeding with pblock")

        timing_context = self.last_timing_context or {}
        computed_args, error = await self._compute_pblock_ranges(dict(arguments), timing_context)
        if error:
            logger.error("pblock action aborted: %s", error)
            return self._failure_json("pblock_range_computation_failed", error, command="pblock")
        assert computed_args is not None
        return await self.call_tool("vivado_create_and_apply_pblock", computed_args, internal=True)

    async def _run_phys_opt_tcl(
        self,
        directive: str = "Default",
        retime: bool = True,
        critical_pin_opt: bool = False,
        critical_cell_opt: bool = False,
        force_replication_on_nets: Optional[str] = None,
    ) -> str:
        if retime and self.phys_opt_retime_supported is False:
            logger.info("Skipping phys_opt -retime because a previous retime attempt was rejected.")
            retime = False
        retime_flag = " -retime" if retime else ""
        pin_swap_flag = " -critical_pin_opt" if critical_pin_opt else ""
        # -critical_cell_opt / -force_replication_on_nets (pipeline audit,
        # 20260804 sweep): replicate_register has ALWAYS sent
        # {"critical_cell_opt": True} through call_tool, and this
        # interception layer silently dropped it -- only directive and
        # critical_pin_opt were ever read -- so "replicate_register" has
        # never once actually run -critical_cell_opt; it ran a plain
        # directive pass indistinguishable from phys_opt_design. That is
        # exactly why it went 0-for-everything on the three excessive_fanout
        # stalled designs (optical-flow/spam-filter/vexriscv_v2) while being
        # the diagnosis layer's ONLY recommended response to that hypothesis.
        cell_opt_flag = " -critical_cell_opt" if critical_cell_opt else ""
        replication_flag = (
            f" -force_replication_on_nets {force_replication_on_nets}"
            if force_replication_on_nets else ""
        )
        # Any of Vivado's "specific optimization options" is genuinely
        # incompatible with -directive (ERROR: [Vivado_Tcl 4-167]) -- drop
        # -directive entirely in that case rather than sending a command
        # guaranteed to fail.
        has_specific_opts = critical_pin_opt or critical_cell_opt or bool(force_replication_on_nets)
        directive_flag = "" if has_specific_opts else f" -directive {directive}"
        command = f"phys_opt_design{directive_flag}{retime_flag}{pin_swap_flag}{cell_opt_flag}{replication_flag}"
        result = await self.call_tool("vivado_run_tcl", {"command": command, "timeout": self._implementation_timeout_s(kind="phys_opt")}, internal=True)
        if self._action_failure(result, default_command=command).get("error_type") == "vivado_license_failure":
            return result
        if self._vivado_output_has_error(result) and retime:
            self.phys_opt_retime_supported = False
            logger.warning("phys_opt retime failed with directive %s; retrying without -retime", directive)
            command = f"phys_opt_design{directive_flag}{pin_swap_flag}{cell_opt_flag}{replication_flag}"
            result = await self.call_tool("vivado_run_tcl", {"command": command, "timeout": self._implementation_timeout_s(kind="phys_opt")}, internal=True)
        elif retime:
            self.phys_opt_retime_supported = True
        return result

    def _is_post_route_physsynth_crash(self, text: str) -> bool:
        """True if route_design died in its own post-route physical-synthesis
        re-optimization pass, not in routing itself.

        Confirmed signature (fir_systolic_transposed_routed_2025.1, runs
        20260801_195142 and 20260803_153128, both post-pblock route_design
        calls with directive 'Explore'): routing itself completes cleanly --
        the log shows "Verifying routed nets: Verification completed
        successfully" -- then Phase 15 "Physical Synthesis in Router" /
        15.1 "Physical Synthesis Initialization" throws
        "ERROR: [Route 35-9] Router encountered a fatal exception of type
        '13HDPLException' - 'Error in placer init in PSFlow'". This is a
        Vivado-side crash in a re-optimization pass that a plainer directive
        doesn't invoke, not a placement/resource problem with the design or
        the pblock -- treating it as a hard pblock/full-replace failure (the
        prior behavior) permanently took the pblock avenue off the table for
        the rest of the run over something a directive retry can route
        around."""
        return "13HDPLException" in text or "Error in placer init in PSFlow" in text

    def _vivado_output_has_error(self, text: str) -> bool:
        failure = self._action_failure(text)
        if failure:
            return True
        upper = text.upper()
        return (
            "ERROR:" in upper
            or "UNKNOWN OPTION" in upper
            or "INVALID" in upper
            or "PHYS_OPT_DESIGN FAILED" in upper
            or "VALID LICENSE WAS NOT FOUND" in upper
            or "FAILED TO GET THE LICENSE" in upper
        )

    def _failure_json(self, error_type: str, message: str, command: str = "") -> str:
        return json.dumps({
            "success": False,
            "error_type": error_type,
            "command": command,
            "message": message,
        }, indent=2)

    def _action_failure(self, text: str, default_command: str = "") -> dict:
        if not text:
            return {}
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and payload.get("success") is False:
                return payload
        except json.JSONDecodeError:
            pass
        lower = text.lower()
        if "valid license was not found" in lower or "failed to get the license" in lower:
            return {
                "success": False,
                "error_type": "vivado_license_failure",
                "command": default_command,
                "message": text.strip(),
            }
        if "phys_opt_design failed" in lower or "error:" in lower:
            return {
                "success": False,
                "error_type": "vivado_command_failure",
                "command": default_command,
                "message": text.strip(),
            }
        # Pipeline audit 2026-07-28: route_design can return cleanly (no
        # "ERROR:" line at all) while leaving most of the design unrouted --
        # rosetta_optical-flow/vtr_mcml/rosetta_spam-filter-scale runs left
        # 28k-70k+ nets unrouted with a clean-looking return, only ever
        # caught later by _verify_routed_state()'s separate
        # report_route_status round-trip, after also paying for a
        # report_timing_summary call on the broken design in between.
        # route_design's own console output already prints a "Number of
        # Unrouted/Failed Nets" summary once per phase checkpoint (0 during
        # early phases, the real count at completion) -- read the LAST
        # occurrence so an all-zero early-phase snapshot can't mask a bad
        # final one, and catch the failure for free off text already in
        # hand instead of a second Vivado call.
        unrouted = re.findall(r"Number of Unrouted Nets\s*=\s*(\d+)", text)
        failed_nets = re.findall(r"Number of Failed Nets\s*=\s*(\d+)", text)
        if unrouted or failed_nets:
            unrouted_n = int(unrouted[-1]) if unrouted else 0
            failed_n = int(failed_nets[-1]) if failed_nets else 0
            if unrouted_n > 0 or failed_n > 0:
                return {
                    "success": False,
                    "error_type": "route_incomplete",
                    "command": default_command,
                    "message": (
                        f"route_design's own summary reports {unrouted_n} unrouted "
                        f"net(s) and {failed_n} failed net(s) -- routing did not "
                        f"actually complete even though no ERROR: line was printed."
                    ),
                }
        return {}

    async def _check_implementation_license(self) -> bool:
        # Vivado 2025.1 Tcl does not reliably expose `get_license`; probing it
        # can mark a usable analysis session as broken. Let implementation
        # commands report structured failures when they are actually invoked.
        return self.implementation_license_available is not False

    def _update_design_state(self, tool_name: str, arguments: dict, result_text: str) -> None:
        """Track whether the live design is unplaced/placed/routed, and whether
        the in-flight action has mutated it. Called for every tool result
        (internal or not), so composite/intercepted flows are covered by their
        leaf calls."""
        if tool_name.startswith("rapidwright_optimize_"):
            self.last_action_mutated_design = True
            return
        if not tool_name.startswith("vivado_"):
            return
        failed = self._vivado_output_has_error(result_text)
        if tool_name == "vivado_open_checkpoint":
            if not failed:
                # Every checkpoint this flow opens (contest input, iter_NNN,
                # best.dcp) is a fully implemented DCP.
                self.design_state = "routed"
            return
        if tool_name == "vivado_place_design":
            self.last_action_mutated_design = True
            if not failed:
                self.design_state = "placed"
            return
        if tool_name == "vivado_route_design":
            self.last_action_mutated_design = True
            if not failed:
                self.design_state = "routed"
                self._harvest_congestion_from_route_log(result_text)
            return
        if tool_name in ("vivado_phys_opt_design", "vivado_create_and_apply_pblock"):
            self.last_action_mutated_design = True
            return
        if tool_name == "vivado_run_tcl":
            command = str(arguments.get("command") or "")
            if "place_design -unplace" in command:
                self.last_action_mutated_design = True
                if not failed:
                    self.design_state = "unplaced"
            elif "route_design" in command:
                self.last_action_mutated_design = True
                if not failed:
                    self.design_state = "routed"
                    self._harvest_congestion_from_route_log(result_text)
            elif "place_design" in command:
                self.last_action_mutated_design = True
                if not failed:
                    self.design_state = "placed"
            elif "phys_opt_design" in command or "create_pblock" in command or "delete_pblocks" in command:
                self.last_action_mutated_design = True

    async def _count_unplaced_cells(self) -> Optional[int]:
        """Number of unplaced primitive cells in the live design, or None if
        the probe output could not be parsed."""
        raw = await self.call_tool(
            "vivado_run_tcl",
            {
                "command": (
                    "set _unplaced [llength [get_cells -quiet -hierarchical "
                    "-filter {IS_PRIMITIVE && STATUS == UNPLACED}]]; "
                    "puts \"STATE_UNPLACED:$_unplaced\""
                ),
                "timeout": 120,
            },
            internal=True,
        )
        match = re.search(r"STATE_UNPLACED:(\d+)", raw)
        return int(match.group(1)) if match else None

    def _unplaced_tolerance(self) -> int:
        """How many unplaced primitives count as benign design artifacts.

        History(16) forensics: this design's input DCP ships with ~6 unplaced
        primitives that have no routable nets -- Vivado's own placer skips
        them, its router reports 0 failed nets around them, and the previous
        run's winning (contest-valid) checkpoint contained them. RapidWright
        round-trips preserve them. A genuinely broken state (failed placer,
        mid-flight unplace) leaves hundreds-to-thousands unplaced, far past
        this tolerance -- and unrouted nets besides."""
        baseline = self.baseline_unplaced_cells or 0
        return max(UNPLACED_CELLS_BENIGN_MAX, baseline + 10)

    async def _retry_incremental_place_for_unplaced(self, action: str, unplaced: int) -> int:
        """RapidWright ECO edits (fanout_split, rapidwright_optimize_cell_
        placement) legitimately create/orphan cells in an unplaced state as
        part of the edit -- an ECO flow expects a follow-up placement pass,
        which neither dispatch handler ever ran before this fix (pipeline
        audit 2026-07-28: both went 0-for-24, always rejected on unplaced-
        cell overflow with no attempt to actually place the strays first).
        Give Vivado's own incremental placer one shot: place_design with no
        -unplace is incremental over the existing (partial) placement, never
        a full re-place, so it only has to seat the cells RapidWright left
        stranded. Returns the (possibly improved) unplaced-cell count."""
        logger.info(
            "%s left %d unplaced cell(s); attempting one incremental place_design "
            "before rejecting the candidate.", action, unplaced,
        )
        place_retry = await self.call_tool(
            "vivado_place_design",
            {"directive": "Default", "timeout": self._implementation_timeout_s(kind="place")},
            internal=True,
        )
        if self._action_failure(place_retry, default_command="vivado_place_design"):
            logger.warning(
                "%s: incremental re-place attempt failed; keeping original unplaced count.",
                action,
            )
            return unplaced
        recovered = await self._count_unplaced_cells()
        return recovered if recovered is not None else unplaced

    async def _verify_routed_state(self) -> tuple[bool, str]:
        """Ask Vivado (not just our client-side tracker) whether the design is
        fully placed and routed. This is the provenance gate a WNS observation
        must pass before it may become iteration history / best_checkpoint.

        Route status is the PRIMARY criterion: if every routable net is
        routed, unplaced primitives (up to _unplaced_tolerance) cannot sit on
        any timing path and are benign. The first version of this gate
        treated ANY unplaced primitive as fatal, which rejected every result
        this design can produce -- including a completed place_design run --
        and zeroed out run history(16)."""
        unplaced = await self._count_unplaced_cells()
        if unplaced is None:
            return False, "could not verify placement state"
        if unplaced > self._unplaced_tolerance():
            return False, (
                f"{unplaced} primitive cell(s) are unplaced "
                f"(tolerance {self._unplaced_tolerance()})"
            )
        route_raw = await self.call_tool("vivado_report_route_status", {"timeout": 300}, internal=True)
        unrouted_match = re.search(r"#\s*of\s+unrouted\s+nets[.\s]*:\s*(\d+)", route_raw, re.IGNORECASE)
        errors_match = re.search(r"#\s*of\s+nets\s+with\s+routing\s+errors[.\s]*:\s*(\d+)", route_raw, re.IGNORECASE)
        unrouted = int(unrouted_match.group(1)) if unrouted_match else None
        errors = int(errors_match.group(1)) if errors_match else None
        if unrouted is None and errors is None:
            if re.search(r"fully routed", route_raw, re.IGNORECASE):
                return True, "fully routed"
            return False, f"could not parse route status: {route_raw[:200]}"
        if (unrouted or 0) > 0 or (errors or 0) > 0:
            return False, f"{unrouted or 0} unrouted net(s), {errors or 0} net(s) with routing errors"
        if unplaced and unplaced > 0:
            logger.warning(
                "Design has %d unplaced primitive cell(s) but route status is clean "
                "(0 unrouted / 0 errors); treating them as benign artifacts -- cells "
                "without routable nets cannot be on a timing path.",
                unplaced,
            )
            return True, f"fully routed ({unplaced} benign unplaced primitive(s))"
        return True, "fully placed and routed"

    async def _restore_best_state(self, reason: str) -> None:
        """Reopen the best checkpoint in Vivado AND resync RapidWright to it,
        so both live sessions match the state the history says is best."""
        if self.checkpoint_manager is None:
            return
        best_ckpt = self.checkpoint_manager.get_best_checkpoint()
        if not best_ckpt:
            return
        logger.warning("Restoring best checkpoint %s (%s).", best_ckpt, reason)
        await self.call_tool(
            "vivado_open_checkpoint", {"dcp_path": best_ckpt, "timeout": 600}, internal=True
        )
        self.active_checkpoint = best_ckpt
        reload_result = await self.call_tool(
            "rapidwright_read_checkpoint", {"dcp_path": best_ckpt}, internal=True
        )
        if "error" in reload_result.lower() and "success" not in reload_result.lower():
            logger.error(
                "Failed to re-sync RapidWright to restored checkpoint %s: %s",
                best_ckpt, reload_result[:300],
            )

    def _next_place_directive(self) -> str:
        """Directive sweep (item 3a): pick the first PLACE_DIRECTIVE_SWEEP entry
        not yet tried this run; once all have been tried, re-run the best
        performer instead of an arbitrary repeat."""
        for directive in PLACE_DIRECTIVE_SWEEP:
            if directive not in self.place_directive_results:
                return directive

        def wns_of(item: tuple[str, dict]) -> float:
            wns = item[1].get("wns_after")
            return wns if isinstance(wns, (int, float)) else float("-inf")

        return max(self.place_directive_results.items(), key=wns_of)[0]

    def _next_phys_opt_directive(self) -> str:
        """Same sweep as _next_place_directive, for the phys_opt directive
        space: pick the first PHYS_OPT_DIRECTIVE_SWEEP entry not yet tried
        this run, else re-run the best performer instead of repeating an
        arbitrary one."""
        for directive in PHYS_OPT_DIRECTIVE_SWEEP:
            if directive not in self.phys_opt_directive_results:
                return directive

        def wns_of(item: tuple[str, dict]) -> float:
            wns = item[1].get("wns_after")
            return wns if isinstance(wns, (int, float)) else float("-inf")

        return max(self.phys_opt_directive_results.items(), key=wns_of)[0]

    def _note_recipe_outcome(self, status: str, wns_after: Optional[float]) -> None:
        """Per-recipe search memory (item 3): record how this attempt's
        parameters performed so the NEXT attempt of the same recipe can move in
        parameter space instead of repeating."""
        action = self.last_action_key
        if action == "place_design_explore" and self.last_place_directive:
            self.place_directive_results[self.last_place_directive] = {
                "status": status,
                "wns_after": wns_after,
                "iteration": self.iteration,
            }
        if (
            action in {"phys_opt_design", "phys_opt_design_retime", "phys_opt_design_pin_swap"}
            and self.last_phys_opt_directive
        ):
            self.phys_opt_directive_results[self.last_phys_opt_directive] = {
                "status": status,
                "wns_after": wns_after,
                "iteration": self.iteration,
                "action": action,
            }
        if action in {"pblock", "pblock_full_replace"}:
            entry = {
                "action": action,
                "iteration": self.iteration,
                "status": status,
                "wns_after": wns_after,
                "targets": [str(t) for t in self.last_targets[:2]],
            }
            if self.last_pblock_sizing:
                entry.update(self.last_pblock_sizing)
            self.pblock_attempt_history.append(entry)
            del self.pblock_attempt_history[:-10]

    def _harvest_congestion_from_route_log(self, route_output: str) -> None:
        """Every successful route_design log ends with the router's own
        congestion report ('Effective congestion level: N' per direction) --
        the most current, most authoritative congestion measurement there is,
        and it costs nothing. Cache it so _fetch_congestion_summary rarely
        needs the (slow, format-unstable) report_design_analysis call."""
        level, detail = self._parse_congestion_report(route_output)
        if level is not None:
            self.last_congestion_info = {
                "iteration": self.iteration,
                "congestion_level": level,
                "detail": f"{detail} (from route_design log)",
            }

    def _load_crossrun_priors(self, input_dcp: Path) -> None:
        """Load this design's action/directive records from previous runs."""
        self.crossrun_design_key = input_dcp.stem
        try:
            store = json.loads(self.crossrun_store_path.read_text(encoding="utf-8"))
            self.crossrun_priors = dict(store.get(self.crossrun_design_key) or {})
        except (OSError, json.JSONDecodeError):
            self.crossrun_priors = {}
        if self.crossrun_priors:
            logger.info(
                "Loaded cross-run priors for %s: %d action record(s), %d directive record(s), %d duration prior(s).",
                self.crossrun_design_key,
                len(self.crossrun_priors.get("actions") or {}),
                len(self.crossrun_priors.get("directives") or {}),
                len(self.crossrun_priors.get("durations") or {}),
            )
        # Classify the design's scale as early as possible: a prior run's
        # measured place duration marks a design "large" before this run has
        # spent a single second learning it the hard way.
        self._refresh_design_scale()
        if self.design_scale != "unknown":
            logger.info("Design scale classified '%s' from cross-run priors.", self.design_scale)

    def _this_run_action_tallies(self) -> dict[str, dict[str, int]]:
        """This run's own good/bad tally per action, bucketed the same way
        _save_crossrun_priors does -- used to keep the cross-run guidance
        text current mid-run instead of only reflecting priors loaded at
        startup (crossrun_priors isn't updated again until the run ends)."""
        tallies: dict[str, dict[str, int]] = {}
        if self.checkpoint_manager is None:
            return tallies
        for record in self.checkpoint_manager.iterations:
            action = str(record.get("llm_chosen_action") or record.get("recipe") or "")
            if not action:
                continue
            bucket = "good" if str(record.get("status")) in ("improved", "marginal") else "bad"
            tally = tallies.setdefault(action, {"good": 0, "bad": 0})
            tally[bucket] = tally.get(bucket, 0) + 1
        return tallies

    def _recent_stall_action_families(self, n: int) -> set[str]:
        """Action names (llm_chosen_action, falling back to recipe) from the
        last n recorded iterations that did NOT improve/marginally-improve --
        used by the structural_override stuck-detector to tell whether the
        stalls it's reacting to were themselves placement/pblock attempts,
        in which case forcing MORE placement is the wrong recovery."""
        if self.checkpoint_manager is None:
            return set()
        recent = self.checkpoint_manager.iterations[-n:]
        return {
            str(record.get("llm_chosen_action") or record.get("recipe"))
            for record in recent
            if str(record.get("status")) not in ("improved", "marginal")
        }

    def _widen_override_with_untried_phys_opt(
        self, structural_allowed: list[str], allowed: list[str]
    ) -> list[str]:
        """Add any phys_opt-family action that is allowed but was NOT among
        the recent stalls -- i.e. genuinely untried, not just walled off by
        the override -- so the forced menu doesn't exclude the one lever
        that hasn't actually failed yet (see _recent_stall_action_families
        and the 20260803_141612 finn_radioml case in _build_timing_context:
        recent stalls were pblock/route_explore/phys_opt_design_pin_swap --
        pin_swap correctly stays excluded as "already tried", but plain
        phys_opt_design and phys_opt_design_retime, never attempted, get
        added back onto a menu that would otherwise have been placement-only).

        Bug fix (pipeline audit, rosetta_optical-flow/rosetta_spam-filter,
        20260804 sweep): the "recent" window here was a fixed 3 iterations
        (STUCK_ITERATION_THRESHOLD), so a phys_opt variant tried earlier than
        that aged out of "recent" and got treated as untried all over again.
        optical-flow's override activated at iter 5 (consecutive_no_
        improvement=3); iter 6's "recent" window (the 3 iterations before it)
        no longer included iters 1-2's replicate_register attempts, so this
        function re-offered it as "genuinely untried" and the LLM picked it
        again -- never once forcing place_design_explore/pblock across 6
        straight phys_opt-only stalled iterations. Looking back across the
        WHOLE current stall streak instead of a fixed 3 fixes this directly:
        as the streak grows, every phys_opt variant that's actually been
        tried during it stays excluded, and once all of them have (there are
        only 4), phys_opt_untried empties out on its own and the override
        reverts to pure structural -- no separate cutoff needed."""
        lookback = max(self.consecutive_no_improvement, STUCK_ITERATION_THRESHOLD)
        recent_stall_actions = self._recent_stall_action_families(lookback)
        phys_opt_untried = [
            action for action in PHYS_OPT_INCREMENTAL_ACTIONS
            if action in allowed
            and action not in recent_stall_actions
            and action not in structural_allowed
        ]
        return structural_allowed + phys_opt_untried

    def _phys_opt_exhausted_this_streak(self, allowed: list[str], high_spread: bool) -> bool:
        """True if every phys_opt-family action currently on offer has
        already appeared somewhere in the current stall streak -- i.e.
        _widen_override_with_untried_phys_opt would have nothing left to add
        back onto a structural-only menu.

        Used to keep the structural override held through what would
        otherwise be its decay window (pipeline audit, rosetta_optical-flow,
        20260804 sweep): the decay timer releases the menu back to phys_opt
        on a fixed schedule regardless of whether phys_opt has anything left
        to offer. Confirmed on optical-flow: by iter 6 every phys_opt variant
        had been tried this streak, but the timer decayed at iter 7 anyway
        and handed the menu back to an already-exhausted phys_opt_design
        instead of the place_design_explore/pblock attempt this design had
        never gotten. If there's no structural action to fall back to
        either, this returns False -- an empty menu is a worse failure mode
        than releasing back to phys_opt, so don't force a dead end."""
        structural_source = RAPIDWRIGHT_PLACEMENT_ACTIONS if high_spread else RAPIDWRIGHT_STRUCTURAL_ACTIONS
        structural_allowed = [action for action in structural_source if action in allowed]
        if not structural_allowed:
            return False
        widened = self._widen_override_with_untried_phys_opt(structural_allowed, allowed)
        return widened == structural_allowed

    def _save_crossrun_priors(self) -> None:
        """Merge this run's outcomes into the persistent per-design store."""
        if self._crossrun_saved or self.checkpoint_manager is None or not self.crossrun_design_key:
            return
        self._crossrun_saved = True
        try:
            store = json.loads(self.crossrun_store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            store = {}
        entry = store.setdefault(self.crossrun_design_key, {})
        actions = entry.setdefault("actions", {})
        directives = entry.setdefault("directives", {})
        for record in self.checkpoint_manager.iterations:
            action = str(record.get("llm_chosen_action") or record.get("recipe") or "")
            status = str(record.get("status"))
            bucket = "good" if status in ("improved", "marginal") else "bad"
            if action:
                action_record = actions.setdefault(action, {"good": 0, "bad": 0})
                action_record[bucket] = int(action_record.get(bucket, 0)) + 1
            targets = record.get("targets") or []
            if targets and str(targets[0]).startswith("directive:"):
                directive = str(targets[0]).split(":", 1)[1]
                directive_record = directives.setdefault(directive, {"good": 0, "bad": 0})
                directive_record[bucket] = int(directive_record.get(bucket, 0)) + 1
        # Cost model: persist this run's longest measured duration per kind so
        # the NEXT run prices place/route correctly from iteration 1 instead
        # of re-learning by burning half its budget. Fresh measurements
        # replace stale priors on purpose (self-corrects after an anomalous
        # hang inflated one).
        durations = entry.setdefault("durations", {})
        for kind, observed in self.action_durations.items():
            if observed:
                durations[kind] = round(max(observed), 1)
        try:
            self.crossrun_store_path.write_text(
                json.dumps(store, indent=2) + "\n", encoding="utf-8"
            )
            logger.info("Saved cross-run priors to %s", self.crossrun_store_path)
        except OSError as exc:
            logger.warning("Could not save cross-run priors: %s", exc)

    def _best_crossrun_directive(self) -> Optional[str]:
        """The place directive with the best cross-run record on this design,
        or None if nothing has ever won."""
        directives = (self.crossrun_priors or {}).get("directives") or {}
        best_name, best_score = None, 0
        for name, record in directives.items():
            score = int(record.get("good", 0)) - int(record.get("bad", 0))
            if int(record.get("good", 0)) > 0 and score > best_score:
                best_name, best_score = name, score
        return best_name

    async def _fetch_congestion_summary(self) -> dict:
        """Item 4: worst router congestion level for the current routed design,
        cached per iteration (report_design_analysis is not cheap). Unknown
        (None) means 'could not measure', never 'no congestion'."""
        if self.last_congestion_info.get("iteration") == self.iteration:
            return self.last_congestion_info
        info: dict = {"iteration": self.iteration, "congestion_level": None, "detail": ""}
        if self.design_state == "routed":
            raw = await self.call_tool(
                "vivado_run_tcl",
                {"command": "report_design_analysis -congestion -return_string", "timeout": 600},
                internal=True,
            )
            level, detail = self._parse_congestion_report(raw)
            info["congestion_level"] = level
            info["detail"] = detail
        if info["congestion_level"] is None and self.last_congestion_info.get("congestion_level") is not None:
            # Keep the most recent REAL measurement (typically harvested from
            # the latest route_design log) instead of replacing it with
            # unknown -- the design hasn't been re-routed since it was taken.
            info = {**self.last_congestion_info, "iteration": self.iteration}
        self.last_congestion_info = info
        return info

    @staticmethod
    def _parse_congestion_report(raw: str) -> tuple[Optional[int], str]:
        """Parse `report_design_analysis -congestion` STRICTLY.

        Run 20260712 lesson: the first version of this parser was 'tolerant'
        (loose direction/level regexes plus an NxN window heuristic) and read
        level 5 out of a report whose router summary said 'Effective
        congestion level: 0' in every direction. That false 5 demoted the
        whole pblock family -- including the recipe family that had produced
        the previous run's best result. A false positive here is WORSE than
        no signal (it vetoes good actions with authority it doesn't have), so
        only two unambiguous formats are accepted; anything else is
        'unknown', which demotes nothing.
        """
        if not raw:
            return None, "empty congestion report"
        levels: list[int] = []
        # The router's own authoritative summary line, one per direction:
        #   "Effective congestion level: 5"
        for match in re.finditer(r"(?i)effective congestion level:\s*(\d+)", raw):
            levels.append(int(match.group(1)))
        # Strict report_design_analysis table rows: "| NORTH | 5 |"
        for match in re.finditer(
            r"(?im)^\s*\|\s*(?:north|south|east|west)\s*\|\s*(\d+)\s*\|", raw
        ):
            levels.append(int(match.group(1)))
        if not levels:
            return None, "no congestion level found in report"
        worst = max(levels)
        return worst, f"worst effective congestion level {worst} (~{2 ** worst}x{2 ** worst} tile window)"

    async def _fetch_clock_regions(self, cell_names: list[str]) -> dict[str, str]:
        """Item 4: CLOCK_REGION per placed cell (first site), one Tcl round-trip
        for the batch. Missing/unplaced cells are simply absent from the map."""
        regions: dict[str, str] = {}
        cells = [str(c) for c in cell_names if c][:20]
        if not cells or self.design_state == "unplaced":
            return regions
        snippets = [
            (
                f"set _s [lindex [get_sites -quiet -of_objects [get_cells -quiet {{{cell}}}]] 0]; "
                f"if {{$_s ne {{}}}} {{puts \"CLOCKREGION|{cell}|[get_property CLOCK_REGION $_s]\"}}"
            )
            for cell in cells
        ]
        raw = await self.call_tool(
            "vivado_run_tcl", {"command": " ; ".join(snippets), "timeout": 120}, internal=True
        )
        for line in raw.splitlines():
            if line.startswith("CLOCKREGION|"):
                parts = line.split("|", 2)
                if len(parts) == 3 and parts[2].strip():
                    regions[parts[1]] = parts[2].strip()
        return regions

    def _time_remaining_s(self) -> Optional[float]:
        if self.checkpoint_manager is None:
            return None
        elapsed = time.time() - self.checkpoint_manager.started_at_epoch_s
        return self.checkpoint_manager.hard_limit_seconds - elapsed

    @staticmethod
    def _duration_kind_for_call(tool_name: str, arguments: dict) -> Optional[str]:
        """Which duration bucket ("place"/"route"/"phys_opt") this tool call's
        wall-time belongs to, or None if it teaches the cost model nothing.
        Covers place/route/phys_opt issued via run_tcl (the re-place flows),
        but not `place_design -unplace`, which is seconds-cheap bookkeeping,
        not a real place."""
        if tool_name == "vivado_place_design":
            return "place"
        if tool_name == "vivado_route_design":
            return "route"
        if tool_name == "vivado_phys_opt_design":
            return "phys_opt"
        if tool_name == "vivado_run_tcl":
            command = str(arguments.get("command") or "")
            if "place_design -unplace" in command:
                return None
            if "route_design" in command:
                return "route"
            if "place_design" in command:
                return "place"
            if "phys_opt_design" in command:
                return "phys_opt"
        return None

    def _note_action_duration(self, kind: str, seconds: float) -> None:
        self.action_durations.setdefault(kind, []).append(round(float(seconds), 1))
        # Refine the scale classification the moment new evidence lands --
        # this is what upgrades "unknown" to "large" right after the warm
        # start's place completes, before the LLM makes its first decision.
        self._refresh_design_scale()

    def _estimated_duration(self, kind: str) -> Optional[float]:
        """Best duration estimate for one action kind: the longest in-run
        observation (a directive change or timeout kill only ever means the
        true cost can be HIGHER), else the cross-run prior for this design,
        else None."""
        observed = self.action_durations.get(kind) or []
        if observed:
            return float(max(observed))
        prior = ((self.crossrun_priors or {}).get("durations") or {}).get(kind)
        try:
            return float(prior) if prior is not None else None
        except (TypeError, ValueError):
            return None

    def _compute_resource_utilization(self, design_info: dict) -> dict:
        """Design resource counts (from rapidwright_get_design_info's
        resource_summary) as a fraction of known xcvu3p device capacity.

        This is a forward-looking companion to the cross-run kill switch in
        _maybe_warm_start_replace: that one only protects a design AFTER it
        has already hung here once, which does nothing for a design seen for
        the first time. Utilization is computable on iteration 0, before any
        action has been attempted -- ispd16_example2's 76% LUT utilization
        was visible from the moment the checkpoint was read, well before its
        first (and every subsequent) place_design call timed out."""
        summary = (design_info or {}).get("resource_summary") or {}
        capacities = {
            "LUT": XCVU3P_LUT_CAPACITY,
            "FF": XCVU3P_FF_CAPACITY,
            "DSP": XCVU3P_DSP_CAPACITY,
            "BRAM": XCVU3P_BRAM_CAPACITY,
        }
        utilization = {}
        for resource, capacity in capacities.items():
            count = summary.get(resource)
            if isinstance(count, (int, float)) and capacity:
                utilization[resource] = round(count / capacity, 4)
        return utilization

    def _refresh_design_scale(self) -> None:
        """Classify the design's scale from whatever evidence exists so far.
        "large" flips on the full re-place cap and pessimistic pricing of
        unmeasured actions; "unknown" keeps the pessimistic pricing only."""
        place_s = self._estimated_duration("place")
        try:
            cell_count = int(self.last_design_info.get("cell_count") or 0)
        except (TypeError, ValueError):
            cell_count = 0
        high_utilization = any(
            fraction >= HIGH_UTILIZATION_FRACTION
            for fraction in (self.resource_utilization or {}).values()
        )
        if (
            (place_s is not None and place_s > LARGE_DESIGN_PLACE_DURATION_S)
            or cell_count > LARGE_DESIGN_PRIMITIVE_COUNT
            or high_utilization
        ):
            self.design_scale = "large"
        elif place_s is not None or cell_count > 0:
            self.design_scale = "small"
        else:
            self.design_scale = "unknown"

    def _estimated_action_cost_s(self, action: str) -> Optional[float]:
        """Estimated wall-clock cost of dispatching `action` now, or None when
        there is nothing to gate on (small design, no measurement -- the
        pre-cost-model behavior). Unmeasured kinds on a large/unknown design
        are priced pessimistically: 2x the longest known duration, at least
        UNKNOWN_EXPENSIVE_ACTION_MIN_S -- an optimistic guess here is how the
        motivating incident burned 30 minutes for zero results."""

        def duration_or_pessimistic(kind: str) -> Optional[float]:
            known = self._estimated_duration(kind)
            if known is not None:
                return known
            if self.design_scale == "small":
                return None
            longest = max(
                (d for d in (self._estimated_duration(k) for k in ("place", "route", "phys_opt")) if d is not None),
                default=None,
            )
            if longest is None:
                return float(UNKNOWN_EXPENSIVE_ACTION_MIN_S)
            return max(float(UNKNOWN_EXPENSIVE_ACTION_MIN_S), 2.0 * longest)

        if action in ("place_design_explore", "pblock_full_replace"):
            place = duration_or_pessimistic("place")
            route = duration_or_pessimistic("route")
            if place is None or route is None:
                return None
            return place + route
        if action == "route_explore":
            return duration_or_pessimistic("route")
        if action in PHYS_OPT_INCREMENTAL_ACTIONS:
            route = duration_or_pessimistic("route")
            if route is None:
                return float(CHEAP_ACTION_COST_S)
            return max(route / 3.0, float(CHEAP_ACTION_COST_S))
        return float(CHEAP_ACTION_COST_S)

    def _full_replace_blocked_reason(self, action: str) -> Optional[str]:
        """Per-run cap on full re-places for large designs (warm start plus at
        most one more, none past 50% of the budget). Returns the human-readable
        block reason, or None when the dispatch is allowed. Small/unknown
        designs keep the current uncapped behavior -- their re-places cost
        minutes, not half the budget."""
        if action not in ("place_design_explore", "pblock_full_replace"):
            return None
        if self.design_scale != "large" or self.checkpoint_manager is None:
            return None
        # 2026-08-01: a design at high device utilization has almost no free
        # space for the placer to legalize into during a full re-place --
        # this is not a timeout-tuning problem, it's a budget-vs-convergence-
        # time mismatch that no cap short of zero fixes. ispd16_example2
        # (76% LUT utilization) hung identically on its first full re-place
        # attempt in three separate runs, each eating the full 20-minute
        # unmeasured-timeout ceiling -- a third of the entire ~58-minute
        # contest budget for zero result. Allow none at all here (not the
        # normal FULL_REPLACE_LARGE_DESIGN_CAP of 2); refinement-only
        # actions (phys_opt_design, phys_opt_design_retime, route_explore,
        # locally-scoped pblock) work directly on the input's own -- already
        # legal -- placement instead of discarding it.
        over_threshold = {
            resource: f"{fraction:.0%}"
            for resource, fraction in (self.resource_utilization or {}).items()
            if fraction >= HIGH_UTILIZATION_FRACTION
        }
        if over_threshold:
            return (
                f"device utilization {over_threshold} leaves too little free space for a "
                f"full re-place to reliably legalize within any contest-realistic timeout; "
                f"refine the input's existing placement instead (phys_opt_design, "
                f"phys_opt_design_retime, route_explore, local pblock)"
            )
        elapsed = time.time() - self.checkpoint_manager.started_at_epoch_s
        budget = float(self.checkpoint_manager.hard_limit_seconds)
        if self.full_replace_attempts >= 1 and elapsed > FULL_REPLACE_BUDGET_FRACTION_CUTOFF * budget:
            return (
                f"large design, {elapsed / 60.0:.0f} of {budget / 60.0:.0f} budget minutes elapsed "
                f"(past the {FULL_REPLACE_BUDGET_FRACTION_CUTOFF:.0%} cutoff) with "
                f"{self.full_replace_attempts} full re-place(s) already spent; no further full "
                f"re-places this run -- refine the best result instead"
            )
        if self.full_replace_attempts >= FULL_REPLACE_LARGE_DESIGN_CAP:
            return (
                f"large design: full re-place cap reached ({self.full_replace_attempts} of "
                f"{FULL_REPLACE_LARGE_DESIGN_CAP} allowed per run, warm start included); "
                f"refine the best result instead"
            )
        return None

    def _maybe_downgrade_route_directive(self, route_directive: str, action_name: str) -> str:
        """Item 5: re-check the budget AFTER a place completes and downgrade
        the requested route directive to Default when the remaining time is
        below ROUTE_DOWNGRADE_FACTOR x the estimated route duration. The
        downgrade is annotated into last_rapidwright_edit_summary so it shows
        up in iteration history and the LLM's next context."""
        if route_directive == "Default":
            return route_directive
        remaining = self._time_remaining_s()
        route_est = self._estimated_duration("route")
        if remaining is None or route_est is None or remaining >= ROUTE_DOWNGRADE_FACTOR * route_est:
            return route_directive
        logger.warning(
            "%s: downgrading route directive %s -> Default (%.0f s remain, route "
            "estimated at %.0f s); a completed Default route beats a killed %s route.",
            action_name, route_directive, remaining, route_est, route_directive,
        )
        self.last_rapidwright_edit_summary = {
            **(self.last_rapidwright_edit_summary or {"action": action_name, "changed_design": True}),
            "route_directive_downgraded": {
                "from": route_directive,
                "to": "Default",
                "remaining_s": round(remaining),
                "estimated_route_s": round(route_est),
            },
        }
        return "Default"

    def _implementation_timeout_s(self, default_s: int = 1200, kind: str = "generic") -> int:
        """Timeout for a single place/route/phys_opt command.

        These used to be a flat 3600 s -- longer than the entire 3500 s run
        budget, so one hung command (e.g. a pexpect prompt desync in the
        Vivado server) silently ate the whole contest run. A healthy full
        place or route on these designs takes 2-4 minutes; cap at 20 minutes
        or the remaining budget, whichever is smaller, so a hang costs
        minutes, fires the VivadoToolCallError recovery (reopen best
        checkpoint), and the run continues.

        Cost model refinement: when this design's own duration for `kind` is
        known (measured this run or from a prior run), size the timeout to
        2.5x that instead -- a large design whose place legitimately takes
        15 minutes must not be killed at 20 while budget remains, and the
        remaining-budget clamp still applies.

        The floor for a KNOWN duration is 600 s, not 1200: run
        20260714_182751 iter 11 had place measured at ~250 s on this design,
        yet the old max(1200, 2.5x) floor let a hung ExtraNetDelay_high
        place burn the full 20 minutes (46% of that run's gamma). With a
        measured baseline, 2.5x it (min 10 min) is already generous; only
        the unmeasured case keeps the conservative 20-minute cap."""
        remaining = self._time_remaining_s()
        known = self._estimated_duration(kind) if kind != "generic" else None
        if known is not None:
            budgeted = max(600.0, 2.5 * known)
            if remaining is not None:
                budgeted = min(budgeted, remaining)
            return int(max(300, budgeted))
        if remaining is None:
            return default_s
        return int(max(300, min(default_s, remaining)))

    async def _set_clock_period(self, period_ns: float) -> bool:
        if not self.target_clock:
            return False
        command = (
            f"set clk_obj [get_clocks -quiet {{{self.target_clock}}}]; "
            f"if {{$clk_obj eq {{}}}} {{puts {{ERROR: clock not found}}}} "
            f"else {{set_property PERIOD {period_ns:.6f} $clk_obj; puts {{CLOCK_PERIOD_UPDATED}}}}"
        )
        result = await self.call_tool("vivado_run_tcl", {"command": command, "timeout": 60}, internal=True)
        if self._vivado_output_has_error(result):
            logger.warning("Failed to set clock period %.6f ns: %s", period_ns, result[:300])
            return False
        self.current_period_ns = period_ns
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.clock_period_ns = period_ns
        return True

    async def _try_close_at_period(self, period_ns: float) -> Optional[float]:
        if not await self._set_clock_period(period_ns):
            return None
        await self._run_phys_opt_with_policy({})
        route = await self.call_tool("vivado_route_design", {"directive": "Default", "timeout": self._implementation_timeout_s(kind="route")}, internal=True)
        if self._vivado_output_has_error(route):
            logger.warning("Route under tightened clock returned errors: %s", route[:300])
        return await self._get_current_wns()

    async def _run_clock_bisection_after_closure(self, passing_wns: float) -> None:
        if self.bisection_active or passing_wns < 0 or self.current_period_ns is None:
            return
        remaining = self._time_remaining_s()
        if remaining is not None and remaining < TIME_BUDGET_RESERVE_S:
            logger.info("Skipping clock bisection: only %.1fs remain", remaining)
            return
        if not self.target_clock:
            logger.info("Skipping clock bisection: no target clock identified")
            return

        self.bisection_active = True
        original_period = self.current_period_ns
        pass_period = original_period
        fail_period = None
        step = max(original_period * CLOCK_TIGHTEN_FRACTION, MIN_CLOCK_TIGHTEN_STEP_NS)
        trial_period = max(MIN_CLOCK_TIGHTEN_STEP_NS, original_period - step)

        try:
            logger.info("Starting clock tightening from %.6f ns to %.6f ns", original_period, trial_period)
            while True:
                remaining = self._time_remaining_s()
                if remaining is not None and remaining < TIME_BUDGET_RESERVE_S:
                    logger.info("Stopping clock tightening: only %.1fs remain", remaining)
                    break
                trial_wns = await self._try_close_at_period(trial_period)
                if trial_wns is None:
                    return
                if trial_wns < 0:
                    fail_period = trial_period
                    self.first_failing_period_ns = fail_period
                    break
                pass_period = trial_period
                self.last_passing_period_ns = pass_period
                await self._record_bisection_pass(pass_period, trial_wns)
                step = max(pass_period * CLOCK_TIGHTEN_FRACTION, MIN_CLOCK_TIGHTEN_STEP_NS)
                next_period = max(MIN_CLOCK_TIGHTEN_STEP_NS, pass_period - step)
                if abs(pass_period - next_period) < MIN_CLOCK_TIGHTEN_STEP_NS:
                    break
                trial_period = next_period

            if fail_period is None:
                await self._set_clock_period(pass_period)
                self.clock_period = pass_period
                return

            self.first_failing_period_ns = fail_period
            await self._set_clock_period(pass_period)

            for _ in range(MAX_BISECT_ITERS):
                remaining = self._time_remaining_s()
                if remaining is not None and remaining < TIME_BUDGET_RESERVE_S:
                    logger.info("Stopping clock bisection: only %.1fs remain", remaining)
                    break
                midpoint = (pass_period + fail_period) / 2.0
                if abs(pass_period - fail_period) < MIN_CLOCK_TIGHTEN_STEP_NS:
                    break
                wns = await self._try_close_at_period(midpoint)
                if wns is None:
                    break
                if wns >= 0:
                    pass_period = midpoint
                    self.last_passing_period_ns = pass_period
                    await self._record_bisection_pass(pass_period, wns)
                else:
                    fail_period = midpoint
                    self.first_failing_period_ns = fail_period

            await self._set_clock_period(pass_period)
            self.clock_period = pass_period
            logger.info("Clock bisection complete; best passing period %.6f ns", pass_period)
        finally:
            self.bisection_active = False

    async def _record_bisection_pass(self, period_ns: float, wns: float) -> None:
        if self.checkpoint_manager is None:
            return
        # Same timing-provenance gate as _record_iteration_timing: a passing
        # WNS under a tightened clock only counts if the design is verifiably
        # fully placed and routed (a re-route that errored out can leave the
        # previous "routed" state flag stale).
        routed_ok, routed_detail = await self._verify_routed_state()
        if not routed_ok:
            logger.warning(
                "Rejecting bisection pass at %.6f ns: %s", period_ns, routed_detail
            )
            return
        achieved_fmax = self.calculate_fmax(wns, period_ns)
        if achieved_fmax is not None and (self.best_fmax_mhz is None or achieved_fmax > self.best_fmax_mhz):
            self.best_fmax_mhz = achieved_fmax
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"bisect_{self.iteration:03d}_{period_ns:.4f}.dcp"
        write_result = await self.call_tool(
            "vivado_write_checkpoint",
            {"dcp_path": str(checkpoint_path), "force": True, "timeout": 600},
            internal=True,
        )
        if "error" in write_result.lower():
            logger.warning("Could not write bisection checkpoint: %s", write_result[:300])
            return
        self.checkpoint_manager.clock_period_ns = period_ns
        iteration = self.checkpoint_manager.record(
            recipe="clock_period_bisection",
            targets=[self.target_clock or "clock"],
            wns_after=wns,
            vivado_runtime_s=self.iteration_tool_elapsed_s,
            checkpoint_path=str(checkpoint_path),
            batch_size=1,
        )
        self.iteration_tool_elapsed_s = 0.0
        self._annotate_latest_history({
            "target_tier": self.target_tier,
            "path_delay_classification": self.path_delay_classification,
            "path_delay_breakdown": self.path_delay_breakdown,
            "clock_period_ns": period_ns,
            "achieved_fmax_mhz": achieved_fmax,
            "bisection": True,
            **self.last_decision_trace,
        })
        logger.info("Recorded bisection pass: %s", iteration)

    def _annotate_latest_history(self, fields: dict) -> None:
        if self.checkpoint_manager is None or not self.checkpoint_manager.iterations:
            return
        self.checkpoint_manager.iterations[-1].update(fields)
        persist = getattr(self.checkpoint_manager, "_persist_history", None)
        if callable(persist):
            persist()

    def _record_failed_action(self, failure: dict) -> None:
        if failure.get("error_type") == "vivado_license_failure":
            self.implementation_license_available = False
            logger.warning("Vivado Implementation license failure observed; future implementation actions will be filtered.")
        # Fix #1: use last_action_key (never renamed) instead of last_recipe
        # (a display label that _remember_recipe() may have rewritten) so the
        # failure gets attributed to the same key that gating logic reads.
        failed_action = str(
            failure.get("command")
            or self.last_decision_trace.get("llm_chosen_action")
            or self.last_action_key
            or self.last_recipe
            or ""
        )
        failed_targets = list(self.last_targets)
        # Fix #14: keep the actual failure text around so the next timing
        # context can show the LLM WHY the action failed, not just that it did.
        self.last_action_failure = {
            "iteration": self.iteration,
            "action": failed_action,
            "error_type": failure.get("error_type"),
            "message": str(failure.get("message") or "")[:1500],
            "targets": failed_targets[:5],
        }
        if self._is_no_action_failure(failure):
            failure_key = (failed_action, tuple(failed_targets), self.iteration)
            if self.last_no_action_failure_key != failure_key:
                self._blacklist_failure_targets(failed_action, failed_targets)
                self._remember_no_action_failure(failed_action, failed_targets)
                self.last_no_action_failure_key = failure_key
        elif failure.get("error_type") != "vivado_license_failure":
            # BUG FIX (failure memory): hard execution failures used to leave
            # action_failure_memory completely untouched -- only "no action
            # target" failures were remembered. So a recipe that blew up
            # identically every time it ran kept full selection priority: run
            # 20260711 chose pblock_full_replace at iterations 4, 6 AND 9 with
            # the exact same ranges and the exact same vivado_command_failure.
            # Route hard failures through the same consecutive-failure /
            # structural-window bookkeeping as no-action failures so repeated
            # broken recipes cool down. Deadlock is impossible by construction:
            # every cooldown expires (cooldown_until_iter), and
            # _build_timing_context's deadlock breaker re-opens forbidden
            # actions if all allowed ones are simultaneously exhausted.
            # (Target blacklisting is intentionally NOT applied here: hard
            # failure targets are things like pblock range strings, and the
            # failure says the recipe broke, not that the cells are bad.)
            failure_key = (failed_action, tuple(failed_targets), self.iteration)
            if self.last_no_action_failure_key != failure_key:
                self._remember_no_action_failure(failed_action, failed_targets)
                self.last_no_action_failure_key = failure_key
        if self.checkpoint_manager is None:
            return
        iteration = {
            "iter": self.checkpoint_manager.current_iter + 1,
            "recipe": self.last_recipe,
            "batch_size": self.last_batch_size,
            "targets": list(self.last_targets),
            "wns_before": self.checkpoint_manager.best_wns,
            "wns_after": None,
            "fmax_before": self.checkpoint_manager.best_fmax_mhz,
            "fmax_after": None,
            "delta_fmax": 0.0,
            "vivado_runtime_s": self.iteration_tool_elapsed_s,
            "status": "failed",
            "reason": failure.get("message", ""),
            "error_type": failure.get("error_type", "vivado_command_failure"),
            "checkpoint": None,
            "rapidwright_edit_summary": self.last_rapidwright_edit_summary,
            "action_failure_memory": self._serializable_action_failure_memory(),
            "action_failure_counts": self._serializable_action_failure_counts(),
            **self.last_decision_trace,
        }
        self.checkpoint_manager.current_iter += 1
        self.checkpoint_manager.stall_count += 1
        self.no_improvement_count += 1
        self.consecutive_no_improvement += 1
        self._note_recipe_outcome("failed", None)
        self.checkpoint_manager.iterations.append(iteration)
        self.iteration_tool_elapsed_s = 0.0
        persist = getattr(self.checkpoint_manager, "_persist_history", None)
        if callable(persist):
             persist()
        logger.warning("Recorded failed optimization action: %s", failure.get("error_type"))
        if self.checkpoint_manager is not None and self.checkpoint_manager.should_escalate():
            message = (
                f"Checkpoint manager observed repeated stalls after a failed action. "
                f"Current summary: {self.checkpoint_manager.summary()} "
                f"The last action ({failure.get('command')}) failed with "
                f"{failure.get('error_type')}. You must choose a different recipe "
                f"or target set this turn."
            )
            self.messages.append({"role": "user", "content": message})

    async def _record_wns_parse_error(self, source: str, reason: str, raw_output: str) -> None:
        """Record a WNS parse failure without treating it as timing regression."""
        if self.checkpoint_manager is None:
            return
        if self.iteration <= 0 or self.iteration in self.recorded_iterations:
            return
        iteration = {
            "iter": self.checkpoint_manager.current_iter + 1,
            "recipe": self.last_recipe,
            "batch_size": self.last_batch_size,
            "targets": list(self.last_targets),
            "wns_before": self.checkpoint_manager.best_wns,
            "wns_after": None,
            "fmax_before": self.checkpoint_manager.best_fmax_mhz,
            "fmax_after": None,
            "delta_fmax": None,
            "vivado_runtime_s": self.last_vivado_runtime_s,
            "status": "wns_parse_error",
            "reason": reason,
            "source": source,
            "raw_wns_output": str(raw_output)[:500],
            "checkpoint": None,
            **self.last_decision_trace,
        }
        self.checkpoint_manager.current_iter += 1
        self.checkpoint_manager.iterations.append(iteration)
        self.recorded_iterations.add(self.iteration)
        persist = getattr(self.checkpoint_manager, "_persist_history", None)
        if callable(persist):
            persist()
        logger.error("Recorded WNS parse error for iteration %s from %s: %s", self.iteration, source, reason)

    def _history_digest(self, max_iterations: int = 40) -> str:
        """One line per past iteration, derived from checkpoint history.

        Used as the replacement text when older conversation turns are pruned
        (Fix #13) -- it preserves the decision-relevant signal (what was tried,
        on what, and what happened) at a tiny fraction of the tokens.
        """
        if self.checkpoint_manager is None or not self.checkpoint_manager.iterations:
            return "(no completed iterations yet)"
        lines: list[str] = []
        for record in self.checkpoint_manager.iterations[-max_iterations:]:
            targets = ", ".join(str(t) for t in (record.get("targets") or [])[:2])
            wns_after = record.get("wns_after")
            wns_text = f"{wns_after:.3f}" if isinstance(wns_after, (int, float)) else "n/a"
            line = (
                f"iter {record.get('iter')}: {record.get('recipe')} -> "
                f"{record.get('status')} (wns_after={wns_text} ns) targets=[{targets}]"
            )
            reason = str(record.get("reason") or "").strip()
            if reason:
                line += f" reason={reason[:160]}"
            lines.append(line)
        lines.append(f"Current state: {self.checkpoint_manager.summary()}")
        return "\n".join(lines)

    def _prune_conversation(self) -> None:
        """Cap LLM conversation growth (Fix #13).

        The conversation used to grow without bound: every iteration appended
        a full timing-context JSON plus the model reply, producing 2.4M-7M
        prompt tokens per run with zero cache hits. Keep the pinned prefix
        (system prompt + initial analysis) and the most recent turns verbatim,
        and collapse everything in between into a compact per-iteration digest
        built from checkpoint history. The stable prefix also makes provider-
        side prompt caching possible again.
        """
        pinned = 2  # system prompt + initial task/analysis message
        if len(self.messages) <= pinned + CONTEXT_KEEP_RECENT_MESSAGES + 1:
            return
        recent = self.messages[-CONTEXT_KEEP_RECENT_MESSAGES:]
        digest_message = {
            "role": "user",
            "content": (
                "PRIOR ITERATION DIGEST (older turns pruned to keep context "
                "small; one line per iteration):\n" + self._history_digest()
            ),
        }
        pruned_count = len(self.messages) - pinned - len(recent)
        self.messages = [*self.messages[:pinned], digest_message, *recent]
        logger.info(
            "Pruned %d old conversation messages into a %d-line history digest.",
            pruned_count, len(digest_message["content"].splitlines()),
        )

    async def _append_iteration_context(self) -> None:
        self._prune_conversation()
        if self.design_state != "routed" and self.checkpoint_manager is not None:
            # Safety net: no iteration may ever start against an unplaced or
            # half-implemented design, whatever the previous action left
            # behind. Restore the validated best and continue from there.
            await self._restore_best_state(
                f"design state was '{self.design_state}' at iteration start"
            )
        current_wns = await self._get_current_wns()
        if current_wns is not None:
            await self._refresh_target_candidates(current_wns)
        await self._classify_worst_path_delay()
        await self._check_implementation_license()
        timing_context = await self._build_timing_context(current_wns)
        self.last_timing_context = timing_context
        prompt = (
            "Given the timing state above, select one action from `allowed_actions` "
            "(a ranked list -- earlier entries carry a stronger prior).\n"
            "You may not choose any action in `forbidden_actions` (hard blocks).\n"
            "If you choose an action listed in `action_guidance`, you must rebut its "
            "guidance reason with evidence from this run (failure history, directive "
            "sweep results, congestion) inside why_this_fits_delay_class.\n\n"
            "Respond in this JSON format only, no other text:\n"
            "{\n"
            "  \"delay_class_acknowledged\": <copy delay_class from input>,\n"
            "  \"endpoint_type_acknowledged\": <copy endpoint_type from input>,\n"
            "  \"chosen_action\": <must be from allowed_actions>,\n"
            "  \"action_parameters\": <object; valid keys per action are documented in action_parameters_schema below -- use them, especially after a failure reported in last_action_failure>,\n"
            "  \"why_this_fits_delay_class\": <one sentence, must reference net_pct or logic_pct; if the action is in action_guidance, also rebut its reason here>,\n"
            "  \"why_not_top_forbidden_action\": <one sentence about the HIGHEST-RANKED action in allowed_actions that you did NOT choose (or the most tempting entry in action_guidance): why not it -- do NOT discuss forbidden_actions here, those are hard-blocked and uninteresting>,\n"
            "  \"confidence\": <1-5>\n"
            "}\n\n"
            f"{json.dumps(timing_context, indent=2)}"
        )
        self.messages.append({"role": "user", "content": prompt})

    async def _build_timing_context(self, current_wns: Optional[float]) -> dict:
        if self.checkpoint_manager is not None:
            self.consecutive_no_improvement = self.checkpoint_manager.stall_count
            self.no_improvement_count = self.checkpoint_manager.stall_count

        worst = self.current_target_candidates[0] if self.current_target_candidates else {}
        endpoint = str(worst.get("endpoint") or "")
        logic_pct = self.path_delay_breakdown.get("logic_pct")
        max_spread = self.last_spread_info.get("max_distance")

        # Analysis Layer (Stage 2): normalize -> cluster -> diagnose (which
        # internally gathers evidence and scores hypotheses). delay_class/
        # endpoint_type/net_pct/avg_spread are computed identically to how
        # this function computed them inline before Stage 1, so the
        # structural_override / stuck-detector logic below is unaffected.
        # allowed/forbidden CAN now differ from the pre-Analysis-Layer
        # baseline: actions_for() calls _allowed_forbidden_actions with
        # those same values first, then applies at most one hypothesis
        # (veto and/or reorder) on top if it cleared CONFIDENCE_FLOOR --
        # see diagnosis.action_adjustment / diagnosis.reasoning_trace below.
        # Item 4: congestion is a pblock/placement decision-changer, so fetch
        # it BEFORE diagnosis runs (hypotheses and actions_for() read
        # last_congestion_info). Skipped for logic-delay-bound designs, where
        # no pblock decision is on the table and the report costs real time.
        if self.path_delay_classification != "logic_delay_bound":
            await self._fetch_congestion_summary()

        failures = self.analysis_engine.normalize(self.current_target_candidates)
        clusters = self.analysis_engine.cluster(failures)
        diagnosis = await self.analysis_engine.diagnose(clusters, current_wns)
        self.last_diagnosis = diagnosis
        endpoint_type = diagnosis.endpoint_type
        net_pct = diagnosis.net_pct
        avg_spread = diagnosis.avg_spread
        delay_class = diagnosis.delay_class
        allowed, forbidden = self.analysis_engine.actions_for(diagnosis, current_wns)
        high_spread = (
            avg_spread is not None
            and net_pct is not None
            and avg_spread > DECISION_SPREAD_TILE_THRESHOLD
            and net_pct > DECISION_SPREAD_NET_THRESHOLD
        )
        structural_override_age = self.consecutive_no_improvement - STUCK_ITERATION_THRESHOLD
        if structural_override_age >= 0:
            cycle_len = STRUCTURAL_OVERRIDE_MAX_ITERS + STUCK_ITERATION_THRESHOLD
            structural_override = (structural_override_age % cycle_len) < STRUCTURAL_OVERRIDE_MAX_ITERS
            # Bug fix (pipeline audit, rosetta_optical-flow, 20260804 sweep):
            # the decay above is a pure timer -- it releases the menu back to
            # phys_opt every (STRUCTURAL_OVERRIDE_MAX_ITERS +
            # STUCK_ITERATION_THRESHOLD) iterations regardless of whether
            # phys_opt has anything left to offer. Confirmed on optical-flow:
            # by iter 6 every phys_opt variant had been tried this streak
            # (_widen_override_with_untried_phys_opt correctly narrowed to
            # zero untried options), but the timer decayed at iter 7 anyway
            # and handed the menu straight back to an already-exhausted
            # phys_opt_design instead of the place_design_explore/pblock
            # attempt this design had never gotten. Once phys_opt is provably
            # exhausted for this streak, hold the override regardless of the
            # timer -- there's nothing left in that family left to protect
            # access to, so decaying just re-tries what already failed.
            if not structural_override and self._phys_opt_exhausted_this_streak(allowed, high_spread):
                structural_override = True
                logger.warning(
                    "Stuck detector: decay timer says off, but every phys_opt "
                    "variant is exhausted this streak (%d iterations); holding "
                    "structural override on instead of releasing it.",
                    self.consecutive_no_improvement,
                )
        else:
            structural_override = False
        self.structural_override_active = structural_override
        if structural_override:
            # Even when forcing a structural action after repeated stalls,
            # still respect the spread-based ordering below - otherwise a
            # design stuck specifically because cell_placement keeps
            # regressing would have that same action handed back to it
            # first, just from a shorter list.
            structural_source = RAPIDWRIGHT_PLACEMENT_ACTIONS if high_spread else RAPIDWRIGHT_STRUCTURAL_ACTIONS
            structural_allowed = [action for action in structural_source if action in allowed]
            # Run 20260803_141612 (finn_radioml): the stalls triggering this
            # override were THEMSELVES placement/pblock attempts (pblock,
            # route_explore, pblock -- all no_improvement). Forcing MORE
            # placement/pblock in response walled phys_opt_design out of
            # allowed_actions for the rest of the override window; it only
            # got tried afterward by the LLM-independent _endgame_polish
            # fallback, where it won +7.4 MHz on the first attempt. When the
            # recent stalls are all structural/placement actions, widen the
            # forced menu to include the untried phys_opt family instead of
            # doubling down on the family that was already failing.
            structural_allowed = self._widen_override_with_untried_phys_opt(structural_allowed, allowed)
            if structural_allowed:
                allowed = structural_allowed
                for action in RAPIDWRIGHT_STRUCTURAL_ACTIONS:
                    if action in forbidden:
                        forbidden.remove(action)
                for action in PHYS_OPT_INCREMENTAL_ACTIONS:
                    if action in forbidden:
                        forbidden.remove(action)
                logger.warning(
                    "Stuck detector: %d iterations without improvement, forcing structural action from: %s",
                    self.consecutive_no_improvement,
                    allowed,
                )
        elif self.consecutive_no_improvement >= STUCK_ITERATION_THRESHOLD:
            logger.warning(
                "Stuck detector: structural override decayed after %d no-improvement iterations; widening action space.",
                self.consecutive_no_improvement,
            )
        allowed = self._filter_exhausted_actions(allowed)
        exhausted_actions = self._active_exhausted_actions()
        # Fix #12 (deadlock breaker): run history showed iterations where
        # EVERY action in allowed_actions was simultaneously in
        # exhausted_actions -- the system prompt forbids exhausted actions
        # even when listed as allowed, so the LLM had literally no legal
        # move and cycled between two cooling-down actions for 30+
        # iterations. When that happens, re-open the forbidden list: any
        # implemented action that is not hard-blocked (license) becomes
        # selectable again. A logic-level or phys_opt attempt the rule
        # engine deprioritized is strictly better than a guaranteed no-op.
        if allowed and all(action in exhausted_actions for action in allowed):
            hard_blocked: set[str] = {"logic_restructure"}  # not implemented by the dispatcher
            if self.implementation_license_available is False:
                hard_blocked |= set(VIVADO_INCREMENTAL_IMPLEMENTATION_ACTIONS) | {"fanout_split"}
            reopened = [
                action for action in forbidden
                if action not in hard_blocked and action not in exhausted_actions
            ]
            if reopened:
                logger.warning(
                    "All allowed actions %s are exhausted; re-opening previously "
                    "forbidden actions %s to break the deadlock.",
                    allowed, reopened,
                )
                allowed = [*allowed, *reopened]
                forbidden = [action for action in forbidden if action not in reopened]
        recommendation = None
        if high_spread:
            recommendation = (
                f"High critical-path spread detected (avg {avg_spread:.1f} tiles, "
                f"threshold {DECISION_SPREAD_TILE_THRESHOLD}) on a net-delay-bound path "
                f"({net_pct:.0%} net delay). Local cell placement nudges are a weak fix "
                f"for this regime and have been observed to regress WNS; prefer the "
                f"pblock track (rapidwright_analyze_fabric_for_pblock -> "
                f"rapidwright_convert_fabric_region_to_pblock -> pblock) unless it is "
                f"already exhausted for this target set."
            )
        # Cost model (items 1-3): the LLM sees the same numbers the dispatch
        # gate enforces, so "costs ~N min" guidance is explainable evidence
        # rather than an invisible veto.
        remaining_budget = self._time_remaining_s()
        known_durations = {
            kind: round(duration, 1)
            for kind in ("place", "route", "phys_opt")
            for duration in (self._estimated_duration(kind),)
            if duration is not None
        }
        return {
            "iteration": self.iteration,
            "wns_ns": current_wns,
            "tns_ns": self.initial_tns,
            "failing_endpoints": self.initial_failing_endpoints,
            "time_remaining_s": int(remaining_budget) if remaining_budget is not None else None,
            "design_scale": self.design_scale,
            "measured_action_durations_s": known_durations,
            "full_replace_attempts": self.full_replace_attempts,
            # Phase 0 signature, minus the multi-KB QoR text (that shipped
            # once in the initial analysis message).
            "design_signature": {
                key: value for key, value in self.design_signature.items()
                if key != "qor_suggestions"
            },
            "clock_period_ns": self.current_period_ns or self.clock_period,
            "worst_path": {
                "slack_ns": worst.get("slack"),
                "logic_delay_ns": self.path_delay_breakdown.get("logic_delay_ns"),
                "net_delay_ns": self.path_delay_breakdown.get("net_delay_ns"),
                "logic_pct": logic_pct,
                "net_pct": net_pct,
                "logic_levels": self.path_delay_breakdown.get("logic_levels"),
                "start_cell": worst.get("startpoint"),
                "end_cell": endpoint.rsplit("/", 1)[0] if "/" in endpoint else endpoint,
                "end_pin": endpoint.rsplit("/", 1)[-1] if "/" in endpoint else endpoint,
                "avg_tile_spread": avg_spread,
                "max_tile_spread": max_spread,
            },
            "delay_class": delay_class,
            "endpoint_type": endpoint_type,
            "stuck_iterations": self.consecutive_no_improvement,
            "consecutive_no_improvement": self.consecutive_no_improvement,
            "structural_override_active": structural_override,
            "structural_override_age": max(0, structural_override_age),
            "structural_override_max_iters": STRUCTURAL_OVERRIDE_MAX_ITERS,
            # Compact view for the LLM (score item D): the full-fingerprint
            # serializer stays on history.json records only.
            "action_failure_memory": self._llm_action_failure_memory(),
            "action_failure_counts": self._serializable_action_failure_counts(),
            "exhausted_actions": exhausted_actions,
            "allowed_actions": allowed,
            "forbidden_actions": forbidden,
            "recommendation": recommendation,
            # Fix #14: give the LLM the information it needs to pick
            # parameters instead of sending {} every turn: the actual error
            # text of the most recent failure, and the valid parameter keys
            # for each action currently on the menu.
            "last_action_failure": self.last_action_failure,
            "action_parameters_schema": {
                action: ACTION_PARAMETERS_SCHEMA.get(action, {})
                for action in allowed
            },
            # Item 1 (priors, not gates): reasons an allowed action is
            # currently discouraged. Choosing one requires rebutting the
            # reason -- see TIMING_DECISION_SYSTEM_PROMPT.
            "action_guidance": dict(self.last_action_guidance),
            # Item 3 (search within a recipe): what this run has already
            # tried, so the next attempt moves in parameter space.
            "place_directives_tried": dict(self.place_directive_results),
            "place_directives_untried": [
                d for d in PLACE_DIRECTIVE_SWEEP if d not in self.place_directive_results
            ],
            "phys_opt_directives_tried": dict(self.phys_opt_directive_results),
            "phys_opt_directives_untried": [
                d for d in PHYS_OPT_DIRECTIVE_SWEEP if d not in self.phys_opt_directive_results
            ],
            # Cross-run memory: measured win/loss records for this design
            # from previous runs -- weigh these like this run's own history.
            "crossrun_action_records": dict((self.crossrun_priors or {}).get("actions") or {}),
            "crossrun_directive_records": dict((self.crossrun_priors or {}).get("directives") or {}),
            "pblock_attempt_history": list(self.pblock_attempt_history[-5:]),
            # Item 4 (decision-changing physical evidence).
            "congestion_level": self.last_congestion_info.get("congestion_level"),
            "congestion_detail": self.last_congestion_info.get("detail"),
            "cluster_clock_regions": sorted(diagnosis.cluster_clock_regions),
            # Analysis Layer (Stage 2): cluster_count/primary_diagnosis/
            # reasoning_trace are always descriptive/logging. action_adjustment
            # is the one field that reflects an actual change to allowed/
            # forbidden above -- None means the primary hypothesis either
            # didn't clear CONFIDENCE_FLOOR or had nothing to veto/reorder.
            "cluster_count": len(diagnosis.clusters),
            "primary_diagnosis": diagnosis.primary_hypothesis.name,
            "diagnosis_reasoning_trace": diagnosis.reasoning_trace,
            "diagnosis_action_adjustment": diagnosis.action_adjustment,
        }

    def _classify_endpoint_type(self, endpoint: str) -> str:
        upper = endpoint.upper()
        if "RAMB" in upper or "BRAM" in upper:
            if any(pin in upper for pin in ("/WE", "/EN", "/RST", "/REGCE", "/ADDR", "/CLK")):
                return "BRAM_CONTROL"
            return "BRAM_CONTROL"
        if "DSP" in upper:
            if any(pin in upper for pin in ("/CEA", "/CEB", "/CEC", "/RST", "/CLK", "/OPMODE", "/ALUMODE")):
                return "DSP_CONTROL"
            return "DSP_CONTROL"
        if "/D" in upper or "FD" in upper or "REG" in upper:
            return "REGISTER"
        return "LUT"

    def _sitting_on_fresh_win(self) -> bool:
        """True while the run holds an unbeaten best worth protecting.

        Originally this only covered stall_count == 0 (the iteration
        immediately after a win), and the moment one refinement stalled the
        demotion vanished: run 20260714_005251 then re-rolled at iters 5, 9
        and 16 and lost -100/-120/-22 MHz (keep-best absorbed it, but each
        burned ~210 s). Score item C: the gamble is against the same unbeaten
        best whether the last iteration stalled or not, so keep full re-rolls
        demoted (still rebuttable guidance, and the stuck-override can still
        force structural actions -- that path won +7.9 MHz at iter 7) for as
        long as best Fmax sits above baseline.

        Used by both the exploit-after-win demotion here and the diagnosis
        layer's promotion block (analysis_layer.actions_for), which must not
        re-promote full re-rolls past this demotion.

        Materiality floor (pipeline audit, rosetta_optical-flow, 20260804
        sweep): the old check was `best > baseline + 1e-9`, so iter 1's
        +0.42 MHz marginal (0.13% over baseline) counted as a "win worth
        protecting" and kept place_design_explore/pblock_full_replace
        demoted for the ENTIRE rest of the run -- the design then stalled
        flat for 7 straight iterations without ever getting the one lever
        (a real re-place) that its balanced logic/net profile hadn't tried.
        The exploit-after-win evidence this demotion encodes came from runs
        holding wins of +98 MHz, not +0.4; protecting a rounding-error gain
        with the same authority is a category error. Require the banked win
        to be material (FRESH_WIN_MATERIALITY_FRACTION over baseline) before
        the demotion engages; below that, the run is treated as not yet
        having found anything, leaving re-rolls at full rank."""
        cm = self.checkpoint_manager
        if cm is None or not cm.iterations:
            return False
        if cm.best_fmax_mhz is None or cm.baseline_fmax_mhz is None:
            return cm.stall_count == 0
        material_floor = cm.baseline_fmax_mhz * (1.0 + FRESH_WIN_MATERIALITY_FRACTION)
        if cm.best_fmax_mhz <= material_floor:
            return False
        if cm.stall_count == 0:
            return True
        return cm.best_fmax_mhz > cm.baseline_fmax_mhz + 1e-9

    def _demote_actions(
        self,
        allowed: list[str],
        actions: list[str] | set[str],
        reason: str,
    ) -> list[str]:
        """Move `actions` to the end of the ranked `allowed` list and record
        `reason` in last_action_guidance. This is the priors replacement for
        the old hard-forbid: the LLM can still choose a demoted action, but
        must rebut the recorded reason (see TIMING_DECISION_SYSTEM_PROMPT)."""
        demoted = [action for action in allowed if action in set(actions)]
        if not demoted:
            return allowed
        for action in demoted:
            existing = self.last_action_guidance.get(action)
            self.last_action_guidance[action] = f"{existing}; {reason}" if existing else reason
        return [action for action in allowed if action not in demoted] + demoted

    def _allowed_forbidden_actions(
        self,
        delay_class: str,
        endpoint_type: str,
        net_pct: Optional[float],
        avg_spread: Optional[float],
        current_wns: Optional[float],
    ) -> tuple[list[str], list[str]]:
        """Rank the full action menu instead of hard-forbidding by delay class.

        Reworked from hard gates to priors: run 20260711's only improvement
        (place_design_explore, -0.92 -> -0.493 ns) came from an action this
        function had put in `forbidden` for net_delay_bound designs -- the LLM
        only got to use it because the deadlock breaker happened to re-open the
        forbidden list. `forbidden` now contains only structural impossibilities
        (unimplemented actions, license blocks); everything else stays in
        `allowed`, ranked by prior, with demotion reasons recorded in
        self.last_action_guidance for the LLM to weigh and rebut.
        """
        self.last_action_guidance = {}
        every_action = [
            *RAPIDWRIGHT_STRUCTURAL_ACTIONS,  # includes place_design_explore
            "route_explore",
            "run_recipe",
            "replicate_register",
            "phys_opt_design",
            "phys_opt_design_retime",
            "phys_opt_design_pin_swap",
            "qor_suggestions",
            "fanout_split",
            "lut_opt",
        ]
        # logic_restructure is named in older prompts but has no dispatcher
        # branch in execute_validated_action -- a true hard block.
        forbidden = ["logic_restructure"]

        if delay_class == "net_delay_bound":
            allowed = list(every_action)
            allowed = self._demote_actions(
                allowed,
                ["lut_opt", "fanout_split"],
                f"delay class is net_delay_bound (net_pct={net_pct}); logic-side changes rarely fix routing-bound paths",
            )
        elif delay_class == "logic_delay_bound":
            preferred = ["lut_opt", "phys_opt_design_retime", "fanout_split"]
            allowed = preferred + [action for action in every_action if action not in preferred]
            allowed = self._demote_actions(
                allowed,
                [
                    "pblock",
                    "pblock_full_replace",
                    "place_design_explore",
                    "rapidwright_optimize_cell_placement",
                ],
                "delay class is logic_delay_bound; placement changes do not reduce logic depth",
            )
        else:
            allowed = [
                *RAPIDWRIGHT_STRUCTURAL_ACTIONS,  # includes place_design_explore
                "phys_opt_design_retime",
                "phys_opt_design",
                "phys_opt_design_pin_swap",
                "route_explore",
                "run_recipe",
                "replicate_register",
                "fanout_split",
                "lut_opt",
            ]

        if endpoint_type in {"BRAM_CONTROL", "DSP_CONTROL"}:
            # NOTE: the pblock range-analysis precursor actions are deliberately
            # NOT re-added here -- choosing "pblock" already runs them
            # internally via _compute_pblock_ranges, and exposing them
            # independently only let the LLM burn iterations on no-op steps.
            for action in [
                "rapidwright_optimize_cell_placement",
                "pblock",
                "replicate_register",
                "place_design_explore",
            ]:
                if action not in allowed:
                    allowed.append(action)
            allowed = self._demote_actions(
                allowed,
                ["fanout_split"],
                f"endpoint is a {endpoint_type} pin; routing to a hard-block control pin needs physical proximity, not net splitting",
            )

        if (
            avg_spread is not None
            and net_pct is not None
            and avg_spread > DECISION_SPREAD_TILE_THRESHOLD
            and net_pct > DECISION_SPREAD_NET_THRESHOLD
        ):
            placement_first = [action for action in RAPIDWRIGHT_PLACEMENT_ACTIONS if action in allowed]
            allowed = placement_first + [action for action in allowed if action not in placement_first]

        # Incremental phys_opt genuinely cannot fix deeply negative WNS, so
        # demote it (with the reason) while structural actions are usable.
        # This is a demotion, not the old hard forbid, and it deliberately no
        # longer touches place_design_explore -- a full re-place is not an
        # incremental optimization (see PHYS_OPT_INCREMENTAL_ACTIONS).
        active_exhausted = set(self._active_exhausted_actions())
        structural_available = any(
            action in allowed and action not in active_exhausted
            for action in RAPIDWRIGHT_STRUCTURAL_ACTIONS
        )
        # Logic-heavy paths are exempt: placement/structural actions cannot
        # reduce logic depth, so demoting retime there points the LLM at
        # levers that cannot work (and the executor guard now honors the
        # same exemption).
        guard_logic_pct = (self.path_delay_breakdown or {}).get("logic_pct")
        guard_logic_heavy = isinstance(guard_logic_pct, (int, float)) and guard_logic_pct >= 0.45
        if (
            current_wns is not None
            and current_wns < PHYS_OPT_MIN_USEFUL_WNS_NS
            and structural_available
            and not guard_logic_heavy
        ):
            allowed = self._demote_actions(
                allowed,
                PHYS_OPT_INCREMENTAL_ACTIONS,
                f"WNS {current_wns:.3f} ns is below {PHYS_OPT_MIN_USEFUL_WNS_NS:.3f} ns; "
                f"incremental phys_opt cannot close that gap while structural actions remain",
            )

        # Cross-run priors: an action with zero wins and repeated losses on
        # THIS design across previous runs starts demoted (with its record as
        # the reason) instead of earning its losses all over again --
        # rapidwright_optimize_cell_placement is 0-for-everything across four
        # runs on the LogicNets benchmark, yet opened every run at full rank.
        # crossrun_priors is loaded once at startup and only merged back to
        # disk at the very end (_save_crossrun_priors), so without folding in
        # this run's own tallies here the guidance text goes stale mid-run:
        # run 20260803_131720 (corescore) showed rapidwright_optimize_cell_placement's
        # guidance still reading "0 wins / 4 losses across previous runs" at
        # iter 14, unchanged from iter 13, even though iter 13's own attempt
        # (this same run) had just failed in between -- a fresher "+1 loss
        # already this run" is a harder number for the LLM to rebut away.
        this_run_tallies = self._this_run_action_tallies()
        for prior_action, prior_record in ((self.crossrun_priors or {}).get("actions") or {}).items():
            good = int(prior_record.get("good", 0))
            bad = int(prior_record.get("bad", 0))
            this_run = this_run_tallies.get(prior_action, {})
            this_run_good = int(this_run.get("good", 0))
            this_run_bad = int(this_run.get("bad", 0))
            if prior_action in allowed and good == 0 and this_run_good == 0 and (bad + this_run_bad) >= 3:
                reason = f"0 wins / {bad} losses across previous runs on this design"
                if this_run_bad:
                    reason += f", plus {this_run_bad} more loss(es) already this run"
                allowed = self._demote_actions(allowed, [prior_action], reason)
        # Directive-level cross-run record: the action-level prior above can't
        # catch "place Default wins, place Explore loses" because both count
        # under place_design_explore. Name the losing directives so the LLM
        # stops re-trying them (Explore lost identically at iter 2 of two
        # consecutive runs on logicnets before this was added).
        losing_directives = [
            f"{name} ({rec.get('good', 0)}/{int(rec.get('good', 0)) + int(rec.get('bad', 0))})"
            for name, rec in (((self.crossrun_priors or {}).get("directives")) or {}).items()
            if int(rec.get("good", 0)) == 0 and int(rec.get("bad", 0)) >= 2
        ]
        if losing_directives and "place_design_explore" in allowed:
            note = (
                "directives with losing cross-run records on this design "
                f"(never improved): {', '.join(sorted(losing_directives))} -- do not re-try these"
            )
            existing = self.last_action_guidance.get("place_design_explore")
            self.last_action_guidance["place_design_explore"] = (
                f"{existing}; {note}" if existing else note
            )

        # This-run yield prior (pipeline audit, 20260802-20260804 sweep):
        # vivado_phys_opt's family was attempted 79/172 iterations across
        # that sweep (46% of all attempts, the single most-used family) yet
        # only 20% improved, while place_design_explore hit 68% on a fifth
        # as many tries. The cross-run "0 wins" kill switch above never
        # catches this because phys_opt_design *does* win sometimes (16/79
        # there) -- it never hits good == 0, and rightly so, since fully
        # excluding it would remove a lever that's genuinely useful ~1 time
        # in 5. Surface each action's own live this-run win rate instead:
        # once an action has enough attempts this run to be a real signal
        # and a currently-allowed alternative is clearly outperforming it,
        # demote (never remove) the laggard with the comparison as the
        # reason, so it stops being the reflex default while remaining
        # available if the better options are exhausted.
        this_run_rates = {
            action: tally["good"] / (tally["good"] + tally["bad"])
            for action, tally in this_run_tallies.items()
            if (tally["good"] + tally["bad"]) > 0
        }
        best_alternative = max(
            (
                (action, this_run_rates[action])
                for action in allowed
                if action in this_run_tallies
                and (this_run_tallies[action]["good"] + this_run_tallies[action]["bad"]) >= RECIPE_YIELD_ALT_MIN_ATTEMPTS
                and this_run_rates[action] >= RECIPE_YIELD_ALT_MIN_RATE
            ),
            key=lambda item: item[1],
            default=None,
        )
        if best_alternative is not None:
            best_action, best_rate = best_alternative
            for action in list(allowed):
                tally = this_run_tallies.get(action)
                if not tally or action == best_action:
                    continue
                total = tally["good"] + tally["bad"]
                rate = this_run_rates.get(action, 0.0)
                if total >= RECIPE_YIELD_MIN_ATTEMPTS and rate < RECIPE_YIELD_LOW_RATE and rate < best_rate:
                    best_total = this_run_tallies[best_action]["good"] + this_run_tallies[best_action]["bad"]
                    allowed = self._demote_actions(
                        allowed,
                        [action],
                        f"this run: {action} is {tally['good']}/{total} good ({rate:.0%}) so far, vs "
                        f"{best_action} at {this_run_tallies[best_action]['good']}/{best_total} ({best_rate:.0%}) "
                        f"-- prefer the higher-yield lever unless {best_action} is exhausted for this target",
                    )

        # Exploit-after-win (run 20260712_051231, measured): from a fresh
        # improvement, full re-place re-rolls went 0/3 (AltSpreadLogic_high
        # -92 MHz, ExtraTimingOpt -91 MHz, Explore -26 MHz) while incremental
        # refinement went 5/6 positive (pblock+re-route +0.25/+5.6/+1.3 MHz,
        # phys_opt +3.9/+1.1 MHz). When the last recorded iteration improved,
        # lead with refinement and demote fresh re-rolls with the reason --
        # polish the win before rolling the dice on it.
        if self._sitting_on_fresh_win():
            refine_first = [
                action for action in
                ("pblock", "phys_opt_design", "qor_suggestions", "route_explore", "phys_opt_design_retime")
                if action in allowed and action not in self.last_action_guidance
            ]
            if refine_first:
                allowed = refine_first + [a for a in allowed if a not in refine_first]
                allowed = self._demote_actions(
                    allowed,
                    [a for a in ("place_design_explore", "pblock_full_replace") if a in allowed],
                    "this run holds an unbeaten best above baseline -- refine it "
                    "(phys_opt, incremental pblock + re-route, qor_suggestions) instead "
                    "of gambling it on a fresh whole-design re-place: measured re-rolls "
                    "from a winning state lost 50-120 MHz far more often than they won",
                )

        # Item 4 (expensive-action cap): once a large design's full-replace
        # cap trips, remove the capped action from `allowed` outright instead
        # of demoting it to the end of the ranking. Demotion still let the
        # LLM pick it, pay for an LLM call, and get rejected in dispatch on a
        # cap decision that had already been made -- boom_soc's 07-19 run
        # spent 3 of its 11 iterations re-proposing place_design_explore/
        # pblock_full_replace after the cap had already fired (each rejected
        # in well under a second by dispatch, but each still cost a full LLM
        # round-trip and an iteration slot that could have tried something
        # else). The reason is still recorded in last_action_guidance for
        # forensic visibility even though the action is no longer offered.
        for capped_action in ("place_design_explore", "pblock_full_replace"):
            cap_reason = self._full_replace_blocked_reason(capped_action)
            if cap_reason:
                if capped_action in allowed:
                    allowed.remove(capped_action)
                existing = self.last_action_guidance.get(capped_action)
                self.last_action_guidance[capped_action] = (
                    f"{existing}; {cap_reason}" if existing else cap_reason
                )

        # Item 3 (affordability): an action that probably cannot finish inside
        # the remaining budget is demoted at 1.3x its estimated cost (and
        # hard-refused at 1.0x on dispatch). Motivating incident: a 15.5 min
        # place Explore followed by a route Explore killed by the budget
        # clamp -- 30+ minutes, zero valid results.
        remaining_budget = self._time_remaining_s()
        if remaining_budget is not None:
            for candidate in list(allowed):
                cost = self._estimated_action_cost_s(candidate)
                if cost is not None and remaining_budget < ACTION_COST_DEMOTE_FACTOR * cost:
                    allowed = self._demote_actions(
                        allowed,
                        [candidate],
                        f"costs ~{cost / 60.0:.0f} min, {max(remaining_budget, 0) / 60.0:.0f} min "
                        f"remain -- pick a refinement that fits",
                    )

        # Empirical kill switch (pipeline audit 2026-07-28, runs 07-19 through
        # 07-28): fanout_split and rapidwright_optimize_cell_placement went
        # 0-for-24 across every run in that window and every design tried --
        # fanout_split either had no valid high-fanout net to act on or its
        # RapidWright edit left hundreds-to-thousands of cells unplaced;
        # rapidwright_optimize_cell_placement did the same or moved zero
        # cells. Demotion alone doesn't stop them: the excessive_fanout
        # hypothesis (analysis_layer.py) recommends fanout_split specifically
        # and never attaches a discouraging reason to it, so the LLM kept
        # picking a 0%-win action with no warning at all. Remove both from
        # the normal menu outright. rapidwright_optimize_cell_placement is
        # restored below as the sole action when the Vivado Implementation
        # license is unavailable, since it is the only action that needs no
        # such license -- actions_for() (analysis_layer.py) already filters
        # its recommended/veto lists against `allowed`, so removing these
        # here safely no-ops any downstream reference to them.
        for dead_action in ("fanout_split", "rapidwright_optimize_cell_placement"):
            if dead_action in allowed:
                allowed.remove(dead_action)

        if self.implementation_license_available is False:
            implementation_actions = set(VIVADO_INCREMENTAL_IMPLEMENTATION_ACTIONS) | {"fanout_split"}
            allowed = [action for action in allowed if action not in implementation_actions]
            for action in sorted(implementation_actions):
                if action not in forbidden:
                    forbidden.append(action)
            if not allowed:
                allowed = ["rapidwright_optimize_cell_placement"]
                forbidden = [action for action in forbidden if action not in allowed]

        return allowed, forbidden

    def _parse_json_result(self, result_text: str) -> dict:
        try:
            payload = json.loads(result_text)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _result_has_error(self, payload: dict) -> bool:
        return bool(payload.get("error") or payload.get("success") is False)

    def _filter_exhausted_actions(self, allowed: list[str]) -> list[str]:
        # BUG FIX: this used to also exclude any action whose *cumulative*
        # action_failure_counts had ever reached ACTION_FAILURE_EXHAUSTION_THRESHOLD,
        # regardless of whether its cooldown had expired. Since that counter
        # is never decremented (only reset to 0 on a later *successful*
        # improving attempt), and an excluded action can never be attempted
        # again to earn that reset, this created a permanent catch-22 lockout:
        # once an action hit the threshold once, it was gone for the rest of
        # the run even though _active_exhausted_actions() (which correctly
        # tracks cooldown_until_iter and target fingerprint) had already
        # cleared it. Rely solely on the cooldown-aware check.
        active_exhausted = set(self._active_exhausted_actions())
        if not active_exhausted:
            return allowed
        filtered = [action for action in allowed if action not in active_exhausted]
        if filtered:
            logger.warning(
                "Action failure memory: suppressing exhausted actions %s until cooldown expires.",
                sorted(active_exhausted),
            )
            return filtered
        structural_fallback = [
            action for action in RAPIDWRIGHT_STRUCTURAL_ACTIONS
            if action not in active_exhausted
        ]
        if structural_fallback:
            logger.warning(
                "Action failure memory: all offered actions were exhausted; offering alternate structural actions %s.",
                structural_fallback,
            )
            return structural_fallback
        return allowed

    def _active_exhausted_actions(self) -> list[str]:
        exhausted: list[str] = []
        current_fingerprint = self._target_fingerprint()
        for action, memory in self.action_failure_memory.items():
            if memory.get("target_fingerprint") != current_fingerprint:
                continue
            cooldown_until = int(memory.get("cooldown_until_iter") or -1)
            if cooldown_until >= self.iteration:
                exhausted.append(action)
        # Fix #7: pblock's overlap cooldown is tracked independently of
        # target_fingerprint (see _compute_pblock_ranges) precisely because
        # fingerprint drift was letting it slip back into allowed_actions
        # every few iterations. Apply it here regardless of fingerprint match.
        if self.iteration <= self.pblock_region_cooldown_until_iter:
            for act in PBLOCK_ACTION_FAMILY:
                if act not in exhausted:
                    exhausted.append(act)
        for act, cooldown_until in self.action_structural_cooldown_until_iter.items():
            if cooldown_until >= self.iteration and act not in exhausted:
                exhausted.append(act)
        return exhausted

    def _llm_action_failure_memory(self) -> dict:
        """Compact, LLM-facing view of action failure memory (score item D).

        The full serializer below embeds the complete multi-KB
        target_fingerprint JSON for every remembered action; shipped in the
        timing context every call, that alone drove per-call prompt cost up
        ~10x by late iterations (run 20260714_005251: $0.30 for 19 iters).
        The LLM only needs WHICH actions failed, HOW often, on WHICH targets
        (a few names), and WHETHER the failure was on the current target set
        -- a short hash answers that last question just as well as the blob.
        history.json keeps the full fingerprints for forensics."""
        current_fp_hash = format(hash(self._target_fingerprint()) & 0xFFFFFFFF, "08x")
        compact: dict = {}
        for action, memory in self._serializable_action_failure_memory().items():
            fp = memory.get("target_fingerprint")
            fp_hash = format(hash(fp) & 0xFFFFFFFF, "08x") if fp else None
            compact[action] = {
                "consecutive_no_action_failures": memory.get("consecutive_no_action_failures"),
                "failed_targets": list(memory.get("failed_targets") or [])[:3],
                "last_failed_iter": memory.get("last_failed_iter"),
                "failed_on_current_targets": fp_hash == current_fp_hash,
            }
            if memory.get("cooldown_until_iter", -1) >= self.iteration:
                compact[action]["cooldown_until_iter"] = memory["cooldown_until_iter"]
        return compact

    def _serializable_action_failure_memory(self) -> dict:
        return {
            action: dict(memory)
            for action, memory in self.action_failure_memory.items()
            if memory.get("consecutive_no_action_failures") or int(memory.get("cooldown_until_iter") or -1) >= self.iteration
        }

    def _serializable_action_failure_counts(self) -> dict[str, int]:
        return {
            action: count
            for action, count in self.action_failure_counts.items()
            if count > 0
        }

    def _is_no_action_failure(self, failure: dict) -> bool:
        error_type = str(failure.get("error_type") or "").lower()
        message = str(failure.get("message") or "").lower()
        return (
            error_type == "no_action_target"
            or error_type == "pblock_range_computation_failed"
            or error_type == "pblock_empty_assignment"
            or "no critical cells were available" in message
            or "no legal placement" in message
            or "no action target" in message
            or "selected but no" in message
            or "overlaps an already-applied pblock" in message
            or "skipping to avoid overlapping" in message
        )

    def _target_fingerprint(self) -> str:
        candidates = []
        for candidate in self.current_target_candidates[:5]:
            candidates.append({
                "slack": candidate.get("slack"),
                "startpoint": candidate.get("startpoint"),
                "endpoint": candidate.get("endpoint"),
            })
        payload = {
            "tier": self.target_tier,
            "delay_class": self.path_delay_classification,
            "candidates": candidates,
        }
        return json.dumps(payload, sort_keys=True)

    def _remember_no_action_failure(self, action: str, targets: list[str]) -> None:
        if not action:
            return
        actions_to_penalize = PBLOCK_ACTION_FAMILY if action in PBLOCK_ACTION_FAMILY else {action}
        for act in actions_to_penalize:
            fingerprint = self._target_fingerprint()
            memory = self.action_failure_memory.get(act, {})
            if memory.get("target_fingerprint") != fingerprint:
                memory = {
                    "consecutive_no_action_failures": 0,
                    "failed_targets": [],
                    "target_fingerprint": fingerprint,
                }
            failed_targets = list(memory.get("failed_targets") or [])
            for target in targets:
                if target and target not in failed_targets:
                    failed_targets.append(target)
            count = int(memory.get("consecutive_no_action_failures") or 0) + 1
            self.action_failure_counts[act] = self.action_failure_counts.get(act, 0) + 1
            memory.update({
                "consecutive_no_action_failures": count,
                "failed_targets": failed_targets,
                "last_failed_iter": self.iteration,
                "target_fingerprint": fingerprint,
            })
            if count >= ACTION_FAILURE_EXHAUSTION_THRESHOLD:
                memory["cooldown_until_iter"] = self.iteration + ACTION_FAILURE_COOLDOWN_ITERS
                logger.warning(
                    "Action failure memory: %s exhausted after %d no-action failures; cooling down until iteration %d.",
                    act, count, memory["cooldown_until_iter"],
                )
            self.action_failure_memory[act] = memory
            history = self.action_structural_failure_iters.setdefault(act, [])
            history.append(self.iteration)
            cutoff = self.iteration - ACTION_STRUCTURAL_FAILURE_WINDOW_ITERS
            self.action_structural_failure_iters[act] = [i for i in history if i >= cutoff]
            recent_count = len(self.action_structural_failure_iters[act])
            if recent_count >= ACTION_STRUCTURAL_FAILURE_THRESHOLD:
                cooldown_until = self.iteration + ACTION_STRUCTURAL_FAILURE_COOLDOWN_ITERS
                if cooldown_until > self.action_structural_cooldown_until_iter.get(act, -1):
                    self.action_structural_cooldown_until_iter[act] = cooldown_until
                    logger.warning(
                        "Structural failure guard: %s failed/regressed %d time(s) within the "
                        "last %d iterations (tracked independent of target_fingerprint drift); "
                        "withholding it until iteration %d regardless of fingerprint changes.",
                        act, recent_count, ACTION_STRUCTURAL_FAILURE_WINDOW_ITERS, cooldown_until,
                    )

    def _reset_action_failure_memory(self, action: str) -> None:
        self.action_failure_counts[action] = 0
        if action in self.action_failure_memory:
            self.action_failure_memory[action]["consecutive_no_action_failures"] = 0
            self.action_failure_memory[action]["cooldown_until_iter"] = -1
        self.action_structural_failure_iters[action] = []
        self.action_structural_cooldown_until_iter[action] = -1

    def _blacklist_failure_targets(self, action: str, targets: list[str]) -> None:
        if self.checkpoint_manager is None or not targets:
            return
        if action in {"rapidwright_optimize_cell_placement", "rapidwright_analyze_net_detour"}:
            for target in targets:
                if self.checkpoint_manager._is_blacklistable_target(target) and target not in self.checkpoint_manager.cells_blacklisted:
                    self.checkpoint_manager.cells_blacklisted.append(str(target))
                # Fix #6b: only stamp the iteration on *first* blacklisting.
                # The original comment here said to refresh the TTL on every
                # repeat failure "so a cell that keeps failing keeps its TTL
                # fresh" -- but when a cell is re-selected as a fallback
                # target purely because it's the only candidate left (no
                # other data was used), that refresh means the TTL never
                # actually elapses. In the FPL run this locked
                # layer1_reg/data_out_reg[72]_rep__0/C and
                # layer2_reg/data_out_reg[84] out for the rest of a 50-iter
                # run (16 consecutive re-blacklists). Stamp once; let it expire.
                if str(target) not in self.cell_blacklist_added_iter:
                    self.cell_blacklist_added_iter[str(target)] = self.iteration
            persist = getattr(self.checkpoint_manager, "_persist_history", None)
            if callable(persist):
                persist()
        elif action in {"rapidwright_optimize_fanout", "fanout_split"}:
            for target in targets:
                if self.checkpoint_manager._is_blacklistable_target(target) and target not in self.checkpoint_manager.nets_blacklisted:
                    self.checkpoint_manager.nets_blacklisted.append(str(target))
            persist = getattr(self.checkpoint_manager, "_persist_history", None)
            if callable(persist):
                persist()

    def _prune_expired_blacklist(self) -> None:
        """Fix #6: drop cells whose TTL has elapsed so the candidate pool for
        rapidwright_optimize_cell_placement can recover instead of shrinking
        monotonically over a long run. Cells with no recorded blacklist
        iteration (e.g. loaded from an older history file) are left alone
        rather than guessed-expired."""
        if self.checkpoint_manager is None:
            return
        current = list(getattr(self.checkpoint_manager, "cells_blacklisted", []) or [])
        if not current:
            return
        still_blacklisted = []
        expired = []
        for cell in current:
            added_at = self.cell_blacklist_added_iter.get(str(cell))
            if added_at is not None and (self.iteration - added_at) >= BLACKLIST_TTL_ITERS:
                expired.append(cell)
            else:
                still_blacklisted.append(cell)
        if expired:
            self.checkpoint_manager.cells_blacklisted = still_blacklisted
            for cell in expired:
                self.cell_blacklist_added_iter.pop(str(cell), None)
            logger.info(
                "Blacklist expiry: %d cell(s) re-eligible after %d+ iterations: %s",
                len(expired), BLACKLIST_TTL_ITERS, expired,
            )
            persist = getattr(self.checkpoint_manager, "_persist_history", None)
            if callable(persist):
                persist()

    def _cell_blacklist(self) -> set[str]:
        if self.checkpoint_manager is None:
            return set()
        self._prune_expired_blacklist()
        return set(self.checkpoint_manager.get_blacklist().get("cells", []))

    def _filter_blacklisted_cells(self, cells: list[str]) -> list[str]:
        blacklist = self._cell_blacklist()
        filtered: list[str] = []
        for cell in cells:
            cell = str(cell).strip()
            if cell and cell not in blacklist and cell not in filtered:
                filtered.append(cell)
        return filtered

    def _critical_path_cell_candidates(self, timing_context: dict, limit: int = 10) -> list[str]:
        candidates: list[str] = []
        critical_pins: dict[str, list[str]] = {}
        worst_path = timing_context.get("worst_path", {})

        def _register(raw_value):
            value = str(raw_value or "").strip("{} ")
            if not value:
                return
            if "/" in value:
                cell, pin = value.rsplit("/", 1)
            else:
                cell, pin = value, None
            if cell and cell not in candidates:
                candidates.append(cell)
            if cell and pin:
                pins = critical_pins.setdefault(cell, [])
                if pin not in pins:
                    pins.append(pin)

        for key in ("start_cell", "end_cell"):
            _register(worst_path.get(key))

        for candidate in self.current_target_candidates[:limit]:
            for key in ("startpoint", "endpoint"):
                _register(candidate.get(key))

        filtered = self._filter_blacklisted_cells(candidates)[:limit]
        # Side-channel: the caller reads this right after calling this method.
        self._last_critical_pins = {c: critical_pins[c] for c in filtered if c in critical_pins}
        return filtered

    def _high_fanout_cell_names(self) -> set[str]:
        """Best-effort mapping from self.high_fanout_nets (net names) to the
        driving cell name, using the same rsplit('/', 1)[0] convention used
        everywhere else in this file for pin -> cell. Used to keep
        rapidwright_optimize_cell_placement's centroid-of-all-connections
        heuristic away from cells whose move would unroute a high-fanout net
        and hit every other sink on it as collateral damage (see the tool's
        own docstring: "this will also unroute any routing going to other
        unrelated cells")."""
        cells: set[str] = set()
        for net_name, _fanout, _path_count in self.high_fanout_nets:
            cleaned = str(net_name).strip("{} ")
            cell = cleaned.rsplit("/", 1)[0] if "/" in cleaned else cleaned
            if cell:
                cells.add(cell)
        return cells

    def _filter_high_fanout_cells(self, cells: list[str]) -> list[str]:
        if not CELL_PLACEMENT_FANOUT_GUARD_ENABLED or not self.high_fanout_nets:
            return cells
        fanout_cells = self._high_fanout_cell_names()
        filtered = [c for c in cells if c not in fanout_cells]
        dropped = [c for c in cells if c in fanout_cells]
        if dropped:
            logger.info(
                "Fanout guard: excluding %d cell(s) from rapidwright_optimize_cell_placement "
                "because they sit on a high-fanout net (moving them would unroute sinks "
                "unrelated to the targeted critical path): %s",
                len(dropped), dropped,
            )
        return filtered

    def _path_identifier_targets(self, timing_context: dict) -> list[str]:
        worst_path = timing_context.get("worst_path", {})
        targets = []
        for key in ("start_cell", "end_cell"):
            value = str(worst_path.get(key) or "").strip("{} ")
            if value and value not in targets:
                targets.append(value)
        if not targets:
            start = str(worst_path.get("start_cell") or "").strip()
            end = str(worst_path.get("end_cell") or "").strip()
            if start or end:
                targets.append(f"{start}->{end}")
        return targets

    async def _extract_critical_path_cells_file(self, num_paths: int = 20) -> Path:
        output_file = Path(self.temp_dir) / f"critical_path_cells_iter_{self.iteration:03d}.json"
        result = await self.call_tool(
            "vivado_extract_critical_path_cells",
            {"num_paths": num_paths, "output_file": str(output_file), "timeout": 300},
            internal=True,
        )
        if self._vivado_output_has_error(result):
            raise RuntimeError(f"Could not extract critical path cells: {result[:300]}")
        return output_file

    async def _live_critical_path_cell_candidates(self, num_paths: int = 20, limit: int = 20) -> list[str]:
        """Extract current timing-path cells from Vivado and apply the cell blacklist."""
        try:
            cells_file = await self._extract_critical_path_cells_file(num_paths=num_paths)
            paths = json.loads(cells_file.read_text())
        except Exception as exc:
            logger.warning("Could not extract live critical path cells: %s", exc)
            return []

        candidates: list[str] = []
        if isinstance(paths, list):
            for path in paths:
                if not isinstance(path, list):
                    continue
                for cell in path:
                    cell_name = str(cell).strip("{} ")
                    if cell_name and cell_name not in candidates:
                        candidates.append(cell_name)

        return self._filter_blacklisted_cells(candidates)[:limit]

    async def _extract_critical_path_pins_file(self, num_paths: int = 20) -> Path:
        output_file = Path(self.temp_dir) / f"critical_path_pins_iter_{self.iteration:03d}.json"
        result = await self.call_tool(
            "vivado_extract_critical_path_pins",
            {"num_paths": num_paths, "output_file": str(output_file), "timeout": 300},
            internal=True,
        )
        if self._vivado_output_has_error(result):
            raise RuntimeError(f"Could not extract critical path pins: {result[:300]}")
        return output_file

    def _bram_dsp_bottleneck_evidence(self) -> dict:
        """Is BRAM/DSP hard-block placement actually the bottleneck right
        now, or just present among the candidates?

        Two independent signals, both required:
        1. dominance -- what FRACTION of current_target_candidates touch a
           BRAM/DSP cell (by start/endpoint name), not just whether at least
           one does. One stray BRAM candidate among 20 net-delay-bound LUT
           paths is noise, not evidence the hard block is the problem.
        2. fixability -- delay on those paths has to actually be placement/
           routing-driven for a BRAM/DSP-aware region to help at all. On a
           logic_delay_bound path (_classify_worst_path_delay already logs
           "skipping pblock" for exactly this reason), no region -- however
           well chosen -- reduces logic depth.

        Used to gate target_bram_count/target_dsp_count sizing in
        _compute_pblock_ranges so hard-block capacity is only requested (and
        therefore only ever fast-failed by _check_pblock_utilization) when
        there's real evidence it matters, not on a single incidental name
        match."""
        candidates = self.current_target_candidates or []
        total = len(candidates)
        bram_cells: set[str] = set()
        dsp_cells: set[str] = set()
        hard_block_candidates = 0
        for candidate in candidates:
            touches_hard_block = False
            for key in ("startpoint", "endpoint"):
                name = str(candidate.get(key) or "").lower()
                cell = name.rsplit("/", 1)[0] if "/" in name else name
                # "fifo" added (pipeline audit, run 20260803_164620 on
                # ispd16_example2): FIFO18E2/FIFO36E2 primitives occupy the
                # same RAMB18/RAMB36 sites as plain BRAM, but this list only
                # matched generic BRAM-ish names, so a candidate touching a
                # FIFO cell never bumped target_bram_count. The pblock request
                # then sailed through _check_pblock_utilization with a BRAM
                # demand of 0, got applied, and only Vivado's own DRC caught
                # the real shortage after the fact ("FIFO: requires 768, only
                # 120 available" / "RAMB36E2: requires 384, only 60
                # available") -- a pblock_region_too_small failure that a
                # correct pre-check would have avoided entirely.
                if any(k in cell for k in ("ram_reg", "_bram", "ramb", "uram", "fifo")):
                    bram_cells.add(cell)
                    touches_hard_block = True
                if "dsp" in cell:
                    dsp_cells.add(cell)
                    touches_hard_block = True
            if touches_hard_block:
                hard_block_candidates += 1

        fraction = (hard_block_candidates / total) if total else 0.0
        dominant = fraction >= BRAM_DSP_BOTTLENECK_FRACTION_THRESHOLD
        fixable = self.path_delay_classification != "logic_delay_bound"
        is_bottleneck = dominant and fixable and bool(bram_cells or dsp_cells)

        if bram_cells or dsp_cells:
            reason = (
                f"{hard_block_candidates}/{total} candidates ({fraction:.0%}) touch a "
                f"BRAM/DSP cell, delay_class={self.path_delay_classification!r}"
            )
            if not dominant:
                reason += f" -- below the {BRAM_DSP_BOTTLENECK_FRACTION_THRESHOLD:.0%} dominance bar"
            if not fixable:
                reason += " -- logic-delay-bound, placement cannot fix this regardless"
        else:
            reason = "no candidate touches a BRAM/DSP cell"

        return {
            "is_bottleneck": is_bottleneck,
            "fraction": fraction,
            "bram_cells": bram_cells,
            "dsp_cells": dsp_cells,
            "reason": reason,
        }

    async def _compute_pblock_ranges(self, params: dict, timing_context: dict) -> tuple[Optional[dict], Optional[str]]:
        """Compute Vivado pblock ranges through RapidWright fabric analysis."""
        if params.get("ranges"):
            return params, None

        # Fix #11: the old "overlap cooldown" early-return that lived here is
        # gone -- overlapping regions are now resolved by deleting the stale
        # pblock (see the overlap branch below) instead of withholding the
        # pblock action for 20 iterations, which run history showed turned
        # into a permanent deadlock once the first pblock landed on the
        # timing-critical region.

        # Fix #5: size the pblock request off the actual number of critical-path
        # candidates we're clustering, not the whole design. last_design_info
        # comes from rapidwright_read_checkpoint (whole-design counts), which is
        # why the old fallback (design lut_count, or 20000) requested a region
        # sized for the entire design and blew the 85% utilization check on
        # every single call regardless of how small the real target was.
        num_candidates = max(len(self.current_target_candidates), 1)
        estimated_lut_count = max(
            num_candidates * PBLOCK_LUTS_PER_CANDIDATE_PATH,
            PBLOCK_MIN_TARGET_LUT_COUNT,
        )
        estimated_ff_count = max(
            num_candidates * PBLOCK_FFS_PER_CANDIDATE_PATH,
            PBLOCK_MIN_TARGET_FF_COUNT,
        )
        # Item 3b (region search memory): if the previous pblock attempt
        # executed cleanly but regressed or changed nothing, the region was
        # most likely too tight -- grow the next auto-sized request instead of
        # recomputing the identical one. An explicit LLM-provided
        # target_lut_count always wins over this heuristic.
        if not params.get("target_lut_count"):
            last_pblock = next(
                (attempt for attempt in reversed(self.pblock_attempt_history)
                 if attempt.get("action") == "pblock"),
                None,
            )
            if last_pblock and last_pblock.get("status") in {"regression", "no_improvement"}:
                prev_lut = int(last_pblock.get("target_lut_count") or 0)
                prev_ff = int(last_pblock.get("target_ff_count") or 0)
                if prev_lut > 0:
                    grown_lut = int(prev_lut * PBLOCK_REGRESSION_GROW_FACTOR)
                    grown_ff = int(max(prev_ff, estimated_ff_count) * PBLOCK_REGRESSION_GROW_FACTOR)
                    logger.info(
                        "pblock search memory: previous attempt (iter %s, %s LUTs) ended in "
                        "'%s'; growing this request to %s LUTs instead of repeating the size.",
                        last_pblock.get("iteration"), prev_lut,
                        last_pblock.get("status"), grown_lut,
                    )
                    estimated_lut_count = max(estimated_lut_count, grown_lut)
                    estimated_ff_count = max(estimated_ff_count, grown_ff)
        target_lut_count = int(params.get("target_lut_count") or estimated_lut_count)
        target_ff_count = int(params.get("target_ff_count") or estimated_ff_count)
        target_dsp_count = int(params.get("target_dsp_count") or 0)
        target_bram_count = int(params.get("target_bram_count") or 0)
        # Hard-block demand (boom_soc run 20260713 iter 4): sizing was
        # LUT/FF-only, so a cluster of BRAM-endpoint paths got a region with
        # 60 RAMB36 sites for 153 BRAMs and failed DRC. The candidates'
        # start/endpoints say when BRAM/DSP cells are being clustered --
        # request sites for them (x2 margin: the server-side cell matching
        # expands beyond the named candidates).
        #
        # Gated on _bram_dsp_bottleneck_evidence (2026-07-28): the original
        # version inflated these targets the moment ANY candidate touched a
        # BRAM/DSP name, even a single stray one among 20 LUT-path
        # candidates. Requesting hard-block capacity that isn't actually
        # needed only exposes the region to a BRAM/DSP utilization rejection
        # (_check_pblock_utilization) for a resource that was never the real
        # bottleneck -- wasted risk for zero benefit.
        if not target_bram_count or not target_dsp_count:
            evidence = self._bram_dsp_bottleneck_evidence()
            bram_cells = evidence["bram_cells"]
            dsp_cells = evidence["dsp_cells"]
            if bram_cells or dsp_cells:
                if evidence["is_bottleneck"]:
                    if bram_cells and not target_bram_count:
                        target_bram_count = 2 * len(bram_cells)
                    if dsp_cells and not target_dsp_count:
                        target_dsp_count = 2 * len(dsp_cells)
                else:
                    logger.info(
                        "BRAM/DSP bottleneck gate: not requesting hard-block capacity (%s).",
                        evidence["reason"],
                    )
        # A previous pblock abort's DRC measured the REAL hard-block demand
        # (server-side cell expansion included); it always outranks the
        # candidate-name estimate above.
        target_bram_count = max(target_bram_count, self.pblock_hard_block_demand.get("bram", 0))
        target_dsp_count = max(target_dsp_count, self.pblock_hard_block_demand.get("dsp", 0))

        # Fix #5 continued: if this sizing (explicit or estimated) still comes
        # back over-utilized, shrink it and retry rather than failing outright
        # on the first miss -- self-corrects instead of depending on the
        # estimate being exactly right.
        last_error: Optional[str] = None
        overlap_attempt = 0
        size_attempt = 0

        # Cluster anchoring (2026-08-01 multi-candidate audit): pass the
        # actual candidate-path cells so the fabric search centers on where
        # THIS cluster lives, not the whole design's centroid -- the same
        # startpoint/endpoint-to-cell-name extraction used elsewhere
        # (_bram_dsp_bottleneck_evidence).
        target_cell_names: list[str] = []
        for candidate in self.current_target_candidates:
            for key in ("startpoint", "endpoint"):
                name = str(candidate.get(key) or "")
                if "/" in name:
                    name = name.rsplit("/", 1)[0]
                if name and name not in target_cell_names:
                    target_cell_names.append(name)

        while True:
            analysis_args = {
                "target_lut_count": target_lut_count,
                "target_ff_count": target_ff_count,
                "target_dsp_count": target_dsp_count,
                "target_bram_count": target_bram_count,
                "target_cell_names": target_cell_names,
            }
            fabric_text = await self.call_tool("rapidwright_analyze_fabric_for_pblock", analysis_args, internal=True)
            fabric = self._parse_json_result(fabric_text)
            if self._result_has_error(fabric):
                return None, f"RapidWright fabric analysis failed: {fabric.get('error') or fabric_text[:300]}"

            required_region_keys = ("col_min", "col_max", "row_min", "row_max")
            region_candidates = [
                candidate for candidate in (fabric.get("candidate_regions") or [fabric.get("recommended_region")])
                if candidate and all(key in candidate for key in required_region_keys)
            ]
            if not region_candidates:
                return None, f"RapidWright fabric analysis did not return recommended_region: {fabric_text[:300]}"

            # Try every candidate region from this fabric scan at the CURRENT
            # target sizing before paying for a whole new RapidWright call --
            # 2026-08-01: previously only the single top recommendation was
            # ever tried, so a region that failed (overlap, clock-region miss
            # resolved but still over-utilized) went straight to shrinking the
            # target instead of just trying the next-best window from the
            # same scan. Different windows can have different available
            # supply even at identical demand, so this applies to BRAM/DSP
            # overshoot too, not just LUT/FF.
            ranges = None
            range_payload = None
            failing_label = None
            utilization_error = None
            for region in region_candidates:
                # --- Fix #2a, reworked (Fix #11): a region overlapping a pblock
                # applied EARLIER this run used to put pblock on a 20-iteration
                # cooldown ("withholding pblock"). Run history showed that once the
                # first pblock landed on the timing-critical region, every future
                # recommendation overlapped it, so the pblock family was
                # effectively dead for the rest of the run -- 25+ consecutive
                # withheld attempts while the design never changed. The critical
                # region is critical precisely because that's where the failing
                # paths live; refusing to ever touch it again is a deadlock, not a
                # safety feature. Instead, delete the stale overlapping pblock
                # from the live design (its cells keep their current placement)
                # and proceed with the newly recommended region. ---
                overlap = self._find_pblock_overlap(region)
                while overlap is not None:
                    overlap_attempt += 1
                    stale_name = str(overlap.get("pblock_name") or "")
                    if stale_name:
                        logger.warning(
                            "pblock region overlap: recommended region %s collides with "
                            "already-applied pblock %s (iteration %s); deleting the stale "
                            "pblock and proceeding with the new region instead of "
                            "withholding the pblock action.",
                            region, stale_name, overlap.get("iteration"),
                        )
                        await self.call_tool(
                            "vivado_run_tcl",
                            {
                                "command": (
                                    f"if {{[llength [get_pblocks -quiet {stale_name}]] > 0}} "
                                    f"{{delete_pblocks [get_pblocks {stale_name}]}}"
                                )
                            },
                            internal=True,
                        )
                    self.applied_pblock_regions = [
                        applied for applied in self.applied_pblock_regions if applied is not overlap
                    ]
                    self.pblock_region_cooldown_until_iter = -1
                    overlap = self._find_pblock_overlap(region)

                convert_args = {
                    "col_min": int(region["col_min"]),
                    "col_max": int(region["col_max"]),
                    "row_min": int(region["row_min"]),
                    "row_max": int(region["row_max"]),
                    "use_clock_regions": bool(params.get("use_clock_regions", False)),
                }
                range_text = await self.call_tool("rapidwright_convert_fabric_region_to_pblock", convert_args, internal=True)
                range_payload = self._parse_json_result(range_text)
                if self._result_has_error(range_payload):
                    return None, f"RapidWright pblock range conversion failed: {range_payload.get('error') or range_text[:300]}"

                ranges = range_payload.get("pblock_ranges")
                if not ranges:
                    return None, f"RapidWright pblock range conversion returned no pblock_ranges: {range_text[:300]}"

                # Run 20260712 lesson (iter 3, WNS -0.978 -> -3.682): the fabric
                # analyzer recommends a region with FREE RESOURCES, which is not
                # necessarily anywhere near the critical cells. Converted to
                # clock-region form that produced CLOCKREGION_X5Y0:X5Y1 while the
                # critical cluster sits in X1Y4/X2Y4 -- constraining the cells
                # into it dragged them across the die. We now measure the
                # cluster's clock regions (item 4), so require the computed
                # clock-region range to actually contain at least one of them.
                cluster_regions = {
                    str(cr)
                    for cr in (timing_context.get("cluster_clock_regions") or [])
                    if cr
                }
                if (
                    "CLOCKREGION" in str(ranges)
                    and cluster_regions
                    and not self._clockregion_ranges_cover(str(ranges), cluster_regions)
                ):
                    # History(16) lesson: failing this check outright burned an
                    # iteration and left recovery to the LLM (which never
                    # retried). The SLICE-range conversion of the SAME fabric
                    # region is what produced the old run's improving pblock, so
                    # fall back to it automatically instead of failing.
                    logger.warning(
                        "Clock-region pblock %s misses the critical cluster's regions %s; "
                        "auto-falling back to site-range conversion of the same fabric region.",
                        ranges, sorted(cluster_regions),
                    )
                    convert_args["use_clock_regions"] = False
                    params["use_clock_regions"] = False
                    range_text = await self.call_tool(
                        "rapidwright_convert_fabric_region_to_pblock", convert_args, internal=True
                    )
                    range_payload = self._parse_json_result(range_text)
                    if self._result_has_error(range_payload):
                        return None, (
                            f"site-range fallback conversion failed after the clock-region "
                            f"result missed the cluster: {range_payload.get('error') or range_text[:300]}"
                        )
                    ranges = range_payload.get("pblock_ranges")
                    if not ranges:
                        return None, (
                            "site-range fallback conversion returned no pblock_ranges after "
                            "the clock-region result missed the cluster."
                        )

                # --- Fix #2b: reject regions that would be packed too densely. ---
                site_counts = range_payload.get("site_counts") or {}
                utilization_failure = self._check_pblock_utilization(
                    site_counts, target_lut_count, target_ff_count,
                    target_dsp_count, target_bram_count,
                )
                if not utilization_failure:
                    last_error = None
                    break  # this candidate works -- stop trying more

                failing_label, utilization_error = utilization_failure
                last_error = utilization_error
                # Try the next candidate region at the SAME target size
                # before shrinking anything.
            else:
                # No `break` fired: every candidate region failed utilization.
                ranges = None

            if ranges is not None and not utilization_failure:
                break  # a candidate worked -- exit the sizing-retry loop too

            # Every candidate region tried at this sizing was over-utilized.
            # Pipeline audit 2026-07-28: BRAM/DSP demand comes from actual
            # hard-block cells that must be in the region
            # (self.pblock_hard_block_demand), not a resizable estimate like
            # LUT/FF -- shrinking target_lut_count/target_ff_count has zero
            # effect on it, so retrying with a smaller LUT/FF target just
            # repeats the identical failure for free (boom_soc: "projected
            # BRAM utilization 464%... gave up after 4 sizing attempts", all
            # 4 identical since none of them touched the BRAM target).
            # Multiple candidate windows were already tried above (different
            # locations can have different DSP/BRAM supply); if ALL of them
            # failed on BRAM/DSP, no amount of LUT/FF shrinking will help --
            # fail immediately instead of burning the remaining sizing attempts.
            if failing_label in ("BRAM", "DSP"):
                last_error = (
                    f"{utilization_error} This is hard-block demand from cells that must "
                    f"be in the region, not a resizable estimate -- retrying with a smaller "
                    f"LUT/FF target cannot fix it, and none of the {len(region_candidates)} "
                    f"candidate region(s) tried had enough {failing_label} capacity; reduce "
                    f"the number of {failing_label} cells being clustered instead."
                )
                break

            if size_attempt < PBLOCK_SIZE_SHRINK_MAX_RETRIES:
                size_attempt += 1
                logger.warning(
                    "pblock sizing attempt %d over-utilized on all %d candidate region(s) (%s); "
                    "shrinking target lut/ff counts by %.0f%% and retrying.",
                    size_attempt, len(region_candidates), utilization_error,
                    100 * (1 - PBLOCK_SIZE_SHRINK_FACTOR),
                )
                target_lut_count = max(int(target_lut_count * PBLOCK_SIZE_SHRINK_FACTOR), PBLOCK_MIN_TARGET_LUT_COUNT // 4)
                target_ff_count = max(int(target_ff_count * PBLOCK_SIZE_SHRINK_FACTOR), PBLOCK_MIN_TARGET_FF_COUNT // 4)
                continue

            break

        if last_error:
            return None, (
                f"{last_error} (gave up after {size_attempt + 1} sizing attempts and "
                f"{overlap_attempt} overlap-avoidance attempts, starting from an estimate "
                f"of {num_candidates} candidate paths)"
            )

        computed = dict(params)
        computed["ranges"] = ranges
        computed.setdefault("pblock_name", f"{DEFAULT_PBLOCK_NAME_PREFIX}_{self.iteration:03d}")
        computed.setdefault("apply_to", "current_design")
        computed.setdefault("is_soft", False)
        # Item 3b: remembered by _note_recipe_outcome alongside the attempt's
        # outcome so the next auto-sizing can move instead of repeating.
        self.last_pblock_sizing = {
            "target_lut_count": target_lut_count,
            "target_ff_count": target_ff_count,
        }
        self.last_rapidwright_edit_summary = {
            "action": "pblock_range_computation",
            "cells_moved": 0,
            "nets_affected": 0,
            "pblock_ranges": ranges,
            "fabric_region": region,
            "target_lut_count": target_lut_count,
            "target_ff_count": target_ff_count,
            "site_counts": range_payload.get("site_counts"),
        }
        # Stash the raw fabric region alongside the computed params so the
        # caller can register it in applied_pblock_regions once the pblock
        # is actually, successfully applied (not just computed).
        computed["_fabric_region"] = region
        logger.info("Computed pblock ranges via RapidWright: %s", ranges)
        return computed, None

    @staticmethod
    def _clockregion_ranges_cover(ranges: str, cluster_regions: set[str]) -> bool:
        """True if any cluster clock region (e.g. 'X1Y4') falls inside any
        CLOCKREGION_XaYb[:CLOCKREGION_XcYd] span in `ranges`. Unparsable
        ranges return True (don't block on formats we don't understand)."""
        spans: list[tuple[int, int, int, int]] = []
        for match in re.finditer(
            r"CLOCKREGION_X(\d+)Y(\d+)(?:\s*:\s*CLOCKREGION_X(\d+)Y(\d+))?", ranges
        ):
            x0, y0 = int(match.group(1)), int(match.group(2))
            x1 = int(match.group(3)) if match.group(3) else x0
            y1 = int(match.group(4)) if match.group(4) else y0
            spans.append((min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)))
        if not spans:
            return True
        for region in cluster_regions:
            region_match = re.fullmatch(r"X(\d+)Y(\d+)", region.strip())
            if not region_match:
                continue
            x, y = int(region_match.group(1)), int(region_match.group(2))
            for x_min, x_max, y_min, y_max in spans:
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True
        return False

    def _find_pblock_overlap(self, region: dict) -> Optional[dict]:
        """Return the first previously-applied pblock region that overlaps `region`, or None."""
        try:
            col_min, col_max = int(region["col_min"]), int(region["col_max"])
            row_min, row_max = int(region["row_min"]), int(region["row_max"])
        except (KeyError, TypeError, ValueError):
            return None
        for applied in self.applied_pblock_regions:
            applied_region = applied.get("region") or {}
            try:
                a_col_min, a_col_max = int(applied_region["col_min"]), int(applied_region["col_max"])
                a_row_min, a_row_max = int(applied_region["row_min"]), int(applied_region["row_max"])
            except (KeyError, TypeError, ValueError):
                continue
            cols_overlap = col_min <= a_col_max and a_col_min <= col_max
            rows_overlap = row_min <= a_row_max and a_row_min <= row_max
            if cols_overlap and rows_overlap:
                return applied
        return None

    # UltraScale/UltraScale+ SLICE composition, used below to approximate LUT/FF
    # capacity from a SLICE count when the fabric-analysis tool only reports
    # site-type counts (see _check_pblock_utilization). These are architecture
    # constants, not measured per-design values -- if RapidWrightMCP's tool
    # ever starts returning explicit lut_capacity/ff_capacity fields directly,
    # prefer those over this approximation.
    LUTS_PER_SLICE = 8
    FFS_PER_SLICE = 16

    def _check_pblock_utilization(
        self,
        site_counts: dict,
        target_lut_count: int,
        target_ff_count: int,
        target_dsp_count: int,
        target_bram_count: int,
    ) -> Optional[tuple[str, str]]:
        """Return (label, error message) if the requested targets would
        over-pack the region, else None. The label lets the caller decide
        whether shrinking the request can actually fix this (pipeline audit
        2026-07-28: it couldn't tell BRAM/DSP overshoot from LUT/FF overshoot,
        so it kept shrinking target_lut_count/target_ff_count for THREE more
        retries on a BRAM/DSP failure those targets have no effect on --
        "Rejected pblock region: projected BRAM utilization 464%... gave up
        after 4 sizing attempts" on boom_soc, always failing the same way)."""
        if not site_counts:
            # No capacity data returned -- nothing to validate against, let it through.
            return None
        # BUG FIX: this previously only looked for "lut_capacity"/"luts",
        # "ff_capacity"/"ffs", "dsp_capacity"/"dsps", "bram_capacity"/"brams" --
        # none of which are keys RapidWrightMCP's analyze_fabric_for_pblock tool
        # actually returns. In practice it returns site-type counts like
        # {"SLICE": 709, "DSP48E2": 0, "RAMB18": 0, "RAMB36": 0, "URAM288": 0},
        # so every lookup above was always None, "if not requested or not
        # capacity: continue" always fired, and this congestion-safety guard
        # never once actually validated a region -- confirmed by the history
        # of a prior run where a region with 709 SLICEs (~5,672 LUT / ~11,344
        # FF capacity) was recommended against a 20,000 LUT / 40,000 FF
        # target (~3.5x over capacity on both) and passed straight through
        # uncaught. Derive LUT/FF capacity from SLICE count using standard
        # UltraScale+ composition when explicit capacity fields aren't present.
        slice_count = site_counts.get("SLICE")
        lut_capacity = site_counts.get("lut_capacity") or site_counts.get("luts")
        ff_capacity = site_counts.get("ff_capacity") or site_counts.get("ffs")
        if lut_capacity is None and slice_count:
            lut_capacity = slice_count * self.LUTS_PER_SLICE
        if ff_capacity is None and slice_count:
            ff_capacity = slice_count * self.FFS_PER_SLICE
        dsp_capacity = (
            site_counts.get("dsp_capacity")
            or site_counts.get("dsps")
            or site_counts.get("DSP48E2")
        )
        bram_capacity = (
            site_counts.get("bram_capacity")
            or site_counts.get("brams")
            or (
                (site_counts.get("RAMB36") or 0) + (site_counts.get("RAMB18") or 0)
            ) or None
        )
        checks = [
            ("LUT", target_lut_count, lut_capacity),
            ("FF", target_ff_count, ff_capacity),
            ("DSP", target_dsp_count, dsp_capacity),
            ("BRAM", target_bram_count, bram_capacity),
        ]
        for label, requested, capacity in checks:
            if not requested or not capacity:
                continue
            utilization = requested / float(capacity)
            if utilization > PBLOCK_MAX_UTILIZATION_FRACTION:
                return label, (
                    f"Rejected pblock region: projected {label} utilization "
                    f"{utilization:.0%} exceeds the {PBLOCK_MAX_UTILIZATION_FRACTION:.0%} "
                    f"congestion-safety limit ({requested} requested vs {capacity} available). "
                    f"Retry with a larger region or lower target_{label.lower()}_count."
                )
        return None

    def _register_applied_pblock(self, region: Optional[dict], pblock_name: Optional[str]) -> None:
        """Record a successfully-applied pblock region so future regions can be checked for overlap."""
        if not region:
            return
        required_region_keys = ("col_min", "col_max", "row_min", "row_max")
        if not all(key in region for key in required_region_keys):
            return
        self.applied_pblock_regions.append({
            "region": dict(region),
            "pblock_name": pblock_name,
            "iteration": self.iteration,
        })

    def _parse_pblock_utilization_counts(self, util_text: str) -> dict[str, int]:
        """Parse the '1.5x Multiplier' section of report_utilization_for_pblock."""
        counts: dict[str, int] = {}
        section = util_text
        marker = "1.5x Multiplier"
        idx = util_text.find(marker)
        if idx >= 0:
            section = util_text[idx:]
        for label, key in (("LUTs", "LUT"), ("FFs", "FF"), ("DSPs", "DSP"), ("BRAMs", "BRAM"), ("URAMs", "URAM")):
            match = re.search(rf"{label}:\s*([\d,]+)", section)
            if match:
                counts[key] = int(match.group(1).replace(",", ""))
        return counts

    async def _execute_pblock_full_replace(self, params: dict) -> str:
        """Whole-design pblock re-place: size a region for the entire design,
        drop all existing pblocks, unplace, constrain everything into the new
        region, then re-place and re-route. Mirrors the test-mode LogicNets
        recipe that agent mode previously could not express."""
        if not await self._check_implementation_license():
            return self._failure_json(
                "vivado_license_failure",
                "Vivado Implementation license is unavailable; pblock_full_replace requires place/route.",
                command="pblock_full_replace",
            )
        self.last_recipe = "pblock_full_replace"
        self.last_batch_size = 1
        ranges = params.get("ranges")
        region: Optional[dict] = None
        self.last_pblock_sizing = None

        if not ranges:
            util_text = await self.call_tool("vivado_report_utilization_for_pblock", {}, internal=True)
            counts = self._parse_pblock_utilization_counts(util_text)
            target_lut_count = int(params.get("target_lut_count") or counts.get("LUT") or 0)
            target_ff_count = int(params.get("target_ff_count") or counts.get("FF") or 0)
            target_dsp_count = int(params.get("target_dsp_count") or counts.get("DSP") or 0)
            target_bram_count = int(params.get("target_bram_count") or counts.get("BRAM") or 0)
            if target_lut_count > 0 and target_ff_count <= 0:
                # Run 20260712_051231 iter 3: the utilization report returned
                # a real LUT count but FFs: 0 (server-side parse miss -- the
                # design plainly has thousands of registers). Don't fail the
                # whole recipe over the secondary number; estimate FFs from
                # LUTs (UltraScale+ slices carry 2 FFs per LUT, so 2x is a
                # safe region-sizing upper bound).
                target_ff_count = target_lut_count * 2
                logger.warning(
                    "Utilization report gave FF count 0 with %d LUTs; estimating "
                    "%d FFs (2x LUTs) for pblock sizing.",
                    target_lut_count, target_ff_count,
                )
            if target_lut_count <= 0:
                return self._failure_json(
                    "full_replace_utilization_unavailable",
                    f"Could not derive whole-design LUT count from utilization report: {util_text[:300]}",
                    command="pblock_full_replace",
                )
            self.last_pblock_sizing = {
                "target_lut_count": target_lut_count,
                "target_ff_count": target_ff_count,
            }
            fabric_text = await self.call_tool(
                "rapidwright_analyze_fabric_for_pblock",
                {
                    "target_lut_count": target_lut_count,
                    "target_ff_count": target_ff_count,
                    "target_dsp_count": target_dsp_count,
                    "target_bram_count": target_bram_count,
                },
                internal=True,
            )
            fabric = self._parse_json_result(fabric_text)
            if self._result_has_error(fabric):
                return self._failure_json(
                    "full_replace_fabric_analysis_failed",
                    f"RapidWright fabric analysis failed: {fabric.get('error') or fabric_text[:300]}",
                    command="pblock_full_replace",
                )
            region = fabric.get("recommended_region") or {}
            if not all(key in region for key in ("col_min", "col_max", "row_min", "row_max")):
                return self._failure_json(
                    "full_replace_fabric_analysis_failed",
                    f"RapidWright fabric analysis did not return recommended_region: {fabric_text[:300]}",
                    command="pblock_full_replace",
                )
            range_text = await self.call_tool(
                "rapidwright_convert_fabric_region_to_pblock",
                {
                    "col_min": int(region["col_min"]),
                    "col_max": int(region["col_max"]),
                    "row_min": int(region["row_min"]),
                    "row_max": int(region["row_max"]),
                    "use_clock_regions": bool(params.get("use_clock_regions", False)),
                },
                internal=True,
            )
            range_payload = self._parse_json_result(range_text)
            if self._result_has_error(range_payload):
                return self._failure_json(
                    "full_replace_range_conversion_failed",
                    f"RapidWright pblock range conversion failed: {range_payload.get('error') or range_text[:300]}",
                    command="pblock_full_replace",
                )
            ranges = range_payload.get("pblock_ranges")
            if not ranges:
                return self._failure_json(
                    "full_replace_range_conversion_failed",
                    f"RapidWright pblock range conversion returned no pblock_ranges: {range_text[:300]}",
                    command="pblock_full_replace",
                )

        pblock_name = str(params.get("pblock_name") or f"pblock_full_replace_{self.iteration:03d}")
        self.last_targets = ["full_design", str(ranges)]
        self.last_rapidwright_edit_summary = {
            "action": "pblock_full_replace",
            "cells_moved": 0,
            "nets_affected": 0,
            "pblock_ranges": ranges,
            "fabric_region": region,
        }

        async def _restore_best_after_failure() -> None:
            # The design is unplaced/half-implemented at this point; put the
            # live Vivado session back on the best known checkpoint so the
            # next iteration doesn't operate on a broken state.
            if self.checkpoint_manager is not None:
                best_ckpt = self.checkpoint_manager.get_best_checkpoint()
                if best_ckpt:
                    await self.call_tool(
                        "vivado_open_checkpoint", {"dcp_path": best_ckpt, "timeout": 600}, internal=True
                    )

        # A whole-design re-place supersedes every pblock applied so far:
        # delete them all and reset the overlap bookkeeping so the fresh
        # region can never collide with stale state.
        delete_result = await self.call_tool(
            "vivado_run_tcl",
            {"command": "if {[llength [get_pblocks -quiet]] > 0} {delete_pblocks [get_pblocks]}"},
            internal=True,
        )
        self.applied_pblock_regions.clear()
        self.pblock_region_cooldown_until_iter = -1

        unplace_result = await self.call_tool(
            "vivado_run_tcl", {"command": "place_design -unplace", "timeout": 600}, internal=True
        )
        if self._vivado_output_has_error(unplace_result):
            await _restore_best_after_failure()
            return self._failure_json(
                "full_replace_unplace_failed",
                f"place_design -unplace failed: {unplace_result[-FAILURE_MESSAGE_CAPTURE_CHARS:]}",
                command="pblock_full_replace",
            )

        # BUG FIX (root cause of the failed full_replace iterations in run
        # 20260711, iters 4/6/9): this call was missing internal=True, so
        # call_tool() intercepted it and routed it through
        # _maybe_run_pblock_or_phys_opt() -- which re-classified the worst
        # path and ran the phys_opt cascade ON THE DESIGN WE JUST UNPLACED
        # (vivado.jou: `place_design -unplace` followed by five failing
        # phys_opt_design attempts). The phys_opt failure text then wasn't
        # JSON, so _parse_json_result() returned {}, the error check passed,
        # and place+route ran WITHOUT any pblock ever being created -- a
        # random whole-design re-place recorded as vivado_command_failure.
        # internal=True sends the call straight to the Vivado server.
        #
        # Region-grow retry (run 20260806_193354, rosetta_optical-flow iter
        # 7): the first full_replace this design ever attempted failed
        # validation by a shortage of TWO FIFO sites ("FIFO: requires 122,
        # only 120 available") and the whole recipe -- the same one that
        # broke rosetta_spam-filter's zero-improvement streak -- was
        # abandoned. When validation fails and we know the region geometry
        # (auto-sized path, not LLM-supplied explicit ranges), grow the
        # region and re-apply instead of giving up: every site type's
        # capacity scales with the window, so a small shortage is almost
        # always covered by one growth step.
        for grow_attempt in range(FULL_REPLACE_REGION_GROW_RETRIES + 1):
            apply_result = await self.call_tool(
                "vivado_create_and_apply_pblock",
                {
                    "pblock_name": pblock_name,
                    "ranges": ranges,
                    "apply_to": "current_design",
                    "is_soft": False,
                },
                internal=True,
            )
            apply_payload = self._parse_json_result(apply_result)
            if self._result_has_error(apply_payload) or self._action_failure(
                apply_result, default_command="vivado_create_and_apply_pblock"
            ):
                logger.error(f"pblock_full_replace apply failed - full result:\n{apply_result}")
                await _restore_best_after_failure()
                return self._failure_json(
                    apply_payload.get("error_type", "full_replace_pblock_failed"),
                    apply_payload.get("message", apply_result[-FAILURE_MESSAGE_CAPTURE_CHARS:]),
                    command="pblock_full_replace",
                )

            # History(16) iter 4: LLM-supplied explicit ranges packed 31370
            # LUTs into a region with 26880 -- the DRC flagged it BEFORE
            # placement, but we placed anyway and burned minutes on a doomed
            # run. The apply payload carries that DRC result; act on it while
            # the failure is still cheap.
            resource_validation = apply_payload.get("resource_validation") or {}
            validation_errors = list(resource_validation.get("errors") or [])
            if not validation_errors:
                break  # region fits -- proceed to place/route

            can_grow = (
                grow_attempt < FULL_REPLACE_REGION_GROW_RETRIES
                and region is not None
                and all(key in region for key in ("col_min", "col_max", "row_min", "row_max"))
            )
            if not can_grow:
                logger.error(
                    "pblock_full_replace resource validation failed pre-placement: %s",
                    validation_errors,
                )
                await _restore_best_after_failure()
                return self._failure_json(
                    "full_replace_region_too_small",
                    (
                        "Pblock resource validation failed before placement: "
                        + "; ".join(str(err)[:200] for err in validation_errors[:3])
                        + f" (after {grow_attempt} region-grow retr{'y' if grow_attempt == 1 else 'ies'}). "
                        "Increase the ranges (or target_lut_count/target_ff_count) and retry."
                    ),
                    command="pblock_full_replace",
                )

            # Delete the too-small pblock, expand the window, re-convert.
            await self.call_tool(
                "vivado_run_tcl",
                {"command": f"if {{[llength [get_pblocks -quiet {pblock_name}]] > 0}} "
                            f"{{delete_pblocks [get_pblocks {pblock_name}]}}"},
                internal=True,
            )
            row_span = int(region["row_max"]) - int(region["row_min"])
            col_span = int(region["col_max"]) - int(region["col_min"])
            row_pad = max(1, int(row_span * FULL_REPLACE_REGION_GROW_FRACTION))
            col_pad = max(1, int(col_span * FULL_REPLACE_REGION_GROW_FRACTION))
            region = {
                **region,
                "row_min": max(0, int(region["row_min"]) - row_pad),
                "row_max": int(region["row_max"]) + row_pad,
                "col_min": max(0, int(region["col_min"]) - col_pad),
                "col_max": int(region["col_max"]) + col_pad,
            }
            logger.warning(
                "pblock_full_replace region too small (%s); growing window to "
                "cols %d-%d rows %d-%d and retrying (%d/%d).",
                "; ".join(str(err)[:120] for err in validation_errors[:2]),
                region["col_min"], region["col_max"], region["row_min"], region["row_max"],
                grow_attempt + 1, FULL_REPLACE_REGION_GROW_RETRIES,
            )
            range_text = await self.call_tool(
                "rapidwright_convert_fabric_region_to_pblock",
                {
                    "col_min": int(region["col_min"]),
                    "col_max": int(region["col_max"]),
                    "row_min": int(region["row_min"]),
                    "row_max": int(region["row_max"]),
                    "use_clock_regions": bool(params.get("use_clock_regions", False)),
                },
                internal=True,
            )
            range_payload = self._parse_json_result(range_text)
            grown_ranges = None if self._result_has_error(range_payload) else range_payload.get("pblock_ranges")
            if not grown_ranges:
                logger.error(
                    "pblock_full_replace region-grow re-conversion failed: %s", range_text[:300]
                )
                await _restore_best_after_failure()
                return self._failure_json(
                    "full_replace_region_too_small",
                    (
                        "Pblock resource validation failed before placement: "
                        + "; ".join(str(err)[:200] for err in validation_errors[:3])
                        + ". A region-grow retry was attempted but range re-conversion failed."
                    ),
                    command="pblock_full_replace",
                )
            ranges = grown_ranges
            self.last_targets = ["full_design", str(ranges)]
            self.last_rapidwright_edit_summary["pblock_ranges"] = ranges
            self.last_rapidwright_edit_summary["fabric_region"] = region
            self.last_rapidwright_edit_summary["region_grow_attempts"] = grow_attempt + 1

        place_result = await self.call_tool(
            "vivado_place_design",
            {"directive": str(params.get("place_directive") or "Default"), "timeout": self._implementation_timeout_s(kind="place")},
            internal=True,
        )
        if self._action_failure(place_result, default_command="vivado_place_design"):
            logger.error(f"pblock_full_replace place failed - full output:\n{place_result}")
            await _restore_best_after_failure()
            return self._failure_json(
                "full_replace_place_failed",
                f"Whole-design pblock applied but placement failed: {place_result[-FAILURE_MESSAGE_CAPTURE_CHARS:]}",
                command="pblock_full_replace",
            )
        # Item 5: re-check the budget now that the place has been paid for.
        route_directive = self._maybe_downgrade_route_directive(
            str(params.get("route_directive") or "Default"), "pblock_full_replace"
        )
        route_result = await self.call_tool(
            "vivado_route_design",
            {"directive": route_directive, "timeout": self._implementation_timeout_s(kind="route")},
            internal=True,
        )
        if (
            self._action_failure(route_result, default_command="vivado_route_design")
            and route_directive != "Default"
            and self._is_post_route_physsynth_crash(route_result)
        ):
            # See _is_post_route_physsynth_crash: Vivado's own post-route
            # re-optimization pass crashing, not a real placement/routing
            # failure -- retry once with Default before giving up.
            logger.warning(
                "pblock_full_replace route_design hit the post-route physical-synthesis "
                "crash (13HDPLException) under -directive %s; retrying once with -directive Default.",
                route_directive,
            )
            route_result = await self.call_tool(
                "vivado_route_design",
                {"directive": "Default", "timeout": self._implementation_timeout_s(kind="route")},
                internal=True,
            )
        if self._action_failure(route_result, default_command="vivado_route_design"):
            logger.error(f"pblock_full_replace route failed - full output:\n{route_result}")
            await _restore_best_after_failure()
            return self._failure_json(
                "full_replace_route_failed",
                f"Whole-design pblock placed but routing failed: {route_result[-FAILURE_MESSAGE_CAPTURE_CHARS:]}",
                command="pblock_full_replace",
            )
        self._register_applied_pblock(region, pblock_name)
        return "\n\n".join(
            part for part in (delete_result, apply_result, place_result, route_result) if part
        )

    def _summarize_cell_placement(self, payload: dict, requested_cells: list[str]) -> dict:
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        moved = [item for item in results if item.get("status") == "success"]
        return {
            "action": "rapidwright_optimize_cell_placement",
            "requested_cells": requested_cells,
            "cells_processed": payload.get("cells_processed", len(requested_cells)),
            "cells_moved": payload.get("cells_moved", len(moved)),
            "moved_cells": [item.get("cell") for item in moved if item.get("cell")],
            "nets_affected": sum(int(item.get("affected_nets") or 0) for item in moved),
            "results": results,
        }

    def _summarize_fanout_split(self, payload: dict, net_name: str, split_factor: int) -> dict:
        return {
            "action": "fanout_split",
            "net_name": net_name,
            "split_factor": split_factor,
            "status": payload.get("status"),
            "cells_moved": int(payload.get("cells_moved") or payload.get("replicas_created") or 0),
            "nets_affected": int(payload.get("nets_affected") or len(payload.get("new_nets") or [])),
            "new_nets": payload.get("new_nets"),
            "replicas_created": payload.get("replicas_created"),
            "message": payload.get("message"),
        }

    def _summarize_pblock_assignment(self, payload: dict, params: dict) -> dict:
        cells_assigned = int(payload.get("cells_assigned") or payload.get("cells_matched") or 0)
        return {
            **(self.last_rapidwright_edit_summary or {}),
            "action": "pblock",
            "changed_design": cells_assigned > 0,
            "cells_moved": cells_assigned,
            "nets_affected": 0,
            "pblock_name": payload.get("pblock_name") or params.get("pblock_name"),
            "pblock_ranges": payload.get("ranges") or params.get("ranges"),
            "apply_to": payload.get("apply_to") or params.get("apply_to", "current_design"),
            "is_soft": payload.get("is_soft") if "is_soft" in payload else params.get("is_soft", False),
            "cells_matched": int(payload.get("cells_matched") or 0),
            "cells_assigned": cells_assigned,
            "resource_validation": payload.get("resource_validation"),
        }

    async def _measure_cluster_spread(self, cell_names: list[str]) -> Optional[dict]:
        """
        Fix #4 helper (cluster-level optimization guard): measure how spread
        out a set of cells targeted together for a move currently is, using
        the same RapidWright spread-analysis tool already used during
        initial design analysis (rapidwright_analyze_critical_path_spread).

        Returns a dict with avg_distance/max_distance, or None if the
        measurement could not be taken (in which case callers should not
        block the action -- absence of data isn't evidence of regression).
        """
        if not cell_names:
            return None
        try:
            cluster_file = Path(self.temp_dir) / f"cluster_spread_iter_{self.iteration:03d}.json"
            cluster_file.write_text(json.dumps([cell_names]))
            spread_result = await self.call_tool(
                "rapidwright_analyze_critical_path_spread",
                {"input_file": str(cluster_file)},
                internal=True,
            )
            spread_data = self._parse_json_result(spread_result)
            if not spread_data or self._result_has_error(spread_data):
                return None
            return {
                "avg_distance": spread_data.get("avg_max_distance"),
                "max_distance": spread_data.get("max_distance_found"),
            }
        except Exception as exc:
            logger.warning("Could not measure cluster spread for %s: %s", cell_names, exc)
            return None

    async def _route_candidate_with_eco(self, arguments: dict) -> str:
        """Apply shields and run preserve-fixed-route ECO routing for RW candidates."""
        if self.eco_router is None:
            return await self.call_tool("vivado_route_design", arguments, internal=True)

        moved_cells: list[str] = []
        preserved_nets: list[str] = []
        # Fix #1: gate on last_action_key (never renamed) rather than the
        # display-only last_recipe, which _remember_recipe() rewrites to
        # "rapidwright_cell_placement" for this same action.
        if self.last_action_key == "rapidwright_optimize_cell_placement":
            moved_cells = list(self.last_targets)

        shield_result = await self._apply_shields_async(moved_cells, preserved_nets)
        logger.info("Shield iteration result: %s", shield_result)

        route_status = await self.call_tool("vivado_report_route_status", {"timeout": 300}, internal=True)
        parsed = ECORouter.parse_route_status(route_status)
        design_cell_count = int(self.last_design_info.get("cell_count") or 0)
        unrouted_net_count = int(parsed.get("unrouted_nets") or 0)

        route_result = await self.eco_router.route_with_fallback_async(
            design_cell_count=design_cell_count,
            unrouted_net_count=unrouted_net_count,
        )
        self.last_route_result = route_result
        self.pending_candidate_checkpoint = None

        return (
            "ECO routing complete.\n"
            f"Directive used: {route_result.directive_used}\n"
            f"Fallback full route used: {route_result.fallback_used}\n"
            f"Routing errors: {route_result.routing_errors}\n"
            f"Unrouted nets: {route_result.unrouted_nets}\n"
            f"Fully routed: {route_result.fully_routed}\n"
            f"Timeout: {route_result.timeout_s}s\n\n"
            f"{route_result.vivado_raw_output}"
        )

    async def _apply_shields_async(self, moved_cells: list[str], preserved_nets: list[str]) -> dict:
        blacklist = self.checkpoint_manager.get_blacklist() if self.checkpoint_manager else {"cells": [], "nets": []}
        cells = [cell for cell in moved_cells if cell not in blacklist.get("cells", [])]
        nets = [net for net in preserved_nets if net not in blacklist.get("nets", [])]
        cells_skipped = len(moved_cells) - len(cells)
        nets_skipped = len(preserved_nets) - len(nets)
        calls = 0

        release_tcl = (
            "set locked_cells [get_cells -filter {IS_LOC_FIXED == 1}]\n"
            "if {[llength $locked_cells] > 0} {\n"
            "    set_property IS_LOC_FIXED 0 $locked_cells\n"
            "    set_property IS_BEL_FIXED 0 $locked_cells\n"
            "}\n"
            "set locked_nets [get_nets -filter {DONT_TOUCH == 1}]\n"
            "if {[llength $locked_nets] > 0} {\n"
            "    set_property DONT_TOUCH 0 $locked_nets\n"
            "}\n"
        )
        await self.call_tool("vivado_run_tcl", {"command": release_tcl, "timeout": 120}, internal=True)
        calls += 1

        if cells:
            quoted_cells = " ".join(escape_tcl_name(cell) for cell in cells)
            cell_tcl = (
                f"set objs [get_cells {{{quoted_cells}}}]\n"
                "set_property IS_LOC_FIXED 1 $objs\n"
                "set_property IS_BEL_FIXED 1 $objs"
            )
            await self.call_tool("vivado_run_tcl", {"command": cell_tcl, "timeout": 120}, internal=True)
            calls += 1

        if nets:
            quoted_nets = " ".join(escape_tcl_name(net) for net in nets)
            net_tcl = f"set objs [get_nets {{{quoted_nets}}}]\nset_property DONT_TOUCH 1 $objs"
            await self.call_tool("vivado_run_tcl", {"command": net_tcl, "timeout": 120}, internal=True)
            calls += 1

        return {
            "cells_locked": len(cells),
            "nets_locked": len(nets),
            "cells_skipped": cells_skipped,
            "nets_skipped": nets_skipped,
            "tcl_calls_made": calls,
        }

    async def _after_tool_success(
        self,
        tool_name: str,
        arguments: dict,
        result_text: str,
        wns_measured: Optional[float],
        elapsed_time: float,
    ) -> None:
        """Update local orchestration state after externally requested tools."""
        if not (
            tool_name == "vivado_create_and_apply_pblock"
            and (
                "skipping pblock" in result_text.lower()
                or "pblock deferred" in result_text.lower()
            )
        ):
            self._remember_recipe(tool_name, arguments)

        if tool_name == "rapidwright_read_checkpoint":
            try:
                self.last_design_info = json.loads(result_text)
            except json.JSONDecodeError:
                self.last_design_info = {}
            # The design's cell count is scale evidence (item 2): >100k
            # primitives marks it "large" before any place has been timed.
            self._refresh_design_scale()

        elif tool_name == "rapidwright_write_checkpoint":
            checkpoint = arguments.get("dcp_path")
            if checkpoint:
                self.pending_candidate_checkpoint = str(Path(checkpoint).resolve())

        elif tool_name == "vivado_open_checkpoint":
            checkpoint = arguments.get("dcp_path")
            if checkpoint:
                self.active_checkpoint = str(Path(checkpoint).resolve())

        elif tool_name == "vivado_write_checkpoint":
            checkpoint = arguments.get("dcp_path")
            if checkpoint:
                self.active_checkpoint = str(Path(checkpoint).resolve())

        if wns_measured is not None:
            await self._record_iteration_timing(wns_measured, self.iteration_tool_elapsed_s)

    def _publish_best_to_output(self) -> None:
        """Copy the best checkpoint so far to the contest output DCP path.

        Called after every improvement and on every stop path so the output
        location always holds the best validated design -- the contest
        evaluates whatever is at that path when the wall-clock budget runs
        out, and a missing/stale output scores zero.
        """
        if self.output_dcp_path is None or self.checkpoint_manager is None:
            return
        best = self.checkpoint_manager.get_best_checkpoint()
        if not best:
            return
        source = Path(best)
        if not source.exists():
            logger.warning("Best checkpoint %s does not exist; cannot publish to output.", source)
            return
        try:
            destination = self.output_dcp_path.resolve()
            if source.resolve(strict=False) == destination:
                return
            shutil.copy2(source, destination)
            logger.info("Published best checkpoint %s to output DCP %s", source, destination)
        except OSError as exc:
            logger.error("Failed to publish best checkpoint to output DCP: %s", exc)

    async def _record_iteration_timing(self, wns: float, vivado_runtime_s: float) -> None:
        """Persist timing/checkpoint history for completed optimizer iterations."""
        if self.checkpoint_manager is None:
            return
        if self.iteration <= 0 or self.iteration in self.recorded_iterations:
            return
        if self.last_recipe == "initial":
            return
        try:
            wns = validate_wns_sanity_static(float(wns), self.current_period_ns or self.clock_period, "iteration_history")
        except (TypeError, ValueError, WNSParseError) as e:
            await self._record_wns_parse_error("iteration_history", str(e), str(wns))
            return

        # Timing provenance gate: a WNS observation may only become iteration
        # history (and potentially best_checkpoint) if the live design is
        # verifiably fully placed and routed. The client-side tracker catches
        # the cheap cases; _verify_routed_state() asks Vivado itself, so a
        # half-implemented state left by a buggy/failed action can never be
        # checkpointed as "best" on the strength of estimated timing.
        #
        # Run 20260712 lesson: a rejected observation must be handled as a
        # FAILED ACTION (failure memory penalty, stall counter, LLM feedback)
        # plus a restore of the best checkpoint. The first version of this
        # gate recorded a passive "wns_parse_error" instead -- no rollback, no
        # penalty, nothing the outcome loop counts -- so the optimizer kept
        # re-running the same mutant-producing recipe on top of the mutant
        # design for four straight iterations while learning nothing.
        if self.design_state != "routed":
            self.recorded_iterations.add(self.iteration)
            self._record_failed_action({
                "error_type": "invalid_design_state",
                "command": self.last_action_key,
                "message": (
                    f"Action completed but design state is '{self.design_state}', not routed; "
                    f"timing observation {wns:.3f} ns rejected."
                ),
            })
            await self._restore_best_state(
                f"design state '{self.design_state}' after {self.last_action_key}"
            )
            return
        routed_ok, routed_detail = await self._verify_routed_state()
        if not routed_ok:
            self.recorded_iterations.add(self.iteration)
            self._record_failed_action({
                "error_type": "invalid_design_state",
                "command": self.last_action_key,
                "message": (
                    f"Action completed but left the design in an invalid state ({routed_detail}); "
                    f"timing observation {wns:.3f} ns rejected."
                ),
            })
            await self._restore_best_state(
                f"invalid design state after {self.last_action_key} ({routed_detail})"
            )
            return

        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"iter_{self.iteration:03d}.dcp"
        period_for_record = self.current_period_ns or self.clock_period
        achieved_fmax = self.calculate_fmax(wns, period_for_record)
        if achieved_fmax is not None and (self.best_fmax_mhz is None or achieved_fmax > self.best_fmax_mhz):
            self.best_fmax_mhz = achieved_fmax

        write_result = await self.call_tool(
            "vivado_write_checkpoint",
            {"dcp_path": str(checkpoint_path), "force": True, "timeout": 600},
            internal=True,
        )
        if "error" in write_result.lower():
            logger.warning(f"Could not write iteration checkpoint: {write_result[:300]}")
            return

        if period_for_record is not None:
            self.checkpoint_manager.clock_period_ns = period_for_record
        # Keep the persisted benchmark_score's beta term current: record()
        # snapshots llm_cost_usd into history.json.
        self.checkpoint_manager.llm_cost_usd = max(0.0, float(self.total_cost or 0.0))
        iteration = self.checkpoint_manager.record(
            recipe=self.last_recipe,
            targets=self.last_targets,
            wns_after=wns,
            vivado_runtime_s=vivado_runtime_s,
            checkpoint_path=str(checkpoint_path),
            batch_size=self.last_batch_size,
        )
        self.iteration_tool_elapsed_s = 0.0
        self.no_improvement_count = self.checkpoint_manager.stall_count
        self.consecutive_no_improvement = self.checkpoint_manager.stall_count
        self.last_recorded_wns = wns
        self._note_recipe_outcome(str(iteration.get("status")), wns)
        if iteration.get("status") in {"improved", "marginal"}:
            self._publish_best_to_output()
        # Fix #1: gate the reset/remember-failure bookkeeping on
        # last_action_key (the never-renamed dispatch key), not last_recipe
        # (a display label _remember_recipe() may have rewritten). Before
        # this fix, e.g. rapidwright_optimize_cell_placement's failures were
        # recorded under "rapidwright_cell_placement", a key that nothing
        # reading action_failure_memory during action-selection ever checks,
        # so a 100%-regression action was never actually suppressed.
        action_key = self.last_action_key or self.last_recipe
        # BUG FIX: analysis-only actions (rapidwright_analyze_net_detour,
        # rapidwright_analyze_fabric_for_pblock,
        # rapidwright_convert_fabric_region_to_pblock) never change the
        # design by themselves (they set last_rapidwright_edit_summary's
        # "changed_design" to False) - they exist purely to compute inputs
        # for a *later* real action. Judging them by "did WNS improve" and
        # feeding a "no improvement" verdict into the same failure-memory
        # mechanism used for genuine regressions was mis-attributing an
        # always-true tautology (an analysis step never moves WNS) as a
        # failure. In practice this exhausted the pblock family (which
        # lacked any offsetting self-reset) after only 3 routine analysis
        # calls, while rapidwright_analyze_net_detour happened to dodge this
        # by resetting its own counter right after execution - an
        # inconsistency that isn't a real fix. Skip this bookkeeping
        # entirely for actions that never attempted a design change.
        is_analysis_only = bool(
            self.last_rapidwright_edit_summary
            and self.last_rapidwright_edit_summary.get("changed_design") is False
        )
        if is_analysis_only:
            pass
        elif iteration.get("status") in {"improved", "marginal"}:
            self._reset_action_failure_memory(action_key)
        else:
            # The action executed successfully but made WNS worse (or did
            # nothing useful) and got rolled back. This used to only bump the
            # generic stall counter, which doesn't discriminate between
            # actions - an action that empirically regresses this exact
            # target set could keep getting picked again every iteration.
            # Route it through the same failure-memory/cooldown mechanism
            # used for hard tool failures so it actually gets suppressed
            # after repeated regressions on the same targets.
            self._remember_no_action_failure(action_key, self.last_targets)

        history_fields = {
            "target_tier": self.target_tier,
            "target_candidate_count": len(self.current_target_candidates),
            "path_delay_classification": self.path_delay_classification,
            "path_delay_breakdown": self.path_delay_breakdown,
            "clock_period_ns": period_for_record,
            "achieved_fmax_mhz": achieved_fmax,
            "no_improvement_count": self.no_improvement_count,
            "consecutive_no_improvement": self.consecutive_no_improvement,
            "action_failure_memory": self._serializable_action_failure_memory(),
            "action_failure_counts": self._serializable_action_failure_counts(),
            **self.last_decision_trace,
        }
        if self.last_rapidwright_edit_summary is not None:
            history_fields["rapidwright_edit_summary"] = self.last_rapidwright_edit_summary
        self._annotate_latest_history(history_fields)
        self.recorded_iterations.add(self.iteration)

        if self.checkpoint_manager.should_rollback():
            best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
            logger.info(f"Iteration regressed; rolling back to best checkpoint: {best_checkpoint}")
            await self.call_tool(
                "vivado_open_checkpoint",
                {"dcp_path": best_checkpoint, "timeout": 600},
                internal=True,
            )
            if self.checkpoint_manager.best_checkpoint == best_checkpoint:
                self.active_checkpoint = best_checkpoint
            # CRITICAL: rapidwright_read_checkpoint is only ever called once, during
            # perform_initial_analysis. Without reloading here, RapidWright's
            # in-memory _current_design keeps every previously-attempted (including
            # just-rejected) cell move baked in, completely diverged from the
            # checkpoint Vivado just reopened. Every subsequent RapidWright-side
            # action (cell placement, spread measurement) would then be computed
            # against a design that has nothing to do with "best" -- compounding
            # damage from a rolled-back state instead of ever actually resetting.
            reload_result = await self.call_tool(
                "rapidwright_read_checkpoint",
                {"dcp_path": best_checkpoint},
                internal=True,
            )
            if "error" in reload_result.lower() and "success" not in reload_result.lower():
                logger.error(
                    "Failed to re-sync RapidWright's design state to the rolled-back "
                    "checkpoint %s: %s. Subsequent RapidWright-side actions this run "
                    "may be operating on stale/diverged design state.",
                    best_checkpoint, reload_result[:300],
                )
            else:
                logger.info("Re-synced RapidWright design state to rolled-back checkpoint %s", best_checkpoint)

        if self.checkpoint_manager.should_escalate():
            message = (
                f"Checkpoint manager observed repeated stalls. Current summary: "
                f"{self.checkpoint_manager.summary()} Try a different recipe or target set."
            )
            self.messages.append({"role": "user", "content": message})

        logger.info(f"Recorded optimization iteration: {iteration}")
        if wns >= 0 and not self.bisection_active:
            await self._run_clock_bisection_after_closure(wns)
    
    async def process_response(self, response) -> tuple[str, bool]:
        """Process LLM response, execute tool calls, return final text and done flag."""
        # Validate response structure with detailed logging
        try:
            if not response:
                raise ValueError("Response is None")
            if not hasattr(response, 'choices'):
                raise ValueError(f"Response has no 'choices' attribute. Response type: {type(response)}, Response: {response}")
            if response.choices is None:
                raise ValueError("Response.choices is None")
            if len(response.choices) == 0:
                raise ValueError("Response choices list is empty")
            
            message = response.choices[0].message
            if not message:
                raise ValueError("Message is None")
        except Exception as e:
            logger.error(f"Failed to parse response structure: {e}")
            logger.error(f"Response object: {response}")
            raise
        
        # Convert message to dict, excluding None values which can cause issues
        message_dict = message.model_dump(exclude_none=True)
        self.messages.append(message_dict)
        
        if self.debug:
            logger.debug(f"Added message to conversation: {json.dumps(message_dict, indent=2)[:500]}...")
        
        # Check for tool calls
        if message.tool_calls:
            tool_results = []
            
            for tool_call in message.tool_calls:
                # Validate tool_call structure
                if not tool_call or not hasattr(tool_call, 'function') or not tool_call.function:
                    logger.warning(f"Invalid tool_call structure: {tool_call}")
                    continue
                
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    tool_args = {}
                
                result = await self.call_tool(tool_name, tool_args)
                
                # Truncate very long results to avoid API issues
                MAX_RESULT_LENGTH = 50000  # characters
                if len(result) > MAX_RESULT_LENGTH:
                    logger.warning(f"Tool result from {tool_name} is {len(result)} chars, truncating to {MAX_RESULT_LENGTH}")
                    result = result[:MAX_RESULT_LENGTH] + f"\n...[truncated {len(result) - MAX_RESULT_LENGTH} characters]"
                
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result
                })
                
                # Debug logging
                if self.debug:
                    logger.debug(f"Tool {tool_name} result: {result[:500]}...")
            
            # Add tool results to messages
            self.messages.extend(tool_results)
            
            # Continue conversation
            return await self.get_completion()
        
        # No tool calls - check if we're done
        content = message.content or ""
        
        # Check for completion indicators
        is_done = any(phrase in content.lower() for phrase in [
            "optimization complete",
            "timing is met",
            "wns >= 0",
            "no more optimizations",
            "design meets timing",
            "successfully saved",
            "final design saved"
        ])
        
        return content, is_done
    
    async def perform_initial_analysis(self, input_dcp: Path) -> str:
        """
        Perform initial analysis without LLM:
        1. Initialize RapidWright
        2. Open checkpoint in Vivado
        3. Report timing summary
        4. Get critical high fanout nets
        
        Returns a formatted summary of the analysis.
        """
        logger.info("Performing initial design analysis...")
        print("\n=== Initial Design Analysis ===\n")
        
        # Step 1: Initialize RapidWright
        logger.info("Initializing RapidWright...")
        print("Initializing RapidWright...")
        result = await self.call_tool("rapidwright_initialize_rapidwright", {})
        if "error" in result.lower() and "success" not in result.lower():
            raise RuntimeError(f"Failed to initialize RapidWright: {result}")
        print("✓ RapidWright initialized\n")
        
        # Step 2: Open checkpoint in Vivado
        logger.info(f"Opening checkpoint: {input_dcp}")
        print(f"Opening checkpoint: {input_dcp.name}")
        result = await self.call_tool("vivado_open_checkpoint", {
            "dcp_path": str(input_dcp.resolve())
        })
        if "error" in result.lower() and "opened successfully" not in result.lower():
            raise RuntimeError(f"Failed to open checkpoint: {result}")
        print("✓ Checkpoint opened in Vivado\n")
        
        # Step 3: Report timing summary
        logger.info("Analyzing timing...")
        print("Analyzing timing...")
        timing_report = await self.call_tool("vivado_report_timing_summary", {})
        
        # Parse timing
        timing_info = parse_timing_summary_static(timing_report)
        self.initial_tns = timing_info["tns"]
        self.initial_failing_endpoints = timing_info["failing_endpoints"]
        
        # Get clock period for fmax calculation (also detects target clock)
        self.clock_period = await super().get_clock_period(self._call_vivado_tool)
        
        # Get WNS for the target clock domain
        target_wns = await super().get_wns_for_target_clock(self._call_vivado_tool)
        if target_wns is not None:
            self.initial_wns = target_wns
        else:
            self.initial_wns = timing_info["wns"]
        self.best_wns = self.initial_wns if self.initial_wns is not None else float('-inf')
        self.current_period_ns = self.clock_period
        self.best_fmax_mhz = self.calculate_fmax(self.initial_wns, self.clock_period)
        self.last_passing_period_ns = self.clock_period if self.initial_wns is not None and self.initial_wns >= 0 else None

        await self._run_constraint_audit()
        await self._refresh_target_candidates(self.initial_wns)
        await self._classify_worst_path_delay()
        
        clock_info = f" (clock: {self.target_clock})" if self.target_clock else ""
        print(f"✓ Timing analyzed:")
        if self.clock_period is not None:
            target_fmax = 1000.0 / self.clock_period
            print(f"  - Clock period: {self.clock_period:.3f} ns (target fmax: {target_fmax:.2f} MHz)")
        if self.target_clock:
            print(f"  - Target clock: {self.target_clock}")
        if self.initial_wns is not None:
            print(f"  - WNS{clock_info}: {self.initial_wns:.3f} ns")
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            if initial_fmax is not None:
                print(f"  - Achievable fmax: {initial_fmax:.2f} MHz")
        if self.initial_tns is not None:
            print(f"  - TNS: {self.initial_tns:.3f} ns")
        if self.initial_failing_endpoints is not None:
            print(f"  - Failing endpoints: {self.initial_failing_endpoints}")
        print(f"  - Target tier: {self.target_tier} ({len(self.current_target_candidates)} candidates)")
        print(f"  - Path delay class: {self.path_delay_classification}")
        print()
        
        # Step 4: Get critical high fanout nets
        logger.info("Identifying critical high fanout nets...")
        print("Identifying critical high fanout nets...")
        nets_report = await self.call_tool("vivado_get_critical_high_fanout_nets", {
            "num_paths": 50,
            "min_fanout": 100
        })
        
        # Parse high fanout nets
        self.high_fanout_nets = self.parse_high_fanout_nets(nets_report)
        print(f"✓ Found {len(self.high_fanout_nets)} high fanout nets (>100 fanout)\n")
        
        # Step 5: Load design in RapidWright for spread analysis
        critical_path_spread_info = None  # Initialize
        
        logger.info("Loading design in RapidWright...")
        print("Loading design in RapidWright for spread analysis...")
        result = await self.call_tool("rapidwright_read_checkpoint", {
            "dcp_path": str(input_dcp.resolve())
        })
        if "error" in result.lower() and "success" not in result.lower():
            print(f"⚠ Warning: Could not load design in RapidWright: {result}")
        else:
            print("✓ Design loaded in RapidWright\n")
            
            # Step 6: Extract critical path cells and analyze spread
            logger.info("Extracting and analyzing critical path spread...")
            print("Analyzing critical path spread...")
            
            # Extract critical path cells from Vivado
            temp_path = Path(self.temp_dir) / "initial_critical_paths.json"
            cells_json = await self.call_tool("vivado_extract_critical_path_cells", {
                "num_paths": 50,
                "output_file": str(temp_path)
            })
            
            # Analyze spread in RapidWright
            spread_result = await self.call_tool("rapidwright_analyze_critical_path_spread", {
                "input_file": str(temp_path)
            })
            
            # Parse spread results
            import json
            try:
                spread_data = json.loads(spread_result)
                critical_path_spread_info = {
                    "max_distance": spread_data.get("max_distance_found", 0),
                    "avg_distance": spread_data.get("avg_max_distance", 0),
                    "paths_analyzed": spread_data.get("paths_analyzed", 0)
                }
                self.last_spread_info = dict(critical_path_spread_info)
                print(f"✓ Critical path spread analyzed:")
                print(f"  - Max distance: {critical_path_spread_info['max_distance']} tiles")
                print(f"  - Avg distance: {critical_path_spread_info['avg_distance']:.1f} tiles")
                print(f"  - Paths analyzed: {critical_path_spread_info['paths_analyzed']}")
                print()
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠ Warning: Could not parse spread results: {e}")
                critical_path_spread_info = None
        
        # Create concise summary for LLM
        summary = []
        summary.append("=== Initial Design Analysis ===\n")
        
        # Timing status
        summary.append("TIMING STATUS:")
        if self.clock_period is not None:
            target_fmax = 1000.0 / self.clock_period
            summary.append(f"  Clock period: {self.clock_period:.3f} ns (target fmax: {target_fmax:.2f} MHz)")
        if self.initial_wns is not None:
            if self.initial_wns >= 0:
                summary.append(f"  WNS: {self.initial_wns:.3f} ns - TIMING MET ✓")
            else:
                summary.append(f"  WNS: {self.initial_wns:.3f} ns - TIMING VIOLATED")
            # Add fmax information
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            if initial_fmax is not None:
                summary.append(f"  Achievable fmax: {initial_fmax:.2f} MHz")
        if self.initial_tns is not None:
            summary.append(f"  TNS: {self.initial_tns:.3f} ns")
        if self.initial_failing_endpoints is not None:
            summary.append(f"  Failing endpoints: {self.initial_failing_endpoints}")
        summary.append("")
        
        # Critical path spread analysis
        summary.append("TARGET SELECTION:")
        summary.append(f"  Tier: {self.target_tier}")
        summary.append(f"  Candidates: {len(self.current_target_candidates)}")
        for i, candidate in enumerate(self.current_target_candidates[:5]):
            summary.append(
                f"  {i+1}. slack={candidate.get('slack')} "
                f"start={candidate.get('startpoint')} end={candidate.get('endpoint')}"
            )
        summary.append("")

        summary.append("PATH DELAY CLASSIFICATION:")
        summary.append(f"  Class: {self.path_delay_classification}")
        summary.append(f"  Breakdown: {self.path_delay_breakdown}")
        summary.append("")

        if critical_path_spread_info:
            summary.append("CRITICAL PATH SPREAD ANALYSIS:")
            summary.append(f"  Max cell distance: {critical_path_spread_info['max_distance']} tiles")
            summary.append(f"  Avg cell distance: {critical_path_spread_info['avg_distance']:.1f} tiles")
            summary.append(f"  Paths analyzed: {critical_path_spread_info['paths_analyzed']}")
            
            # Recommendation based on spread
            if critical_path_spread_info['avg_distance'] > 70 and critical_path_spread_info['paths_analyzed'] >= 5:
                summary.append(f"  ⚠ RECOMMENDATION: Use PBLOCK strategy (high spread detected)")
            summary.append("")
        
        # High fanout nets (show top 10)
        if self.high_fanout_nets:
            summary.append("CRITICAL HIGH FANOUT NETS (top 10):")
            for i, (net_name, fanout, path_count) in enumerate(self.high_fanout_nets[:10]):
                summary.append(f"  {i+1}. {net_name}")
                summary.append(f"     Fanout: {fanout}, Critical paths: {path_count}")
            if len(self.high_fanout_nets) > 10:
                summary.append(f"  ... and {len(self.high_fanout_nets) - 10} more nets")
        else:
            summary.append("CRITICAL HIGH FANOUT NETS: None found")
        
        summary.append("")
        summary.append(f"Total nets available for optimization: {len(self.high_fanout_nets)}")
        
        summary_text = "\n".join(summary)
        print(summary_text)
        print()
        
        return summary_text
    
    async def get_completion(self) -> tuple[str, bool]:
        """Get LLM completion and process it."""
        try:
            self.llm_call_count += 1
            logger.info(f"LLM API call #{self.llm_call_count}")
            
            # Request usage accounting from OpenRouter
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=4096,
                extra_body={
                    "usage": {
                        "include": True
                    }
                }
            )
            
            # Validate response immediately
            if response is None:
                raise ValueError("API returned None response")
            
            # Extract token usage information from OpenRouter
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                # Update cumulative totals
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                
                # Get actual cost from OpenRouter (in credits/dollars)
                call_cost = 0.0
                if hasattr(response.usage, 'cost') and response.usage.cost is not None:
                    call_cost = float(response.usage.cost)
                    self.total_cost += call_cost
                else:
                    logger.warning("OpenRouter did not provide cost information")
                
                # Extract additional usage details if available
                cached_tokens = 0
                reasoning_tokens = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    if hasattr(response.usage.prompt_tokens_details, 'cached_tokens'):
                        cached_tokens = response.usage.prompt_tokens_details.cached_tokens or 0
                if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                    if hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                        reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0
                
                # Store details for this call
                call_detail = {
                    "call_number": self.llm_call_count,
                    "iteration": self.iteration,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": call_cost,
                    "cached_tokens": cached_tokens,
                    "reasoning_tokens": reasoning_tokens
                }
                self.api_call_details.append(call_detail)
                
                # Log token usage
                cache_info = f", Cached: {cached_tokens:,}" if cached_tokens > 0 else ""
                reasoning_info = f", Reasoning: {reasoning_tokens:,}" if reasoning_tokens > 0 else ""
                cost_info = f" | Cost: ${call_cost:.4f}" if call_cost > 0 else ""
                
                logger.info(f"API call #{self.llm_call_count} - Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total{cost_info}{cache_info}{reasoning_info}")
                print(f"[API Call #{self.llm_call_count}] Tokens: {total_tokens:,} (Prompt: {prompt_tokens:,}, Completion: {completion_tokens:,}{cache_info}{reasoning_info}){cost_info}")
            else:
                logger.warning("No usage information in API response")
            
            # Debug logging
            if self.debug:
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response: {response}")
            
            # Check if response has error
            if hasattr(response, 'error') and response.error:
                raise ValueError(f"API returned error: {response.error}")
            
            return await self.process_response(response)
            
        except Exception as e:
            logger.error(f"Error in get_completion: {e}")
            logger.error(f"Number of messages in conversation: {len(self.messages)}")
            if self.messages:
                logger.error(f"Last message: {self.messages[-1]}")
            raise

    async def get_validated_action_decision(self, timing_context: dict) -> dict:
        reprompt_count: int | str = 0
        decision, raw_text = await self._request_action_json()
        valid, reason = self.validate_llm_action(decision, timing_context)
        if not valid:
            chosen = decision.get("chosen_action")
            logger.warning(
                "LLM chose invalid action %s for %s path: %s. Re-prompting.",
                chosen,
                timing_context.get("delay_class"),
                reason,
            )
            self.messages.append({
                "role": "user",
                "content": (
                    f"Your previous response chose {chosen} which is forbidden or invalid for "
                    f"{timing_context.get('delay_class')} paths. Failure reason: {reason}. "
                    f"Try again. You must choose from: {timing_context.get('allowed_actions')}."
                ),
            })
            reprompt_count = 1
            decision, raw_text = await self._request_action_json()
            valid, reason = self.validate_llm_action(decision, timing_context)

        if not valid:
            logger.warning("LLM_OVERRIDE: fell back to deterministic choice after 2 failed validations.")
            decision = {
                "delay_class_acknowledged": timing_context.get("delay_class"),
                "endpoint_type_acknowledged": timing_context.get("endpoint_type"),
                "chosen_action": timing_context["allowed_actions"][0],
                "action_parameters": {},
                "why_this_fits_delay_class": "Deterministic fallback selected the highest-priority allowed action.",
                "why_not_top_forbidden_action": "The previous model response failed validation.",
                "confidence": 1,
            }
            reprompt_count = "deterministic_fallback"
            reason = "deterministic_fallback"

        self.last_llm_decision = decision
        self.last_decision_trace = {
            "delay_class": timing_context.get("delay_class"),
            "endpoint_type": timing_context.get("endpoint_type"),
            "allowed_actions": list(timing_context.get("allowed_actions", [])),
            "forbidden_actions": list(timing_context.get("forbidden_actions", [])),
            "action_guidance": dict(timing_context.get("action_guidance", {})),
            "congestion_level": timing_context.get("congestion_level"),
            "cluster_clock_regions": list(timing_context.get("cluster_clock_regions", [])),
            "llm_chosen_action": decision.get("chosen_action"),
            "structural_override_active": timing_context.get("structural_override_active", False),
            "stuck_iterations": timing_context.get("stuck_iterations", 0),
            "consecutive_no_improvement": timing_context.get("consecutive_no_improvement", 0),
            "exhausted_actions": list(timing_context.get("exhausted_actions", [])),
            "action_failure_memory": dict(timing_context.get("action_failure_memory", {})),
            "action_failure_counts": dict(timing_context.get("action_failure_counts", {})),
            "validation_result": reason,
            "reprompt_count": reprompt_count,
            "why_not_top_forbidden_action": decision.get("why_not_top_forbidden_action"),
            "raw_response": raw_text,
            "cluster_count": timing_context.get("cluster_count"),
            "primary_diagnosis": timing_context.get("primary_diagnosis"),
            "diagnosis_reasoning_trace": list(timing_context.get("diagnosis_reasoning_trace", [])),
            "diagnosis_action_adjustment": timing_context.get("diagnosis_action_adjustment"),
        }
        logger.info(
            "Action selection: offered=%s picked=%s structural_override=%s stuck_iterations=%s",
            self.last_decision_trace["allowed_actions"],
            self.last_decision_trace["llm_chosen_action"],
            self.last_decision_trace["structural_override_active"],
            self.last_decision_trace["stuck_iterations"],
        )
        if self.last_decision_trace["structural_override_active"]:
            logger.warning(
                "Stuck detector: %s iterations without improvement, forcing structural action: %s",
                self.last_decision_trace["stuck_iterations"],
                self.last_decision_trace["llm_chosen_action"],
            )
        return decision

    async def _request_action_json(self) -> tuple[dict, str]:
        self.llm_call_count += 1
        logger.info(f"LLM action-selection call #{self.llm_call_count}")
        try:
            # The last unbounded wait in the loop: without a timeout, a stalled
            # OpenRouter request froze the entire run with no recovery path
            # (every stop condition is checked between iterations, so a hang
            # inside one is forever). On failure, return an empty decision --
            # validation fails, one reprompt happens, and after that the
            # deterministic fallback picks the top-ranked allowed action.
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=self.messages,
                max_tokens=1200,
                timeout=180.0,
                extra_body={"usage": {"include": True}},
            )
        except Exception as exc:
            logger.error("LLM action-selection call failed or timed out: %s", exc)
            return {}, ""
        if hasattr(response, "usage") and response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
            self.total_tokens += response.usage.total_tokens
            if hasattr(response.usage, "cost") and response.usage.cost is not None:
                self.total_cost += float(response.usage.cost)
        content = response.choices[0].message.content or "{}"
        self.messages.append({"role": "assistant", "content": content})
        try:
            return json.loads(content), content
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0)), content
                except json.JSONDecodeError:
                    pass
        return {}, content

    def validate_llm_action(self, response: dict, timing_context: dict) -> tuple[bool, str]:
        chosen = response.get("chosen_action")
        forbidden = timing_context.get("forbidden_actions", [])
        allowed = timing_context.get("allowed_actions", [])
        acknowledged_class = response.get("delay_class_acknowledged")
        actual_class = timing_context.get("delay_class")

        if chosen in forbidden:
            logger.warning(f"LLM chose forbidden action {chosen} for {actual_class} path. Re-prompting.")
            return False, "chosen_action_is_forbidden"
        if chosen not in allowed:
            logger.warning(f"LLM chose {chosen} which is not in allowed_actions. Re-prompting.")
            return False, "chosen_action_not_in_allowed_list"
        if acknowledged_class != actual_class:
            logger.warning(f"LLM acknowledged wrong delay class: {acknowledged_class} vs {actual_class}.")
            return False, "delay_class_mismatch"
        # Place directives are a closed set: an invalid one (run
        # 20260714_182751 iter 9: "AggressiveExplore", a phys_opt directive)
        # costs an unplace + failed place + restore at dispatch. Reject it
        # here so the reprompt can pick a real one.
        params = response.get("action_parameters") or {}
        place_directive = None
        if chosen == "place_design_explore":
            place_directive = params.get("directive")
        elif chosen in ("pblock_full_replace", "run_recipe"):
            place_directive = params.get("place_directive")
        if place_directive and str(place_directive) not in VALID_PLACE_DIRECTIVES:
            logger.warning(
                "LLM chose invalid place directive %r for %s. Re-prompting.",
                place_directive, chosen,
            )
            return False, (
                f"invalid_place_directive: {place_directive!r} is not a place_design "
                f"directive; choose one of {sorted(VALID_PLACE_DIRECTIVES)}"
            )
        guidance = timing_context.get("action_guidance") or {}
        if chosen in guidance:
            try:
                confidence = int(response.get("confidence") or 0)
            except (TypeError, ValueError):
                confidence = 0
            if confidence < DISCOURAGED_ACTION_MIN_CONFIDENCE:
                logger.warning(
                    "LLM chose discouraged action %s with confidence %d/5 (guidance: %s). Re-prompting.",
                    chosen, confidence, guidance[chosen],
                )
                return False, (
                    f"discouraged_action_low_confidence: {chosen} is discouraged ({guidance[chosen]}) "
                    f"and was chosen with confidence {confidence}/5 (minimum {DISCOURAGED_ACTION_MIN_CONFIDENCE} "
                    "required to override guidance); either give a substantially stronger rebuttal of the "
                    "guidance's specific record, or pick a different allowed action"
                )
        return True, "ok"

    async def execute_validated_action(self, decision: dict, timing_context: dict) -> str:
        action = decision.get("chosen_action")
        params = decision.get("action_parameters") or {}
        self.last_rapidwright_edit_summary = None
        self.last_action_mutated_design = False
        # Fix #1: set the never-renamed dispatch key exactly once per action,
        # before any of the per-action branches below run. _remember_recipe()
        # is still allowed to rewrite self.last_recipe for display purposes;
        # self.last_action_key is the one all gating logic should read.
        self.last_action_key = str(action)
        # Item 4 (expensive-action cap): the demotion in
        # _allowed_forbidden_actions is rebuttable by design; this refusal is
        # not -- a third full re-place on a large design cannot fit the run
        # no matter how good the argument for it sounds.
        cap_reason = self._full_replace_blocked_reason(str(action))
        if cap_reason:
            return self._failure_json("full_replace_cap_reached", cap_reason, command=str(action))
        # Item 3 (affordability): never start an action that cannot finish in
        # the remaining budget -- a cheap recorded failure the LLM can react
        # to, instead of an expensive half-finished one killed by a timeout.
        remaining_budget = self._time_remaining_s()
        estimated_cost = self._estimated_action_cost_s(str(action))
        if (
            remaining_budget is not None
            and estimated_cost is not None
            and remaining_budget < estimated_cost
        ):
            # Pipeline audit (20260802-20260804 sweep): this is the single
            # biggest failure cause (14/47 across that sweep) -- ispd16_
            # example2 spent its LAST 15 OF 19 ITERATIONS re-proposing an
            # action already refused here, because the demotion in
            # _allowed_forbidden_actions (same threshold, 1.3x) is soft and
            # the LLM kept choosing it anyway. Feed the refusal into the
            # same cooldown machinery other repeat-failing actions already
            # use, so it's actually removed from allowed_actions after
            # ACTION_FAILURE_EXHAUSTION_THRESHOLD refusals instead of
            # merely re-ranked with a reason the LLM can (and did) ignore.
            self._remember_no_action_failure(str(action), [])
            return self._failure_json(
                "insufficient_budget",
                f"{action} is estimated to cost ~{estimated_cost / 60.0:.0f} min but only "
                f"{max(remaining_budget, 0) / 60.0:.0f} min of budget remain; choose a "
                f"cheaper refinement action that fits.",
                command=str(action),
            )
        if action in ("place_design_explore", "pblock_full_replace"):
            # Item 4: count every full re-place dispatch (warm start and
            # run_recipe stages included) toward the large-design cap.
            self.full_replace_attempts += 1
        if action in {"phys_opt_design", "phys_opt_design_retime", "phys_opt_design_pin_swap"}:
            if not await self._check_implementation_license():
                return self._failure_json(
                    "vivado_license_failure",
                    "Vivado Implementation license is unavailable; phys_opt_design disabled.",
                    command="phys_opt_design",
                )
            self.last_recipe = action
            self.last_targets = [timing_context.get("delay_class", "timing_path")]
            self.last_batch_size = 1
            call_params = dict(params)
            # Directive sweep (mirrors place_design_explore's _next_place_directive):
            # default to the next untried PHYS_OPT_DIRECTIVE_SWEEP entry when the
            # LLM omits one, and always record whichever directive actually ran
            # so _note_recipe_outcome can track its result and the untried list
            # stays accurate even when the LLM supplies its own directive string.
            directive = str(call_params.get("directive") or self._next_phys_opt_directive())
            call_params["directive"] = directive
            self.last_phys_opt_directive = directive
            if action == "phys_opt_design_pin_swap":
                # LUT pin-swapping (remap logical to physical pins within a
                # SLICE to reduce routing congestion on critical nets) --
                # Vivado's -critical_pin_opt flag, registered in the MCP
                # schema (VivadoMCP/vivado_mcp_server.py) but never
                # exercised until now: every existing phys_opt call goes
                # through _run_phys_opt_with_policy, which always builds a
                # -directive-based Tcl command, and the server treats
                # directive/bool-flags as mutually exclusive. Naming this as
                # its own action (mirroring phys_opt_design_retime) forces
                # critical_pin_opt=True through unconditionally, giving it
                # its own trackable win/loss record instead of being a
                # silent, invisible parameter on phys_opt_design.
                call_params["critical_pin_opt"] = True
            return await self.call_tool("vivado_phys_opt_design", call_params)
        if action == "place_design_explore":
            # History(16) forensics: Vivado's place_design is INCREMENTAL over
            # an existing placement -- on a fully placed design it is a
            # near-total no-op regardless of directive (history(16) iter 3
            # measured WNS identical to baseline after "AltSpreadLogic_high").
            # The 501 MHz result credited to this action actually came from an
            # accidental unplace + place + route. So: unplace first, always.
            # That makes this the plain whole-design re-place recipe (the
            # pblock-less variant of pblock_full_replace), and makes the
            # directive sweep actually sweep something.
            if not await self._check_implementation_license():
                return self._failure_json(
                    "vivado_license_failure",
                    "Vivado Implementation license is unavailable; place_design/route_design disabled.",
                    command="place_design/route_design",
                )
            directive = str(params.get("directive") or self._next_place_directive())
            route_directive = str(params.get("route_directive") or "Explore")
            self.last_recipe = action
            self.last_place_directive = directive
            self.last_targets = [f"directive:{directive}"]
            self.last_batch_size = 1
            unplace_result = await self.call_tool(
                "vivado_run_tcl", {"command": "place_design -unplace", "timeout": 600}, internal=True
            )
            if self._vivado_output_has_error(unplace_result):
                return self._failure_json(
                    "replace_unplace_failed",
                    f"place_design -unplace failed: {unplace_result[-FAILURE_MESSAGE_CAPTURE_CHARS:]}",
                    command="place_design_explore",
                )
            place = await self.call_tool("vivado_place_design", {"directive": directive, "timeout": self._implementation_timeout_s(kind="place")})
            if self._action_failure(place, default_command="vivado_place_design"):
                return self._failure_json(
                    "replace_place_failed",
                    f"place_design -directive {directive} failed after unplace: {place[-FAILURE_MESSAGE_CAPTURE_CHARS:]}",
                    command="place_design_explore",
                )
            # Item 5 (adaptive route directive): the place just consumed real
            # budget; if what is left cannot comfortably fit the requested
            # route directive, downgrade to Default -- a completed Default
            # route always beats an Explore route killed by the budget clamp
            # (the motivating incident's exact failure mode).
            route_directive = self._maybe_downgrade_route_directive(route_directive, "place_design_explore")
            route = await self.call_tool("vivado_route_design", {"directive": route_directive, "timeout": self._implementation_timeout_s(kind="route")})
            if self._action_failure(route, default_command="vivado_route_design"):
                return self._failure_json(
                    "replace_route_failed",
                    f"route_design -directive {route_directive} failed after re-place: {route[-FAILURE_MESSAGE_CAPTURE_CHARS:]}",
                    command="place_design_explore",
                )
            return place + "\n\n" + route
        if action == "route_explore":
            # Route-only refinement: rip up and re-route the CURRENT placement
            # with a stronger directive, never touching cell locations. The
            # lowest-variance move in the menu -- the placement that earned
            # the current WNS is preserved by construction, so the downside
            # is bounded by one route cycle (rolled back on regression).
            if not await self._check_implementation_license():
                return self._failure_json(
                    "vivado_license_failure",
                    "Vivado Implementation license is unavailable; route_design disabled.",
                    command="route_explore",
                )
            if self.design_state != "routed":
                return self._failure_json(
                    "invalid_design_state",
                    f"route_explore requires a routed design to re-route; state is '{self.design_state}'.",
                    command="route_explore",
                )
            directive = str(params.get("directive") or "Explore")
            self.last_recipe = action
            self.last_targets = [f"directive:{directive}"]
            self.last_batch_size = 1
            route = await self.call_tool(
                "vivado_route_design",
                {"directive": directive, "timeout": self._implementation_timeout_s(kind="route")},
                internal=True,
            )
            if self._action_failure(route, default_command="vivado_route_design"):
                return self._failure_json(
                    "route_explore_failed",
                    f"route_design -directive {directive} failed: {route[-FAILURE_MESSAGE_CAPTURE_CHARS:]}",
                    command="route_explore",
                )
            return route
        if action == "qor_suggestions":
            return await self._execute_qor_suggestions()
        if action == "run_recipe":
            return await self._execute_run_recipe(dict(params), timing_context)
        if action == "pblock_full_replace":
            return await self._execute_pblock_full_replace(dict(params))
        if action == "pblock":
            if not await self._check_implementation_license():
                return self._failure_json(
                    "vivado_license_failure",
                    "Vivado Implementation license is unavailable; pblock flow requiring implementation is disabled.",
                    command="create_and_apply_pblock",
                )
            self.last_recipe = action
            params, error = await self._compute_pblock_ranges(dict(params), timing_context)
            if error:
                logger.error("pblock action aborted: %s", error)
                return self._failure_json("pblock_range_computation_failed", error, command="pblock")
            assert params is not None
            # Fix #2: _compute_pblock_ranges tucks the raw fabric region into
            # "_fabric_region" so it can be registered as applied (for future
            # overlap checks) once we know the pblock call actually succeeded
            # -- pop it out here so it never gets sent to Vivado as an arg.
            fabric_region = params.pop("_fabric_region", None)
            self.last_recipe = action
            self.last_targets = [str(params.get("pblock_name", "pblock_opt")), str(params.get("ranges"))]
            self.last_batch_size = 1
            # internal=True: same interception bug class as pblock_full_replace.
            # Without it, call_tool rerouted this through
            # _maybe_run_pblock_or_phys_opt, which re-classified the path and
            # ran hidden phys_opt passes first (run 20260712_051231 iters
            # 12/13: recipe mislabeled "vivado_phys_opt_mixed_path", stale
            # ranges reapplied, 0 cells matched). The ranges were already
            # computed above; send the call straight to the server.
            result = await self.call_tool("vivado_create_and_apply_pblock", params, internal=True)
            payload = self._parse_json_result(result)
            if self._result_has_error(payload):
                logger.error(f"pblock_assignment_failed - full result:\n{result}")
                return self._failure_json(
                    payload.get("error_type", "pblock_assignment_failed"),
                    payload.get("message", result[-FAILURE_MESSAGE_CAPTURE_CHARS:]),
                    command="pblock",
                )
            self.last_rapidwright_edit_summary = self._summarize_pblock_assignment(payload, params)
            if int(self.last_rapidwright_edit_summary.get("cells_assigned") or 0) <= 0:
                return self._failure_json(
                    "pblock_empty_assignment",
                    "Pblock action completed but assigned zero cells.",
                    command="pblock",
                )
            # Abort BEFORE paying for place_design when the apply payload's
            # DRC already says the region cannot fit the assigned cells
            # (boom_soc run 20260713 iter 4: 153 RAMB36 assigned to a region
            # with 60 sites -- resource_validation flagged it, we placed
            # anyway, and the placer burned budget just to fail on the same
            # DRC). The region generator sizes by LUT/FF only, so hard-block
            # (BRAM/DSP/URAM) overflow is exactly the failure it can't see.
            pblock_validation = payload.get("resource_validation") or {}
            pblock_validation_errors = list(pblock_validation.get("errors") or [])
            if pblock_validation_errors:
                # Harvest the DRC's own `required` counts as the demand floor
                # for every later pblock sizing this run -- the candidate-based
                # estimate has no visibility into the server-side cell
                # expansion that produced them.
                for issue_key, issue in (pblock_validation.get("resource_issues") or {}).items():
                    required = int((issue or {}).get("required") or 0)
                    if required <= 0:
                        continue
                    key_upper = str(issue_key).upper()
                    if any(k in key_upper for k in ("RAMB", "FIFO", "URAM")):
                        kind = "bram"
                    elif "DSP" in key_upper:
                        kind = "dsp"
                    else:
                        continue
                    if required > self.pblock_hard_block_demand.get(kind, 0):
                        self.pblock_hard_block_demand[kind] = required
                        logger.info(
                            "pblock demand floor learned from DRC: %s requires %d sites.",
                            kind, required,
                        )
                stale_name = params.get("pblock_name")
                if stale_name:
                    await self.call_tool(
                        "vivado_run_tcl",
                        {"command": f"if {{[llength [get_pblocks -quiet {stale_name}]] > 0}} "
                                    f"{{delete_pblocks [get_pblocks {stale_name}]}}"},
                        internal=True,
                    )
                logger.error(
                    "pblock resource validation failed pre-placement: %s",
                    pblock_validation_errors,
                )
                return self._failure_json(
                    "pblock_region_too_small",
                    (
                        "Pblock resource validation failed before placement: "
                        + "; ".join(str(err)[:200] for err in pblock_validation_errors[:3])
                        + ". The region cannot fit the assigned cells (typically BRAM/DSP "
                        "shortage); widen the ranges or exclude hard-block cells."
                    ),
                    command="pblock",
                )
            # Creating/applying the pblock only constrains a region for
            # future placement - it does not move any cells by itself, so
            # WNS cannot change as a result of this call alone. Re-place
            # (now respecting the new pblock) and re-route before returning,
            # or this action can never have any measurable timing effect.
            place_result = await self.call_tool(
                "vivado_place_design", {"directive": "Explore", "timeout": self._implementation_timeout_s(kind="place")}, internal=True
            )
            if self._action_failure(place_result, default_command="vivado_place_design"):
                # BUG FIX: this used to cap the stored message at 300 chars,
                # which for a real Vivado placer error just captures the
                # license-acquisition boilerplate and cuts off before the
                # actual ERROR/CRITICAL WARNING line further down in the
                # report -- exactly the info needed to diagnose *why*
                # placement failed after a pblock was applied. Log the full,
                # untruncated text (recoverable from the run log even if the
                # JSON history record trims it) and widen the stored message
                # itself well past where a real placer error would appear.
                logger.error(f"pblock_place_failed - full place_design output:\n{place_result}")
                return self._failure_json(
                    "pblock_place_failed",
                    f"Pblock applied but re-placement failed: {place_result[-FAILURE_MESSAGE_CAPTURE_CHARS:]}",
                    command="pblock",
                )
            route_result = await self.call_tool(
                "vivado_route_design", {"directive": "Explore", "timeout": self._implementation_timeout_s(kind="route")}, internal=True
            )
            if self._action_failure(route_result, default_command="vivado_route_design"):
                if self._is_post_route_physsynth_crash(route_result):
                    # See _is_post_route_physsynth_crash: this is Vivado's own
                    # post-route re-optimization pass crashing, not the
                    # design/pblock -- retry once with Default, which doesn't
                    # invoke that pass, instead of taking pblock off the
                    # table for the rest of the run over an unrelated crash.
                    logger.warning(
                        "pblock route_design hit the post-route physical-synthesis "
                        "crash (13HDPLException) under -directive Explore; retrying "
                        "once with -directive Default."
                    )
                    route_result = await self.call_tool(
                        "vivado_route_design", {"directive": "Default", "timeout": self._implementation_timeout_s(kind="route")}, internal=True
                    )
            if self._action_failure(route_result, default_command="vivado_route_design"):
                logger.error(f"pblock_route_failed - full route_design output:\n{route_result}")
                return self._failure_json(
                    "pblock_route_failed",
                    f"Pblock applied and re-placed but routing failed: {route_result[-FAILURE_MESSAGE_CAPTURE_CHARS:]}",
                    command="pblock",
                )
            # Fix #2: only now, after placement + routing succeeded, register
            # this region as applied so future pblock recommendations are
            # checked against it for overlap.
            self._register_applied_pblock(fabric_region, params.get("pblock_name"))
            return result + "\n\n" + place_result + "\n\n" + route_result
        if action == "rapidwright_analyze_fabric_for_pblock":
            self.last_recipe = action 
            params, error = await self._compute_pblock_ranges(dict(params), timing_context)
            if error:
                logger.error("RapidWright fabric/pblock analysis aborted: %s", error)
                return self._failure_json("pblock_range_computation_failed", error, command=action)
            assert params is not None
            params.pop("_fabric_region", None)
            self.last_targets = [str(params.get("ranges"))]
            self.last_batch_size = 1
            self.last_rapidwright_edit_summary = {
                **(self.last_rapidwright_edit_summary or {}),
                "action": action,
                "cells_moved": 0,
                "nets_affected": 0,
                "pblock_ranges": params.get("ranges"),
                "changed_design": False,
            }
            return json.dumps({"success": True, "pblock_parameters": params}, indent=2)
        if action == "rapidwright_convert_fabric_region_to_pblock":
            self.last_recipe = action
            params, error = await self._compute_pblock_ranges(dict(params), timing_context)
            if error:
                logger.error("RapidWright pblock range conversion aborted: %s", error)
                return self._failure_json("pblock_range_computation_failed", error, command=action)
            assert params is not None
            params.pop("_fabric_region", None)
            self.last_targets = [str(params.get("ranges"))]
            self.last_batch_size = 1
            self.last_rapidwright_edit_summary = {
                **(self.last_rapidwright_edit_summary or {}),
                "action": action,
                "cells_moved": 0,
                "nets_affected": 0,
                "pblock_ranges": params.get("ranges"),
                "changed_design": False,
            }
            return json.dumps({"success": True, "pblock_parameters": params}, indent=2)
        if action == "rapidwright_analyze_net_detour":
            pins_file = await self._extract_critical_path_pins_file(num_paths=int(params.get("num_paths") or 20))
            detour_result = await self.call_tool(
                "rapidwright_analyze_net_detour",
                {
                    "input_file": str(pins_file),
                    "detour_threshold": float(params.get("detour_threshold") or 2.0),
                },
            )
            payload = self._parse_json_result(detour_result)
            if self._result_has_error(payload):
                return self._failure_json(
                    "rapidwright_detour_analysis_failed",
                    payload.get("error", detour_result[:300]),
                    command=action,
                )
            candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
            self.last_recipe = action
            self.last_targets = self._filter_blacklisted_cells([str(item.get("cell")) for item in candidates[:10] if item.get("cell")])
            self.last_batch_size = len(self.last_targets)
            self.last_rapidwright_edit_summary = {
                "action": action,
                "changed_design": False,
                "cells_moved": 0,
                "nets_affected": 0,
                "cells_analyzed": payload.get("cells_analyzed"),
                "candidates_found": payload.get("candidates_found", len(candidates)),
                "top_candidates": candidates[:10],
            }
            self._reset_action_failure_memory(action)
            return detour_result
        if action == "rapidwright_optimize_cell_placement":
            attempted_cells = [str(cell) for cell in params.get("cell_names", []) if str(cell).strip()]
            cell_names = self._filter_high_fanout_cells(self._filter_blacklisted_cells(attempted_cells))
            if not cell_names:
                try:
                    pins_file = await self._extract_critical_path_pins_file(num_paths=int(params.get("num_paths") or 20))
                    detour_result = await self.call_tool(
                        "rapidwright_analyze_net_detour",
                        {
                            "input_file": str(pins_file),
                            "detour_threshold": float(params.get("detour_threshold") or 2.0),
                        },
                        internal=True,
                    )
                    detour_payload = self._parse_json_result(detour_result)
                    candidates = detour_payload.get("candidates") if isinstance(detour_payload.get("candidates"), list) else []
                    cell_names = self._filter_high_fanout_cells(self._critical_path_cell_candidates(timing_context))
                    attempted_cells = cell_names or attempted_cells
                except Exception as exc:
                    logger.warning("Could not derive placement cells from detour analysis: %s", exc)
            if not cell_names:
                cell_names = self._critical_path_cell_candidates(timing_context)
                attempted_cells = cell_names or attempted_cells
            if not cell_names:
                live_cells = await self._live_critical_path_cell_candidates(
                    num_paths=int(params.get("num_paths") or 20),
                    limit=int(params.get("max_candidates") or 20),
                )
                if live_cells:
                    cell_names = self._filter_high_fanout_cells(live_cells)
                    attempted_cells = cell_names
            if not cell_names:
                fallback_targets = attempted_cells or self._path_identifier_targets(timing_context)
                self.last_recipe = action
                self.last_targets = fallback_targets
                self.last_batch_size = len(fallback_targets)
                self._blacklist_failure_targets(action, fallback_targets)
                self._remember_no_action_failure(action, fallback_targets)
                self.last_no_action_failure_key = (action, tuple(fallback_targets), self.iteration)
                return self._failure_json(
                    "no_action_target",
                    "rapidwright_optimize_cell_placement selected but no critical cells were available.",
                    command=action,
                )
            max_candidates = int(params.get("max_candidates") or min(10, len(cell_names)))
            cell_names = cell_names[:max_candidates]

            critical_pins = {
            cell: pins
                for cell, pins in getattr(self, "_last_critical_pins", {}).items()
                if cell in cell_names
            }
            # --- Fix #4 (cluster-level optimization guard) ---
            # rapidwright_optimize_cell_placement moves each cell in
            # cell_names independently. If these cells are tightly coupled
            # (on the same critical path), moving them one at a time can
            # increase the distance between them even though each move
            # looked locally reasonable -- this was the likely root cause of
            # the 11/11 regression rate observed for this action on the
            # LogicNets benchmark. Measure the cluster's spread before and
            # after the move; if the move made the cluster measurably worse
            # spread out, reject it before it is ever routed (saving a full
            # route cycle and a checkpoint we already know regressed).
            spread_before = await self._measure_cluster_spread(cell_names)
            self.last_recipe = action
            self.last_targets = list(cell_names)
            self.last_batch_size = len(cell_names)
            result = await self.call_tool(
                "rapidwright_optimize_cell_placement",
                {"cell_names": cell_names, "max_candidates": max_candidates, "critical_pins": critical_pins, "max_move_distance": 15,},
            )
            payload = self._parse_json_result(result)
            if self._result_has_error(payload):
                message = str(payload.get("error") or payload.get("message") or result[:300])
                error_type = "no_action_target" if self._is_no_action_failure({"error_type": payload.get("error_type"), "message": message}) else "rapidwright_cell_placement_failed"
                if error_type == "no_action_target":
                    self._blacklist_failure_targets(action, cell_names)
                    self._remember_no_action_failure(action, cell_names)
                    self.last_no_action_failure_key = (action, tuple(cell_names), self.iteration)
                return self._failure_json(
                    error_type,
                    message,
                    command=action,
                )
            self.last_rapidwright_edit_summary = self._summarize_cell_placement(payload, cell_names)
            if int(self.last_rapidwright_edit_summary.get("cells_moved") or 0) > 0:
                self._reset_action_failure_memory(action)
            else:
                self._blacklist_failure_targets(action, cell_names)
                self._remember_no_action_failure(action, cell_names)
                self.last_no_action_failure_key = (action, tuple(cell_names), self.iteration)
                return self._failure_json(
                    "no_action_target",
                    "rapidwright_optimize_cell_placement completed but moved zero cells.",
                    command=action,
                )
            # Fix #4 continued: now that cells have actually moved, re-measure
            # cluster spread and compare. If it got meaningfully worse, treat
            # this as a proactive regression -- reject before writing a
            # checkpoint or spending a route cycle on a move we can already
            # tell hurt locality. The RapidWright session still has the
            # moved (unwritten) placement active; the caller's normal
            # rollback-on-regression path (via checkpoint history) will
            # restore the last-good checkpoint on the *next* successful
            # write, but we still want the explicit failure recorded now so
            # action_failure_memory learns from it immediately rather than
            # waiting a full route+timing cycle to find out.
            moved_cell_names = self.last_rapidwright_edit_summary.get("moved_cells") or cell_names
            spread_after = await self._measure_cluster_spread(moved_cell_names)
            if (
                spread_before
                and spread_after
                and spread_before.get("avg_distance")
                and spread_after.get("avg_distance") is not None
            ):
                before_val = float(spread_before["avg_distance"])
                after_val = float(spread_after["avg_distance"])
                if before_val > 0 and (after_val - before_val) / before_val > CLUSTER_SPREAD_REGRESSION_FRACTION:
                    logger.warning(
                        "Cluster spread guard: cell placement increased avg cluster spread from "
                        "%.1f to %.1f tiles (+%.0f%%) for cells %s; rejecting before routing.",
                        before_val,
                        after_val,
                        100.0 * (after_val - before_val) / before_val,
                        moved_cell_names,
                    )
                    self._blacklist_failure_targets(action, moved_cell_names)
                    self._remember_no_action_failure(action, moved_cell_names)
                    self.last_no_action_failure_key = (action, tuple(moved_cell_names), self.iteration)
                    return self._failure_json(
                        "cluster_spread_regression",
                        (
                            f"rapidwright_optimize_cell_placement moved cells independently and "
                            f"increased average cluster spread from {before_val:.1f} to {after_val:.1f} "
                            f"tiles (+{100.0 * (after_val - before_val) / before_val:.0f}%), which is "
                            f"expected to worsen net delay; rejected before routing."
                        ),
                        command=action,
                    )
            dcp = self.run_dir / f"cell_placement_iter_{self.iteration:03d}.dcp"
            result += "\n\n" + await self.call_tool("rapidwright_write_checkpoint", {"dcp_path": str(dcp), "overwrite": True})
            result += "\n\n" + await self.call_tool("vivado_open_checkpoint", {"dcp_path": str(dcp), "timeout": 600})
            # Fail fast only on candidates with unplaced primitives BEYOND the
            # design's benign baseline artifacts (see _unplaced_tolerance --
            # this design always shows ~6, plus one per failed move; the
            # post-route provenance gate re-verifies with route status).
            candidate_unplaced = await self._count_unplaced_cells()
            if candidate_unplaced and candidate_unplaced > self._unplaced_tolerance():
                candidate_unplaced = await self._retry_incremental_place_for_unplaced(action, candidate_unplaced)
            if candidate_unplaced and candidate_unplaced > self._unplaced_tolerance():
                return self._failure_json(
                    "invalid_design_state",
                    (
                        f"rapidwright_optimize_cell_placement produced a candidate with "
                        f"{candidate_unplaced} unplaced primitive cell(s) (tolerance "
                        f"{self._unplaced_tolerance()}) even after an incremental re-place "
                        f"attempt; refusing to route or measure it."
                    ),
                    command=action,
                )
            result += "\n\n" + await self.call_tool("vivado_route_design", {"directive": "Default", "timeout": self._implementation_timeout_s(kind="route")})
            return result
        if action == "fanout_split":
            if not await self._check_implementation_license():
                return self._failure_json(
                    "vivado_license_failure",
                    "Vivado Implementation license is unavailable; fanout_split flow cannot reroute the edited design.",
                    command="fanout_split/route_design",
                )
            if not self.high_fanout_nets:
                # Fanout is a netlist property: no placement or routing action
                # can create high-fanout nets, so retrying this action later in
                # the run is guaranteed to fail again (vexriscv/corescore runs
                # 20260713 each burned 2 iterations rediscovering this).
                # Exhaust it for the rest of the run, not just a short cooldown.
                self.action_structural_cooldown_until_iter["fanout_split"] = 10 ** 9
                logger.warning(
                    "fanout_split selected but no high-fanout nets exist in this design; "
                    "exhausting fanout_split for the remainder of the run."
                )
                return self._failure_json(
                    "no_action_target",
                    "fanout_split selected but no high-fanout nets are available; "
                    "action withheld for the rest of the run.",
                    command="fanout_split",
                )
            net_name, fanout, _ = self.high_fanout_nets[0]
            split_factor = int(params.get("split_factor") or max(2, min(8, fanout // 100)))
            self.last_recipe = action
            self.last_targets = [net_name]
            self.last_batch_size = 1
            result = await self.call_tool("rapidwright_optimize_fanout", {"net_name": net_name, "split_factor": split_factor})
            payload = self._parse_json_result(result)
            self.last_rapidwright_edit_summary = self._summarize_fanout_split(payload, net_name, split_factor)
            dcp = self.run_dir / f"fanout_split_iter_{self.iteration:03d}.dcp"
            result += "\n\n" + await self.call_tool("rapidwright_write_checkpoint", {"dcp_path": str(dcp), "overwrite": True})
            result += "\n\n" + await self.call_tool("vivado_open_checkpoint", {"dcp_path": str(dcp), "timeout": 600})
            # Same tolerance-aware fail-fast as the cell-placement flow.
            candidate_unplaced = await self._count_unplaced_cells()
            if candidate_unplaced and candidate_unplaced > self._unplaced_tolerance():
                candidate_unplaced = await self._retry_incremental_place_for_unplaced(action, candidate_unplaced)
            if candidate_unplaced and candidate_unplaced > self._unplaced_tolerance():
                return self._failure_json(
                    "invalid_design_state",
                    (
                        f"fanout_split produced a candidate with {candidate_unplaced} unplaced "
                        f"primitive cell(s) (tolerance {self._unplaced_tolerance()}) even after "
                        f"an incremental re-place attempt; refusing to route or measure it."
                    ),
                    command=action,
                )
            result += "\n\n" + await self.call_tool("vivado_route_design", {"directive": "Default", "timeout": self._implementation_timeout_s(kind="route")})
            return result
        if action == "lut_opt":
            pins = list(params.get("hierarchical_input_pins") or [])
            if not pins:
                # Auto-derive from the current worst-path candidates' own
                # endpoint pins (Vivado ENDPOINT_PIN, "/"-separated hierarchy
                # -- the same format rapidwright's getHierPortInstFromName
                # expects) instead of refusing outright: see
                # LUT_OPT_DEFAULT_MAX_PINS above for why the LLM alone can't
                # supply this.
                pins = []
                for candidate in self.current_target_candidates:
                    endpoint = str(candidate.get("endpoint") or "").strip()
                    if endpoint and endpoint not in pins:
                        pins.append(endpoint)
                    if len(pins) >= LUT_OPT_DEFAULT_MAX_PINS:
                        break
                if pins:
                    logger.info(
                        "lut_opt selected with no hierarchical_input_pins; defaulting to "
                        "the current worst-path candidates' endpoint pins: %s", pins,
                    )
                else:
                    logger.warning("lut_opt selected but no pins provided and none could be derived.")
                    return self._failure_json(
                        "missing_action_parameters",
                        "lut_opt selected but no hierarchical_input_pins were provided and none "
                        "could be derived from current_target_candidates.",
                        command="lut_opt",
                    )
            self.last_recipe = action
            self.last_targets = [str(pin) for pin in pins]
            self.last_batch_size = len(pins)
            return await self.call_tool("rapidwright_optimize_lut_input_cone", {"hierarchical_input_pins": pins})
        if action == "replicate_register":
            self.last_recipe = action
            self.last_targets = [str(timing_context["worst_path"].get("end_cell"))]
            self.last_batch_size = 1
            # Targeted replication via -force_replication_on_nets was tried
            # here (20260804) and REVERTED after run 20260806_193354
            # (rosetta_optical-flow iter 1) failed with the definitive
            # verdict: "ERROR: [Vivado_Tcl 4-265] Option
            # -force_replication_on_nets is specified but not supported yet
            # for post-route physical synthesis." This pipeline only ever
            # runs phys_opt on routed designs (_run_phys_opt_with_policy
            # refuses any other design_state), so the option is categorically
            # unusable in this flow -- it turned a weak-but-harmless action
            # into a guaranteed hard failure. -critical_cell_opt (which 4-265
            # does NOT reject post-route, and which this action claimed to
            # send all along but the policy layer used to silently drop) is
            # the strongest replication lever actually available post-route.
            return await self.call_tool("vivado_phys_opt_design", {"critical_cell_opt": True})
        return self._failure_json(
            "unsupported_action",
            f"Action {action!r} is not implemented by the orchestrator dispatch layer.",
            command=str(action),
        )
    
    async def _execute_qor_suggestions(self) -> str:
        """Run Vivado's own ML QoR advisor and apply its strategy.

        report_qor_suggestions is Vivado's built-in expert: it analyzes the
        placed+routed design and emits an RQS (Report QoR Suggestions) file of
        timing/utilization/congestion strategy recommendations. We write it,
        read it back so the suggestions become active, then apply them with a
        phys_opt pass using the RQS directive (which selects the recommended
        strategy) followed by a re-route. If the RQS directive is unsupported
        on this design (e.g. out-of-context DCPs), phys_opt falls back to a
        standard directive, so the action degrades gracefully to a bounded
        phys_opt+route refinement rather than failing. Keep-best/rollback in
        _record_iteration_timing gates the outcome like any other action.
        """
        if not await self._check_implementation_license():
            return self._failure_json(
                "vivado_license_failure",
                "Vivado Implementation license is unavailable; qor_suggestions disabled.",
                command="qor_suggestions",
            )
        if self.design_state != "routed":
            return self._failure_json(
                "invalid_design_state",
                f"qor_suggestions needs a placed+routed design to analyze; state is '{self.design_state}'.",
                command="qor_suggestions",
            )
        self.last_recipe = action = "qor_suggestions"
        self.last_targets = ["qor_suggestions"]
        self.last_batch_size = 1

        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        rqs_path = checkpoint_dir / f"qor_iter_{self.iteration:03d}.rqs"
        rqs_tcl = str(rqs_path).replace("\\", "/")

        # 1. Generate suggestions in memory. (Vivado 2025.1's
        # report_qor_suggestions has NO -rqs_files option -- run
        # 20260714_182751 iter 6 failed on exactly that; the file is written
        # separately by write_qor_suggestions.)
        gen = await self.call_tool(
            "vivado_run_tcl",
            {"command": "report_qor_suggestions -max_paths 30", "timeout": 300},
            internal=True,
        )
        if self._vivado_output_has_error(gen):
            return self._failure_json(
                "no_action_target",
                f"report_qor_suggestions produced no usable suggestions: {gen[:600]}",
                command="qor_suggestions",
            )

        # 2. Persist + reload them so the RQS phys_opt directive can select
        #    their strategy (best-effort: the suggestions are already active
        #    in memory, so a write/read hiccup should not abort the action).
        write = await self.call_tool(
            "vivado_run_tcl",
            {"command": f"write_qor_suggestions -force {{{rqs_tcl}}}", "timeout": 120},
            internal=True,
        )
        if self._vivado_output_has_error(write):
            logger.warning("qor_suggestions: write_qor_suggestions failed (continuing): %s", write[:300])
        elif rqs_path.exists():
            await self.call_tool(
                "vivado_run_tcl",
                {"command": f"read_qor_suggestions {{{rqs_tcl}}}", "timeout": 120},
                internal=True,
            )

        # 3. Apply via phys_opt with the RQS strategy directive; fall back to a
        #    standard aggressive directive if RQS is unsupported here.
        phys = await self.call_tool(
            "vivado_phys_opt_design", {"directive": "RQS", "timeout": self._implementation_timeout_s(kind="phys_opt")},
            internal=True,
        )
        if self._action_failure(phys, default_command="vivado_phys_opt_design"):
            logger.info("qor_suggestions: RQS directive unsupported here; falling back to AggressiveExplore phys_opt.")
            phys = await self.call_tool(
                "vivado_phys_opt_design",
                {"directive": "AggressiveExplore", "timeout": self._implementation_timeout_s(kind="phys_opt")},
                internal=True,
            )
            if self._action_failure(phys, default_command="vivado_phys_opt_design"):
                return self._failure_json(
                    "qor_suggestions_failed",
                    f"phys_opt after QoR suggestions failed (RQS and fallback): {phys[-FAILURE_MESSAGE_CAPTURE_CHARS:]}",
                    command="qor_suggestions",
                )

        # 4. Re-route to realize the phys_opt changes.
        route = await self.call_tool(
            "vivado_route_design",
            {"directive": "Explore", "timeout": self._implementation_timeout_s(kind="route")},
            internal=True,
        )
        if self._action_failure(route, default_command="vivado_route_design"):
            return self._failure_json(
                "qor_suggestions_failed",
                f"route after QoR suggestions failed: {route[-FAILURE_MESSAGE_CAPTURE_CHARS:]}",
                command="qor_suggestions",
            )
        return "\n\n".join(part for part in (gen, phys, route) if part)

    async def _execute_run_recipe(self, params: dict, timing_context: dict) -> str:
        """Item 8 (recipe architecture, tranche 1): execute a whitelisted
        stage pipeline in ONE LLM decision instead of one stage per
        iteration. Stages run the way _endgame_polish steps do: a per-stage
        affordability check (the recipe stops cleanly when the next stage no
        longer fits the budget), per-stage failure recording + restore-best,
        and a normal timing probe after each successful stage so keep-best/
        rollback applies per stage, not per recipe."""
        stages = params.get("stages")
        if not isinstance(stages, list) or not stages:
            return self._failure_json(
                "invalid_recipe",
                "run_recipe requires a non-empty 'stages' list of {action, params} objects.",
                command="run_recipe",
            )
        if len(stages) > RUN_RECIPE_MAX_STAGES:
            return self._failure_json(
                "invalid_recipe",
                f"run_recipe allows at most {RUN_RECIPE_MAX_STAGES} stages, got {len(stages)}.",
                command="run_recipe",
            )
        parsed: list[tuple[str, dict]] = []
        for stage in stages:
            stage_action = str(stage.get("action") or "") if isinstance(stage, dict) else ""
            if stage_action not in RUN_RECIPE_STAGE_WHITELIST:
                return self._failure_json(
                    "invalid_recipe",
                    f"run_recipe stage action {stage_action!r} is not allowed; "
                    f"whitelist: {sorted(RUN_RECIPE_STAGE_WHITELIST)}.",
                    command="run_recipe",
                )
            stage_params = stage.get("params")
            parsed.append((stage_action, dict(stage_params) if isinstance(stage_params, dict) else {}))

        self.last_recipe = "run_recipe"
        self.last_targets = [stage_action for stage_action, _ in parsed]
        self.last_batch_size = len(parsed)
        outcomes: list[dict] = []
        stop_reason: Optional[str] = None
        for position, (stage_action, stage_params) in enumerate(parsed):
            remaining = self._time_remaining_s()
            cost = self._estimated_action_cost_s(stage_action)
            if remaining is not None and cost is not None and remaining < cost:
                stop_reason = (
                    f"stage {position} ({stage_action}) costs ~{cost / 60.0:.0f} min but only "
                    f"{max(remaining, 0) / 60.0:.0f} min remain"
                )
                logger.info("run_recipe: stopping cleanly -- %s.", stop_reason)
                break
            if position > 0:
                # Each stage is its own recorded iteration (the first one
                # rides the iteration the main loop already opened).
                self.iteration += 1
            self.last_decision_trace = {
                "llm_chosen_action": stage_action,
                "run_recipe_stage": position,
                "validation_result": "run_recipe",
            }
            response = await self.execute_validated_action(
                {"chosen_action": stage_action, "action_parameters": stage_params},
                timing_context,
            )
            failure = self._action_failure(response, default_command=stage_action)
            if failure:
                self._record_failed_action(failure)
                # Without this, the main loop's post-recipe timing probe would
                # record the failed stage's iteration a second time as a bogus
                # no-improvement entry measured on the restored-best design.
                self.recorded_iterations.add(self.iteration)
                outcomes.append({
                    "stage": position,
                    "action": stage_action,
                    "status": "failed",
                    "error_type": failure.get("error_type"),
                })
                if self.last_action_mutated_design:
                    await self._restore_best_state(f"run_recipe stage {stage_action} failed")
                continue
            # Normal recording between stages: the timing summary routes
            # through _after_tool_success -> _record_iteration_timing, so an
            # improving stage becomes the new best (and is published) and a
            # regressing one rolls back before the next stage builds on it.
            await self.call_tool("vivado_report_timing_summary", {"timeout": 300})
            outcomes.append({"stage": position, "action": stage_action, "status": "executed"})

        if not outcomes:
            self._remember_no_action_failure("run_recipe", [])
            return self._failure_json(
                "insufficient_budget",
                f"run_recipe could not start: {stop_reason or 'no stages executed'}.",
                command="run_recipe",
            )
        self.last_action_key = "run_recipe"
        self.last_recipe = "run_recipe"
        self.last_targets = [str(outcome["action"]) for outcome in outcomes]
        return json.dumps({
            "success": True,
            "recipe_stages": outcomes,
            "stopped_early": stop_reason,
        }, indent=2)

    def _known_loser_reason(self, action: str) -> Optional[str]:
        """Reason string if `action` has a decisively losing record, else None.

        Two independent sources of evidence:
        - cross-run: 0 wins / >= 3 losses on this design in previous runs;
        - this run: >= 2 regressions AND more regressions than wins.
        The second clause matters: an action with stale cross-run losses can
        still win (measured: pblock_full_replace at 0/3 cross-run gained
        +7.9 MHz at iter 7 of run 20260714_005251), but one that keeps
        regressing against THIS run's best has fresh evidence against it
        (same run, iters 5/9: -100 and -120 MHz)."""
        record = ((self.crossrun_priors or {}).get("actions") or {}).get(action) or {}
        if int(record.get("good", 0)) == 0 and int(record.get("bad", 0)) >= 3:
            return f"0 wins / {int(record['bad'])} losses cross-run"
        if self.checkpoint_manager is not None:
            wins = regressions = 0
            for it in self.checkpoint_manager.iterations:
                if str(it.get("llm_chosen_action") or it.get("recipe") or "") != action:
                    continue
                status = str(it.get("status"))
                if status in ("improved", "marginal"):
                    wins += 1
                elif status == "regression":
                    regressions += 1
            if regressions >= 2 and regressions > wins:
                return f"{regressions} regressions vs {wins} wins this run"
        return None

    def _menu_collapse_reason(self) -> Optional[str]:
        """Score item A: reason string when the action menu has collapsed to
        known losers, else None.

        Run 20260714_005251 iter 16: after exhaustion cooldowns removed every
        refinement action, the menu was exactly the three actions with
        decisively losing records, the stuck-override forced one, and it
        regressed -22 MHz; iters 17-19 then stalled to the wall. When we are
        already stalled AND every remaining choice is a known loser, the
        expected value of continuing is negative (gamma + beta cost, no
        plausible alpha) -- publish the best and stop."""
        if self.consecutive_no_improvement < CONVERGENCE_MIN_STALLS:
            return None
        allowed = list((self.last_timing_context or {}).get("allowed_actions") or [])
        if not allowed:
            return None
        verdicts = []
        for action in allowed:
            reason = self._known_loser_reason(action)
            if reason is None:
                return None
            verdicts.append(f"{action}: {reason}")
        return (
            f"menu collapsed after {self.consecutive_no_improvement} stalls -- every "
            f"remaining action has a decisively losing record ({'; '.join(verdicts)})"
        )

    def _at_logic_ceiling(self) -> Optional[str]:
        """Score item 2: return a reason string if the best Fmax is within
        LOGIC_CEILING_STOP_FRACTION of the design's zero-interconnect logic
        Fmax ceiling (measured in phase-0 diagnostics), else None.

        The logic ceiling is the Fmax with all interconnect delay modeled as
        zero -- the hard limit no placement or routing can beat, because it is
        pure logic depth. Once achieved Fmax reaches it, every further
        iteration is guaranteed to be a stall, so continuing only spends gamma
        (runtime) and beta (LLM cost) for no possible alpha gain."""
        ceiling = (self.design_signature or {}).get("logic_fmax_ceiling_mhz")
        best = self.best_fmax_mhz
        if self.checkpoint_manager is not None and self.checkpoint_manager.best_fmax_mhz is not None:
            best = self.checkpoint_manager.best_fmax_mhz
        if not ceiling or not best or ceiling <= 0:
            return None
        if best >= LOGIC_CEILING_STOP_FRACTION * ceiling:
            return (
                f"best Fmax {best:.1f} MHz is within {LOGIC_CEILING_STOP_FRACTION:.0%} of the "
                f"zero-interconnect logic ceiling {ceiling:.1f} MHz -- no placement or routing "
                f"gain is physically possible, so further iterations only cost runtime/budget"
            )
        return None

    async def _finalize_run(self) -> None:
        """Shared stop sequence: polish the best checkpoint, reopen it as the
        live design, publish it to the contest output, and print the summary.
        Callers return True after this."""
        await self._endgame_polish()
        if self.checkpoint_manager is not None:
            best_ckpt = self.checkpoint_manager.get_best_checkpoint()
            if best_ckpt:
                await self.call_tool(
                    "vivado_open_checkpoint", {"dcp_path": best_ckpt}, internal=True
                )
        self._publish_best_to_output()
        self.end_time = time.time()
        self._print_optimization_summary()

    async def _reconstrain_focus_pass(self) -> bool:
        """Score item 1: on a deeply-unmet design, re-constrain the clock to a
        barely-unmet period so place/route/phys_opt get an honest target and
        concentrate effort on the genuinely-critical paths instead of spreading
        it across thousands of equally-violating ones.

        Safety invariant: the relaxed clock is ONLY an internal optimization
        guide. The contest period is captured up front and ALWAYS restored
        before the timing report that records the result, so best_fmax and the
        published DCP are only ever measured against the real contest clock
        (Fmax = 1000/(period-WNS) is period-invariant, but calculate_fmax
        clamps positive WNS -- restoring the contest period keeps the unmet
        design's WNS negative and the recorded Fmax honest). Downside is one
        phys_opt+route cycle, gated by the normal keep-best/rollback path.
        Runs once per run."""
        if self._reconstrain_focus_done:
            return False
        contest_period = self.clock_period
        if contest_period is None or not self.target_clock:
            return False
        if self.implementation_license_available is False:
            return False
        current_wns = await self._get_current_wns()
        if current_wns is None or current_wns >= RECONSTRAIN_MIN_UNMET_WNS_NS:
            # Not deeply unmet: the tools already have a workable target and
            # this maneuver would just add a phys_opt+route cycle for nothing.
            return False
        remaining = self._time_remaining_s()
        if remaining is not None and remaining < ENDGAME_MIN_REMAINING_S:
            return False
        self._reconstrain_focus_done = True
        # Claim a fresh iteration slot: _record_iteration_timing early-returns
        # if self.iteration is already in recorded_iterations, which would
        # silently drop this pass's result (and its potential new best).
        self.iteration += 1
        self.last_decision_trace = {
            "llm_chosen_action": "reconstrain_focus",
            "reconstrain_focus": True,
            "validation_result": "reconstrain_focus",
        }

        # Achieved delay against the contest clock, and a barely-unmet target.
        achieved_delay_ns = contest_period - current_wns
        relaxed_period = max(MIN_CLOCK_TIGHTEN_STEP_NS, achieved_delay_ns * RECONSTRAIN_RELAX)
        logger.info(
            "Reconstrain focus pass: WNS %.3f ns vs contest period %.3f ns (achieved "
            "delay %.3f ns); re-constraining to %.3f ns to focus timing-driven effort.",
            current_wns, contest_period, achieved_delay_ns, relaxed_period,
        )
        print("=== Reconstrain focus pass: honest clock target for place/route effort ===\n")
        self.last_recipe = "reconstrain_focus"
        self.last_targets = [f"period:{relaxed_period:.4f}"]
        self.last_batch_size = 1
        try:
            if not await self._set_clock_period(relaxed_period):
                return False
            # phys_opt honors the relaxed target; route realizes it. Both are
            # internal so the contest period is restored before recording.
            await self._run_phys_opt_with_policy({"directive": "AggressiveExplore"})
            route = await self.call_tool(
                "vivado_route_design",
                {"directive": "Explore", "timeout": self._implementation_timeout_s(kind="route")},
                internal=True,
            )
            if self._vivado_output_has_error(route):
                logger.warning("Reconstrain focus pass: route returned errors: %s", route[:300])
        finally:
            # ALWAYS restore the contest clock before anything measures timing.
            await self._set_clock_period(contest_period)
            self.clock_period = contest_period

        # Now measure/record against the real contest clock. This routes
        # through _record_iteration_timing (keep-best + rollback on regression).
        await self.call_tool("vivado_report_timing_summary", {"timeout": 300})
        return True

    async def _endgame_polish(self) -> None:
        """Stall limit reached but wall-clock remains: spend it polishing the
        best checkpoint with cheap incremental passes instead of exiting
        early. Contest score only pays for the final DCP; unspent minutes are
        worthless, and every pass here is measured through the normal
        recording machinery (improvements become the new published best,
        regressions roll back)."""
        if self.checkpoint_manager is None:
            return
        remaining = self._time_remaining_s()
        if remaining is None or remaining < ENDGAME_MIN_REMAINING_S:
            return
        if self.implementation_license_available is False:
            return
        # Score item 1: before the cheap polish chain, try one honest-target
        # re-constrain pass on a deeply-unmet design -- it can unlock gains the
        # incremental passes below can't, and restores best on regression.
        await self._restore_best_state("reconstrain focus pass on best checkpoint")
        try:
            await self._reconstrain_focus_pass()
        except Exception as exc:
            logger.warning("Reconstrain focus pass failed (continuing to polish): %s", exc)
            await self._set_clock_period(self.clock_period)
        # Score item 2: if the reconstrain pass (or the run so far) already
        # reached the physical logic ceiling, the polish chain cannot help --
        # skip it and save the runtime.
        ceiling_reason = self._at_logic_ceiling()
        if ceiling_reason:
            logger.info("Endgame polish skipped: %s", ceiling_reason)
            await self._restore_best_state("at logic ceiling; no polish needed")
            return
        logger.info(
            "Endgame polish: stall limit hit with %.0f s of budget left; "
            "running incremental passes on the best checkpoint.",
            remaining,
        )
        print("=== Endgame polish: spending remaining budget on the best checkpoint ===\n")
        await self._restore_best_state("endgame polish on best checkpoint")
        polish_steps = [
            ("phys_opt_design", {"directive": "Explore"}),
            ("route_explore", {"directive": "Explore"}),
            ("phys_opt_design", {"directive": "AggressiveExplore"}),
            ("route_explore", {"directive": "NoTimingRelaxation"}),
            ("phys_opt_design", {"directive": "Default"}),
        ]
        # Score item 2: stop the chain after two consecutive steps that don't
        # improve the best -- the later directives are weaker variants of the
        # earlier ones, so two failures in a row means the remaining steps are
        # very unlikely to help and only cost runtime.
        best_before_step = self.checkpoint_manager.best_fmax_mhz
        consecutive_flat = 0
        for polish_action, polish_params in polish_steps:
            remaining = self._time_remaining_s()
            if remaining is not None and remaining < ENDGAME_MIN_REMAINING_S:
                logger.info("Endgame polish: %.0f s left, stopping the polish chain.", remaining or 0)
                break
            self.iteration += 1
            self.last_decision_trace = {
                "llm_chosen_action": polish_action,
                "endgame_polish": True,
                "validation_result": "endgame_polish",
            }
            response = await self.execute_validated_action(
                {"chosen_action": polish_action, "action_parameters": dict(polish_params)},
                {"worst_path": {}, "delay_class": self.path_delay_classification},
            )
            failure = self._action_failure(response, default_command=polish_action)
            if failure:
                self._record_failed_action(failure)
                if self.last_action_mutated_design:
                    await self._restore_best_state("endgame polish step failed")
                consecutive_flat += 1
                if consecutive_flat >= 2:
                    logger.info("Endgame polish: 2 consecutive steps without a new best; stopping the chain.")
                    break
                continue
            await self.call_tool("vivado_report_timing_summary", {"timeout": 300})
            best_now = self.checkpoint_manager.best_fmax_mhz
            improved = best_before_step is None or (best_now is not None and best_now > best_before_step)
            if improved:
                consecutive_flat = 0
                best_before_step = best_now
            else:
                consecutive_flat += 1
                if consecutive_flat >= 2:
                    logger.info("Endgame polish: 2 consecutive steps without a new best; stopping the chain.")
                    break

    async def _initial_diagnostics(self) -> None:
        """Phase 0 diagnostic battery (item 7): one-time probes run before the
        first LLM decision, hard-capped at INITIAL_DIAGNOSTICS_BUDGET_S total.
        Every probe is individually wrapped: a failed probe is a SKIPPED
        probe, never a failed run. Results land in self.design_signature,
        which rides along in the initial analysis message and (compactly, QoR
        text excluded) in every timing context."""
        started = time.time()
        signature: dict = {}

        def _over_budget() -> bool:
            return (time.time() - started) > INITIAL_DIAGNOSTICS_BUDGET_S

        # (a) Zero-interconnect logic floor: with interconnect delay modeled
        # as zero, the remaining WNS is pure logic depth -- the fmax ceiling
        # that NO amount of placement/routing work can beat. The delay model
        # is restored in a finally: leaving it on "none" would turn every
        # later WNS observation this run into fantasy. Read the slack via a
        # raw sentinel (not vivado_get_wns/_get_current_wns) so the strongly
        # positive floor value can never ratchet best_wns or trip the
        # positive-WNS sanity limit.
        try:
            set_result = await self.call_tool(
                "vivado_run_tcl",
                {"command": "set_delay_model -interconnect none", "timeout": 120},
                internal=True,
            )
            try:
                if not self._vivado_output_has_error(set_result):
                    raw = await self.call_tool(
                        "vivado_run_tcl",
                        {"command": (
                            "set _p [lindex [get_timing_paths -max_paths 1 -setup] 0]; "
                            "if {$_p ne {}} {puts \"LOGIC_FLOOR_WNS:[get_property SLACK $_p]\"}"
                        ), "timeout": 180},
                        internal=True,
                    )
                    match = re.search(r"LOGIC_FLOOR_WNS:([-+]?\d+(?:\.\d+)?)", raw)
                    if match:
                        logic_floor_wns = float(match.group(1))
                        signature["logic_floor_wns_ns"] = logic_floor_wns
                        period = self.current_period_ns or self.clock_period
                        if period is not None and period - logic_floor_wns > 0:
                            # Deliberately NOT calculate_fmax(): that clamps
                            # positive-WNS results to 1/period, but the whole
                            # point of the floor is the headroom past it.
                            signature["logic_fmax_ceiling_mhz"] = round(
                                1000.0 / (period - logic_floor_wns), 2
                            )
            finally:
                # The input DCP is routed, so "actual" is the model every
                # real measurement this run must use.
                await self.call_tool(
                    "vivado_run_tcl",
                    {"command": "set_delay_model -interconnect actual", "timeout": 120},
                    internal=True,
                )
        except Exception as exc:
            logger.warning("Phase 0 logic-floor probe failed (skipped): %s", exc)

        # (b) Critical-net fanout profile, from data initial analysis already
        # collected (no tool calls): high replication pressure on the worst
        # paths says fanout_split/replication before placement heroics.
        try:
            if self.high_fanout_nets:
                fanouts = sorted(int(fanout) for _, fanout, _ in self.high_fanout_nets)
                signature["critical_fanout_max"] = fanouts[-1]
                signature["critical_fanout_median"] = fanouts[len(fanouts) // 2]
                signature["critical_high_fanout_nets"] = len(fanouts)
            signature["worst_path_candidates"] = len(self.current_target_candidates)
        except Exception as exc:
            logger.warning("Phase 0 fanout profile failed (skipped): %s", exc)

        # (c) Vivado's own QoR suggestions -- cheap expert hints, but the
        # report itself can take minutes on a large design, so skip it when
        # the run cannot spare them.
        try:
            remaining = self._time_remaining_s()
            if _over_budget():
                logger.info(
                    "Phase 0: over the %d s diagnostics budget; skipping QoR suggestions.",
                    INITIAL_DIAGNOSTICS_BUDGET_S,
                )
            elif (
                self.design_scale == "large"
                and remaining is not None
                and remaining < QOR_SUGGESTIONS_MIN_REMAINING_S
            ):
                logger.info(
                    "Phase 0: large design with only %.0f min remaining; skipping QoR suggestions.",
                    remaining / 60.0,
                )
            else:
                raw = await self.call_tool(
                    "vivado_run_tcl",
                    {"command": "report_qor_suggestions -return_string", "timeout": 300},
                    internal=True,
                )
                if raw and not self._vivado_output_has_error(raw):
                    signature["qor_suggestions"] = raw.strip()[:2000]
        except Exception as exc:
            logger.warning("Phase 0 QoR suggestions probe failed (skipped): %s", exc)

        # (d) Device resource utilization -- cheap (pure RapidWright, no
        # Vivado round-trip) and the earliest possible warning that a full
        # re-place is risky on THIS design, independent of whether it has
        # ever been seen before (see _compute_resource_utilization).
        try:
            design_info_text = await self.call_tool("rapidwright_get_design_info", {}, internal=True)
            design_info = self._parse_json_result(design_info_text)
            if not self._result_has_error(design_info):
                self.resource_utilization = self._compute_resource_utilization(design_info)
                if self.resource_utilization:
                    signature["resource_utilization"] = self.resource_utilization
                    self._refresh_design_scale()
                    over_threshold = {
                        resource: fraction
                        for resource, fraction in self.resource_utilization.items()
                        if fraction >= HIGH_UTILIZATION_FRACTION
                    }
                    if over_threshold:
                        logger.warning(
                            "High device utilization detected (%s); full re-places on this "
                            "design carry real hang/non-convergence risk (limited legalization "
                            "headroom), independent of this design's own run history.",
                            {k: f"{v:.0%}" for k, v in over_threshold.items()},
                        )
        except Exception as exc:
            logger.warning("Phase 0 resource utilization probe failed (skipped): %s", exc)

        self.design_signature = signature
        if signature:
            logger.info(
                "Phase 0 design signature: %s",
                {key: value for key, value in signature.items() if key != "qor_suggestions"},
            )

    async def _maybe_warm_start_replace(self) -> bool:
        """Deterministic opening move, GATED on the design's own signature --
        not unconditional, because every DCP is different.

        A whole-design unplace + re-place is the only recipe with recorded
        wins on net-delay-bound designs (403 -> 501 MHz accidentally in run
        20260711, 403 -> 521 MHz in test mode), and its downside is bounded:
        one place/route cycle, rolled back on regression by the normal
        checkpoint machinery. So when the input shows exactly that signature
        (failing WNS + net-delay-bound critical path), run it once
        deterministically before handing control to the LLM. Designs that
        already meet timing, or whose paths are logic-bound, skip this
        entirely -- discarding a good placement is the one case where this
        bet is negative.
        """
        if self.initial_wns is None or self.initial_wns >= 0:
            logger.info("Warm start skipped: timing met or initial WNS unknown.")
            return False
        classification = await self._classify_worst_path_delay()
        if classification != "net_delay_bound":
            logger.info(
                "Warm start skipped: delay class is '%s', not net_delay_bound; "
                "the input placement is not the demonstrated bottleneck.",
                classification,
            )
            return False

        # Utilization kill switch (2026-08-01 large-design audit): the
        # forward-looking sibling of the cross-run check below. This one
        # doesn't need the design to have already failed here -- ANY design
        # whose LUT/FF/DSP/BRAM usage leaves the device this full carries
        # real legalization risk on a full re-place, known from the moment
        # the checkpoint is read (see _compute_resource_utilization). The
        # normal LLM loop's full-replace path has its own per-action
        # affordability check (_estimated_action_cost_s); the deterministic
        # warm start has none, so it's the one place a bare bet on an
        # untested, unbounded-risk design is not appropriate.
        over_threshold = {
            resource: fraction
            for resource, fraction in (self.resource_utilization or {}).items()
            if fraction >= HIGH_UTILIZATION_FRACTION
        }
        if over_threshold:
            logger.warning(
                "Warm start skipped: device utilization %s -- limited legalization headroom "
                "makes a full re-place's hang/non-convergence risk too high to spend the "
                "un-budget-checked opening move on it.",
                {k: f"{v:.0%}" for k, v in over_threshold.items()},
            )
            return False

        # Cross-run kill switch: don't bet 15-20+ minutes on a full re-place
        # that cross-run history already shows never completes or helps on
        # this exact design (see WARM_START_SKIP_AFTER_LOSSES). The main
        # loop's own full-replace gating (_full_replace_blocked_reason,
        # cross-run action demotion) still applies from here on, so this
        # isn't giving up on full re-places for this design -- just refusing
        # to spend the deterministic, un-budget-checked opening move on one
        # that's already proven itself a guaranteed loss.
        place_record = (self.crossrun_priors or {}).get("actions", {}).get("place_design_explore") or {}
        if (
            int(place_record.get("good", 0)) == 0
            and int(place_record.get("bad", 0)) >= WARM_START_SKIP_AFTER_LOSSES
        ):
            logger.warning(
                "Warm start skipped: place_design_explore has 0 wins / %d loss(es) on this "
                "design across previous runs -- a full re-place has never once completed or "
                "helped here.",
                place_record.get("bad", 0),
            )
            return False

        self.iteration = 1
        # Cross-run priors trump the hardcoded default: if a directive has a
        # winning record on this exact design from previous runs, open with it.
        warm_directive = self._best_crossrun_directive() or "Default"
        logger.info(
            "Warm start: input is net-delay-bound with failing WNS (%.3f ns); "
            "running deterministic whole-design re-place (place %s / route Default) "
            "as iteration 1.",
            self.initial_wns, warm_directive,
        )
        print("=== Warm start: deterministic whole-design re-place (iteration 1) ===\n")
        self.last_decision_trace = {
            "llm_chosen_action": "place_design_explore",
            "warm_start": True,
            "validation_result": "warm_start",
        }
        decision = {
            "chosen_action": "place_design_explore",
            "action_parameters": {"directive": warm_directive, "route_directive": "Default"},
        }
        response_text = await self.execute_validated_action(
            decision, {"worst_path": {}, "delay_class": classification}
        )
        failure = self._action_failure(response_text, default_command="place_design_explore")
        if failure:
            self._record_failed_action(failure)
            print(f"\nWarm start failed: {failure.get('error_type')}\n{failure.get('message', '')}\n")
            if self.last_action_mutated_design:
                await self._restore_best_state("warm-start re-place failed after mutating the design")
            summary = (
                f"A deterministic warm-start whole-design re-place was attempted as iteration 1 "
                f"and FAILED ({failure.get('error_type')}): {str(failure.get('message'))[:300]}"
            )
        else:
            self.cheap_failure_streak = 0
            await self.call_tool("vivado_report_timing_summary", {"timeout": 300})
            summary = (
                "A deterministic warm-start whole-design re-place (unplace + place Default + "
                "route Default) was executed as iteration 1 -- it is the recipe with the best "
                "recorded results on net-delay-bound designs. Its outcome is reflected in the "
                "timing state and place_directives_tried. Build on it: if it improved, REFINE "
                "the winning placement first (phys_opt_design, route_explore, incremental "
                "pblock) -- fresh whole-design re-places from a winning state have regressed "
                "~50-100 MHz every time they were tried on past runs; if it regressed, choose "
                "a different strategy family."
            )
        self.messages.append({"role": "user", "content": summary})
        return True

    async def optimize(self, input_dcp: Path, output_dcp: Path) -> bool:
        """Run the optimization workflow."""
        # Start timing the optimization process
        self.start_time = time.time()
        
        # Perform initial analysis without LLM
        try:
            initial_analysis = await self.perform_initial_analysis(input_dcp)
        except Exception as e:
            logger.exception(f"Initial analysis failed: {e}")
            print(f"\n✗ Initial analysis failed: {e}\n")
            self.end_time = time.time()
            return False

        self._initialize_run_helpers(input_dcp)
        self._load_crossrun_priors(input_dcp)
        # Fix #10: publish immediately (the input design itself at first) so
        # the contest output path is never empty, then re-publish on every
        # improvement and stop path.
        self.output_dcp_path = output_dcp
        self._publish_best_to_output()

        # Probe the input's own unplaced-primitive count so the provenance
        # gate can distinguish this design's benign artifacts from breakage
        # (see _unplaced_tolerance).
        try:
            self.baseline_unplaced_cells = await self._count_unplaced_cells()
            if self.baseline_unplaced_cells:
                logger.info(
                    "Input DCP has %d unplaced primitive cell(s) at baseline; "
                    "tolerating up to %d as benign design artifacts.",
                    self.baseline_unplaced_cells, self._unplaced_tolerance(),
                )
        except Exception as exc:
            logger.warning("Could not probe baseline unplaced-cell count: %s", exc)

        # Phase 0 diagnostic battery (item 7): budget-capped, failure-tolerant
        # one-time probes. Runs after _load_crossrun_priors so the QoR probe
        # can respect a prior-run "large" classification.
        try:
            await self._initial_diagnostics()
        except Exception as exc:
            logger.warning("Phase 0 diagnostics failed entirely (continuing): %s", exc)
        if self.design_signature:
            initial_analysis += (
                "\n\nDESIGN SIGNATURE (phase-0 diagnostics):\n"
                + json.dumps(self.design_signature, indent=2)
            )

        # If timing is already met, continue anyway: this contest flow pushes Fmax
        # by tightening the target clock instead of stopping at closure.
        if self.initial_wns is not None and self.initial_wns >= 0:
            print("✓ Design meets timing; continuing with Tier 2 worst-slack optimization and clock tightening.\n")
            logger.info("Design meets timing; entering Fmax-push flow")
            try:
                await self._run_clock_bisection_after_closure(self.initial_wns)
            except VivadoToolCallError as e:
                # Same class of gap as the warm-start guard above: this runs
                # place/route cycles before the main loop's try/except
                # exists, so a hang here would otherwise crash the whole run
                # instead of recovering. Fall back to the pristine input
                # checkpoint and proceed into the normal loop.
                logger.error(
                    "Clock bisection failed with a Vivado-side error (%s); restoring "
                    "the input checkpoint and proceeding into the normal loop without it.",
                    e,
                )
                await self._restore_best_state("clock bisection failed with a Vivado tool error")
        
        # Initialize conversation with analysis results
        self.messages = [
            {"role": "system", "content": TIMING_DECISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Optimize this FPGA design for timing and Fmax.

PATHS:
- Input DCP: {input_dcp.resolve()}
- Output DCP (save final result here): {output_dcp.resolve()}
- Run directory (for intermediate files): {self.temp_dir}

CURRENT STATE:
- Vivado has the input design ALREADY OPEN and analyzed
- RapidWright has the input design ALREADY LOADED (from initial analysis)

INITIAL ANALYSIS RESULTS:
{initial_analysis}

Proceed by selecting exactly one validated action per timing-context turn."""
            }
        ]

        # Signature-gated deterministic opening move (see the method's
        # docstring). Consumes iteration 1 when it runs; the LLM loop then
        # continues from iteration 2 with the warm start's outcome in its
        # history, directive-sweep memory, and conversation.
        try:
            await self._maybe_warm_start_replace()
        except VivadoToolCallError as e:
            # 2026-08-01 ispd16_example2 incident: a place_design hang during
            # warm start crashed the whole run with an unhandled
            # VivadoToolCallError. call_tool's hang/crash auto-recovery
            # already restarted the Vivado PROCESS before raising this, but
            # this call site sits entirely outside the main loop below --
            # nothing here reopens a design into the fresh session, and
            # nothing catches the exception. Fall back to the pristine input
            # checkpoint (best_checkpoint before any iteration has recorded)
            # and let the main loop proceed normally -- a skipped warm start
            # is a minor optimization loss, not a reason to abort the run.
            logger.error(
                "Warm start failed with a Vivado-side error (%s); restoring the "
                "input checkpoint and proceeding into the normal loop without it.",
                e,
            )
            await self._restore_best_state("warm start failed with a Vivado tool error")

        max_iterations = 50  # Safety limit

        print("=== Starting LLM-Driven Optimization ===\n")
        
        while self.iteration < max_iterations:
            self.iteration += 1
            iteration_started_at = time.time()
            logger.info(f"=== Iteration {self.iteration} ===")

            try:
                await self._append_iteration_context()
                # Score item A: if the menu has collapsed to known losers
                # while stalled, stop before paying for an LLM call and a
                # doomed Vivado cycle -- publish the best instead.
                collapse_reason = self._menu_collapse_reason()
                if collapse_reason:
                    logger.info("Convergence early-exit: %s", collapse_reason)
                    print(f"\n=== Converged: {collapse_reason} ===\n")
                    await self._finalize_run()
                    return True
                decision = await self.get_validated_action_decision(self.last_timing_context)
                response_text = await self.execute_validated_action(decision, self.last_timing_context)
                failure = self._action_failure(response_text, default_command=str(decision.get("chosen_action")))
                if failure:
                    self._record_failed_action(failure)
                    print(f"\nAction failed: {failure.get('error_type')}\n{failure.get('message', '')}\n")
                    if self.last_action_mutated_design:
                        # The failed action already issued mutating commands
                        # (place/route/phys_opt/pblock/unplace or a RapidWright
                        # edit), so the live session no longer matches
                        # best_checkpoint -- and some failure paths (e.g. an
                        # error detected only in the combined output) leave a
                        # half-transformed design live. Never let the next
                        # iteration build on that.
                        await self._restore_best_state(
                            f"action {failure.get('command')} failed after mutating the design"
                        )
                    # BUG FIX: this `continue` used to skip straight to the next
                    # loop iteration, bypassing the ABSOLUTE_STALL_HARD_LIMIT
                    # check below entirely -- since _record_failed_action already
                    # incremented consecutive_no_improvement, a run that fails at
                    # the action-execution level (rather than completing but not
                    # improving WNS) could stall indefinitely (up to
                    # max_iterations) without ever hitting the hard stop. Runs
                    # dominated by "failed" iterations (e.g. every offered action
                    # cooling down / geometrically invalid) are exactly the case
                    # this limit exists for, so check it here too before moving on.
                    #
                    # History(16) refinement: the 5-stall budget was calibrated
                    # for stalls that each burn a full place/route (real score
                    # damage). Validation failures that abort in seconds must
                    # not consume it -- history(16) died after 5 failures that
                    # used almost no Vivado time, abandoning the whole
                    # remaining budget. Cheap failures count against a
                    # separate, much larger cap instead.
                    iteration_elapsed = time.time() - iteration_started_at
                    if iteration_elapsed < CHEAP_FAILURE_RUNTIME_S:
                        self.cheap_failure_streak += 1
                    else:
                        self.cheap_failure_streak = 0
                    effective_stalls = self.consecutive_no_improvement - self.cheap_failure_streak
                    if (
                        effective_stalls >= ABSOLUTE_STALL_HARD_LIMIT
                        or self.consecutive_no_improvement >= STALL_LIMIT_CHEAP_FAILURES
                    ):
                        logger.error(
                            "Hard stall limit reached (%d effective stalls, %d total "
                            "consecutive failures of which %d were cheap); "
                            "stopping and restoring best checkpoint.",
                            effective_stalls,
                            self.consecutive_no_improvement,
                            self.cheap_failure_streak,
                        )
                        await self._endgame_polish()
                        if self.checkpoint_manager is not None:
                            best_ckpt = self.checkpoint_manager.get_best_checkpoint()
                            if best_ckpt:
                                await self.call_tool(
                                    "vivado_open_checkpoint", {"dcp_path": best_ckpt}, internal=True
                                )
                        self._publish_best_to_output()
                        self.end_time = time.time()
                        self._print_optimization_summary()
                        return True
                    continue
                # An action executed without failing: the cheap-failure streak
                # is broken regardless of whether WNS improved.
                self.cheap_failure_streak = 0
                await self.call_tool("vivado_report_timing_summary", {"timeout": 300})
                print(f"\n{response_text}\n")
                
                current_wns = await self._get_current_wns()
                if current_wns is not None and current_wns >= 0 and self.current_period_ns is not None:
                    await self._run_clock_bisection_after_closure(current_wns)

                # Score item 2: stop the moment the design reaches its physical
                # logic ceiling -- no further alpha is possible, so every extra
                # iteration is pure gamma/beta cost.
                ceiling_reason = self._at_logic_ceiling()
                if ceiling_reason:
                    logger.info("Convergence early-exit: %s", ceiling_reason)
                    print(f"\n=== Converged: {ceiling_reason} ===\n")
                    await self._finalize_run()
                    return True

                if self.checkpoint_manager is not None and not self.checkpoint_manager.should_continue():
                    logger.info("Optimization workflow completed")
                    self._publish_best_to_output()
                    self.end_time = time.time()
                    self._print_optimization_summary()
                    return True

                if self.consecutive_no_improvement >= ABSOLUTE_STALL_HARD_LIMIT:
                    logger.error(
                        "Hard stall limit (%d) reached with no improvement; "
                        "stopping and restoring best checkpoint.",
                        ABSOLUTE_STALL_HARD_LIMIT,
                    )
                    await self._endgame_polish()
                    if self.checkpoint_manager is not None:
                        best_ckpt = self.checkpoint_manager.get_best_checkpoint()
                        if best_ckpt:
                            await self.call_tool(
                                "vivado_open_checkpoint", {"dcp_path": best_ckpt}, internal=True
                            )
                    self._publish_best_to_output()
                    self.end_time = time.time()
                    self._print_optimization_summary()
                    return True

            except VivadoToolCallError as e:
                # A real Vivado/RapidWright-side failure occurred (e.g. a pipe
                # desync + auto-restart on the MCP server). The live Vivado
                # session may now be empty/stateless, so resync by reopening
                # the last-known-good checkpoint before continuing, rather than
                # letting subsequent commands run blind against nothing.
                logger.error(f"Vivado tool call failed ({e.tool_name}); reopening last good checkpoint to resync state.")
                best_ckpt = self.checkpoint_manager.get_best_checkpoint() if self.checkpoint_manager else None
                if best_ckpt:
                    try:
                        await self.call_tool("vivado_open_checkpoint", {"dcp_path": best_ckpt}, internal=True)
                        await self.call_tool("rapidwright_read_checkpoint", {"dcp_path": best_ckpt}, internal=True)
                    except Exception as reopen_exc:
                        logger.exception(f"Failed to reopen checkpoint after desync recovery: {reopen_exc}")
                        self._publish_best_to_output()
                        self.end_time = time.time()
                        raise
                else:
                    logger.error("No known-good checkpoint to reopen; aborting run.")
                    self._publish_best_to_output()
                    self.end_time = time.time()
                    raise
                continue

            except Exception as e:
                logger.exception(f"Error during optimization: {e}")
                self._publish_best_to_output()
                self.end_time = time.time()
                raise
        
        logger.warning("Reached maximum iterations")
        self._publish_best_to_output()
        self.end_time = time.time()
        self._print_optimization_summary(max_iterations_reached=True)
        return False
    
    def save_token_usage_report(self, output_path: Path):
        """Save detailed token usage report to JSON file."""
        # Calculate total cached and reasoning tokens
        total_cached = sum(detail.get('cached_tokens', 0) for detail in self.api_call_details)
        total_reasoning = sum(detail.get('reasoning_tokens', 0) for detail in self.api_call_details)
        
        # Calculate tool call statistics
        total_tool_time = sum(detail['elapsed_time'] for detail in self.tool_call_details)
        tool_counts = {}
        for detail in self.tool_call_details:
            tool_name = detail['tool_name']
            if tool_name not in tool_counts:
                tool_counts[tool_name] = 0
            tool_counts[tool_name] += 1
        
        # Calculate total runtime
        total_runtime = None
        if self.start_time is not None:
            total_runtime = (self.end_time or time.time()) - self.start_time
        
        # Calculate fmax values from VALIDATED checkpoint history when
        # available (see _print_optimization_summary for why the raw
        # self.best_wns/self.best_fmax_mhz ratchets are not trustworthy).
        cm = self.checkpoint_manager
        if cm is not None and cm.best_wns is not None:
            initial_fmax = cm.baseline_fmax_mhz
            best_fmax = cm.best_fmax_mhz
            best_wns = cm.best_wns
            initial_wns = cm.baseline_wns
            best_checkpoint = cm.get_best_checkpoint()
        else:
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            best_fmax = self.best_fmax_mhz
            if best_fmax is None and self.best_wns > float('-inf'):
                best_fmax = self.calculate_fmax(self.best_wns, self.clock_period)
            best_wns = self.best_wns if self.best_wns > float('-inf') else None
            initial_wns = self.initial_wns
            best_checkpoint = None
        fmax_improvement = (best_fmax - initial_fmax) if (initial_fmax is not None and best_fmax is not None) else None
        
        report = {
            "model": self.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_runtime_seconds": total_runtime,
                "total_llm_calls": self.llm_call_count,
                "total_iterations": self.iteration,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "total_cached_tokens": total_cached,
                "total_reasoning_tokens": total_reasoning,
                "total_cost": self.total_cost,
                "clock_period_ns": self.clock_period,
                "current_period_ns": self.current_period_ns,
                "initial_wns": initial_wns,
                "best_wns": best_wns,
                "wns_improvement": (best_wns - initial_wns) if (best_wns is not None and initial_wns is not None) else None,
                "initial_fmax_mhz": initial_fmax,
                "best_fmax_mhz": best_fmax,
                "fmax_improvement_mhz": fmax_improvement,
                "best_checkpoint": best_checkpoint,
                "total_tool_calls": len(self.tool_call_details),
                "total_tool_time_seconds": total_tool_time,
                "tool_call_counts": tool_counts
            },
            "per_llm_call_details": self.api_call_details,
            "per_tool_call_details": self.tool_call_details
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Token usage report saved to {output_path}")
    
    def _print_optimization_summary(self, max_iterations_reached: bool = False):
        """Print detailed optimization summary including token usage and costs."""
        self._save_crossrun_priors()
        # Final benchmark_score persist: pick up the end-of-run wall clock and
        # the full LLM spend so history.json's score matches contest scoring.
        if self.checkpoint_manager is not None and self.checkpoint_manager.baseline_fmax_mhz is not None:
            try:
                self.checkpoint_manager.set_llm_cost_usd(float(self.total_cost or 0.0))
            except Exception as exc:
                logger.warning("Could not persist final benchmark score: %s", exc)
        title = "Optimization Summary (Max Iterations Reached)" if max_iterations_reached else "Optimization Summary"
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")
        
        # Calculate total runtime
        if self.start_time is not None:
            total_runtime = (self.end_time or time.time()) - self.start_time
            print(f"\nTOTAL RUNTIME: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
        
        # Summary correctness: report ONLY validated numbers. The raw
        # self.best_wns / self.best_fmax_mhz ratchets record every WNS ever
        # observed mid-flight -- including observations from states that were
        # later rolled back or rejected -- so they can only overstate the
        # result. CheckpointManager.best_wns/best_fmax_mhz are updated
        # exclusively by record() for iterations whose checkpoint was written
        # and classified improved/marginal, and best_checkpoint is the DCP
        # those numbers were measured on.
        cm = self.checkpoint_manager
        if cm is not None and cm.best_wns is not None:
            result_lines = self._format_fmax_results(
                cm.clock_period_ns, cm.baseline_wns, cm.best_wns, result_label="Best"
            )
            if result_lines:
                print(f"\nFMAX RESULTS (validated from checkpoint history):")
                print("\n".join(result_lines))
            if cm.best_fmax_mhz is not None:
                print(f"  {'Best achieved Fmax:':<21s}{cm.best_fmax_mhz:8.2f} MHz")
            score_block = cm.benchmark_score()
            if score_block.get("score") is not None:
                print(
                    f"\nBENCHMARK SCORE: {score_block['score']:.3f}  "
                    f"(alpha={score_block['alpha_delta_fmax_mhz']:.2f} MHz, "
                    f"beta=${score_block['beta_openrouter_cost_usd']:.4f}, "
                    f"gamma={score_block['gamma_runtime_hours']:.4f} h)"
                )
            print(f"  {'Best checkpoint:':<21s}{cm.get_best_checkpoint()}")
        else:
            best_wns = self.best_wns if self.best_wns > float('-inf') else None
            result_lines = self._format_fmax_results(
                self.clock_period, self.initial_wns, best_wns, result_label="Best"
            )
            if result_lines:
                print(f"\nFMAX RESULTS (unvalidated -- no checkpoint history):")
                print("\n".join(result_lines))
            if self.best_fmax_mhz is not None:
                print(f"  {'Best achieved Fmax:':<21s}{self.best_fmax_mhz:8.2f} MHz")
        if self.current_period_ns is not None:
            print(f"  {'Current period:':<21s}{self.current_period_ns:8.3f} ns")
        
        # Iteration stats
        print(f"\nITERATION STATS:")
        print(f"  Total iterations:    {self.iteration}")
        print(f"  LLM API calls:       {self.llm_call_count}")
        
        # Token usage
        print(f"\nTOKEN USAGE:")
        print(f"  Prompt tokens:       {self.total_prompt_tokens:,}")
        print(f"  Completion tokens:   {self.total_completion_tokens:,}")
        print(f"  Total tokens:        {self.total_tokens:,}")
        
        # Calculate total cached and reasoning tokens
        total_cached = sum(detail.get('cached_tokens', 0) for detail in self.api_call_details)
        total_reasoning = sum(detail.get('reasoning_tokens', 0) for detail in self.api_call_details)
        
        if total_cached > 0:
            print(f"  Cached tokens:       {total_cached:,} (saved cost)")
        if total_reasoning > 0:
            print(f"  Reasoning tokens:    {total_reasoning:,}")
        
        # Cost
        print(f"\nCOST:")
        print(f"  Model:               {self.model}")
        if self.total_cost > 0:
            print(f"  Total cost:          ${self.total_cost:.4f}")
        else:
            print(f"  Total cost:          Not available")
        
        # Tool call summary
        if self.tool_call_details:
            print(f"\nTOOL CALLS SUMMARY:")
            print(f"  Total tool calls:    {len(self.tool_call_details)}")
            
            # Calculate total time spent in tool calls
            total_tool_time = sum(detail['elapsed_time'] for detail in self.tool_call_details)
            print(f"  Total tool time:     {total_tool_time:.2f}s")
            
            # Count by tool type
            tool_counts = {}
            for detail in self.tool_call_details:
                tool_name = detail['tool_name']
                if tool_name not in tool_counts:
                    tool_counts[tool_name] = 0
                tool_counts[tool_name] += 1
            
            print(f"\n  Tool call breakdown:")
            for tool_name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
                print(f"    {tool_name}: {count}")
            
            # Detailed tool call list
            print(f"\n  Detailed tool call log:")
            print(f"  {'#':<5} {'Iter':<6} {'Tool':<40} {'Time (s)':<12} {'WNS (ns)':<12} {'Status':<10}")
            print(f"  {'-'*5} {'-'*6} {'-'*40} {'-'*12} {'-'*12} {'-'*10}")
            
            for i, detail in enumerate(self.tool_call_details, 1):
                tool_name = detail['tool_name']
                iteration = detail.get('iteration', 0)
                elapsed = detail['elapsed_time']
                wns = detail.get('wns')
                error = detail.get('error', False)
                
                # Format WNS column
                wns_str = f"{wns:.3f}" if wns is not None else "-"
                
                # Format status
                status_str = "ERROR" if error else "OK"
                
                print(f"  {i:<5} {iteration:<6} {tool_name:<40} {elapsed:<12.2f} {wns_str:<12} {status_str:<10}")
                
                # If error, show error message on next line
                if error and 'error_message' in detail:
                    print(f"        Error: {detail['error_message'][:80]}")
        
        # Per-call breakdown if debug mode
        if self.debug and self.api_call_details:
            print(f"\nPER-CALL BREAKDOWN:")
            
            # Check if we have cached or reasoning tokens to display
            has_cached = any(detail.get('cached_tokens', 0) > 0 for detail in self.api_call_details)
            has_reasoning = any(detail.get('reasoning_tokens', 0) > 0 for detail in self.api_call_details)
            has_cost = any(detail.get('cost', 0) > 0 for detail in self.api_call_details)
            
            # Build header
            header = f"  {'Call':<6} {'Iter':<6} {'Prompt':<10} {'Completion':<12}"
            if has_cached:
                header += f" {'Cached':<10}"
            if has_reasoning:
                header += f" {'Reasoning':<10}"
            header += f" {'Total':<10}"
            if has_cost:
                header += f" {'Cost':<12}"
            print(header)
            
            # Build separator
            separator = f"  {'-'*6} {'-'*6} {'-'*10} {'-'*12}"
            if has_cached:
                separator += f" {'-'*10}"
            if has_reasoning:
                separator += f" {'-'*10}"
            separator += f" {'-'*10}"
            if has_cost:
                separator += f" {'-'*12}"
            print(separator)
            
            # Print details
            for detail in self.api_call_details:
                line = (f"  {detail['call_number']:<6} {detail['iteration']:<6} "
                       f"{detail['prompt_tokens']:<10,} {detail['completion_tokens']:<12,}")
                if has_cached:
                    line += f" {detail.get('cached_tokens', 0):<10,}"
                if has_reasoning:
                    line += f" {detail.get('reasoning_tokens', 0):<10,}"
                line += f" {detail['total_tokens']:<10,}"
                if has_cost:
                    cost = detail.get('cost', 0)
                    line += f" ${cost:<11.4f}" if cost > 0 else f" {'N/A':<12}"
                print(line)
        
        print(f"\n{'='*70}\n")
        
        # Save detailed report to JSON in run directory
        try:
            report_path = self.run_dir / "token_usage.json"
            self.save_token_usage_report(report_path)
            print(f"Detailed token usage report saved to: {report_path}\n")
        except Exception as e:
            logger.warning(f"Failed to save token usage report: {e}")
    


class FPGAOptimizerTest(DCPOptimizerBase):
    """
    Test mode for FPGA Design Optimization - hardcodes all tool calls to diagnose issues.
    
    This class runs a deterministic optimization flow without using any LLM, 
    making it easier to identify where MCP servers or Vivado might hang.
    """
    
    def __init__(self, debug: bool = False, run_dir: Optional[Path] = None):
        super().__init__(debug=debug, run_dir=run_dir)
        self.final_wns = None
    
    async def start_servers(self):
        """Start and connect to both MCP servers."""
        await super().start_servers(log_prefix="[TEST]")
    
    async def call_vivado_tool(self, tool_name: str, arguments: dict, timeout: float = 300.0) -> str:
        """Execute a Vivado tool call with timing and logging."""
        logger.info(f"[VIVADO] Calling {tool_name} with args: {json.dumps(arguments)[:200]}...")
        print(f"[TEST] Calling vivado_{tool_name}...")
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.vivado_session.call_tool(tool_name, arguments),
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[VIVADO] {tool_name} completed in {elapsed:.2f}s")
            print(f"[TEST] vivado_{tool_name} completed in {elapsed:.2f}s")
            
            # Extract text content from result
            if result.content:
                text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                return "\n".join(text_parts)
            return "(no output)"
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[VIVADO] {tool_name} TIMED OUT after {elapsed:.2f}s")
            print(f"[TEST] ERROR: vivado_{tool_name} TIMED OUT after {elapsed:.2f}s")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VIVADO] {tool_name} FAILED after {elapsed:.2f}s: {e}")
            print(f"[TEST] ERROR: vivado_{tool_name} failed after {elapsed:.2f}s: {e}")
            raise
    
    async def call_rapidwright_tool(self, tool_name: str, arguments: dict, timeout: float = 300.0) -> str:
        """Execute a RapidWright tool call with timing and logging."""
        logger.info(f"[RAPIDWRIGHT] Calling {tool_name} with args: {json.dumps(arguments)[:200]}...")
        print(f"[TEST] Calling rapidwright_{tool_name}...")
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.rapidwright_session.call_tool(tool_name, arguments),
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[RAPIDWRIGHT] {tool_name} completed in {elapsed:.2f}s")
            print(f"[TEST] rapidwright_{tool_name} completed in {elapsed:.2f}s")
            
            # Extract text content from result
            if result.content:
                text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                return "\n".join(text_parts)
            return "(no output)"
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[RAPIDWRIGHT] {tool_name} TIMED OUT after {elapsed:.2f}s")
            print(f"[TEST] ERROR: rapidwright_{tool_name} TIMED OUT after {elapsed:.2f}s")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[RAPIDWRIGHT] {tool_name} FAILED after {elapsed:.2f}s: {e}")
            print(f"[TEST] ERROR: rapidwright_{tool_name} failed after {elapsed:.2f}s: {e}")
            raise
    
    def parse_wns_from_timing_report(self, timing_report: str) -> Optional[float]:
        """Extract WNS from timing report using shared parsing logic."""
        return parse_timing_summary_static(timing_report)["wns"]
    
    async def _call_vivado_for_clock(self, tool_name: str, arguments: dict) -> str:
        """Helper to call Vivado tools for clock period query."""
        return await self.call_vivado_tool(tool_name, arguments, timeout=60.0)
    
    async def fetch_clock_period(self) -> Optional[float]:
        """Query clock period with test-mode logging."""
        period = await super().get_clock_period(self._call_vivado_for_clock)
        if period is not None:
            clock_info = f" (target clock: {self.target_clock})" if self.target_clock else ""
            print(f"[TEST] Clock period: {period:.3f} ns{clock_info}")
        else:
            print("[TEST] WARNING: Could not parse clock period from Vivado")
        return period
    
    async def run_test(self, input_dcp: Path, output_dcp: Path, max_nets_to_optimize: int = 5) -> bool:
        """
        Run the deterministic test optimization flow.
        
        Steps:
        1. Open the input DCP in Vivado
        2. Report timing in Vivado
        3. Get the critical high fan out nets from Vivado
        4. Open the DCP in RapidWright
        5. Apply the fanout optimization for each high fanout net
        6. Write a DCP out from RapidWright
        7. Read the RapidWright generated DCP into Vivado
        8. Route the design in Vivado
        9. Report timing and compare WNS
        """
        print("\n" + "="*70)
        print("FPGA OPTIMIZER TEST MODE")
        print("="*70)
        print(f"Input DCP:  {input_dcp}")
        print(f"Output DCP: {output_dcp}")
        print(f"Temp dir:   {self.temp_dir}")
        print(f"Max nets to optimize: {max_nets_to_optimize}")
        print("="*70 + "\n")
        
        overall_start = time.time()
        
        try:
            # ================================================================
            # Step 0: Initialize RapidWright (Vivado starts automatically)
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 0: Initialize RapidWright")
            print("-"*60)
            
            # Initialize RapidWright (Vivado will auto-start when first used)
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            print(f"RapidWright init result:\n{result[:500]}...")
            logger.info(f"RapidWright init result: {result}")
            
            # ================================================================
            # Step 1: Open the input DCP in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 1: Open input DCP in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"Open checkpoint result:\n{result}")
            logger.info(f"Open checkpoint result: {result}")
            
            # ================================================================
            # Step 2: Report timing in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 2: Report timing in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Initial timing summary: {result}")
            
            # Get clock period for fmax calculation (also detects target clock)
            self.clock_period = await self.fetch_clock_period()
            
            # Get WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                self.initial_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Initial", self.initial_wns)
            logger.info(f"Initial WNS: {self.initial_wns} ns")
            print()
            
            # ================================================================
            # Step 3: Get critical high fanout nets
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 3: Get critical high fanout nets")
            print("-"*60)
            
            result = await self.call_vivado_tool("get_critical_high_fanout_nets", {
                "num_paths": 50,
                "min_fanout": 100,
                "exclude_clocks": True
            }, timeout=600.0)
            print(f"High fanout nets report:\n{result}")
            logger.info(f"High fanout nets: {result}")
            
            # Parse the nets
            self.high_fanout_nets = self.parse_high_fanout_nets(result)
            print(f"\nParsed {len(self.high_fanout_nets)} high fanout nets")
            
            if not self.high_fanout_nets:
                print("WARNING: No high fanout nets found to optimize!")
                logger.warning("No high fanout nets found to optimize")
            
            # Select top nets to optimize
            nets_to_optimize = self.high_fanout_nets[:max_nets_to_optimize]
            print(f"Will optimize {len(nets_to_optimize)} nets:")
            for net_name, fanout, path_count in nets_to_optimize:
                print(f"  - {net_name} (fanout={fanout}, paths={path_count})")
            
            # ================================================================
            # Step 4: Open the DCP in RapidWright
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 4: Open DCP in RapidWright")
            print("-"*60)
            
            result = await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"RapidWright read checkpoint result:\n{result}")
            logger.info(f"RapidWright read checkpoint: {result}")
            
            # ================================================================
            # Step 5: Apply fanout optimization for each high fanout net
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 5: Apply fanout optimizations in RapidWright")
            print("-"*60)
            
            successful_optimizations = 0
            for i, (net_name, fanout, path_count) in enumerate(nets_to_optimize):
                print(f"\n[{i+1}/{len(nets_to_optimize)}] Optimizing net: {net_name}")
                print(f"    Fanout: {fanout}, Critical paths: {path_count}")
                
                # Calculate split factor: fanout/100, min 2, max 8
                split_factor = max(2, min(8, fanout // 100))
                print(f"    Split factor: {split_factor}")
                
                try:
                    result = await self.call_rapidwright_tool("optimize_fanout", {
                        "net_name": net_name,
                        "split_factor": split_factor
                    }, timeout=300.0)
                    print(f"    Result: {result[:500]}...")
                    logger.info(f"Optimize fanout {net_name}: {result}")
                    
                    # Check if successful
                    if "error" not in result.lower() or "success" in result.lower():
                        successful_optimizations += 1
                except Exception as e:
                    print(f"    FAILED: {e}")
                    logger.error(f"Failed to optimize {net_name}: {e}")
            
            print(f"\nSuccessfully optimized {successful_optimizations}/{len(nets_to_optimize)} nets")
            
            # ================================================================
            # Step 6: Write DCP from RapidWright
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 6: Write DCP from RapidWright")
            print("-"*60)
            
            rapidwright_dcp = Path(self.temp_dir) / "rapidwright_optimized.dcp"
            result = await self.call_rapidwright_tool("write_checkpoint", {
                "dcp_path": str(rapidwright_dcp),
                "overwrite": True
            }, timeout=600.0)
            print(f"Write checkpoint result:\n{result}")
            logger.info(f"RapidWright write checkpoint: {result}")
            
            # Check if the file was created
            if rapidwright_dcp.exists():
                print(f"DCP file created: {rapidwright_dcp} ({rapidwright_dcp.stat().st_size} bytes)")
            else:
                print("WARNING: DCP file was not created!")
                logger.warning("RapidWright DCP file not created")
            
            # ================================================================
            # Step 7: Read RapidWright DCP into Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 7: Read RapidWright DCP into Vivado")
            print("-"*60)
            
            # Note: Opening a RapidWright-generated DCP takes MUCH longer than
            # opening the original DCP because:
            # 1. Vivado must reload encrypted IP blocks from disk
            # 2. Vivado must reconstruct internal data structures
            # For large designs, this can take 10-30 minutes
            RAPIDWRIGHT_DCP_TIMEOUT = 300.0  # 5 minutes
            
            # Check if there's a Tcl script we need to source first (for encrypted IP)
            tcl_script = rapidwright_dcp.with_suffix('.tcl')
            if tcl_script.exists():
                print(f"Found Tcl script for encrypted IP: {tcl_script}")
                print(f"Note: This may take 10-30 minutes for large designs...")
                # Source the Tcl script instead of directly opening the DCP
                result = await self.call_vivado_tool("run_tcl", {
                    "command": f"source {{{tcl_script}}}"
                }, timeout=RAPIDWRIGHT_DCP_TIMEOUT)
                print(f"Source Tcl script result:\n{result}")
            else:
                # Opening a RapidWright-generated DCP can take longer than original
                # because Vivado needs to reconstruct some internal data structures
                result = await self.call_vivado_tool("open_checkpoint", {
                    "dcp_path": str(rapidwright_dcp)
                }, timeout=RAPIDWRIGHT_DCP_TIMEOUT)
                print(f"Open RapidWright DCP result:\n{result}")
            logger.info(f"Open RapidWright DCP: {result}")
            
            # ================================================================
            # Step 8: Route the design in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 8: Route design in Vivado")
            print("-"*60)
            
            # First check route status
            result = await self.call_vivado_tool("report_route_status", {
                "show_unrouted": True,
                "show_errors": True,
                "max_nets": 20
            }, timeout=300.0)
            print(f"Route status before routing:\n{result[:1500]}...")
            logger.info(f"Route status before routing: {result}")
            
            # Route the design
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default",
            }, timeout=600.0)  # 2 hour timeout for routing
            print(f"Route design result:\n{result}")
            logger.info(f"Route design: {result}")
            
            # Check route status again
            result = await self.call_vivado_tool("report_route_status", {
                "show_unrouted": True,
                "show_errors": True,
                "max_nets": 20
            }, timeout=300.0)
            print(f"Route status after routing:\n{result[:1500]}...")
            logger.info(f"Route status after routing: {result}")
            
            # ================================================================
            # Step 9: Report final timing
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 9: Report final timing")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Final timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Final timing summary: {result}")
            
            # Get final WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                self.final_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Final", self.final_wns)
            logger.info(f"Final WNS: {self.final_wns} ns")
            print()
            
            # ================================================================
            # Write final DCP and report results
            # ================================================================
            self.print_wns_change(self.initial_wns, self.final_wns, self.clock_period)
            
            # Always write the final checkpoint (regardless of improvement)
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            print(f"Write final DCP result:\n{result}")
            
            # ================================================================
            # Summary
            # ================================================================
            elapsed = time.time() - overall_start
            self.print_test_summary(
                title="TEST SUMMARY",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Nets optimized: {successful_optimizations}/{len(nets_to_optimize)}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"Test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False
    
    async def run_test_logicnets(self, input_dcp: Path, output_dcp: Path) -> bool:
        """
        Run the pblock-based optimization flow for LogicNets designs.
        
        Steps:
        1. Open the input DCP in Vivado
        2. Report timing in Vivado (Initialize WNS)
        3. Run the Vivado tool extract_critical_path_cells
        4. Run the RapidWright tool analyze_critical_path_spread
        5. Use known-optimal pblock range for LogicNets (SLICE_X55Y60:SLICE_X111Y254)
        6. Unplace the design in Vivado
        7. Create and apply pblock to entire design
        8. Place the design in Vivado
        9. Route the design in Vivado
        10. Report timing in Vivado (compare against initial WNS)
        """
        pblock_ranges = "SLICE_X55Y60:SLICE_X111Y254"
        
        print("\n" + "="*70)
        print("FPGA OPTIMIZER TEST MODE - LOGICNETS PBLOCK FLOW")
        print("="*70)
        print(f"Input DCP:  {input_dcp}")
        print(f"Output DCP: {output_dcp}")
        print(f"Temp dir:   {self.temp_dir}")
        print("="*70 + "\n")
        
        overall_start = time.time()
        
        try:
            # ================================================================
            # Step 0: Initialize RapidWright (Vivado starts automatically)
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 0: Initialize RapidWright")
            print("-"*60)
            
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            print(f"RapidWright init result:\n{result[:500]}...")
            logger.info(f"RapidWright init result: {result}")
            
            # ================================================================
            # Step 1: Open the input DCP in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 1: Open input DCP in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"Open checkpoint result:\n{result}")
            logger.info(f"Open checkpoint result: {result}")
            
            # ================================================================
            # Step 2: Report timing in Vivado (Initialize WNS)
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 2: Report initial timing in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Initial timing summary: {result}")
            
            # Get clock period for fmax calculation (also detects target clock)
            self.clock_period = await self.fetch_clock_period()
            
            # Get WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                self.initial_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Initial", self.initial_wns)
            logger.info(f"Initial WNS: {self.initial_wns} ns")
            print()
            
            # ================================================================
            # Step 3: Extract critical path cells from Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 3: Extract critical path cells")
            print("-"*60)
            
            # Write to a file for efficient data transfer
            critical_paths_file = Path(self.temp_dir) / "critical_paths.json"
            result = await self.call_vivado_tool("extract_critical_path_cells", {
                "num_paths": 50,
                "output_file": str(critical_paths_file)
            }, timeout=600.0)
            print(f"Extract critical paths result:\n{result[:2000]}...")
            logger.info(f"Extract critical paths: {result}")
            
            # ================================================================
            # Step 4: Open DCP in RapidWright and analyze critical path spread
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 4: Analyze critical path spread in RapidWright")
            print("-"*60)
            
            # First, open the DCP in RapidWright
            result = await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"RapidWright read checkpoint result:\n{result}")
            logger.info(f"RapidWright read checkpoint: {result}")
            
            # Analyze critical path spread
            result = await self.call_rapidwright_tool("analyze_critical_path_spread", {
                "input_file": str(critical_paths_file)
            }, timeout=300.0)
            print(f"Critical path spread analysis:\n{result[:3000] if isinstance(result, str) else str(result)[:3000]}...")
            logger.info(f"Critical path spread: {result}")
            
            # Parse the spread analysis result to check if pblock is recommended
            spread_result = result if isinstance(result, str) else str(result)
            pblock_recommended = "spread-out" in spread_result.lower() or "pblock" in spread_result.lower()
            print(f"\n*** Pblock optimization {'RECOMMENDED' if pblock_recommended else 'may not be needed'} ***")
            
            # ================================================================
            # Step 5: Apply pblock constraint for LogicNets
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 5: Apply pblock for LogicNets")
            print("-"*60)
            
            print(f"Using pblock range: {pblock_ranges}")
            
            # ================================================================
            # Step 6: Unplace the design in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 6: Unplace the design in Vivado")
            print("-"*60)
            
            # Use place_design -unplace to remove all placement
            result = await self.call_vivado_tool("run_tcl", {
                "command": "place_design -unplace"
            }, timeout=300.0)
            print(f"Unplace result:\n{result}")
            logger.info(f"Unplace result: {result}")
            
            # ================================================================
            # Step 7: Create and apply pblock to entire design
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 7: Create and apply pblock to entire design")
            print("-"*60)
            
            result = await self.call_vivado_tool("create_and_apply_pblock", {
                "pblock_name": "pblock_opt",
                "ranges": pblock_ranges,
                "apply_to": "current_design",  # Apply to entire design
                "is_soft": False  # Hard constraint
            }, timeout=300.0)
            print(f"Create and apply pblock result:\n{result}")
            logger.info(f"Create pblock result: {result}")
            
            # ================================================================
            # Step 8: Place the design in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 8: Place the design in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("place_design", {
                "directive": "Default"
            }, timeout=3600.0)  # 1 hour timeout for placement
            print(f"Place design result:\n{result}")
            logger.info(f"Place design: {result}")
            
            # ================================================================
            # Step 9: Route the design in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 9: Route the design in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default"
            }, timeout=3600.0)  # 1 hour timeout for routing
            print(f"Route design result:\n{result}")
            logger.info(f"Route design: {result}")
            
            # Check route status
            result = await self.call_vivado_tool("report_route_status", {}, timeout=300.0)
            print(f"Route status after routing:\n{result[:1500]}...")
            logger.info(f"Route status after routing: {result}")
            
            # ================================================================
            # Step 10: Report timing and compare WNS
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 10: Report final timing")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Final timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Final timing summary: {result}")
            
            # Get final WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                self.final_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Final", self.final_wns)
            logger.info(f"Final WNS: {self.final_wns} ns")
            print()
            
            # ================================================================
            # Write final DCP and report results
            # ================================================================
            self.print_wns_change(self.initial_wns, self.final_wns, self.clock_period)
            
            # Always write the final checkpoint
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            print(f"Write final DCP result:\n{result}")
            
            # ================================================================
            # Summary
            # ================================================================
            elapsed = time.time() - overall_start
            self.print_test_summary(
                title="TEST SUMMARY - LOGICNETS PBLOCK OPTIMIZATION",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Pblock applied: {pblock_ranges}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"LogicNets test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False

    async def run_test_vexriscv(self, input_dcp: Path, output_dcp: Path) -> bool:
        """
        Cell re-placement optimization flow for VexRiscv.
        
        Mirrors the script in docs/optimization_example.md:
          Step 1 — Vivado baseline (open, get Fmax, extract critical path pins)
          Step 2 — RapidWright analysis (analyze_net_detour, filter candidates)
          Step 3 — RapidWright optimization (optimize_cell_placement, write DCP)
          Step 4 — Vivado verification (open optimized DCP, route, measure Fmax)
        """
        overall_start = time.time()
        
        try:
            # ==============================================================
            # Step 1: Vivado baseline
            # ==============================================================
            print("=" * 60)
            print("Step 1  Vivado baseline")
            print("=" * 60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            logger.info(f"Open checkpoint result: {result}")
            
            self.clock_period = await self.fetch_clock_period()
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.initial_wns = self.parse_wns_from_timing_report(ts)
            
            baseline_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            print(f"  Clock period:   {self.clock_period} ns")
            print(f"  Baseline WNS:   {self.initial_wns} ns")
            if baseline_fmax is not None:
                print(f"  Baseline Fmax:  {baseline_fmax:.2f} MHz")
            
            pins_file = Path(self.temp_dir) / "critical_path_pins.json"
            result = await self.call_vivado_tool("extract_critical_path_pins", {
                "num_paths": 10,
                "output_file": str(pins_file)
            }, timeout=600.0)
            
            critical_paths = json.loads(Path(pins_file).read_text()) if pins_file.exists() else json.loads(result)
            print(f"  Extracted {len(critical_paths)} critical path pin lists")
            
            # ==============================================================
            # Step 2: RapidWright analysis
            # ==============================================================
            print("\n" + "=" * 60)
            print("Step 2  RapidWright analysis")
            print("=" * 60)
            
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            logger.info(f"RapidWright init: {result}")
            
            result = await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            logger.info(f"RapidWright read checkpoint: {result}")
            
            result = await self.call_rapidwright_tool("analyze_net_detour", {
                "input_file": str(pins_file),
                "detour_threshold": 2.0
            }, timeout=300.0)
            logger.info(f"analyze_net_detour: {result}")
            
            analysis = json.loads(result) if isinstance(result, str) else result
            if "error" in analysis:
                raise RuntimeError(f"analyze_net_detour failed: {analysis['error']}")
            candidates = analysis.get("candidates", [])
            print(f"  Cells analyzed: {analysis.get('cells_analyzed', '?')}")
            print(f"  Candidates (detour > 2.0): {len(candidates)}")
            for c in candidates[:5]:
                print(f"    {str(c['cell']):55s}  ratio={c['max_detour_ratio']}")
            
            if not candidates:
                print("\n  No candidates found — nothing to optimize")
                self.final_wns = self.initial_wns
                return True
            
            worst_path_cells = list(set(
                str(c["cell"]) for c in candidates if c.get("path", 0) <= 2
            ))
            if not worst_path_cells:
                worst_path_cells = [str(candidates[0]["cell"])]
            
            print(f"\n  Targeting {len(worst_path_cells)} cells on paths 1-2:")
            for name in worst_path_cells:
                print(f"    {name}")
            
            # ==============================================================
            # Step 3: RapidWright optimization
            # ==============================================================
            print("\n" + "=" * 60)
            print("Step 3  RapidWright optimization")
            print("=" * 60)
            
            result = await self.call_rapidwright_tool("optimize_cell_placement", {
                "cell_names": worst_path_cells
            }, timeout=300.0)
            logger.info(f"optimize_cell_placement: {result}")
            
            opt_result = json.loads(result) if isinstance(result, str) else result
            for r in opt_result.get("results", []):
                print(f"  {r['cell']}: {r['status']} — {r['message']}")
            
            rw_output = Path(self.temp_dir) / "vexriscv_rw_optimized.dcp"
            result = await self.call_rapidwright_tool("write_checkpoint", {
                "dcp_path": str(rw_output)
            }, timeout=600.0)
            print(f"  Wrote {rw_output.name}")
            
            # ==============================================================
            # Step 4: Vivado verification
            # ==============================================================
            print("\n" + "=" * 60)
            print("Step 4  Vivado verification")
            print("=" * 60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(rw_output)
            }, timeout=600.0)
            logger.info(f"Open optimized checkpoint: {result}")
            
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default"
            }, timeout=3600.0)
            logger.info(f"Route design: {result}")
            
            route_result = await self.call_vivado_tool("report_route_status", {}, timeout=300.0)
            error_match = re.search(r"# of nets with routing errors.*?:\s+(\d+)", route_result)
            error_count = int(error_match.group(1)) if error_match else -1
            
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.final_wns = self.parse_wns_from_timing_report(ts)
            
            new_fmax = self.calculate_fmax(self.final_wns, self.clock_period)
            
            print(f"  Routing errors:  {error_count}")
            if baseline_fmax is not None and new_fmax is not None:
                print(f"  Baseline WNS:    {self.initial_wns} ns  →  Fmax {baseline_fmax:.2f} MHz")
                print(f"  Optimized WNS:   {self.final_wns} ns  →  Fmax {new_fmax:.2f} MHz")
                delta = new_fmax - baseline_fmax
                print(f"  Fmax improvement: {delta:+.2f} MHz")
            else:
                print(f"  Baseline WNS:  {self.initial_wns} ns")
                print(f"  Optimized WNS: {self.final_wns} ns")
            
            # Write final DCP
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            
            # Summary
            elapsed = time.time() - overall_start
            cells_info = ", ".join(worst_path_cells)
            self.print_test_summary(
                title="TEST SUMMARY - VEXRISCV CELL RE-PLACEMENT",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Cells re-placed: {cells_info}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"VexRiscv test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False

    async def cleanup(self):
        """Clean up resources."""
        print("\n[TEST] Cleaning up...")
        await super().cleanup()
        print(f"[TEST] Run directory preserved at: {self.run_dir}")


async def run_test_mode(input_dcp: Path, output_dcp: Path, debug: bool = False, max_nets: int = 5, run_dir: Optional[Path] = None):
    """Run the test mode optimization.
    
    Detects which example DCP is being used and applies the appropriate optimization flow:
    - logicnets_jscl: Pblock-based placement optimization flow
    - vexriscv_re-place: Cell re-placement flow (same recipe as docs/optimization_example.md)
    """
    # Detect which DCP is being used based on filename
    dcp_name = input_dcp.name.lower()
    
    if "logicnets" in dcp_name:
        design_type = "logicnets"
        print(f"[TEST] Detected LogicNets design - using pblock optimization flow")
    elif "vexriscv" in dcp_name:
        design_type = "vexriscv"
        print(f"[TEST] Detected VexRiscv design - using cell re-placement flow")
    else:
        print(f"\n[TEST] ERROR: Unsupported DCP file: {input_dcp.name}")
        print(f"[TEST] Test mode supports these benchmark DCPs:")
        print(f"[TEST]   - fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp")
        print(f"[TEST]   - fpl26_contest_benchmarks/vexriscv_re-place_2025.1.dcp")
        print(f"[TEST]")
        print(f"[TEST] For custom DCPs, run without --test to use the LLM-guided optimizer.")
        return 1
    
    tester = FPGAOptimizerTest(debug=debug, run_dir=run_dir)
    
    try:
        await tester.start_servers()
        
        if design_type == "logicnets":
            success = await tester.run_test_logicnets(input_dcp, output_dcp)
        else:
            success = await tester.run_test_vexriscv(input_dcp, output_dcp)
        
        if success:
            print("\n[TEST] Test completed successfully")
            print(f"\n[TEST] Output files:")
            print(f"[TEST]   Optimized DCP: {output_dcp}")
            print(f"[TEST]   Run directory: {tester.run_dir}")
            return 0
        else:
            print("\n[TEST] Test failed")
            print(f"[TEST] Run directory: {tester.run_dir}")
            return 1
            
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
        print(f"[TEST] Run directory: {tester.run_dir}")
        return 130
    except Exception as e:
        logger.exception(f"Test mode fatal error: {e}")
        print(f"\n[TEST] Fatal error: {e}")
        print(f"[TEST] Run directory: {tester.run_dir}")
        return 1
    finally:
        await tester.cleanup()


async def main():
    parser = argparse.ArgumentParser(
        description="FPGA Design Optimization Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dcp_optimizer.py input.dcp
  python dcp_optimizer.py input.dcp --output output.dcp
  python dcp_optimizer.py input.dcp --model anthropic/claude-sonnet-4
  python dcp_optimizer.py input.dcp --debug
  python dcp_optimizer.py fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp --test
  python dcp_optimizer.py fpl26_contest_benchmarks/vexriscv_re-place_2025.1.dcp --test
        """
    )
    parser.add_argument("input_dcp", type=Path, help="Input design checkpoint (.dcp)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        dest="output_dcp",
        help="Output optimized checkpoint (.dcp). Default: <input_name>_optimized-<timestamp>.dcp in same directory as input"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (default: OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--budget-seconds",
        type=int,
        default=None,
        help=(
            "Override the run's wall-clock budget (default: 3500 s, the contest limit). "
            "For exploratory/offline runs only -- e.g. letting a large design run "
            "overnight (--budget-seconds 36000) to see whether a full re-place ever "
            "converges given real time. NOT contest-compliant; never set this for an "
            "actual submission run."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging, save intermediate checkpoints)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: run without LLM. Pblock for LogicNets, cell re-placement for VexRiscv (see docs/optimization_example.md)."
    )
    parser.add_argument(
        "--max-nets",
        type=int,
        default=5,
        help="Maximum number of high fanout nets to optimize in test mode (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_dcp.exists():
        print(f"Error: Input file not found: {args.input_dcp}", file=sys.stderr)
        sys.exit(1)
    
    # Generate default output DCP name if not provided
    if args.output_dcp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        input_stem = args.input_dcp.stem  # Filename without extension
        input_dir = args.input_dcp.parent  # Directory of input file
        args.output_dcp = input_dir / f"{input_stem}_optimized-{timestamp}.dcp"
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if needed
    args.output_dcp.parent.mkdir(parents=True, exist_ok=True)
    
    # Test mode - run without LLM
    if args.test:
        # Create run directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path.cwd() / f"dcp_optimizer_run-{timestamp}"
        
        print(f"FPGA Design Optimization - TEST MODE")
        print(f"=====================================")
        print(f"Input:       {args.input_dcp.resolve()}")
        print(f"Output:      {args.output_dcp.resolve()}")
        print(f"Run dir:     {run_dir}")
        print(f"Max nets to optimize: {args.max_nets}")
        print()
        
        exit_code = await run_test_mode(
            args.input_dcp, 
            args.output_dcp, 
            debug=args.debug,
            max_nets=args.max_nets,
            run_dir=run_dir
        )
        sys.exit(exit_code)
    
    # Normal mode - requires API key and LLM
    if not args.api_key:
        print("Error: OpenRouter API key required. Set OPENROUTER_API_KEY or use --api-key", file=sys.stderr)
        print("       Use --test flag to run in test mode without LLM", file=sys.stderr)
        sys.exit(1)
    
    if OpenAI is None:
        print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)
    
    # Create run directory with timestamp (before creating optimizer so we can show it)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path.cwd() / f"dcp_optimizer_run-{timestamp}"
    
    print(f"FPGA Design Optimization Agent")
    print(f"================================")
    print(f"Input:       {args.input_dcp.resolve()}")
    print(f"Output:      {args.output_dcp.resolve()}")
    print(f"Run dir:     {run_dir}")
    print(f"Model:       {args.model}")
    print()
    
    optimizer = DCPOptimizer(
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
        run_dir=run_dir,
        hard_limit_seconds=args.budget_seconds,
    )
    
    try:
        await optimizer.start_servers()
        success = await optimizer.optimize(args.input_dcp, args.output_dcp)
        
        if success:
            print("\n✓ Optimization completed successfully")
            print(f"\nOutput files:")
            print(f"  Optimized DCP: {args.output_dcp}")
            print(f"  Run directory: {run_dir}")
            sys.exit(0)
        else:
            print("\n✗ Optimization did not complete successfully")
            print(f"\nRun directory: {run_dir}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        print(f"Run directory: {run_dir}")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"Run directory: {run_dir}")
        sys.exit(1)
    finally:
        await optimizer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())