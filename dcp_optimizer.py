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
PHYS_OPT_SECONDARY_DIRECTIVE = "ExploreWithRemap"
DECISION_NET_DELAY_BOUND_THRESHOLD = 0.70
DECISION_LOGIC_DELAY_BOUND_THRESHOLD = 0.70
DECISION_SPREAD_NET_THRESHOLD = 0.60
DECISION_SPREAD_TILE_THRESHOLD = 30
PHYS_OPT_MIN_USEFUL_WNS_NS = -0.5
WNS_SANITY_ABS_LIMIT_NS = 50.0
WNS_SANITY_POSITIVE_CLOCK_FRACTION = 0.10
STUCK_ITERATION_THRESHOLD = 3
STRUCTURAL_OVERRIDE_MAX_ITERS = 6
WNS_IMPROVEMENT_EPSILON_NS = 1e-4
ACTION_FAILURE_EXHAUSTION_THRESHOLD = 3
ACTION_FAILURE_COOLDOWN_ITERS = 5
ABSOLUTE_STALL_HARD_LIMIT = 15
# --- Cluster-aware cell placement guard (fix #4) ---
# rapidwright_optimize_cell_placement moves each requested cell independently.
# If cells on the same critical path are tightly coupled, moving one without
# the others can increase the spread between them even though each individual
# move looked locally reasonable. This threshold gates the move: if the
# post-move spread across the targeted cluster is worse than before the move
# by more than this fraction, the move is rejected before it is ever routed
# (saving a full route cycle and avoiding a checkpoint that we know regressed).
CLUSTER_SPREAD_REGRESSION_FRACTION = 0.15
# --- Pblock region validation (fix #2) ---
# Reject RapidWright-recommended pblock regions that would be packed too
# densely (congestion risk) or that overlap a pblock already applied earlier
# in this run (Vivado handles overlapping pblocks poorly and it creates
# ambiguous, hard-to-solve placement scenarios).
PBLOCK_MAX_UTILIZATION_FRACTION = 0.85
RAPIDWRIGHT_STRUCTURAL_ACTIONS = [
    "rapidwright_optimize_cell_placement",
    "rapidwright_analyze_net_detour",
    "rapidwright_analyze_fabric_for_pblock",
    "rapidwright_convert_fabric_region_to_pblock",
    "pblock",
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
    "rapidwright_analyze_fabric_for_pblock",
    "rapidwright_convert_fabric_region_to_pblock",
    "pblock",
    "rapidwright_analyze_net_detour",
    "rapidwright_optimize_cell_placement",
]
VIVADO_INCREMENTAL_IMPLEMENTATION_ACTIONS = {
    "phys_opt_design",
    "phys_opt_design_retime",
    "replicate_register",
    "place_design_explore",
}
DEFAULT_PBLOCK_TARGET_LUT_COUNT = 20000
DEFAULT_PBLOCK_TARGET_FF_COUNT = 40000
DEFAULT_PBLOCK_NAME_PREFIX = "pblock_net_delay"


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

You follow these rules without exception. Violating them is a harder failure than choosing
a suboptimal action.

RULE 1 - DELAY CLASS BINDING:
  If delay_class == "net_delay_bound" (net_pct > 0.70):
    PRIMARY actions must be from allowed_actions, especially RapidWright structural
    placement actions such as rapidwright_optimize_cell_placement, pblock, and the
    pblock range-analysis actions.
    FORBIDDEN as primary: [lut_opt, logic_restructure, fanout_split]
    Rationale: routing-bound paths need cell movement or placement constraints, not logic optimization.

  If delay_class == "logic_delay_bound" (logic_pct > 0.70):
    PRIMARY actions must be from: [lut_opt, phys_opt_design_retime, fanout_split]
    FORBIDDEN as primary: [pblock, place_design_explore]
    Rationale: placement changes do not reduce logic depth.

  If delay_class == "mixed":
    No forbidden actions, but phys_opt_design with -retime is preferred first.

RULE 2 - ENDPOINT BINDING:
  If endpoint_type == "BRAM_CONTROL" or "DSP_CONTROL":
    fanout_split is only valid as a SECONDARY action.
    PRIMARY must address physical co-location: pblock, replicate_register, place_design_explore.
    Append to rationale: "Routing to hard block control pin requires physical proximity, not net splitting."

RULE 3 - SPREAD BINDING:
  If avg_tile_spread > 30 AND net_pct > 0.60:
    First action must address placement. Prefer rapidwright_optimize_cell_placement
    or a pblock flow with computed ranges before any routing-only fix.

You must not propose actions that contradict these rules. If you find yourself wanting to
choose a forbidden action, that is a signal your reasoning has drifted from the timing data -
re-read delay_class and endpoint_type and correct course.
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
        run_dir: Optional[Path] = None
    ):
        super().__init__(debug=debug, run_dir=run_dir)
        
        self.api_key = api_key
        self.model = model
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
            
            # Track WNS from timing reports and get_wns calls
            if tool_name == "vivado_report_timing_summary":
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

        tcl_cmd = (
            f"set paths [get_timing_paths -max_paths {TIER2_TOP_PATHS_DEFAULT} -setup{tcl_filter}]; "
            "if {[llength $paths] == 0} {set paths [get_timing_paths -max_paths "
            f"{TIER2_TOP_PATHS_DEFAULT} -setup]}}; "
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
        before_guard_wns = await self._get_current_wns()
        if before_guard_wns is not None and before_guard_wns < PHYS_OPT_MIN_USEFUL_WNS_NS:
            message = (
                f"Skipping phys_opt_design because current WNS {before_guard_wns:.3f} ns is below "
                f"{PHYS_OPT_MIN_USEFUL_WNS_NS:.3f} ns; use structural placement actions first."
            )
            logger.info(message)
            return self._failure_json("phys_opt_below_useful_wns", message, command="phys_opt_design")
        if not await self._check_implementation_license():
            return self._failure_json(
                "vivado_license_failure",
                "Vivado Implementation license is unavailable; phys_opt_design disabled.",
                command="phys_opt_design",
            )
        before_wns = before_guard_wns
        directive = arguments.get("directive") or PHYS_OPT_PRIMARY_DIRECTIVE
        primary = await self._run_phys_opt_tcl(directive=directive, retime=True)
        if self._action_failure(primary, default_command="phys_opt_design").get("error_type") == "vivado_license_failure":
            return primary
        if self._vivado_output_has_error(primary) and directive != "Default":
            logger.warning("phys_opt directive %s unsupported or failed; falling back to Default", directive)
            primary = await self._run_phys_opt_tcl(directive="Default", retime=True)

        after_wns = await self._get_current_wns()
        output = [primary]
        if before_wns is not None and after_wns is not None and after_wns <= before_wns:
            secondary = await self._run_phys_opt_tcl(directive=PHYS_OPT_SECONDARY_DIRECTIVE, retime=True)
            if self._action_failure(secondary, default_command="phys_opt_design").get("error_type") == "vivado_license_failure":
                output.append(secondary)
                return "\n\n".join(output)
            if self._vivado_output_has_error(secondary):
                logger.warning("phys_opt directive %s unsupported or failed; falling back to Default", PHYS_OPT_SECONDARY_DIRECTIVE)
                secondary = await self._run_phys_opt_tcl(directive="Default", retime=True)
            output.append(secondary)
        return "\n\n".join(output)

    async def _maybe_run_pblock_or_phys_opt(self, arguments: dict) -> str:
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

    async def _run_phys_opt_tcl(self, directive: str = "Default", retime: bool = True) -> str:
        if retime and self.phys_opt_retime_supported is False:
            logger.info("Skipping phys_opt -retime because a previous retime attempt was rejected.")
            retime = False
        retime_flag = " -retime" if retime else ""
        command = f"phys_opt_design -directive {directive}{retime_flag}"
        result = await self.call_tool("vivado_run_tcl", {"command": command, "timeout": 3600}, internal=True)
        if self._action_failure(result, default_command=command).get("error_type") == "vivado_license_failure":
            return result
        if self._vivado_output_has_error(result) and retime:
            self.phys_opt_retime_supported = False
            logger.warning("phys_opt retime failed with directive %s; retrying without -retime", directive)
            command = f"phys_opt_design -directive {directive}"
            result = await self.call_tool("vivado_run_tcl", {"command": command, "timeout": 3600}, internal=True)
        elif retime:
            self.phys_opt_retime_supported = True
        return result

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
        return {}

    async def _check_implementation_license(self) -> bool:
        # Vivado 2025.1 Tcl does not reliably expose `get_license`; probing it
        # can mark a usable analysis session as broken. Let implementation
        # commands report structured failures when they are actually invoked.
        return self.implementation_license_available is not False

    def _time_remaining_s(self) -> Optional[float]:
        if self.checkpoint_manager is None:
            return None
        elapsed = time.time() - self.checkpoint_manager.started_at_epoch_s
        return self.checkpoint_manager.hard_limit_seconds - elapsed

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
        route = await self.call_tool("vivado_route_design", {"directive": "Default", "timeout": 3600}, internal=True)
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
            vivado_runtime_s=0.0,
            checkpoint_path=str(checkpoint_path),
            batch_size=1,
        )
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
        if self._is_no_action_failure(failure):
            failure_key = (failed_action, tuple(failed_targets), self.iteration)
            if self.last_no_action_failure_key != failure_key:
                self._blacklist_failure_targets(failed_action, failed_targets)
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
            "vivado_runtime_s": 0.0,
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
        self.checkpoint_manager.iterations.append(iteration)
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

    async def _append_iteration_context(self) -> None:
        current_wns = await self._get_current_wns()
        if current_wns is not None:
            await self._refresh_target_candidates(current_wns)
        await self._classify_worst_path_delay()
        await self._check_implementation_license()
        timing_context = self._build_timing_context(current_wns)
        self.last_timing_context = timing_context
        prompt = (
            "Given the timing state above, select one action from `allowed_actions`.\n"
            "You may not choose any action in `forbidden_actions`.\n\n"
            "Respond in this JSON format only, no other text:\n"
            "{\n"
            "  \"delay_class_acknowledged\": <copy delay_class from input>,\n"
            "  \"endpoint_type_acknowledged\": <copy endpoint_type from input>,\n"
            "  \"chosen_action\": <must be from allowed_actions>,\n"
            "  \"action_parameters\": { ... },\n"
            "  \"why_this_fits_delay_class\": <one sentence, must reference net_pct or logic_pct>,\n"
            "  \"why_not_top_forbidden_action\": <one sentence explaining why the most tempting forbidden action does not apply here>,\n"
            "  \"confidence\": <1-5>\n"
            "}\n\n"
            f"{json.dumps(timing_context, indent=2)}"
        )
        self.messages.append({"role": "user", "content": prompt})

    def _build_timing_context(self, current_wns: Optional[float]) -> dict:
        if self.checkpoint_manager is not None:
            self.consecutive_no_improvement = self.checkpoint_manager.stall_count
            self.no_improvement_count = self.checkpoint_manager.stall_count

        worst = self.current_target_candidates[0] if self.current_target_candidates else {}
        endpoint = str(worst.get("endpoint") or "")
        endpoint_type = self._classify_endpoint_type(endpoint)
        logic_pct = self.path_delay_breakdown.get("logic_pct")
        net_pct = self.path_delay_breakdown.get("net_pct")
        avg_spread = self.last_spread_info.get("avg_distance")
        max_spread = self.last_spread_info.get("max_distance")
        delay_class = self.path_delay_classification
        if delay_class == "unknown":
            delay_class = "mixed"
        allowed, forbidden = self._allowed_forbidden_actions(
            delay_class,
            endpoint_type,
            net_pct,
            avg_spread,
            current_wns,
        )
        structural_override_age = self.consecutive_no_improvement - STUCK_ITERATION_THRESHOLD
        structural_override = (
            structural_override_age >= 0
            and structural_override_age < STRUCTURAL_OVERRIDE_MAX_ITERS
        )
        self.structural_override_active = structural_override
        high_spread = (
            avg_spread is not None
            and net_pct is not None
            and avg_spread > DECISION_SPREAD_TILE_THRESHOLD
            and net_pct > DECISION_SPREAD_NET_THRESHOLD
        )
        if structural_override:
            # Even when forcing a structural action after repeated stalls,
            # still respect the spread-based ordering below - otherwise a
            # design stuck specifically because cell_placement keeps
            # regressing would have that same action handed back to it
            # first, just from a shorter list.
            structural_source = RAPIDWRIGHT_PLACEMENT_ACTIONS if high_spread else RAPIDWRIGHT_STRUCTURAL_ACTIONS
            structural_allowed = [action for action in structural_source if action in allowed]
            if structural_allowed:
                allowed = structural_allowed
                for action in RAPIDWRIGHT_STRUCTURAL_ACTIONS:
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
        return {
            "iteration": self.iteration,
            "wns_ns": current_wns,
            "tns_ns": self.initial_tns,
            "failing_endpoints": self.initial_failing_endpoints,
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
            "action_failure_memory": self._serializable_action_failure_memory(),
            "action_failure_counts": self._serializable_action_failure_counts(),
            "exhausted_actions": exhausted_actions,
            "allowed_actions": allowed,
            "forbidden_actions": forbidden,
            "recommendation": recommendation,
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

    def _allowed_forbidden_actions(
        self,
        delay_class: str,
        endpoint_type: str,
        net_pct: Optional[float],
        avg_spread: Optional[float],
        current_wns: Optional[float],
    ) -> tuple[list[str], list[str]]:
        if delay_class == "net_delay_bound":
            allowed = [
                *RAPIDWRIGHT_STRUCTURAL_ACTIONS,
                "place_design_explore",
                "replicate_register",
                "phys_opt_design",
            ]
            forbidden = ["lut_opt", "logic_restructure", "fanout_split"]
        elif delay_class == "logic_delay_bound":
            allowed = ["lut_opt", "phys_opt_design_retime", "fanout_split"]
            forbidden = [
                "pblock",
                "place_design_explore",
                "rapidwright_optimize_cell_placement",
                "rapidwright_analyze_net_detour",
                "rapidwright_analyze_fabric_for_pblock",
                "rapidwright_convert_fabric_region_to_pblock",
            ]
        else:
            allowed = [
                *RAPIDWRIGHT_STRUCTURAL_ACTIONS,
                "phys_opt_design_retime",
                "phys_opt_design",
                "place_design_explore",
                "fanout_split",
                "lut_opt",
            ]
            forbidden = []

        if endpoint_type in {"BRAM_CONTROL", "DSP_CONTROL"}:
            allowed = [action for action in allowed if action != "fanout_split"]
            for action in [
                "rapidwright_optimize_cell_placement",
                "pblock",
                "rapidwright_analyze_fabric_for_pblock",
                "rapidwright_convert_fabric_region_to_pblock",
                "replicate_register",
                "place_design_explore",
            ]:
                if action not in allowed:
                    allowed.append(action)
            if "fanout_split" not in forbidden:
                forbidden.append("fanout_split")

        if (
            avg_spread is not None
            and net_pct is not None
            and avg_spread > DECISION_SPREAD_TILE_THRESHOLD
            and net_pct > DECISION_SPREAD_NET_THRESHOLD
        ):
            placement_first = [action for action in RAPIDWRIGHT_PLACEMENT_ACTIONS if action in allowed]
            allowed = placement_first + [action for action in allowed if action not in placement_first]

        structural_available = any(action in allowed for action in RAPIDWRIGHT_STRUCTURAL_ACTIONS)
        if (
            current_wns is not None
            and current_wns < PHYS_OPT_MIN_USEFUL_WNS_NS
            and structural_available
        ):
            phys_actions = set(VIVADO_INCREMENTAL_IMPLEMENTATION_ACTIONS)
            allowed = [action for action in allowed if action not in phys_actions]
            for action in sorted(phys_actions):
                if action not in forbidden:
                    forbidden.append(action)
            logger.info(
                "Skipping phys_opt candidates because WNS %.3f ns is below %.3f ns and structural actions are available.",
                current_wns,
                PHYS_OPT_MIN_USEFUL_WNS_NS,
            )

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
        active_exhausted = set(self._active_exhausted_actions())
        active_exhausted.update(
            action
            for action, count in self.action_failure_counts.items()
            if count >= ACTION_FAILURE_EXHAUSTION_THRESHOLD
        )
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
        return exhausted

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

    def _reset_action_failure_memory(self, action: str) -> None:
        self.action_failure_counts[action] = 0
        if action in self.action_failure_memory:
            self.action_failure_memory[action]["consecutive_no_action_failures"] = 0
            self.action_failure_memory[action]["cooldown_until_iter"] = -1

    def _blacklist_failure_targets(self, action: str, targets: list[str]) -> None:
        if self.checkpoint_manager is None or not targets:
            return
        if action in {"rapidwright_optimize_cell_placement", "rapidwright_analyze_net_detour"}:
            for target in targets:
                if self.checkpoint_manager._is_blacklistable_target(target) and target not in self.checkpoint_manager.cells_blacklisted:
                    self.checkpoint_manager.cells_blacklisted.append(str(target))
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

    def _cell_blacklist(self) -> set[str]:
        if self.checkpoint_manager is None:
            return set()
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
        worst_path = timing_context.get("worst_path", {})
        for key in ("start_cell", "end_cell"):
            value = str(worst_path.get(key) or "").strip("{} ")
            if not value:
                continue
            cell = value.rsplit("/", 1)[0] if "/" in value else value
            if cell and cell not in candidates:
                candidates.append(cell)

        for candidate in self.current_target_candidates[:limit]:
            for key in ("startpoint", "endpoint"):
                value = str(candidate.get(key) or "").strip("{} ")
                cell = value.rsplit("/", 1)[0] if "/" in value else value
                if cell and cell not in candidates:
                    candidates.append(cell)

        return self._filter_blacklisted_cells(candidates)[:limit]

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

    async def _compute_pblock_ranges(self, params: dict, timing_context: dict) -> tuple[Optional[dict], Optional[str]]:
        """Compute Vivado pblock ranges through RapidWright fabric analysis."""
        if params.get("ranges"):
            return params, None

        target_lut_count = int(
            params.get("target_lut_count")
            or self.last_design_info.get("lut_count")
            or self.last_design_info.get("luts")
            or DEFAULT_PBLOCK_TARGET_LUT_COUNT
        )
        target_ff_count = int(
            params.get("target_ff_count")
            or self.last_design_info.get("ff_count")
            or self.last_design_info.get("ffs")
            or DEFAULT_PBLOCK_TARGET_FF_COUNT
        )
        analysis_args = {
            "target_lut_count": target_lut_count,
            "target_ff_count": target_ff_count,
            "target_dsp_count": int(params.get("target_dsp_count") or 0),
            "target_bram_count": int(params.get("target_bram_count") or 0),
        }
        fabric_text = await self.call_tool("rapidwright_analyze_fabric_for_pblock", analysis_args, internal=True)
        fabric = self._parse_json_result(fabric_text)
        if self._result_has_error(fabric):
            return None, f"RapidWright fabric analysis failed: {fabric.get('error') or fabric_text[:300]}"

        region = fabric.get("recommended_region") or {}
        required_region_keys = ("col_min", "col_max", "row_min", "row_max")
        if not all(key in region for key in required_region_keys):
            return None, f"RapidWright fabric analysis did not return recommended_region: {fabric_text[:300]}"

        # --- Fix #2a: reject regions that overlap a pblock already applied
        # this run. Vivado handles overlapping pblocks poorly, and it creates
        # complex, hard-to-solve placement scenarios. ---
        overlap = self._find_pblock_overlap(region)
        if overlap is not None:
            return None, (
                f"RapidWright fabric analysis recommended region {region} which overlaps "
                f"an already-applied pblock {overlap.get('pblock_name')} at "
                f"{overlap.get('region')}; skipping to avoid overlapping pblocks."
            )

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

        # --- Fix #2b: reject regions that would be packed too densely.
        # High-utilization pblocks are a known congestion risk; better to ask
        # for a larger/different region than to hand Vivado a region that is
        # very likely to fail to place-and-route cleanly. ---
        site_counts = range_payload.get("site_counts") or {}
        utilization_error = self._check_pblock_utilization(
            site_counts, target_lut_count, target_ff_count,
            int(params.get("target_dsp_count") or 0),
            int(params.get("target_bram_count") or 0),
        )
        if utilization_error:
            return None, utilization_error

        computed = dict(params)
        computed["ranges"] = ranges
        computed.setdefault("pblock_name", f"{DEFAULT_PBLOCK_NAME_PREFIX}_{self.iteration:03d}")
        computed.setdefault("apply_to", "current_design")
        computed.setdefault("is_soft", False)
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

    def _check_pblock_utilization(
        self,
        site_counts: dict,
        target_lut_count: int,
        target_ff_count: int,
        target_dsp_count: int,
        target_bram_count: int,
    ) -> Optional[str]:
        """Return an error string if the requested targets would over-pack the region, else None."""
        if not site_counts:
            # No capacity data returned -- nothing to validate against, let it through.
            return None
        checks = [
            ("LUT", target_lut_count, site_counts.get("lut_capacity") or site_counts.get("luts")),
            ("FF", target_ff_count, site_counts.get("ff_capacity") or site_counts.get("ffs")),
            ("DSP", target_dsp_count, site_counts.get("dsp_capacity") or site_counts.get("dsps")),
            ("BRAM", target_bram_count, site_counts.get("bram_capacity") or site_counts.get("brams")),
        ]
        for label, requested, capacity in checks:
            if not requested or not capacity:
                continue
            utilization = requested / float(capacity)
            if utilization > PBLOCK_MAX_UTILIZATION_FRACTION:
                return (
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
            await self._record_iteration_timing(wns_measured, elapsed_time)

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
        iteration = self.checkpoint_manager.record(
            recipe=self.last_recipe,
            targets=self.last_targets,
            wns_after=wns,
            vivado_runtime_s=vivado_runtime_s,
            checkpoint_path=str(checkpoint_path),
            batch_size=self.last_batch_size,
        )
        self.no_improvement_count = self.checkpoint_manager.stall_count
        self.consecutive_no_improvement = self.checkpoint_manager.stall_count
        self.last_recorded_wns = wns
        # Fix #1: gate the reset/remember-failure bookkeeping on
        # last_action_key (the never-renamed dispatch key), not last_recipe
        # (a display label _remember_recipe() may have rewritten). Before
        # this fix, e.g. rapidwright_optimize_cell_placement's failures were
        # recorded under "rapidwright_cell_placement", a key that nothing
        # reading action_failure_memory during action-selection ever checks,
        # so a 100%-regression action was never actually suppressed.
        action_key = self.last_action_key or self.last_recipe
        if iteration.get("status") in {"improved", "marginal"}:
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
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=1200,
            extra_body={"usage": {"include": True}},
        )
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
        return True, "ok"

    async def execute_validated_action(self, decision: dict, timing_context: dict) -> str:
        action = decision.get("chosen_action")
        params = decision.get("action_parameters") or {}
        self.last_rapidwright_edit_summary = None
        # Fix #1: set the never-renamed dispatch key exactly once per action,
        # before any of the per-action branches below run. _remember_recipe()
        # is still allowed to rewrite self.last_recipe for display purposes;
        # self.last_action_key is the one all gating logic should read.
        self.last_action_key = str(action)
        if action in {"phys_opt_design", "phys_opt_design_retime"}:
            if not await self._check_implementation_license():
                return self._failure_json(
                    "vivado_license_failure",
                    "Vivado Implementation license is unavailable; phys_opt_design disabled.",
                    command="phys_opt_design",
                )
            self.last_recipe = action
            self.last_targets = [timing_context.get("delay_class", "timing_path")]
            self.last_batch_size = 1
            return await self.call_tool("vivado_phys_opt_design", params)
        if action == "place_design_explore":
            current_wns = timing_context.get("wns_ns")
            if current_wns is not None and current_wns < PHYS_OPT_MIN_USEFUL_WNS_NS:
                return self._failure_json(
                    "implementation_action_below_useful_wns",
                    (
                        f"place_design_explore skipped because WNS {current_wns:.3f} ns is below "
                        f"{PHYS_OPT_MIN_USEFUL_WNS_NS:.3f} ns and structural RapidWright actions are required first."
                    ),
                    command="place_design_explore",
                )
            if not await self._check_implementation_license():
                return self._failure_json(
                    "vivado_license_failure",
                    "Vivado Implementation license is unavailable; place_design/route_design disabled.",
                    command="place_design/route_design",
                )
            self.last_recipe = action
            self.last_targets = [str(timing_context["worst_path"].get("end_cell"))]
            self.last_batch_size = 1
            place = await self.call_tool("vivado_place_design", {"directive": "Explore", "timeout": 3600})
            route = await self.call_tool("vivado_route_design", {"directive": "Explore", "timeout": 3600})
            return place + "\n\n" + route
        if action == "pblock":
            if not await self._check_implementation_license():
                return self._failure_json(
                    "vivado_license_failure",
                    "Vivado Implementation license is unavailable; pblock flow requiring implementation is disabled.",
                    command="create_and_apply_pblock",
                )
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
            result = await self.call_tool("vivado_create_and_apply_pblock", params)
            payload = self._parse_json_result(result)
            if self._result_has_error(payload):
                return self._failure_json(
                    payload.get("error_type", "pblock_assignment_failed"),
                    payload.get("message", result[:300]),
                    command="pblock",
                )
            self.last_rapidwright_edit_summary = self._summarize_pblock_assignment(payload, params)
            if int(self.last_rapidwright_edit_summary.get("cells_assigned") or 0) <= 0:
                return self._failure_json(
                    "pblock_empty_assignment",
                    "Pblock action completed but assigned zero cells.",
                    command="pblock",
                )
            # Creating/applying the pblock only constrains a region for
            # future placement - it does not move any cells by itself, so
            # WNS cannot change as a result of this call alone. Re-place
            # (now respecting the new pblock) and re-route before returning,
            # or this action can never have any measurable timing effect.
            place_result = await self.call_tool(
                "vivado_place_design", {"directive": "Explore", "timeout": 3600}, internal=True
            )
            if self._action_failure(place_result, default_command="vivado_place_design"):
                return self._failure_json(
                    "pblock_place_failed",
                    f"Pblock applied but re-placement failed: {place_result[:300]}",
                    command="pblock",
                )
            route_result = await self.call_tool(
                "vivado_route_design", {"directive": "Explore", "timeout": 3600}, internal=True
            )
            if self._action_failure(route_result, default_command="vivado_route_design"):
                return self._failure_json(
                    "pblock_route_failed",
                    f"Pblock applied and re-placed but routing failed: {route_result[:300]}",
                    command="pblock",
                )
            # Fix #2: only now, after placement + routing succeeded, register
            # this region as applied so future pblock recommendations are
            # checked against it for overlap.
            self._register_applied_pblock(fabric_region, params.get("pblock_name"))
            return result + "\n\n" + place_result + "\n\n" + route_result
        if action == "rapidwright_analyze_fabric_for_pblock":
            params, error = await self._compute_pblock_ranges(dict(params), timing_context)
            if error:
                logger.error("RapidWright fabric/pblock analysis aborted: %s", error)
                return self._failure_json("pblock_range_computation_failed", error, command=action)
            assert params is not None
            params.pop("_fabric_region", None)
            self.last_recipe = action
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
            params, error = await self._compute_pblock_ranges(dict(params), timing_context)
            if error:
                logger.error("RapidWright pblock range conversion aborted: %s", error)
                return self._failure_json("pblock_range_computation_failed", error, command=action)
            assert params is not None
            params.pop("_fabric_region", None)
            self.last_recipe = action
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
            cell_names = self._filter_blacklisted_cells(attempted_cells)
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
                    attempted_cells = [str(item.get("cell")) for item in candidates if item.get("cell")]
                    cell_names = self._filter_blacklisted_cells(attempted_cells)
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
                    cell_names = live_cells
                    attempted_cells = live_cells
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
                {"cell_names": cell_names, "max_candidates": max_candidates},
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
            result += "\n\n" + await self.call_tool("vivado_route_design", {"directive": "Default", "timeout": 3600})
            return result
        if action == "fanout_split":
            if not await self._check_implementation_license():
                return self._failure_json(
                    "vivado_license_failure",
                    "Vivado Implementation license is unavailable; fanout_split flow cannot reroute the edited design.",
                    command="fanout_split/route_design",
                )
            if not self.high_fanout_nets:
                logger.warning("fanout_split selected but no high-fanout nets are available; falling back to phys_opt.")
                return self._failure_json(
                    "no_action_target",
                    "fanout_split selected but no high-fanout nets are available.",
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
            result += "\n\n" + await self.call_tool("vivado_route_design", {"directive": "Default", "timeout": 3600})
            return result
        if action == "lut_opt":
            pins = params.get("hierarchical_input_pins") or []
            if not pins:
                logger.warning("lut_opt selected but no pins provided; falling back to phys_opt retime.")
                return self._failure_json(
                    "missing_action_parameters",
                    "lut_opt selected but no hierarchical_input_pins were provided.",
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
            return await self.call_tool("vivado_phys_opt_design", {"critical_cell_opt": True})
        return self._failure_json(
            "unsupported_action",
            f"Action {action!r} is not implemented by the orchestrator dispatch layer.",
            command=str(action),
        )
    
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
        
        # If timing is already met, continue anyway: this contest flow pushes Fmax
        # by tightening the target clock instead of stopping at closure.
        if self.initial_wns is not None and self.initial_wns >= 0:
            print("✓ Design meets timing; continuing with Tier 2 worst-slack optimization and clock tightening.\n")
            logger.info("Design meets timing; entering Fmax-push flow")
            await self._run_clock_bisection_after_closure(self.initial_wns)
        
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
        
        max_iterations = 50  # Safety limit
        
        print("=== Starting LLM-Driven Optimization ===\n")
        
        while self.iteration < max_iterations:
            self.iteration += 1
            logger.info(f"=== Iteration {self.iteration} ===")
            
            try:
                await self._append_iteration_context()
                decision = await self.get_validated_action_decision(self.last_timing_context)
                response_text = await self.execute_validated_action(decision, self.last_timing_context)
                failure = self._action_failure(response_text, default_command=str(decision.get("chosen_action")))
                if failure:
                    self._record_failed_action(failure)
                    print(f"\nAction failed: {failure.get('error_type')}\n{failure.get('message', '')}\n")
                    continue
                await self.call_tool("vivado_report_timing_summary", {"timeout": 300})
                print(f"\n{response_text}\n")
                
                current_wns = await self._get_current_wns()
                if current_wns is not None and current_wns >= 0 and self.current_period_ns is not None:
                    await self._run_clock_bisection_after_closure(current_wns)

                if self.checkpoint_manager is not None and not self.checkpoint_manager.should_continue():
                    logger.info("Optimization workflow completed")
                    self.end_time = time.time()
                    self._print_optimization_summary()
                    return True

                    if self.consecutive_no_improvement >= ABSOLUTE_STALL_HARD_LIMIT:
                    logger.error(
                        "Hard stall limit (%d) reached with no improvement; "
                        "stopping and restoring best checkpoint.",
                        ABSOLUTE_STALL_HARD_LIMIT,
                    )
                    if self.checkpoint_manager is not None:
                        best_ckpt = self.checkpoint_manager.get_best_checkpoint()
                        if best_ckpt:
                            await self.call_tool(
                                "vivado_open_checkpoint", {"dcp_path": best_ckpt}, internal=True
                            )
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
                    except Exception as reopen_exc:
                        logger.exception(f"Failed to reopen checkpoint after desync recovery: {reopen_exc}")
                        self.end_time = time.time()
                        raise
                else:
                    logger.error("No known-good checkpoint to reopen; aborting run.")
                    self.end_time = time.time()
                    raise
                continue

            except Exception as e:
                logger.exception(f"Error during optimization: {e}")
                self.end_time = time.time()
                raise
        
        logger.warning("Reached maximum iterations")
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
        
        # Calculate fmax values
        initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
        best_fmax = self.best_fmax_mhz
        if best_fmax is None and self.best_wns > float('-inf'):
            best_fmax = self.calculate_fmax(self.best_wns, self.clock_period)
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
                "initial_wns": self.initial_wns,
                "best_wns": self.best_wns,
                "wns_improvement": self.best_wns - self.initial_wns if self.initial_wns is not None else None,
                "initial_fmax_mhz": initial_fmax,
                "best_fmax_mhz": best_fmax,
                "fmax_improvement_mhz": fmax_improvement,
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
        title = "Optimization Summary (Max Iterations Reached)" if max_iterations_reached else "Optimization Summary"
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")
        
        # Calculate total runtime
        if self.start_time is not None:
            total_runtime = (self.end_time or time.time()) - self.start_time
            print(f"\nTOTAL RUNTIME: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
        
        best_wns = self.best_wns if self.best_wns > float('-inf') else None
        result_lines = self._format_fmax_results(
            self.clock_period, self.initial_wns, best_wns, result_label="Best"
        )
        if result_lines:
            print(f"\nFMAX RESULTS:")
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
        run_dir=run_dir
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