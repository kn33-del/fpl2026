from dataclasses import dataclass
import logging
import re
import time
from typing import List, Optional
import asyncio
import inspect

PRESERVE_FLAG = "-preserve_fixed_routes"

# Compiled regex patterns for parsing Vivado outputs
_RE_ROUTE_ERRORS = re.compile(r"# of nets with routing errors\s*:\s*(\d+)")
_RE_UNROUTED_NETS = re.compile(r"# of unrouted nets\s*:\s*(\d+)")
_RE_ANTENNA = re.compile(r"antenna violations\s*:\s*(\d+)", re.IGNORECASE)
_RE_OPTIMIZED_PATHS = re.compile(r"Optimized\s+(\d+)\s+timing paths", re.IGNORECASE)
_RE_WNS = re.compile(r"WNS[:=]?\s*([+-]?\d+\.\d+)")

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    directive_used: str
    routing_errors: int
    unrouted_nets: int
    fully_routed: bool
    fallback_used: bool
    timeout_s: int
    vivado_raw_output: str


@dataclass
class PhysOptResult:
    directives_run: List[str]
    paths_optimized: List[int]
    wns_trajectory: List[float]


class ECORouter:
    def __init__(self, vivado, base_timeout_s: int = 600, large_design_threshold: int = 300_000):
        self.vivado = vivado
        self.base_timeout_s = int(base_timeout_s)
        self.large_design_threshold = int(large_design_threshold)

    def _compute_timeout(self, design_cell_count: int, unrouted_net_count: int) -> int:
        timeout = int(self.base_timeout_s)
        if design_cell_count > self.large_design_threshold:
            timeout *= 2
        if unrouted_net_count > 500:
            timeout += 120
        return timeout

    def route_incremental(self, design_cell_count: int, unrouted_net_count: int, directive: str = "Default") -> RouteResult:
        timeout = self._compute_timeout(design_cell_count, unrouted_net_count)
        tcl = f"route_design {PRESERVE_FLAG} -directive {{{directive}}}"
        raw = self.vivado.run_tcl_command(tcl, timeout=timeout)
        # Immediately query route status
        status_raw = self.vivado.run_tcl_command("report_route_status -return_string", timeout=30)
        parsed = self.parse_route_status(status_raw)
        return RouteResult(
            directive_used=directive,
            routing_errors=parsed.get("routing_errors", -1),
            unrouted_nets=parsed.get("unrouted_nets", -1),
            fully_routed=parsed.get("fully_routed", False),
            fallback_used=False,
            timeout_s=timeout,
            vivado_raw_output=status_raw,
        )

    async def _call_vivado(self, tcl: str, timeout: int):
        # Support both sync and async vivado.run_tcl_command implementations
        try:
            res = self.vivado.run_tcl_command(tcl, timeout=timeout)
            if inspect.isawaitable(res):
                return await res
            return res
        except Exception:
            # re-raise for caller
            raise

    async def route_incremental_async(self, design_cell_count: int, unrouted_net_count: int, directive: str = "Default") -> RouteResult:
        timeout = self._compute_timeout(design_cell_count, unrouted_net_count)
        tcl = f"route_design {PRESERVE_FLAG} -directive {{{directive}}}"
        await self._call_vivado(tcl, timeout=timeout)
        status_raw = await self._call_vivado("report_route_status -return_string", timeout=30)
        parsed = self.parse_route_status(status_raw)
        return RouteResult(
            directive_used=directive,
            routing_errors=parsed.get("routing_errors", -1),
            unrouted_nets=parsed.get("unrouted_nets", -1),
            fully_routed=parsed.get("fully_routed", False),
            fallback_used=False,
            timeout_s=timeout,
            vivado_raw_output=status_raw,
        )

    def route_with_fallback(self, design_cell_count: int, unrouted_net_count: int) -> RouteResult:
        directives = ["Default", "Explore", "AggressiveExplore"]
        timeout = self._compute_timeout(design_cell_count, unrouted_net_count)
        # Try preserve attempts
        for d in directives:
            tcl = f"route_design {PRESERVE_FLAG} -directive {{{d}}}"
            raw = self.vivado.run_tcl_command(tcl, timeout=timeout)
            status_raw = self.vivado.run_tcl_command("report_route_status -return_string", timeout=30)
            parsed = self.parse_route_status(status_raw)
            if parsed.get("routing_errors", 1) == 0:
                return RouteResult(
                    directive_used=d,
                    routing_errors=parsed.get("routing_errors", 0),
                    unrouted_nets=parsed.get("unrouted_nets", 0),
                    fully_routed=parsed.get("fully_routed", False),
                    fallback_used=False,
                    timeout_s=timeout,
                    vivado_raw_output=status_raw,
                )

        # All preserve attempts failed — fallback to full route_design without preserve
        logger.warning("All preserve-fixed-routes attempts failed; falling back to full route_design (this will overwrite existing routing)")
        raw = self.vivado.run_tcl_command(f"route_design -directive {{Default}}", timeout=timeout * 2)
        status_raw = self.vivado.run_tcl_command("report_route_status -return_string", timeout=60)
        parsed = self.parse_route_status(status_raw)
        return RouteResult(
            directive_used="Default",
            routing_errors=parsed.get("routing_errors", -1),
            unrouted_nets=parsed.get("unrouted_nets", -1),
            fully_routed=parsed.get("fully_routed", False),
            fallback_used=True,
            timeout_s=timeout * 2,
            vivado_raw_output=status_raw,
        )

    async def route_with_fallback_async(self, design_cell_count: int, unrouted_net_count: int) -> RouteResult:
        directives = ["Default", "Explore", "AggressiveExplore"]
        timeout = self._compute_timeout(design_cell_count, unrouted_net_count)
        for d in directives:
            tcl = f"route_design {PRESERVE_FLAG} -directive {{{d}}}"
            await self._call_vivado(tcl, timeout=timeout)
            status_raw = await self._call_vivado("report_route_status -return_string", timeout=30)
            parsed = self.parse_route_status(status_raw)
            if parsed.get("routing_errors", 1) == 0:
                return RouteResult(
                    directive_used=d,
                    routing_errors=parsed.get("routing_errors", 0),
                    unrouted_nets=parsed.get("unrouted_nets", 0),
                    fully_routed=parsed.get("fully_routed", False),
                    fallback_used=False,
                    timeout_s=timeout,
                    vivado_raw_output=status_raw,
                )

        logger.warning("All preserve-fixed-routes attempts failed; falling back to full route_design (this will overwrite existing routing)")
        await self._call_vivado(f"route_design -directive {{Default}}", timeout=timeout * 2)
        status_raw = await self._call_vivado("report_route_status -return_string", timeout=60)
        parsed = self.parse_route_status(status_raw)
        return RouteResult(
            directive_used="Default",
            routing_errors=parsed.get("routing_errors", -1),
            unrouted_nets=parsed.get("unrouted_nets", -1),
            fully_routed=parsed.get("fully_routed", False),
            fallback_used=True,
            timeout_s=timeout * 2,
            vivado_raw_output=status_raw,
        )

    def run_phys_opt(self, directives: Optional[List[str]] = None) -> PhysOptResult:
        if directives is None:
            directives = ["Default", "AggressiveExplore"]
        paths = []
        wns_traj: List[float] = []
        run_list: List[str] = []
        for d in directives:
            tcl = f"phys_opt_design -directive {{{d}}}"
            raw = self.vivado.run_tcl_command(tcl, timeout=300)
            run_list.append(d)
            m = _RE_OPTIMIZED_PATHS.search(raw)
            if m:
                paths.append(int(m.group(1)))
            else:
                paths.append(0)
            m2 = _RE_WNS.search(raw)
            if m2:
                try:
                    wns_traj.append(float(m2.group(1)))
                except Exception:
                    pass
        return PhysOptResult(directives_run=run_list, paths_optimized=paths, wns_trajectory=wns_traj)

    async def run_phys_opt_async(self, directives: Optional[List[str]] = None) -> PhysOptResult:
        if directives is None:
            directives = ["Default", "AggressiveExplore"]
        paths = []
        wns_traj: List[float] = []
        run_list: List[str] = []
        for d in directives:
            tcl = f"phys_opt_design -directive {{{d}}}"
            raw = await self._call_vivado(tcl, timeout=300)
            run_list.append(d)
            m = _RE_OPTIMIZED_PATHS.search(raw)
            if m:
                paths.append(int(m.group(1)))
            else:
                paths.append(0)
            m2 = _RE_WNS.search(raw)
            if m2:
                try:
                    wns_traj.append(float(m2.group(1)))
                except Exception:
                    pass
        return PhysOptResult(directives_run=run_list, paths_optimized=paths, wns_trajectory=wns_traj)

    @staticmethod
    def parse_route_status(raw_output: str) -> dict:
        routing_errors = 0
        unrouted = 0
        antenna = 0
        m = _RE_ROUTE_ERRORS.search(raw_output)
        if m:
            try:
                routing_errors = int(m.group(1))
            except Exception:
                routing_errors = -1
        m = _RE_UNROUTED_NETS.search(raw_output)
        if m:
            try:
                unrouted = int(m.group(1))
            except Exception:
                unrouted = -1
        m = _RE_ANTENNA.search(raw_output)
        if m:
            try:
                antenna = int(m.group(1))
            except Exception:
                antenna = 0
        fully = (routing_errors == 0 and unrouted == 0)
        return {
            "routing_errors": routing_errors,
            "unrouted_nets": unrouted,
            "nets_with_antenna": antenna,
            "fully_routed": fully,
        }
