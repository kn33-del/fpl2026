from dataclasses import dataclass
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


@dataclass
class ShieldResult:
    cells_locked: int
    nets_locked: int
    cells_skipped: int
    nets_skipped: int
    tcl_calls_made: int
    success: bool
    error_message: str = ""


def escape_tcl_name(name: str) -> str:
    # Use {} quoting to preserve special characters in Vivado Tcl
    if name.startswith("{") and name.endswith("}"):
        return name
    return "{" + name + "}"


def _build_tcl_set_property(obj_type: str, prop: str, value: int, names: List[str]) -> str:
    # obj_type: get_cells or get_nets
    if not names:
        return ""
    quoted = " ".join(escape_tcl_name(n) for n in names)
    tcl = f"set objs [{obj_type} {{{quoted}}}]\nset_property {prop} {value} $objs"
    return tcl


class ShieldGenerator:
    def __init__(self, vivado, checkpoint_manager):
        self.vivado = vivado
        self.manager = checkpoint_manager

    def parse_get_cells_output(self, raw: str) -> List[str]:
        if raw is None:
            return []
        raw = raw.strip()
        if raw == "" or raw == '""':
            return []
        # split on whitespace, newlines
        parts = [p for p in raw.replace('\n', ' ').split(' ') if p]
        return parts

    async def get_currently_locked(self, filter_type: str = "both") -> Dict[str, List[str]]:
        locked_cells = []
        locked_nets = []
        if filter_type in ("cells", "both"):
            out = await self.vivado.run_tcl_command("get_cells -filter {IS_LOC_FIXED == 1}")
            locked_cells = self.parse_get_cells_output(out)
        if filter_type in ("nets", "both"):
            out = await self.vivado.run_tcl_command("get_nets -filter {DONT_TOUCH == 1}")
            locked_nets = self.parse_get_cells_output(out)
        return {"locked_cells": locked_cells, "locked_nets": locked_nets}

    async def release_all_shields(self) -> ShieldResult:
        tcl = (
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
        try:
            out = await self.vivado.run_tcl_command(tcl)
            # don't treat warnings as fatal
            return ShieldResult(cells_locked=0, nets_locked=0, cells_skipped=0, nets_skipped=0, tcl_calls_made=1, success=True)
        except Exception as e:
            logger.warning(f"release_all_shields failed: {e}")
            return ShieldResult(cells_locked=0, nets_locked=0, cells_skipped=0, nets_skipped=0, tcl_calls_made=0, success=False, error_message=str(e))

    async def lock_moved_cells(self, cell_names: List[str], lock_bel: bool = True) -> ShieldResult:
        if not cell_names:
            return ShieldResult(0, 0, 0, 0, 0, True)

        blacklist = self.manager.get_blacklist() if self.manager else {"cells": [], "nets": []}
        skip = 0
        to_lock = []
        for c in cell_names:
            if c in blacklist.get("cells", []):
                skip += 1
            else:
                to_lock.append(c)

        if not to_lock:
            return ShieldResult(cells_locked=0, nets_locked=0, cells_skipped=skip, nets_skipped=0, tcl_calls_made=0, success=True)

        tcl = _build_tcl_set_property("get_cells", "IS_LOC_FIXED", 1, to_lock)
        if lock_bel:
            tcl += "\nset_property IS_BEL_FIXED 1 $objs"
        try:
            out = await self.vivado.run_tcl_command(tcl)
            return ShieldResult(cells_locked=len(to_lock), nets_locked=0, cells_skipped=skip, nets_skipped=0, tcl_calls_made=1, success=True)
        except Exception as e:
            logger.warning(f"lock_moved_cells failed: {e}")
            return ShieldResult(cells_locked=0, nets_locked=0, cells_skipped=skip, nets_skipped=0, tcl_calls_made=0, success=False, error_message=str(e))

    async def lock_good_routes(self, net_names: List[str]) -> ShieldResult:
        if not net_names:
            return ShieldResult(0, 0, 0, 0, 0, True)

        blacklist = self.manager.get_blacklist() if self.manager else {"cells": [], "nets": []}
        skip = 0
        to_lock = []
        for n in net_names:
            if n in blacklist.get("nets", []):
                skip += 1
            else:
                to_lock.append(n)

        if not to_lock:
            return ShieldResult(cells_locked=0, nets_locked=0, cells_skipped=0, nets_skipped=skip, tcl_calls_made=0, success=True)

        tcl = _build_tcl_set_property("get_nets", "DONT_TOUCH", 1, to_lock)
        try:
            out = await self.vivado.run_tcl_command(tcl)
            return ShieldResult(cells_locked=0, nets_locked=len(to_lock), cells_skipped=0, nets_skipped=skip, tcl_calls_made=1, success=True)
        except Exception as e:
            logger.warning(f"lock_good_routes failed: {e}")
            return ShieldResult(cells_locked=0, nets_locked=0, cells_skipped=0, nets_skipped=skip, tcl_calls_made=0, success=False, error_message=str(e))

    async def shield_iteration(self, moved_cells: List[str], preserved_nets: List[str]) -> ShieldResult:
        """
        Orchestrate a single iteration's shield sequence.

        Order (CRITICAL):
        1. RapidWright: apply ECOs, unroute affected nets
        2. RapidWright: write_checkpoint("candidate.dcp")
        3. Vivado: open_checkpoint("candidate.dcp")
        4. ShieldGenerator: shield_iteration(moved_cells, preserved_nets)
        5. ECORouter: route_incremental(...)
        6. Vivado: report_timing_summary
        7. CheckpointManager: record(...)

        This method implements steps 1-4 (release previous shields, lock moved cells, lock preserved nets).
        """
        # release previous iteration shields
        res_release = await self.release_all_shields()
        res_cells = await self.lock_moved_cells(moved_cells)
        res_nets = await self.lock_good_routes(preserved_nets)

        total_cells_locked = res_cells.cells_locked
        total_nets_locked = res_nets.nets_locked
        total_cells_skipped = res_cells.cells_skipped
        total_nets_skipped = res_nets.nets_skipped
        tcl_calls = res_release.tcl_calls_made + res_cells.tcl_calls_made + res_nets.tcl_calls_made
        success = res_release.success and res_cells.success and res_nets.success
        error_msgs = ", ".join([m for m in (res_release.error_message, res_cells.error_message, res_nets.error_message) if m])
        return ShieldResult(cells_locked=total_cells_locked, nets_locked=total_nets_locked, cells_skipped=total_cells_skipped, nets_skipped=total_nets_skipped, tcl_calls_made=tcl_calls, success=success, error_message=error_msgs)