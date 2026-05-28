import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shield_generator import ShieldGenerator, ShieldResult, escape_tcl_name


class MockVivado:
    def __init__(self):
        self.calls = []

    def run_tcl_command(self, tcl: str, timeout: int = 0):
        self.calls.append(tcl)
        # Simulate Vivado returning empty for get_cells when none
        if tcl.startswith("get_cells") or "get_cells" in tcl:
            return ""
        if tcl.startswith("get_nets") or "get_nets" in tcl:
            return ""
        return "OK"


class MockManager:
    def __init__(self, blacklist=None):
        self._blacklist = blacklist or {"cells": [], "nets": []}

    def get_blacklist(self):
        return self._blacklist


def test_lock_cells_batched():
    viv = MockVivado()
    mgr = MockManager()
    shield = ShieldGenerator(vivado=viv, checkpoint_manager=mgr)
    cells = [f"cell{i}" for i in range(5)]
    res = shield.lock_moved_cells(cells)
    assert res.tcl_calls_made == 1
    assert len(viv.calls) == 1
    assert all(name in viv.calls[0] for name in cells)


def test_empty_cell_list_no_tcl():
    viv = MockVivado()
    mgr = MockManager()
    shield = ShieldGenerator(vivado=viv, checkpoint_manager=mgr)
    res = shield.lock_moved_cells([])
    assert res.cells_locked == 0
    assert len(viv.calls) == 0


def test_blacklisted_cells_skipped():
    viv = MockVivado()
    mgr = MockManager(blacklist={"cells": ["bad1", "bad2"], "nets": []})
    shield = ShieldGenerator(vivado=viv, checkpoint_manager=mgr)
    cells = ["good1", "bad1", "good2", "bad2", "good3"]
    res = shield.lock_moved_cells(cells)
    assert res.cells_skipped == 2
    assert res.cells_locked == 3
    # ensure bad1/bad2 not in tcl
    assert "bad1" not in viv.calls[0]
    assert "bad2" not in viv.calls[0]


def test_release_all_clears_both():
    viv = MockVivado()
    mgr = MockManager()
    shield = ShieldGenerator(vivado=viv, checkpoint_manager=mgr)
    res = shield.release_all_shields()
    assert res.tcl_calls_made == 1
    assert "IS_LOC_FIXED" in viv.calls[0]
    assert "DONT_TOUCH" in viv.calls[0]


def test_shield_iteration_order():
    viv = MockVivado()
    mgr = MockManager()
    shield = ShieldGenerator(vivado=viv, checkpoint_manager=mgr)
    shield.shield_iteration(["a"], [])
    # first call is release_all_shields
    assert "IS_LOC_FIXED == 1" in viv.calls[0]
    # second call is lock_moved_cells
    assert "set_property IS_LOC_FIXED" in viv.calls[1]


def test_special_chars_in_cell_name():
    name = "module/cell[3]"
    esc = escape_tcl_name(name)
    assert esc.startswith("{") and esc.endswith("}")


def test_parse_get_cells_empty():
    from shield_generator import ShieldGenerator
    viv = MockVivado()
    mgr = MockManager()
    shield = ShieldGenerator(vivado=viv, checkpoint_manager=mgr)
    assert shield.parse_get_cells_output('') == []
    assert shield.parse_get_cells_output('""') == []


def test_get_currently_locked_parses_names():
    class MockV2(MockVivado):
        def run_tcl_command(self, tcl: str, timeout: int = 0):
            if "get_cells" in tcl:
                return "cell_a cell_b\ncell_c"
            if "get_nets" in tcl:
                return "net1 net2"
            return super().run_tcl_command(tcl, timeout)

    viv = MockV2()
    shield = ShieldGenerator(vivado=viv, checkpoint_manager=MockManager())
    locked = shield.get_currently_locked()
    assert len(locked["locked_cells"]) == 3
    assert len(locked["locked_nets"]) == 2
