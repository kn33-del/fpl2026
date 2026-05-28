from shield_generator import ShieldGenerator, ShieldResult


class MockVivado:
    def __init__(self):
        self.calls = []

    def run_tcl_command(self, tcl: str, timeout: int = 0):
        self.calls.append(tcl)
        # Return empty for get_cells/get_nets
        return ""


class MockManager:
    def __init__(self):
        self._blacklist = {"cells": ["bad_cell"], "nets": []}

    def get_blacklist(self):
        return self._blacklist


def main():
    viv = MockVivado()
    mgr = MockManager()
    shield = ShieldGenerator(vivado=viv, checkpoint_manager=mgr)
    res = shield.shield_iteration(["good_cell_a", "bad_cell", "good_cell_b"], [])
    # bad_cell should be skipped
    all_tcl = "\n".join(viv.calls)
    assert "bad_cell" not in all_tcl
    assert "IS_LOC_FIXED" in all_tcl
    assert res.cells_skipped == 1
    # release_all_shields should be present
    assert any("IS_LOC_FIXED == 1" in c for c in viv.calls)
    print("PASS")


if __name__ == "__main__":
    main()
