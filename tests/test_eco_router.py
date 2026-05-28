import logging
import pytest
import sys
from pathlib import Path

# Ensure project root is on sys.path for test discovery when pytest is invoked
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eco_router import ECORouter, PRESERVE_FLAG


class MockVivado:
    def __init__(self, responses=None):
        # responses: list of (tcl_substr, return_string) or callable
        self.calls = []
        self._responses = responses or []

    def run_tcl_command(self, tcl: str, timeout: int = 0):
        self.calls.append((tcl, timeout))
        # If responses is callable, call it
        for pattern, resp in self._responses:
            if pattern in tcl:
                if callable(resp):
                    return resp(tcl)
                return resp
        return ""


MOCK_ROUTE_STATUS_CLEAN = """
# of nets with routing errors:          0
# of unrouted nets:                     0
"""

MOCK_ROUTE_STATUS_ERROR = """
# of nets with routing errors:          3
# of unrouted nets:                     5
"""


def test_preserve_flag_always_present():
    viv = MockVivado(responses=[("route_design", MOCK_ROUTE_STATUS_CLEAN), ("report_route_status", MOCK_ROUTE_STATUS_CLEAN)])
    router = ECORouter(vivado=viv)
    res = router.route_with_fallback(design_cell_count=1000, unrouted_net_count=10)
    # ensure preserve flag present in every route_design call
    for tcl, _ in viv.calls:
        if "route_design" in tcl:
            assert PRESERVE_FLAG in tcl
    assert res.routing_errors == 0


def test_timeout_scaling_large_design():
    viv = MockVivado(responses=[("route_design", MOCK_ROUTE_STATUS_CLEAN), ("report_route_status", MOCK_ROUTE_STATUS_CLEAN)])
    router = ECORouter(vivado=viv, base_timeout_s=600, large_design_threshold=300_000)
    _ = router.route_incremental(design_cell_count=400_000, unrouted_net_count=10)
    # find first call timeout
    first_timeout = viv.calls[0][1]
    assert first_timeout > 600


def test_timeout_scaling_many_nets():
    viv = MockVivado(responses=[("route_design", MOCK_ROUTE_STATUS_CLEAN), ("report_route_status", MOCK_ROUTE_STATUS_CLEAN)])
    router = ECORouter(vivado=viv, base_timeout_s=600)
    _ = router.route_incremental(design_cell_count=1000, unrouted_net_count=600)
    first_timeout = viv.calls[0][1]
    assert first_timeout >= 720  # 600 + 120


def test_parse_route_status_clean():
    parsed = ECORouter.parse_route_status(MOCK_ROUTE_STATUS_CLEAN)
    assert parsed["routing_errors"] == 0
    assert parsed["fully_routed"] is True


def test_parse_route_status_errors():
    parsed = ECORouter.parse_route_status(MOCK_ROUTE_STATUS_ERROR)
    assert parsed["routing_errors"] == 3
    assert parsed["fully_routed"] is False


def test_fallback_progression():
    # First two preserve attempts return errors, third returns clean
    responses = [
        ("route_design", lambda t: ""),
        ("report_route_status", MOCK_ROUTE_STATUS_ERROR),
    ]

    # We'll implement a stateful responder
    state = {"count": 0}

    def responder(tcl):
        if "report_route_status" in tcl:
            state["count"] += 1
            if state["count"] < 3:
                return MOCK_ROUTE_STATUS_ERROR
            return MOCK_ROUTE_STATUS_CLEAN
        return ""

    viv = MockVivado(responses=[("route_design", ""), ("report_route_status", responder)])
    router = ECORouter(vivado=viv)
    res = router.route_with_fallback(design_cell_count=1000, unrouted_net_count=10)
    assert res.directive_used == "AggressiveExplore"
    assert res.fallback_used is False


def test_full_fallback_logs_warning(caplog):
    caplog.set_level(logging.WARNING)
    # Make all preserve attempts fail; final full route also fails
    def bad_status(tcl):
        return MOCK_ROUTE_STATUS_ERROR

    viv = MockVivado(responses=[("route_design", ""), ("report_route_status", bad_status)])
    router = ECORouter(vivado=viv)
    res = router.route_with_fallback(design_cell_count=1000, unrouted_net_count=10)
    assert res.fallback_used is True
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_phys_opt_parses_paths_optimized():
    output = "Optimized 14 timing paths\nSome other text"
    viv = MockVivado(responses=[("phys_opt_design", output)])
    router = ECORouter(vivado=viv)
    phys = router.run_phys_opt(directives=["Default"])  # single directive
    assert phys.paths_optimized == [14]
