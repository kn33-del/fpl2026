from eco_router import ECORouter, MOCK_ROUTE_STATUS_CLEAN  # type: ignore


class MockVivado:
    def run_tcl_command(self, tcl, timeout=0):
        if "route_design" in tcl:
            assert "-preserve_fixed_routes" in tcl, "FAIL: preserve flag missing from route_design call"
            return """
# of nets with routing errors:          0
# of unrouted nets:                     0
"""
        if "report_route_status" in tcl:
            return """
# of nets with routing errors:          0
# of unrouted nets:                     0
"""
        if "phys_opt_design" in tcl:
            return "Optimized 2 timing paths\nWNS: -0.12"
        return ""


def main():
    viv = MockVivado()
    router = ECORouter(vivado=viv)
    r = router.route_incremental(50_000, 12)
    wf = router.route_with_fallback(50_000, 12)
    phys = router.run_phys_opt()
    assert isinstance(r.routing_errors, int)
    assert isinstance(wf.fully_routed, bool)
    assert isinstance(phys.paths_optimized, list)
    print("PASS")


if __name__ == "__main__":
    main()
