#!/usr/bin/env python3
"""
Analysis Layer for dcp_optimizer.py

Stage 2 of the rollout plan: normalize -> cluster -> diagnose -> gather_evidence
-> rank_hypotheses -> actions_for, now with real scored hypotheses (design doc
section 3/4) instead of Stage 1's single pass-through hypothesis.

What's real and what's still deferred, and why:

  - localized_congestion / long_interconnect / excessive_fanout /
    placement_already_compact are now scored against ACTUAL evidence:
      * cluster-level spread: reuses DCPOptimizer._measure_cluster_spread(),
        which already wraps rapidwright_analyze_critical_path_spread --
        no new tool call, just a new caller (cluster cells instead of a
        single move's target cells).
      * routing detour: a new, gated call to rapidwright_analyze_net_detour
        (confirmed present in rapidwright_tools.py), only fetched when cheap
        evidence leaves the top two hypotheses within CONFIDENCE_GAP of each
        other and neither has hit CONFIDENCE_FLOOR -- this is section 7's
        "expensive evidence only if it would change the answer" gate.
      * net_pct / logic_pct: already in memory (path_delay_breakdown).

  - clock_region and true multi-cell bbox are still NOT implemented.
    rapidwright_tools.py has no clock-region-of-cell or bbox-of-cell-set
    call; analyze_critical_path_spread only returns max Manhattan distance
    between *sequential* cells in a supplied path plus a top-10
    "worst_paths" sample, not full per-cell coordinates for an arbitrary
    cluster. Building a real bbox would mean adding a new tool to
    rapidwright_tools.py (e.g. exposing tile col/row for a cell list the
    way analyze_fabric_for_pblock already walks device.getAllTiles()) --
    that's a server-side change, out of scope here. TimingFailure.bbox /
    clock_region stay None so this can slot in later without touching
    callers.

  - action_failure_memory / diagnosis_outcome_history integration (design
    doc section 8) is still not wired in. checkpoint_manager.py's
    CheckpointManager.record()/iterations schema has no diagnosis-name
    field yet, and adding one means deciding how it interacts with the
    existing blacklist/cooldown bookkeeping in dcp_optimizer.py -- flagging
    for a follow-up rather than bolting it on here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dcp_optimizer import DCPOptimizer

# A cell counted as a "fanout hotspot" within a cluster if it appears as the
# start_cell or end_cell of at least this many members of that cluster.
FANOUT_HOTSPOT_MIN_MEMBERS = 3

HYPOTHESIS_PRIOR = 0.5
# Matches design doc section 7: stop gathering evidence once the top
# hypothesis clears this, or once cheap evidence is exhausted and the gap
# is already decisive.
CONFIDENCE_FLOOR = 0.75
# Only fetch the expensive (net-detour) evidence if the top two hypotheses
# are within this of each other after cheap evidence -- doc section 6/7.
CONFIDENCE_GAP_FOR_EXPENSIVE_EVIDENCE = 0.15
# rapidwright_analyze_net_detour's own default; a detour ratio above this
# means the routed path is meaningfully longer than the straight-line
# distance between source and sink.
DETOUR_RATIO_FLAG_THRESHOLD = 2.0
# Below this fraction of extracted candidates, a recurring cell is fanout
# but not dominant enough to override the base delay-class thresholds.
FANOUT_DOMINANCE_FRACTION = 0.30


@dataclass
class TimingFailure:
    """One row from current_target_candidates, normalized.

    `cells` / `clock_region` / `bbox` are placeholders for richer per-path
    data that rapidwright_tools.py does not currently expose a call for
    (see module docstring) -- left empty/None so downstream code can
    already type against them.
    """

    startpoint: str
    endpoint: str
    start_cell: str
    end_cell: str
    slack_ns: Optional[float]
    endpoint_type: str
    cells: list[str] = field(default_factory=list)
    clock_region: Optional[str] = None
    bbox: Optional[dict] = None


@dataclass
class FailureCluster:
    """A group of TimingFailures that share at least one start/end cell."""

    id: str
    members: list[TimingFailure]
    shared_cells: set[str]
    fanout_hotspots: list[str]
    bbox_union: Optional[dict] = None
    clock_regions: set[str] = field(default_factory=set)


@dataclass
class RootCauseHypothesis:
    name: str
    cluster_id: str
    confidence: float
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    recommended_actions: list[str]
    evidence_requests: list[str]
    # Actions this hypothesis actively forbids, distinct from
    # recommended_actions (which only reorders within what's already
    # allowed). Only placement_already_compact / excessive_fanout use this
    # today -- see design doc section 3's "negative hypothesis" case.
    veto_actions: list[str] = field(default_factory=list)


@dataclass
class Diagnosis:
    clusters: list[FailureCluster]
    hypotheses: list[RootCauseHypothesis]
    primary_cluster_id: str
    primary_hypothesis: RootCauseHypothesis
    reasoning_trace: list[str]
    # Carried alongside the hypothesis so actions_for() can call the
    # optimizer's existing _allowed_forbidden_actions with the same base
    # inputs _build_timing_context always computed.
    delay_class: str
    endpoint_type: str
    net_pct: Optional[float]
    avg_spread: Optional[float]
    # Set by actions_for() after it runs; a one-line human-readable
    # explanation of what, if anything, the primary hypothesis changed
    # versus the base _allowed_forbidden_actions output.
    action_adjustment: Optional[str] = None


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ----------------------------------------------------------------------
# Hypothesis rule functions (design doc section 3). Each takes the primary
# cluster plus a shared evidence dict and returns a scored hypothesis.
# ----------------------------------------------------------------------

def _spread_thresholds():
    # Deferred import: dcp_optimizer.py imports this module at load time,
    # so a top-level `import dcp_optimizer` here would be circular. By the
    # time these rule functions actually run, dcp_optimizer is fully
    # loaded, so importing it lazily inside the function is safe -- and
    # keeps DECISION_SPREAD_TILE_THRESHOLD / DECISION_SPREAD_NET_THRESHOLD
    # single-sourced instead of duplicated as magic numbers here.
    import dcp_optimizer as _dcp
    return _dcp.DECISION_SPREAD_TILE_THRESHOLD, _dcp.DECISION_SPREAD_NET_THRESHOLD


def _rule_localized_congestion(cluster: FailureCluster, evidence: dict) -> RootCauseHypothesis:
    spread_threshold, net_threshold = _spread_thresholds()
    confidence = HYPOTHESIS_PRIOR
    supporting: list[str] = []
    contradicting: list[str] = []
    avg_spread = evidence.get("cluster_avg_spread")
    net_pct = evidence.get("net_pct")
    member_count = evidence.get("member_count", 0)

    if avg_spread is not None and net_pct is not None:
        if avg_spread <= spread_threshold and net_pct > net_threshold and member_count >= 2:
            confidence += 0.25
            supporting.append(
                f"{member_count} clustered paths sit close together (avg spread "
                f"{avg_spread:.1f} tiles <= {spread_threshold}) but are still "
                f"net-delay-bound ({net_pct:.0%} net delay) -- consistent with routing "
                f"congestion rather than physical distance."
            )
        elif avg_spread > spread_threshold:
            confidence -= 0.2
            contradicting.append(
                f"avg spread {avg_spread:.1f} tiles exceeds the congestion-regime "
                f"threshold ({spread_threshold}); these cells aren't actually close."
            )
    if member_count < 2:
        confidence -= 0.15
        contradicting.append("fewer than 2 members in this cluster; congestion needs multiple co-located paths.")

    return RootCauseHypothesis(
        name="localized_congestion",
        cluster_id=cluster.id,
        confidence=max(0.0, min(1.0, confidence)),
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        recommended_actions=["pblock"],
        evidence_requests=["cluster_avg_spread", "net_pct"],
    )


def _rule_long_interconnect(cluster: FailureCluster, evidence: dict) -> RootCauseHypothesis:
    spread_threshold, net_threshold = _spread_thresholds()
    confidence = HYPOTHESIS_PRIOR
    supporting: list[str] = []
    contradicting: list[str] = []
    avg_spread = evidence.get("cluster_avg_spread")
    net_pct = evidence.get("net_pct")
    evidence_requests = ["cluster_avg_spread", "net_pct"]

    if avg_spread is not None and net_pct is not None:
        if avg_spread > spread_threshold and net_pct > net_threshold:
            confidence += 0.25
            supporting.append(
                f"avg spread {avg_spread:.1f} tiles exceeds {spread_threshold} on a "
                f"net-delay-bound cluster ({net_pct:.0%} net delay) -- cells are "
                f"genuinely far apart, not just congested."
            )
        else:
            confidence -= 0.15
            contradicting.append(
                f"avg spread {avg_spread:.1f} tiles / net_pct {net_pct} doesn't clear "
                f"the long-interconnect regime (spread > {spread_threshold}, net_pct > {net_threshold})."
            )

    detour_candidates = evidence.get("detour_candidates")
    if detour_candidates:
        evidence_requests.append("rapidwright_analyze_net_detour")
        cluster_cells = {c for m in cluster.members for c in (m.start_cell, m.end_cell) if c}
        hits = [
            d for d in detour_candidates
            if any(cell and cell in str(d.get("cell", "")) for cell in cluster_cells)
            and float(d.get("max_detour_ratio", 0) or 0) > DETOUR_RATIO_FLAG_THRESHOLD
        ]
        if hits:
            worst = max(float(h.get("max_detour_ratio", 0) or 0) for h in hits)
            confidence += 0.2
            supporting.append(
                f"{len(hits)} cell(s) in this cluster route at > {DETOUR_RATIO_FLAG_THRESHOLD}x "
                f"detour (worst {worst:.2f}x) -- the routed path is much longer than straight-line "
                f"distance, consistent with poor placement locality."
            )
        else:
            confidence -= 0.1
            contradicting.append("no cluster cell showed a flagged routing detour ratio.")

    return RootCauseHypothesis(
        name="long_interconnect",
        cluster_id=cluster.id,
        confidence=max(0.0, min(1.0, confidence)),
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        recommended_actions=["rapidwright_optimize_cell_placement"],
        evidence_requests=evidence_requests,
    )


def _rule_excessive_fanout(cluster: FailureCluster, evidence: dict) -> RootCauseHypothesis:
    confidence = HYPOTHESIS_PRIOR
    supporting: list[str] = []
    contradicting: list[str] = []
    member_count = evidence.get("member_count", 0)
    total_candidates = max(evidence.get("total_candidates", member_count), 1)
    hotspots = cluster.fanout_hotspots

    if hotspots and member_count >= 2:
        fraction = member_count / total_candidates
        if fraction >= FANOUT_DOMINANCE_FRACTION:
            confidence += 0.3
            supporting.append(
                f"{len(hotspots)} cell(s) ({hotspots[:3]}) recur across {member_count}/"
                f"{total_candidates} extracted candidates ({fraction:.0%}) -- consistent "
                f"with a shared high-fanout driver rather than independent placement issues."
            )
        else:
            confidence -= 0.1
            contradicting.append(
                f"recurring cells only account for {fraction:.0%} of extracted candidates "
                f"(< {FANOUT_DOMINANCE_FRACTION:.0%}) -- fanout is present but not dominant."
            )
    else:
        confidence -= 0.2
        contradicting.append("no cell recurs across multiple candidates in this cluster.")

    return RootCauseHypothesis(
        name="excessive_fanout",
        cluster_id=cluster.id,
        confidence=max(0.0, min(1.0, confidence)),
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        recommended_actions=["fanout_split", "replicate_register"],
        evidence_requests=["fanout_hotspots", "total_candidates"],
        # design doc section 3: excessive_fanout should suppress
        # clustering-based pblock/placement actions, since moving one
        # instance of a fanout net doesn't fix the other paths.
        veto_actions=["pblock", "rapidwright_optimize_cell_placement"],
    )


def _rule_placement_already_compact(cluster: FailureCluster, evidence: dict) -> RootCauseHypothesis:
    spread_threshold, _net_threshold = _spread_thresholds()
    confidence = HYPOTHESIS_PRIOR
    supporting: list[str] = []
    contradicting: list[str] = []
    avg_spread = evidence.get("cluster_avg_spread")
    logic_pct = evidence.get("logic_pct")

    if avg_spread is not None and logic_pct is not None:
        if avg_spread <= spread_threshold and logic_pct >= 0.5:
            confidence += 0.3
            supporting.append(
                f"cluster is already tightly placed (avg spread {avg_spread:.1f} tiles "
                f"<= {spread_threshold}) and delay is logic-dominated ({logic_pct:.0%} logic) "
                f"-- moving cells further is unlikely to help; look at logic restructuring instead."
            )
        else:
            confidence -= 0.15
            contradicting.append(
                f"avg spread {avg_spread:.1f} tiles / logic_pct {logic_pct} doesn't support "
                f"'placement is already fine' -- either cells are spread out or delay is net-bound."
            )

    return RootCauseHypothesis(
        name="placement_already_compact",
        cluster_id=cluster.id,
        confidence=max(0.0, min(1.0, confidence)),
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        recommended_actions=[],
        evidence_requests=["cluster_avg_spread", "logic_pct"],
        veto_actions=["pblock", "rapidwright_optimize_cell_placement"],
    )


_HYPOTHESIS_RULES = [
    _rule_localized_congestion,
    _rule_long_interconnect,
    _rule_excessive_fanout,
    _rule_placement_already_compact,
]


class AnalysisEngine:
    """Owned once per DCPOptimizer instance; reuses the optimizer's own
    state and helpers (call_tool, checkpoint_manager, current_target_candidates,
    _classify_endpoint_type, _allowed_forbidden_actions, _measure_cluster_spread,
    _extract_critical_path_pins_file) rather than duplicating them."""

    def __init__(self, optimizer: "DCPOptimizer"):
        self.opt = optimizer

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------
    def normalize(self, candidates: list[dict]) -> list[TimingFailure]:
        failures: list[TimingFailure] = []
        for candidate in candidates:
            endpoint = str(candidate.get("endpoint") or "")
            startpoint = str(candidate.get("startpoint") or "")
            end_cell = endpoint.rsplit("/", 1)[0] if "/" in endpoint else endpoint
            start_cell = startpoint.rsplit("/", 1)[0] if "/" in startpoint else startpoint
            failures.append(
                TimingFailure(
                    startpoint=startpoint,
                    endpoint=endpoint,
                    start_cell=start_cell,
                    end_cell=end_cell,
                    slack_ns=candidate.get("slack"),
                    endpoint_type=self.opt._classify_endpoint_type(endpoint),
                )
            )
        return failures

    # ------------------------------------------------------------------
    # cluster
    # ------------------------------------------------------------------
    def cluster(self, failures: list[TimingFailure]) -> list[FailureCluster]:
        """Union-find over failures; edges added when two failures share a
        start_cell or end_cell (design doc section 5, edge type #1 -- the
        cheapest one, and the only one available without a bbox/clock-region
        tool -- see module docstring)."""
        n = len(failures)
        if n == 0:
            return []

        uf = _UnionFind(n)
        cell_to_indices: dict[str, list[int]] = {}
        for i, failure in enumerate(failures):
            for cell in (failure.start_cell, failure.end_cell):
                if cell:
                    cell_to_indices.setdefault(cell, []).append(i)
        for indices in cell_to_indices.values():
            for a, b in zip(indices, indices[1:]):
                uf.union(a, b)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(uf.find(i), []).append(i)

        clusters: list[FailureCluster] = []
        for cluster_index, indices in enumerate(groups.values()):
            members = [failures[i] for i in indices]
            cell_sets = [
                {cell for cell in (m.start_cell, m.end_cell) if cell} for m in members
            ]
            shared_cells: set[str] = set.intersection(*cell_sets) if cell_sets else set()

            cell_counts = Counter(cell for cell_set in cell_sets for cell in cell_set)
            # NOTE: clusters are formed by union-find on shared cells, so any
            # cluster with >=2 members trivially has at least one cell shared
            # by ALL members (that's how they got unioned in the first
            # place). Scaling the threshold down to len(members) (as an
            # earlier version of this did) made every 2-member cluster
            # spuriously count as a "fanout hotspot" -- two paths converging
            # on one register is normal fan-in, not excessive fanout. Use
            # the absolute threshold so this only fires when a cell recurs
            # across a genuinely large number of paths (design doc's "the
            # same register appears in 37 critical paths" case).
            fanout_hotspots = sorted(
                cell
                for cell, count in cell_counts.items()
                if count >= FANOUT_HOTSPOT_MIN_MEMBERS
                and len(members) > 1
            )

            clusters.append(
                FailureCluster(
                    id=f"cluster_{cluster_index}",
                    members=members,
                    shared_cells=shared_cells,
                    fanout_hotspots=fanout_hotspots,
                )
            )
        return clusters

    def _find_primary_cluster(
        self, clusters: list[FailureCluster], worst_endpoint: str
    ) -> Optional[FailureCluster]:
        """Prefer the cluster containing the current single worst path -- that
        preserves the original design intent of never losing track of what
        today's code already prioritizes (design doc section 5). But if that
        cluster is a singleton, don't give up on clustering entirely: fall back
        to the largest real cluster in the candidate pool instead. Otherwise a
        design where the single worst path happens to be relatively isolated
        (common) permanently starves excessive_fanout/localized_congestion/
        long_interconnect of ever running, even when the rest of the candidate
        pool is dominated by a real shared-cell structure (e.g. a high-fanout
        driver recurring across a dozen other candidates)."""
        worst_cluster = None
        for cluster in clusters:
            if any(m.endpoint == worst_endpoint for m in cluster.members):
                worst_cluster = cluster
                break

        if worst_cluster is not None and len(worst_cluster.members) >= 2:
            return worst_cluster

        non_trivial = [c for c in clusters if len(c.members) >= 2]
        if non_trivial:
            largest = max(non_trivial, key=lambda c: len(c.members))
            return largest

        # Nothing in the pool has real structure -- keep the original
        # worst-path cluster (even if singleton) so independent_failures still
        # fires correctly when the candidates genuinely are all isolated.
        return worst_cluster if worst_cluster is not None else (clusters[0] if clusters else None)

    # ------------------------------------------------------------------
    # gather_evidence
    # ------------------------------------------------------------------
    async def _gather_cheap_evidence(self, cluster: FailureCluster) -> dict:
        """Tier 1/2 evidence from design doc section 6: already-in-memory
        values plus one real RapidWright call (cluster-level spread, reusing
        _measure_cluster_spread) -- no new tool call type, just a new caller."""
        opt = self.opt
        cell_names = sorted({c for m in cluster.members for c in (m.start_cell, m.end_cell) if c})
        evidence: dict = {
            "member_count": len(cluster.members),
            "total_candidates": len(opt.current_target_candidates),
            "net_pct": opt.path_delay_breakdown.get("net_pct"),
            "logic_pct": opt.path_delay_breakdown.get("logic_pct"),
            "cluster_avg_spread": None,
            "cluster_max_spread": None,
        }
        if len(cell_names) >= 2:
            spread = await opt._measure_cluster_spread(cell_names)
            if spread:
                evidence["cluster_avg_spread"] = spread.get("avg_distance")
                evidence["cluster_max_spread"] = spread.get("max_distance")
        return evidence

    async def _gather_detour_evidence(self) -> Optional[list[dict]]:
        """Tier 5 (expensive) evidence from design doc section 6: routing
        detour ratios via rapidwright_analyze_net_detour. Only called when
        cheap evidence leaves the top two hypotheses too close to call."""
        opt = self.opt
        try:
            pins_file = await opt._extract_critical_path_pins_file(num_paths=20)
        except Exception:
            return None
        try:
            result = await opt.call_tool(
                "rapidwright_analyze_net_detour",
                {"input_file": str(pins_file)},
                internal=True,
            )
        except Exception:
            return None
        payload = opt._parse_json_result(result)
        if not payload or opt._result_has_error(payload):
            return None
        return payload.get("all_cells") or payload.get("candidates") or []

    # ------------------------------------------------------------------
    # diagnose (normalize -> cluster already done by caller; this is
    # generate_hypotheses -> gather_evidence -> rank_hypotheses)
    # ------------------------------------------------------------------
    async def diagnose(self, clusters: list[FailureCluster], current_wns: Optional[float]) -> Diagnosis:
        opt = self.opt
        worst = opt.current_target_candidates[0] if opt.current_target_candidates else {}
        worst_endpoint = str(worst.get("endpoint") or "")
        endpoint_type = opt._classify_endpoint_type(worst_endpoint)
        global_net_pct = opt.path_delay_breakdown.get("net_pct")
        global_avg_spread = opt.last_spread_info.get("avg_distance")
        delay_class = opt.path_delay_classification
        if delay_class == "unknown":
            delay_class = "mixed"

        primary_cluster = self._find_primary_cluster(clusters, worst_endpoint)
        total_members = sum(len(c.members) for c in clusters)
        reasoning_trace = [
            f"normalized {total_members} timing-path candidate(s) from current_target_candidates",
            f"clustered into {len(clusters)} cluster(s) via shared start/end cell name",
        ]

        # design doc section 3, "independent_failures": no cluster, or the
        # primary cluster is a singleton -- there's no multi-path structure
        # to diagnose, so fall back to the single-worst-path thresholds
        # exactly as before this module existed.
        if primary_cluster is None or len(primary_cluster.members) < 2:
            reasoning_trace.append(
                "primary cluster has <2 members; falling back to single-path thresholds "
                "(independent_failures -- the safety valve that keeps this pipeline from "
                "ever doing worse than today's behavior when clustering doesn't apply)."
            )
            hypothesis = RootCauseHypothesis(
                name="independent_failures",
                cluster_id=primary_cluster.id if primary_cluster else "none",
                confidence=1.0,
                supporting_evidence=["fewer than 2 members in the primary cluster."],
                contradicting_evidence=[],
                recommended_actions=[],
                evidence_requests=[],
            )
            return Diagnosis(
                clusters=clusters,
                hypotheses=[hypothesis],
                primary_cluster_id=hypothesis.cluster_id,
                primary_hypothesis=hypothesis,
                reasoning_trace=reasoning_trace,
                delay_class=delay_class,
                endpoint_type=endpoint_type,
                net_pct=global_net_pct,
                avg_spread=global_avg_spread,
            )

        worst_path_in_primary = any(m.endpoint == worst_endpoint for m in primary_cluster.members)
        selection_note = (
            "contains the current worst path"
            if worst_path_in_primary
            else "does NOT contain the current worst path -- reassigned to the largest "
                "non-trivial cluster in the pool since the worst path's own cluster was a singleton"
        )
        reasoning_trace.append(
            f"primary cluster '{primary_cluster.id}' {selection_note} "
            f"({len(primary_cluster.members)} member(s), "
            f"{len(primary_cluster.shared_cells)} cell(s) shared across all members)"
        )

        cheap_evidence = await self._gather_cheap_evidence(primary_cluster)
        hypotheses = [rule(primary_cluster, cheap_evidence) for rule in _HYPOTHESIS_RULES]
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        top = hypotheses[0]
        second = hypotheses[1] if len(hypotheses) > 1 else None
        gap_small = second is not None and (top.confidence - second.confidence) < CONFIDENCE_GAP_FOR_EXPENSIVE_EVIDENCE
        if gap_small and top.confidence < CONFIDENCE_FLOOR:
            reasoning_trace.append(
                f"top two hypotheses ({top.name}={top.confidence:.2f}, "
                f"{second.name}={second.confidence:.2f}) within "
                f"{CONFIDENCE_GAP_FOR_EXPENSIVE_EVIDENCE} of each other and neither hit "
                f"the {CONFIDENCE_FLOOR} floor -- fetching rapidwright_analyze_net_detour "
                f"as tie-break evidence."
            )
            detour_candidates = await self._gather_detour_evidence()
            if detour_candidates:
                cheap_evidence["detour_candidates"] = detour_candidates
                hypotheses = [rule(primary_cluster, cheap_evidence) for rule in _HYPOTHESIS_RULES]
                hypotheses.sort(key=lambda h: h.confidence, reverse=True)
            else:
                reasoning_trace.append("net-detour evidence unavailable; keeping cheap-evidence ranking.")

        primary_hypothesis = hypotheses[0]
        reasoning_trace.append(
            "scored hypotheses: "
            + ", ".join(f"{h.name}={h.confidence:.2f}" for h in hypotheses)
        )
        reasoning_trace.append(
            f"primary hypothesis: {primary_hypothesis.name} (confidence {primary_hypothesis.confidence:.2f})"
        )
        for line in primary_hypothesis.supporting_evidence:
            reasoning_trace.append(f"  + {line}")
        for line in primary_hypothesis.contradicting_evidence:
            reasoning_trace.append(f"  - {line}")

        return Diagnosis(
            clusters=clusters,
            hypotheses=hypotheses,
            primary_cluster_id=primary_cluster.id,
            primary_hypothesis=primary_hypothesis,
            reasoning_trace=reasoning_trace,
            delay_class=delay_class,
            endpoint_type=endpoint_type,
            net_pct=global_net_pct,
            avg_spread=global_avg_spread,
        )

    # ------------------------------------------------------------------
    # actions_for (diagnosis_to_allowed_actions)
    # ------------------------------------------------------------------
    def actions_for(
        self, diagnosis: Diagnosis, current_wns: Optional[float]
    ) -> tuple[list[str], list[str]]:
        """Base allowed/forbidden comes from the optimizer's existing,
        unchanged _allowed_forbidden_actions -- that's the safety net every
        hypothesis adjusts on top of, never replaces. A hypothesis only
        takes effect once it clears CONFIDENCE_FLOOR, and even then a veto
        is skipped if it would empty `allowed` entirely (better to fall
        back to the base thresholds than strand the optimizer with zero
        options)."""
        allowed, forbidden = self.opt._allowed_forbidden_actions(
            diagnosis.delay_class,
            diagnosis.endpoint_type,
            diagnosis.net_pct,
            diagnosis.avg_spread,
            current_wns,
        )
        hyp = diagnosis.primary_hypothesis
        adjustment_notes: list[str] = []

        if hyp.confidence >= CONFIDENCE_FLOOR:
            if hyp.veto_actions:
                vetoed = [a for a in hyp.veto_actions if a in allowed]
                if vetoed and len(vetoed) < len(allowed):
                    allowed = [a for a in allowed if a not in hyp.veto_actions]
                    for action in vetoed:
                        if action not in forbidden:
                            forbidden.append(action)
                    adjustment_notes.append(
                        f"{hyp.name} (confidence {hyp.confidence:.2f}) vetoed {vetoed}"
                    )
                elif vetoed:
                    adjustment_notes.append(
                        f"{hyp.name} would veto every allowed action ({vetoed}); "
                        f"skipping the veto rather than stranding the optimizer."
                    )
            if hyp.recommended_actions:
                preferred = [a for a in hyp.recommended_actions if a in allowed]
                if preferred:
                    allowed = preferred + [a for a in allowed if a not in preferred]
                    adjustment_notes.append(
                        f"{hyp.name} (confidence {hyp.confidence:.2f}) prioritized {preferred}"
                    )

        diagnosis.action_adjustment = "; ".join(adjustment_notes) if adjustment_notes else None
        hyp.recommended_actions = hyp.recommended_actions or list(allowed)
        return allowed, forbidden