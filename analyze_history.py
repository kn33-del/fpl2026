#!/usr/bin/env python3
"""One-command digest of an optimizer run's history.json.

Usage:
    python analyze_history.py <path/to/history.json>

Prints: run summary, per-iteration timeline, win/loss per action family,
directive results, failure-type histogram, and an explore-vs-exploit split
(fresh re-places vs incremental refinement) -- the numbers that took hours
of manual JSON reading during the 2026-07-11/12 debugging sessions.
Stdlib only, so it runs on the contest box as-is.
"""
import json
import sys
from collections import Counter, defaultdict

REROLL_RECIPES = {"place_design_explore", "pblock_full_replace"}
REFINE_RECIPES = {"vivado_pblock", "pblock", "vivado_phys_opt", "phys_opt_design",
                  "phys_opt_design_retime", "vivado_phys_opt_mixed_path", "route_explore"}

GOOD = {"improved", "marginal"}
# "design_unchanged": WNS came back bit-identical, i.e. the action provably
# moved nothing measurable (distinct from no_improvement, where it did move
# something that didn't pay).
BAD = {"regression", "no_improvement", "design_unchanged", "failed", "wns_parse_error"}


def fmt_wns(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) else "  -  "


def main(path: str) -> None:
    with open(path, encoding="utf-8") as fh:
        history = json.load(fh)

    iterations = history.get("iterations", [])
    baseline_fmax = history.get("baseline_fmax_mhz")
    best_fmax = history.get("best_fmax_mhz")

    print("=" * 78)
    print(f"RUN SUMMARY  ({history.get('input_dcp', '?').rsplit('/', 1)[-1]})")
    print("=" * 78)
    print(f"  baseline : WNS {fmt_wns(history.get('baseline_wns'))} ns  "
          f"({baseline_fmax:.1f} MHz)" if baseline_fmax else "  baseline : ?")
    if best_fmax and baseline_fmax:
        print(f"  best     : WNS {fmt_wns(history.get('best_wns'))} ns  "
              f"({best_fmax:.1f} MHz, {best_fmax - baseline_fmax:+.1f})")
    print(f"  best ckpt: {history.get('best_checkpoint', '?').rsplit('/', 1)[-1]}")
    print(f"  iterations: {len(iterations)}   stall_count: {history.get('stall_count')}")
    score_block = history.get("benchmark_score") or {}
    if score_block.get("score") is not None:
        print(f"  score    : {score_block['score']:.3f}  "
              f"(alpha={score_block.get('alpha_delta_fmax_mhz', 0):.2f} MHz, "
              f"beta=${score_block.get('beta_openrouter_cost_usd', 0):.4f}, "
              f"gamma={score_block.get('gamma_runtime_hours', 0):.4f} h)")

    print("\nTIMELINE")
    print(f"  {'it':>3} {'recipe':<28} {'target/directive':<28} "
          f"{'wns_after':>9} {'status':<15}")
    for record in iterations:
        target = ", ".join(str(t) for t in (record.get("targets") or [])[:1])[:28]
        print(f"  {record.get('iter'):>3} {str(record.get('recipe'))[:28]:<28} "
              f"{target:<28} {fmt_wns(record.get('wns_after')):>9} "
              f"{str(record.get('status'))[:15]:<15}")

    action_outcomes: dict = defaultdict(Counter)
    failure_types: Counter = Counter()
    directive_results = {}
    explore = Counter()
    exploit = Counter()
    for record in iterations:
        recipe = str(record.get("recipe"))
        status = str(record.get("status"))
        action_outcomes[recipe][status] += 1
        if status == "failed":
            failure_types[str(record.get("error_type"))] += 1
        targets = record.get("targets") or []
        if targets and str(targets[0]).startswith("directive:"):
            directive_results[str(targets[0]).split(":", 1)[1]] = (
                status, record.get("wns_after"))
        bucket = "good" if status in GOOD else "bad" if status in BAD else "other"
        if recipe in REROLL_RECIPES:
            explore[bucket] += 1
        elif recipe in REFINE_RECIPES:
            exploit[bucket] += 1

    print("\nWIN/LOSS PER ACTION")
    for recipe, outcomes in sorted(action_outcomes.items()):
        wins = sum(outcomes[s] for s in GOOD)
        total = sum(outcomes.values())
        detail = ", ".join(f"{count}x {status}" for status, count in outcomes.most_common())
        print(f"  {recipe:<30} {wins}/{total} good   ({detail})")

    if directive_results:
        print("\nPLACE DIRECTIVES")
        for directive, (status, wns) in directive_results.items():
            print(f"  {directive:<24} {status:<15} wns_after={fmt_wns(wns)}")

    if failure_types:
        print("\nFAILURE TYPES")
        for error_type, count in failure_types.most_common():
            print(f"  {count}x {error_type}")

    print("\nEXPLORE vs EXPLOIT")
    print(f"  fresh re-places (re-roll): {explore['good']}/{sum(explore.values())} good")
    print(f"  incremental refinement:    {exploit['good']}/{sum(exploit.values())} good")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
