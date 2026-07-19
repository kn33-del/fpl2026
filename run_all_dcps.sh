#!/usr/bin/env bash
#
# run_all_dcps.sh -- run the optimizer over every benchmark DCP, unattended.
#
# Designed to be started and left overnight: one crash cannot abort the batch,
# every run is logged separately, a hung run is killed by a per-design timeout,
# and you wake up to a scorecard (initial/best Fmax, improvement, runtime, cost)
# for all designs.
#
# Usage:
#   bash run_all_dcps.sh                 # run everything
#   RESUME=1 bash run_all_dcps.sh        # skip designs already done in the newest batch
#   PER_DESIGN_TIMEOUT=5400 bash run_all_dcps.sh   # 90 min hard cap per design
#   ONLY="logicnets vtr_mcml" bash run_all_dcps.sh # only matching designs
#
# Config via environment (all optional):
#   BENCH_DIR           benchmark directory            (default: fpl26_contest_benchmarks)
#   PER_DESIGN_TIMEOUT  seconds before a run is killed  (default: 4500; 0 = no cap)
#   ONLY                space-separated substrings; run only matching filenames
#   RESUME              if 1, skip designs whose log already shows a finished run
#
# NOTE: goes through `make run_optimizer`, which sets up JAVA_HOME/Vivado for you.
# Requires OPENROUTER_API_KEY in the environment (the LLM-guided flow needs it).

set -u  # deliberately NOT `set -e`: a single design failing must not stop the batch.

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BENCH_DIR="${BENCH_DIR:-fpl26_contest_benchmarks}"
PER_DESIGN_TIMEOUT="${PER_DESIGN_TIMEOUT:-4500}"   # 75 min; internal budget is ~1h
ONLY="${ONLY:-}"
RESUME="${RESUME:-0}"

BATCH_TS="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="batch_run-${BATCH_TS}"
SUMMARY_TSV="${BATCH_DIR}/summary.tsv"
MASTER_LOG="${BATCH_DIR}/batch.log"

# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------
if [[ ! -d "$BENCH_DIR" ]]; then
    echo "ERROR: benchmark directory '$BENCH_DIR' not found. Run 'make setup' first." >&2
    exit 1
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "WARNING: OPENROUTER_API_KEY is not set; the LLM-guided optimizer will fail." >&2
    echo "         Export it before running, or expect every design to error out." >&2
fi

TIMEOUT_BIN=""
if [[ "$PER_DESIGN_TIMEOUT" != "0" ]] && command -v timeout >/dev/null 2>&1; then
    TIMEOUT_BIN="timeout --signal=INT --kill-after=120 ${PER_DESIGN_TIMEOUT}"
fi

mkdir -p "$BATCH_DIR"
printf 'benchmark\tstatus\texit\twall_s\tinit_fmax\tbest_fmax\timprovement\tcost_usd\trun_dir\n' > "$SUMMARY_TSV"

# Log everything (stdout+stderr of this script) to the master log as well as the terminal.
exec > >(tee -a "$MASTER_LOG") 2>&1

# Stop cleanly on Ctrl-C: kill the current child, then exit the batch.
CURRENT_PID=""
on_interrupt() {
    echo ""
    echo "!!! Interrupted -- stopping batch. Killing current run (${CURRENT_PID:-none})."
    [[ -n "$CURRENT_PID" ]] && kill -INT "$CURRENT_PID" 2>/dev/null
    exit 130
}
trap on_interrupt INT

# ----------------------------------------------------------------------------
# Discover input DCPs (skip optimizer outputs)
# ----------------------------------------------------------------------------
declare -a DCPS=()
for dcp in "$BENCH_DIR"/*.dcp; do
    [[ -e "$dcp" ]] || continue
    fn="$(basename "$dcp")"
    [[ "$fn" == *optimized* ]] && continue          # skip our own outputs
    if [[ -n "$ONLY" ]]; then                         # optional name filter
        match=0
        for pat in $ONLY; do [[ "$fn" == *"$pat"* ]] && match=1; done
        [[ "$match" == 1 ]] || continue
    fi
    DCPS+=("$dcp")
done

TOTAL=${#DCPS[@]}
if [[ "$TOTAL" -eq 0 ]]; then
    echo "No input DCPs found in '$BENCH_DIR' (after filters). Nothing to do." >&2
    exit 1
fi

echo "======================================================================"
echo " Batch start: $BATCH_TS"
echo " Designs:     $TOTAL      (dir: $BENCH_DIR)"
echo " Per-design:  ${PER_DESIGN_TIMEOUT}s cap  |  logs: $BATCH_DIR/"
echo "======================================================================"

BATCH_START=$(date +%s)
idx=0

# ----------------------------------------------------------------------------
# Run each design
# ----------------------------------------------------------------------------
for dcp in "${DCPS[@]}"; do
    idx=$((idx + 1))
    fn="$(basename "$dcp")"
    stem="${fn%.dcp}"
    design_log="${BATCH_DIR}/${stem}.log"

    if [[ "$RESUME" == "1" && -f "$design_log" ]] && grep -q "=== DONE ===" "$design_log" 2>/dev/null; then
        echo "[$idx/$TOTAL] SKIP (resume): $fn already completed."
        continue
    fi

    echo ""
    echo "======================================================================"
    echo "[$idx/$TOTAL] $(date '+%H:%M:%S')  Running: $fn"
    echo "======================================================================"

    run_start=$(date +%s)

    # Run through make (inherits Java/Vivado setup). Per-design log + terminal.
    # Backgrounded so the INT trap can target it; we wait immediately.
    $TIMEOUT_BIN make run_optimizer DCP="$dcp" > >(tee "$design_log") 2>&1 &
    CURRENT_PID=$!
    wait "$CURRENT_PID"
    rc=$?
    CURRENT_PID=""

    run_end=$(date +%s)
    wall=$((run_end - run_start))

    # Classify outcome.
    if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
        status="TIMEOUT"
    elif [[ "$rc" -eq 0 ]]; then
        status="OK"
    else
        status="FAIL"
    fi
    echo "=== DONE === $fn  status=$status exit=$rc wall=${wall}s"

    # Find the run dir this design just produced (newest by mtime) and pull the
    # Fmax/cost numbers from its token_usage.json for the scorecard.
    run_dir="$(ls -dt dcp_optimizer_run-*/ 2>/dev/null | head -n1)"
    run_dir="${run_dir%/}"
    init_fmax="-"; best_fmax="-"; improvement="-"; cost="-"
    tu=""
    [[ -n "$run_dir" && -f "$run_dir/token_usage.json" ]] && tu="$run_dir/token_usage.json"
    if [[ -n "$tu" ]]; then
        read -r init_fmax best_fmax improvement cost < <(python3 - "$tu" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))["summary"]
    def f(x): return "-" if x is None else f"{x:.2f}"
    print(f(s.get("initial_fmax_mhz")), f(s.get("best_fmax_mhz")),
          f(s.get("fmax_improvement_mhz")), f(s.get("total_cost")))
except Exception:
    print("- - - -")
PY
)
        # Copy the optimized DCP + report into the batch dir for easy review.
        cp "$run_dir/token_usage.json" "${BATCH_DIR}/${stem}.token_usage.json" 2>/dev/null || true
    fi
    newest_opt="$(ls -t "$BENCH_DIR/${stem}"_optimized-*.dcp 2>/dev/null | head -n1)"
    [[ -n "$newest_opt" ]] && cp "$newest_opt" "${BATCH_DIR}/${stem}_optimized.dcp" 2>/dev/null || true

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$stem" "$status" "$rc" "$wall" "$init_fmax" "$best_fmax" "$improvement" "$cost" "${run_dir:-none}" \
        >> "$SUMMARY_TSV"

    echo "[$idx/$TOTAL] $fn -> $status | init=$init_fmax best=$best_fmax dFmax=$improvement cost=\$$cost"
done

# ----------------------------------------------------------------------------
# Scorecard
# ----------------------------------------------------------------------------
BATCH_END=$(date +%s)
TOTAL_WALL=$((BATCH_END - BATCH_START))

echo ""
echo "======================================================================"
echo " Batch complete in $((TOTAL_WALL / 60)) min $((TOTAL_WALL % 60)) s"
echo "======================================================================"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$SUMMARY_TSV"
else
    cat "$SUMMARY_TSV"
fi
echo ""
echo "Logs, per-design token_usage.json, and optimized DCPs are in: $BATCH_DIR/"
