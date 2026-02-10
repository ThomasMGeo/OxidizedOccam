#!/usr/bin/env bash
#
# Benchmark script for OxidizedOccam
#
# Runs all three example inversions for both Rust and Fortran,
# records timing, and outputs results to a CSV file.
#
# Usage:
#   ./benchmark.sh                  # default: 10 runs per example
#   ./benchmark.sh 5                # 5 runs per example
#   ./benchmark.sh 1 results.csv    # 1 run, custom output file

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
RUST_BINARY="$REPO_DIR/target/release/occam1d"
FORTRAN_BINARY="$REPO_DIR/Occam1DCSEM_matlab/Source/OCCAM1DCSEM"
N_RUNS="${1:-10}"
OUTPUT="${2:-benchmark_results.csv}"
TMPDIR="$(mktemp -d)"

trap 'rm -rf "$TMPDIR"' EXIT

# Build Rust release if needed
if [ ! -f "$RUST_BINARY" ]; then
    echo "Building Rust release binary..."
    (cd "$REPO_DIR" && cargo build --release 2>/dev/null)
fi

# Build Fortran if needed
if [ ! -f "$FORTRAN_BINARY" ]; then
    echo "Building Fortran binary (gfortran -O2, native)..."
    (cd "$REPO_DIR/Occam1DCSEM_matlab/Source" && \
     make clean && \
     make OCCAM1DCSEM FC=gfortran "FCFLAGS=-O2" "LIBS=-framework Accelerate") 2>/dev/null
fi

echo "Fortran binary: $(file "$FORTRAN_BINARY" | sed 's/.*: //')"
echo "Rust binary:    $(file "$RUST_BINARY" | sed 's/.*: //')"

# Examples to benchmark: (name, directory, startup_file)
EXAMPLES=(
    "csem_unbounded|$REPO_DIR/examples/canonical_reservoir|startup"
    "csem_bounded|$REPO_DIR/examples/canonical_reservoir_bounded|startup"
    "mt|$REPO_DIR/examples/canonical_mt|startup"
)

echo ""
echo "OxidizedOccam Benchmark"
echo "======================="
echo "Runs per example: $N_RUNS"
echo "Output: $OUTPUT"
echo ""

# CSV header
echo "engine,example,run,iterations,wall_time_s,cpu_time_s,final_rms,ms_per_iter" > "$OUTPUT"

# ---------------------------------------------------------------
# Run Rust benchmarks
# ---------------------------------------------------------------
echo "=== Rust ==="
for entry in "${EXAMPLES[@]}"; do
    IFS='|' read -r name dir startup <<< "$entry"
    echo "--- $name ---"

    for run in $(seq 1 "$N_RUNS"); do
        run_tmp="$TMPDIR/rust_${name}_run${run}"
        mkdir -p "$run_tmp"

        output=$( { cd "$dir" && /usr/bin/time -p "$RUST_BINARY" invert \
            --startup "$startup" \
            -o "$run_tmp/iter" \
            -v; } 2>&1 )

        iterations=$(echo "$output" | grep -oE '[0-9]+ iterations' | grep -oE '[0-9]+' || echo "0")
        wall_time=$(echo "$output" | grep 'Total wall time' | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "NA")
        user_time=$(echo "$output" | grep '^user ' | awk '{print $2}' || echo "0")
        sys_time=$(echo "$output" | grep '^sys ' | awk '{print $2}' || echo "0")
        cpu_time=$(echo "$user_time $sys_time" | awk '{printf "%.2f", $1 + $2}')
        final_line=$(echo "$output" | grep -E '(converged|target)' | tail -1)
        final_rms=$(echo "$final_line" | awk '{print $2}' || echo "NA")

        if [ "$iterations" -gt 0 ] 2>/dev/null && [ "$wall_time" != "NA" ]; then
            ms_per_iter=$(echo "$wall_time $iterations" | awk '{printf "%.2f", $1 / $2 * 1000}')
        else
            ms_per_iter="NA"
        fi

        echo "  run $run: ${wall_time}s wall, ${iterations} iters, ${ms_per_iter} ms/iter"
        echo "rust,$name,$run,$iterations,$wall_time,$cpu_time,$final_rms,$ms_per_iter" >> "$OUTPUT"
    done
    echo ""
done

# ---------------------------------------------------------------
# Run Fortran benchmarks
# ---------------------------------------------------------------
echo "=== Fortran ==="
for entry in "${EXAMPLES[@]}"; do
    IFS='|' read -r name dir startup <<< "$entry"
    echo "--- $name ---"

    for run in $(seq 1 "$N_RUNS"); do
        run_tmp="$TMPDIR/fortran_${name}_run${run}"
        mkdir -p "$run_tmp"

        # Fortran writes ITER_*.iter to cwd, so run from a temp dir with symlinked inputs
        ln -sf "$dir/$startup" "$run_tmp/startup"
        ln -sf "$dir/model" "$run_tmp/model" 2>/dev/null || true
        # Symlink all data files
        for f in "$dir"/*.data; do
            [ -f "$f" ] && ln -sf "$f" "$run_tmp/$(basename "$f")"
        done

        output=$( { cd "$run_tmp" && /usr/bin/time -p "$FORTRAN_BINARY" startup; } 2>&1 )

        # Parse Fortran output: "Total Time (s):   3.745"
        wall_time=$(echo "$output" | grep 'Total Time' | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "NA")
        user_time=$(echo "$output" | grep '^user ' | awk '{print $2}' || echo "0")
        sys_time=$(echo "$output" | grep '^sys ' | awk '{print $2}' || echo "0")
        cpu_time=$(echo "$user_time $sys_time" | awk '{printf "%.2f", $1 + $2}')

        # Parse iteration count from last "Iteration:" line
        iterations=$(echo "$output" | grep '^ *Iteration:' | tail -1 | grep -oE '[0-9]+' || echo "0")

        # Parse final RMS from last "RMS Misfit:" line
        final_rms=$(echo "$output" | grep 'RMS Misfit:' | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "NA")

        if [ "$iterations" -gt 0 ] 2>/dev/null && [ "$wall_time" != "NA" ]; then
            ms_per_iter=$(echo "$wall_time $iterations" | awk '{printf "%.2f", $1 / $2 * 1000}')
        else
            ms_per_iter="NA"
        fi

        echo "  run $run: ${wall_time}s wall, ${iterations} iters, ${ms_per_iter} ms/iter"
        echo "fortran,$name,$run,$iterations,$wall_time,$cpu_time,$final_rms,$ms_per_iter" >> "$OUTPUT"
    done
    echo ""
done

# ---------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------
echo "Results written to: $OUTPUT"
echo ""
echo "Summary (median values):"
echo "========================"
printf "%-18s %-8s %6s %10s %10s %12s %8s\n" "Example" "Engine" "Iters" "Wall(s)" "CPU(s)" "ms/iter" "Ratio"
printf "%-18s %-8s %6s %10s %10s %12s %8s\n" "-------" "------" "-----" "-------" "------" "-------" "-----"

for entry in "${EXAMPLES[@]}"; do
    IFS='|' read -r name dir startup <<< "$entry"

    for engine in rust fortran; do
        rows=$(grep "^$engine,$name," "$OUTPUT")
        median_wall=$(echo "$rows" | cut -d, -f5 | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
        median_cpu=$(echo "$rows" | cut -d, -f6 | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
        median_iters=$(echo "$rows" | cut -d, -f4 | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
        median_ms=$(echo "$rows" | cut -d, -f8 | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')

        # Store for ratio computation
        if [ "$engine" = "rust" ]; then
            rust_ms="$median_ms"
            printf "%-18s %-8s %6s %10s %10s %12s %8s\n" "$name" "$engine" "$median_iters" "$median_wall" "$median_cpu" "$median_ms" ""
        else
            if [ "$rust_ms" != "NA" ] && [ "$median_ms" != "NA" ] && [ "$rust_ms" != "0.00" ]; then
                ratio=$(echo "$median_ms $rust_ms" | awk '{printf "%.1fx", $1 / $2}')
            else
                ratio="N/A"
            fi
            printf "%-18s %-8s %6s %10s %10s %12s %8s\n" "" "$engine" "$median_iters" "$median_wall" "$median_cpu" "$median_ms" "$ratio"
        fi
    done
done
