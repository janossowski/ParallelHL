#!/usr/bin/env bash
set -euo pipefail

SEED=42
RUNS=1000
SIZES=(1000 700 300 100 70 30 10)

mkdir -p results

for N in "${SIZES[@]}"; do
    ./bin/rphast_benchmark_cuda benchmark usa.ch \
        --sources "$N" --targets "$N" --queries "$RUNS" --seed "$SEED" \
        --backend ch-parallel \
        --output "results/parallel_ch_usa_${N}_${N}.csv"

    ./bin/rphast_benchmark_cuda benchmark usa.ch \
        --sources "$N" --targets "$N" --queries "$RUNS" --seed "$SEED" \
        --backend cuda-g32-opt \
        --output "results/rphast_usa_${N}_${N}.csv"

    ./bin/hl_benchmark_cuda benchmark usa.hlbin \
        --random "$N" "$N" --runs "$RUNS" --seed "$SEED" \
        --cuda-kernel none --cpu-pair \
        --output "results/parallel_hl_usa_${N}_${N}.csv"

    # kernel1 = pair, kernel2 = hub-source, kernel3 = hub-triple
    declare -A KERNEL_LABELS=( [pair]=kernel1 [hub-source]=kernel2 [hub-triple]=kernel3 )
    for KERNEL in pair hub-source hub-triple; do
        ./bin/hl_benchmark_cuda benchmark usa.hlbin \
            --random "$N" "$N" --runs "$RUNS" --seed "$SEED" \
            --cuda-kernel "$KERNEL" \
            --output "results/${KERNEL_LABELS[$KERNEL]}_usa_${N}_${N}.csv"
    done
done