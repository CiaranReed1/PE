#!/bin/bash
#SBATCH -c 128                # total cores
#SBATCH --mem=128G             # total memory
#SBATCH --job-name=benchmark
#SBATCH -o benchmark.out
#SBATCH -e benchmark.err
#SBATCH -t 10:00:00            # 10 hours
#SBATCH -p multi

module purge
module load gcc
module load likwid/5.2.0

# -----------------------
# Configuration
# -----------------------

# L1/L2 per-core working set (in bytes)
L1_PER_CORE=$((16 * 1024))      # 16 KB
L2_PER_CORE=$((256 * 1024))     # 256 KB

# L3 cache parameters
L3_SLICE=$((16 * 1024 * 1024))  # 16 MB per slice
SLICE_SHARING=4                 # cores per L3 slice
FILL_FRACTION=2/3               # fraction of slice to fill

# Memory benchmark
MEM_PER_SOCKET=$((1 * 1024 * 1024 * 1024))  # 1 GB per socket

# NUMA sockets
SOCKET_CORES=64

# Output directory
mkdir -p data

# -----------------------
# Benchmark function
# -----------------------
run_benchmark() {
    local level=$1
    local cores=$2
    local w0=$3
    local w1=$4
    local outfile=$5

    if [[ -n "$w1" ]]; then
        bw=$(likwid-bench -t clload -w "$w0" -w "$w1" 2>/dev/null \
            | awk '/MByte\/s/ {sum += $NF} END {print sum}')
    else
        bw=$(likwid-bench -t clload -w "$w0" 2>/dev/null \
            | awk '/MByte\/s/ {sum += $NF} END {print sum}')
    fi

    echo "$cores $bw" >> "$outfile"
}

# -----------------------
# Main loop: 1 → 128 cores
# -----------------------
for level in L1 L2 L3 MEM; do
    OUTFILE="data/${level}.dat"
    rm -f "$OUTFILE"

    for cores in $(seq 1 128); do

        # Determine cores per socket
        if (( cores <= SOCKET_CORES )); then
            c0=$cores
            c1=0
        else
            c0=$SOCKET_CORES
            c1=$(( cores - SOCKET_CORES ))
        fi

        # -----------------------
        # L1/L2 private caches
        # -----------------------
        if [[ "$level" == "L1" || "$level" == "L2" ]]; then
            size_per_core=$([[ "$level" == "L1" ]] && echo $L1_PER_CORE || echo $L2_PER_CORE)

            w0=""
            w1=""
            if (( c0 > 0 )); then
                total0=$((size_per_core * c0))
                w0="S0:${total0}B:${c0}"
            fi
            if (( c1 > 0 )); then
                total1=$((size_per_core * c1))
                w1="S1:${total1}B:${c1}"
            fi

        # -----------------------
        # L3 shared cache
        # -----------------------
        elif [[ "$level" == "L3" ]]; then
            slices0=$(( (c0 + SLICE_SHARING - 1) / SLICE_SHARING ))
            slices1=$(( (c1 + SLICE_SHARING - 1) / SLICE_SHARING ))

            ws_per_slice=$(echo "$L3_SLICE * $FILL_FRACTION" | bc)

            w0=""
            w1=""
            if (( c0 > 0 )); then
                total0=$(echo "$slices0 * $ws_per_slice / 1" | bc)
                w0="S0:${total0}B:${c0}"
            fi
            if (( c1 > 0 )); then
                total1=$(echo "$slices1 * $ws_per_slice / 1" | bc)
                w1="S1:${total1}B:${c1}"
            fi

        # -----------------------
        # Main memory
        # -----------------------
        elif [[ "$level" == "MEM" ]]; then
            w0=""
            w1=""
            if (( c0 > 0 )); then
                w0="S0:${MEM_PER_SOCKET}B:${c0}"
            fi
            if (( c1 > 0 )); then
                w1="S1:${MEM_PER_SOCKET}B:${c1}"
            fi
        fi

        # Run the benchmark
        run_benchmark "$level" "$cores" "$w0" "$w1" "$OUTFILE"

    done
done

