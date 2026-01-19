#!/bin/bash

#SBATCH -c 128
#SBATCH --mem=64G
#SBATCH --job-name=benchmark
#SBATCH -o /dev/null
#SBATCH -e benchmark.err
#SBATCH -t 00:15:00
#SBATCH -p multi

module purge
module load gcc
module load likwid/5.2.0

# Private caches
declare -A cache_sizes=( ["L1"]=16384 ["L2"]=262144 )  # bytes per core

# Shared caches
L3_SLICE=16777216       # 16 MB per slice
SLICE_SHARING=4         # cores per slice
FILL_FRACTION=2/3       # fraction of slice to fill

# Memory benchmark (big enough to exceed caches)
MEM_SIZE=$((1024*1024*1024))   # 1 GB

# Function to run a benchmark and save to file
run_benchmark() {
    local level=$1
    local total_size=$2
    local cores=$3
    local outfile=$4

    bw=$(likwid-bench -t clload -w S0:${total_size}:${cores} 2>/dev/null \
        | grep MByte/s | awk '{print $3}')

    echo "$cores $total_size $bw" >> $outfile
}

# -----------------------
# L1 & L2 (private caches)
# -----------------------
for level in L1 L2; do
    OUTFILE="${level}.dat"
    rm -f $OUTFILE

    size_per_core=${cache_sizes[$level]}

    for cores in 1 2 4 8 16 32 64; do
        total_size=$((size_per_core * cores))
        run_benchmark $level $total_size $cores $OUTFILE
    done
done

# -----------------------
# L3 (shared caches)
# -----------------------
OUTFILE=L3.dat
rm -f $OUTFILE

for cores in 1 2 4 8 16 32 64; do
    # number of L3 slices needed
    slices=$(( (cores + SLICE_SHARING - 1) / SLICE_SHARING ))
    
    # working set per slice
    ws_per_slice=$(echo "$L3_SLICE * $FILL_FRACTION" | bc)
    
    # total array size
    total_size=$(echo "$slices * $ws_per_slice / 1" | bc)
    
    run_benchmark "L3" $total_size $cores $OUTFILE
done

# -----------------------
# Main memory (big array)
# -----------------------
OUTFILE=MEM.dat
rm -f $OUTFILE

for cores in 1 2 4 8 16 32 64; do
    total_size=$MEM_SIZE
    run_benchmark "MEM" $total_size $cores $OUTFILE
done


# -----------------------
# Multi-socket benchmark (128 cores)
# -----------------------
OUTFILE=ALLSOCKETS.dat
rm -f $OUTFILE

for level in L1 L2 L3 MEM; do
    # Determine per-socket array sizes
    case $level in
        L1) size_per_core=16384 ;;
        L2) size_per_core=262144 ;;
        L3) 
            slices_per_socket=$(( (64 + SLICE_SHARING - 1) / SLICE_SHARING ))
            ws_per_slice=$(echo "$L3_SLICE * $FILL_FRACTION" | bc)
            size_per_socket=$(echo "$slices_per_socket * $ws_per_slice / 1" | bc)
            ;;
        MEM) size_per_socket=$((1024*1024*1024)) ;;  # 1 GB per socket
    esac

    # Private caches total size per socket
    if [[ "$level" == "L1" || "$level" == "L2" ]]; then
        size_per_socket=$((size_per_core * 64))
    fi

    # Run LIKWID on both sockets
    bw=$(likwid-bench -t clload \
        -w S0:${size_per_socket}:64,S1:${size_per_socket}:64 \
        2>/dev/null | awk '/MByte\/s/ {sum += $NF} END {print sum}')


    echo "128 $level $bw" >> $OUTFILE
done
