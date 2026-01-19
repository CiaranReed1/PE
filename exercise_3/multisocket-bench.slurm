#!/bin/bash
#SBATCH -c 128              # 128 cores (2 sockets × 64)
#SBATCH --mem=3G             # 1 GB per socket → 2 GB total
#SBATCH --job-name=multi_socket_bench
#SBATCH -o multi_socket.out
#SBATCH -e multi_socket.err
#SBATCH -t 01:00:00
#SBATCH -p multi             # Adjust queue if needed

module purge
module load gcc
module load likwid/5.2.0

# -----------------------
# Parameters
# -----------------------
# Private caches per core
declare -A cache_sizes=( ["L1"]=16384 ["L2"]=262144 )

# Shared L3 cache
L3_SLICE=16777216       # 16 MB per slice
SLICE_SHARING=4         # cores per slice
FILL_FRACTION_NUM=2    # numerator of 2/3
FILL_FRACTION_DEN=3    # denominator of 2/3

# Memory (per socket)
MEM_SIZE=$((1024*1024*1024))   # 1 GB

OUTFILE=ALLSOCKETS.dat
rm -f $OUTFILE

# -----------------------
# Loop over cache/memory levels
# -----------------------
for level in L1 L2 L3 MEM; do
    case $level in
        L1)
            size_per_core=${cache_sizes[$level]}
            size_per_socket=$((size_per_core * 64))
            ;;
        L2)
            size_per_core=${cache_sizes[$level]}
            size_per_socket=$((size_per_core * 64))
            ;;
        L3)
            slices_per_socket=$(( (64 + SLICE_SHARING - 1) / SLICE_SHARING ))
            ws_per_slice=$(( L3_SLICE * FILL_FRACTION_NUM / FILL_FRACTION_DEN ))
            size_per_socket=$(( slices_per_socket * ws_per_slice ))
            ;;
        MEM)
            size_per_socket=$MEM_SIZE
            ;;
    esac

    # Run LIKWID on both sockets
    bw=$(likwid-bench -t clload \
        -w S0:${size_per_socket}:64,S1:${size_per_socket}:64 \
        2>/dev/null \
        | awk -F'MByte/s: ' '{if($2!="") bw=$2} END{print bw}')

    echo "128 $level $bw" >> $OUTFILE
done
