#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=2G #up to 250G on shared queue
#SBATCH --job-name="instrument-test"
#SBATCH -o instrument.out #use /dev/null to suppress output
#SBATCH -e instrument.err #can use /dev/null to suppress error output
#SBATCH -t 00:15:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module purge 
module load gcc/12.2 likwid/5.2.0
gcc -std=c99 -mfma -O1 -DLIKWID_PERFMON -fno-inline -march=native \
    -o simple_instrumented simple_instrumented.c -llikwid

for perf_group in MEM_DP CACHE TLB L2; do
    echo "Testing ${perf_group}"

    likwid-perfctr \
        -V 2 \
        -m \
        -g "${perf_group}" \
        -C 0 \
        ./simple_instrumented \
        10000 \
    || echo "FAILED: ${perf_group}"
done
 
rm simple_instrumented
