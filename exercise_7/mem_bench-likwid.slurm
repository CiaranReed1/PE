#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="likwid-transpose-bench"
#SBATCH -o /dev/null #can use /dev/null to suppress output
#SBATCH -e likwid-bench.err #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module purge
module load intel/2022.2
module load likwid/5.2.0
icx -O1 -std=c99 -DLIKWID_PERFMON -pthread -o transpose transpose.c -llikwid
icx -O1 -std=c99 -DLIKWID_PERFMON -pthread -o transpose-blocked transpose-blocked.c -llikwid

#likwid-perfctr -g MEM -C 0 -m ./transpose 4096 4096 > likwid_transpose_4096.out
likwid-perfctr -g MEM -C 0 -m ./transpose-blocked 4096 4096 > likwid_transpose-blocked-32_4096.out
#likwid-perfctr -g MEM -C 0 -m ./transpose 5000 5000 > likwid_transpose_5000.out
likwid-perfctr -g MEM -C 0 -m ./transpose-blocked 5000 5000 > likwid_transpose-blocked-32_5000.out

rm transpose transpose-blocked

