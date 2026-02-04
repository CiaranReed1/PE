#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=2G #up to 250G on shared queue
#SBATCH --job-name="instrument-test"
#SBATCH -o instrument.out use /dev/null to suppress output
#SBATCH -e instrument.err #can use /dev/null to suppress error output
#SBATCH -t 00:15:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module purge 
module load gcc/12.2 likwid/5.2.0
gcc -std=c99 -mfma -O1 -DLIKWID_PERFMON -fno-inline -march=native \
    -o simple_instrumented simple_instrumented.c -llikwid
likwid-perfctr -m -g "MEM_DP" -C 0 ./simple_instrumented 100000  
