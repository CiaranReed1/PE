#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="stream-test"
#SBATCH -o test.out #can use /dev/null to suppress output
#SBATCH -e test.err #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test
module purge
module load gcc/12.2
module load likwid/5.2.0
gcc -std=c99 -mfma -O1 -DLIKWID_PERFMON -fno-inline -march=native -o stream stream.c -llikwid
likwid-perfctr -m -g "LS_DISPATCH_LOADS:PMC0" -C 0 ./stream 1000000 sca
rm stream

