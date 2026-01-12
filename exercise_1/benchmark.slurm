#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="benchmark"
#SBATCH -o benchmark.out #can use /dev/null to suppress output
#SBATCH -e benchmark.err #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test
module purge
module load gcc
module load likwid/5.2.0
likwid-bench -t sum_sp -w N:16kB:1

