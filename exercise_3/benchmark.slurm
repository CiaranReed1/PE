#!/bin/bash

# 1 node
#SBATCH -c 64  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="benchmark"
#SBATCH -o /dev/null #can use /dev/null to suppress output
#SBATCH -e benchmark.err #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test
module purge
module load gcc
module load likwid/5.2.0
for cores in 1 2 4 8 16 32 64; do
	for size in 16KB 256KB; do
  		echo $cores $size $(likwid-bench -t clload -w S0:($size*$cores):$cores 2> /dev/null | grep MByte/s | cut -f 3) >> bench.dat
	done
done
