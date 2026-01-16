#!/bin/bash

# 1 node
#SBATCH -c 4  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="benchmark"
#SBATCH -o /dev/null #can use /dev/null to suppress output
#SBATCH -e benchmark.err #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test
module purge
module load gcc
module load likwid/5.2.0
for size in 16kB; do
  echo $size $(likwid-bench -t clstore -w S0:$size:4 ) > 4cores.dat
done
