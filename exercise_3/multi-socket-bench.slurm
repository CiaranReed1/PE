#!/bin/bash

# 1 node
#SBATCH -c 128  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="benchmark"
#SBATCH -o /dev/null #can use /dev/null to suppress output
#SBATCH -e benchmark.err #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p multi  #Queue: shared, multi, long, bigmem, test
module purge
module load gcc
module load likwid/5.2.0
for size in 1GB; do
  echo $size $(likwid-bench -t clload -w S0:$size:64 S1:$size:64 ) > multisocket.dat
done
