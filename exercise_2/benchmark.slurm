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
for size in 1kB 2kB 4kB 8kB 16kB 64kB 256kB 1MB 4MB 16MB 64MB 128MB 512MB 1GB; do
  echo $size $(likwid-bench -t clcopy -w N:$size:1 2> /dev/null | grep MByte/s | cut -f 3)
done
