#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="transpose-bench"
#SBATCH -o /dev/null #can use /dev/null to suppress output
#SBATCH -e mem-bench.err #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module load intel/2022.2
icx -O1 -std=c99 -o transpose transpose.c
icx -O1 -std=c99 -o transpose-blocked transpose-blocked.c

echo "Nrow Ncol EffectiveBW EffectiveLoadMBytes/s EffectiveStoreMBytes/s" > transpose_mem.dat
echo "Nrow Ncol EffectiveBW EffectiveLoadMBytes/s EffectiveStoreMBytes/s" > transpose-blocked_mem.dat
for size in 100 200 400 800 1600 3200 6400 12800 25600; do
./transpose $size $size >> transpose_mem.dat
./transpose-blocked $size $size >> transpose-blocked_mem.dat
done
for size in 128 256 512 1024 2048 4096 8192 16384 32768; do
./transpose $size $size >> transpose_mem.dat
./transpose-blocked $size $size >> transpose-blocked_mem.dat
done
rm transpose transpose-blocked

