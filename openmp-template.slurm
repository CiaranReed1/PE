#!/bin/bash

# 1 node
#SBATCH -c 32  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="<jobname>"
#SBATCH -o <outputfile.out> #can use /dev/null to suppress output
#SBATCH -e <outputfile.err> #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module load gcc

export OMP_PLACES=cores  #sets where threads are, cores = 1 process per core, threads = allowes hyperthreading, sockets = one process per socket
export OMP_PROC_BIND=close #how threads are bound to place specified above, close = as close as possible (fills cores sequentially), spread = spread out as much as possible (for bandwidth)
export OMP_DISPLAY_ENV=TRUE # display OpenMP environment variables

echo "<outputheaders>" > "<outputfile.dat>"
gcc -O2 -lm -fopenmp "<source.c>" -o r
./r >> "<outputfile.dat>"
rm r

