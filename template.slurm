#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="<jobname>"
#SBATCH -o <outputfile.out> #can use /dev/null to suppress output
#SBATCH -e <outputfile.err> #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module load gcc
gcc -O2 "<source.c>" -o r
./r >> "<outputfile.dat>"
rm r

