#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=2G #up to 250G on shared queue
#SBATCH --job-name="dmvm"
#SBATCH -o test.out #can use /dev/null to suppress output
#SBATCH -e test.err #can use /dev/null to suppress error output
#SBATCH -t 00:15:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module load gcc
gcc -O0 "dmvm.c" -o r
./r 4000 4000 
rm r

