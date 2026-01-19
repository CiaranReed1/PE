#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="roofline"
#SBATCH -o roofline.out #can use /dev/null to suppress output
#SBATCH -e roofline.err #can use /dev/null to suppress error output
#SBATCH -t 00:15:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module purge
module load gcc
gcc dmvm.c -O0 -o r 
for size in 1000 2500 5000 7500 10000 12500 15000 17500 20000 25000 50000 75000 100000; do
    ./r $size 10000 
done
rm r
