#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="roofline"
#SBATCH -o roofline.out #can use /dev/null to suppress output
#SBATCH -e roofline.err #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module purge
module load gcc
gcc dmvm.c -O0 -o r0
gcc dmvm.c -O1 -o r1
gcc dmvm.c -O2 -o r2
gcc dmvm.c -O3 -o r3 
echo "#iter rows cols flops" > roofline_O0.out
echo "#iter rows cols flops" > roofline_O1.out
echo "#iter rows cols flops" > roofline_O2.out
echo "#iter rows cols flops" > roofline_O3.out

for size in 1000 2500 5000 7500 10000 12500 15000 17500 20000 25000 50000 75000 100000; do
    ./r0 $size 10000 >> roofline_O0.out
    ./r1 $size 10000 >> roofline_O1.out
    ./r2 $size 10000 >> roofline_O2.out
    ./r3 $size 10000 >> roofline_O3.out
done
rm r0 r1 r2 r3
