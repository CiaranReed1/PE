#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=200G #up to 250G on shared queue
#SBATCH --job-name="roofline-block"
#SBATCH -o roofline-block.out #can use /dev/null to suppress output
#SBATCH -e roofline-block.err #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module purge
module load gcc
gcc dmvm-blocked.c -O0 -o rb0
gcc dmvm-blocked.c -O1 -o rb1
gcc dmvm-blocked.c -O2 -o rb2
gcc dmvm-blocked.c -O3 -o rb3 
echo "#iter rows cols flops" > roofline-blocked_O0.dat
echo "#iter rows cols flops" > roofline-blocked_O1.dat
echo "#iter rows cols flops" > roofline-blocked_O2.dat
echo "#iter rows cols flops" > roofline-blocked_O3.dat

for size in 1000 2500 5000 7500 10000 12500 15000 17500 20000 25000 50000 75000 100000 1000000; do
    ./rb0 $size 10000 >> roofline-blocked_O0.dat
    ./rb1 $size 10000 >> roofline-blocked_O1.dat
    ./rb2 $size 10000 >> roofline-blocked_O2.dat
    ./rb3 $size 10000 >> roofline-blocked_O3.dat
done
rm rb0 rb1 rb2 rb3
