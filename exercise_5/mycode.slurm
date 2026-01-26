#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="mycode"
#SBATCH -o /dev/null #can use /dev/null to suppress output
#SBATCH -e mycode.err #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test
module purge
module load gcc/12.2
module load likwid/5.2.0
gcc -std=c99 -mfma -O1 -DLIKWID_PERFMON -fno-inline -march=native -o mycode mycode.c -llikwid
likwid-perfctr -m -g "LS_DISPATCH_LOADS:PMC0,LS_DISPATCH_STORES:PMC1" -C 0 ./mycode 1000000 > mycode.out 2>&1
rm mycode

