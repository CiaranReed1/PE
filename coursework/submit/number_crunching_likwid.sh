#!/bin/bash
# 1 node
#SBATCH -c 1  
#SBATCH --mem=128G 
#SBATCH --job-name="number-crunching_likwid"
#SBATCH -o "number-crunching_likwid.out"
#SBATCH -e "number-crunching_likwid.err" 
#SBATCH -t 00:15:00 
#SBATCH -p test 

module purge
module load gcc/12.2 likwid/5.2.0
mkdir -p likwid_txt
g++ -mfma -O1 -DLIKWID_PERFMON -fno-inline -march=native \
    -o instrumented number_crunching_likwid.cpp -llikwid
likwid-perfctr -m -g "MEM_DP" -C 0 ./instrumented 10000   > likwid_txt/number_crunching_likwid.out
rm instrumented