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

g++ number_crunching_likwid.cpp -mfma -D LIKWID_PERFMON -fno-inline -O1 -march=native -o number_crunching_likwid_O1 -llikwid
likwid-perfctr -m -g "LS_DISPATCH_LOADS:PMC0,LS_DISPATCH_STORES:PMC1" -C 0 ./number_crunching_likwid_O1 10000 > likwid_txt/likwid_O1_N10000.txt
rm -f number_crunching_likwid_O1