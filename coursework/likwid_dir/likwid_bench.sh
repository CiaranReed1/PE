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
g++ -mfma -O0 -DLIKWID_PERFMON -fno-inline -march=native -o instrumented_00 number_crunching_likwid.cpp -llikwid
g++ -mfma -O1 -DLIKWID_PERFMON -fno-inline -march=native -o instrumented_01 number_crunching_likwid.cpp -llikwid
g++ -mfma -O2 -DLIKWID_PERFMON -fno-inline -march=native -o instrumented_02 number_crunching_likwid.cpp -llikwid
g++ -mfma -O3 -DLIKWID_PERFMON -fno-inline -march=native -o instrumented_03 number_crunching_likwid.cpp -llikwid

likwid-perfctr -m -g "MEM_DP" -C 0 ./instrumented_00 10000   > likwid_txt/likwid_bench_00.out
likwid-perfctr -m -g "MEM_DP" -C 0 ./instrumented_01 10000   > likwid_txt/likwid_bench_01.out
likwid-perfctr -m -g "MEM_DP" -C 0 ./instrumented_02 10000   > likwid_txt/likwid_bench_02.out
likwid-perfctr -m -g "MEM_DP" -C 0 ./instrumented_03 10000   > likwid_txt/likwid_bench_03.out
rm instrumented_00
rm instrumented_01
rm instrumented_02
rm instrumented_03