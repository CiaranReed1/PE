#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=64G #up to 250G on shared queue
#SBATCH --job-name="profile-test"
#SBATCH -o profile-test.out #can use /dev/null to suppress output
#SBATCH -e profile-test.err #can use /dev/null to suppress error output
#SBATCH -t 01:00:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module purge 
module load gcc/12.2 likwid/5.2.0
gcc simple_code.c -O0 -pg -march=native  -o simple_code_O0
gcc simple_code.c -O3 -pg -march=native  -o simple_code_O3
time ./simple_code_O0 100000 > simple_O0.out
time ./simple_code_O3 100000 > simple_O3.out
rm simple_code_O0 simple_code_O3

