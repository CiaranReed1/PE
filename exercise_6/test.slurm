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
gcc simple_code.c -o simple_code
time ./simple_code
rm simple_code 

