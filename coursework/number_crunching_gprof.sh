#!/bin/bash
# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=128G #up to 250G on shared queue
#SBATCH --job-name="number-crunching_gprof"
#SBATCH -o /dev/null #can use /dev/null to suppress output
#SBATCH -e "number-crunching_gprof.err" #can use /dev/null to suppress error output
#SBATCH -t 00:15:00 #allowed time
#SBATCH -p test  #Queue: shared, multi, long, bigmem, test

module purge
module load gcc/12.2 likwid/5.2.0
mkdir -p gprof_txt

g++ number_crunching.cpp -O0 -pg -march=native -o number_crunching_O0

for k in 1 2 3 4
do
    N=$((k * 10000))
    ./number_crunching_O0 $N
    gprof number_crunching_O0 gmon.out > gprof_txt/gprof_O0_N${N}.txt
    rm -f gmon.out
done

rm -f number_crunching_O0