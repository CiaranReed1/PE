#!/bin/bash
# 1 node
#SBATCH -c 1  
#SBATCH --mem=128G 
#SBATCH --job-name="number-crunching_gprof"
#SBATCH -o "number-crunching_gprof.out"
#SBATCH -e "number-crunching_gprof.err" 
#SBATCH -t 00:15:00 
#SBATCH -p test 

module purge
module load gcc/12.2 likwid/5.2.0
mkdir -p gprof_txt

g++ number_crunching.cpp -fno-inline -fno-reorder-functions -O3 -pg -march=native -o number_crunching_O3
g++ number_crunching.cpp -fno-inline -fno-reorder-functions -O0 -pg -march=native -o number_crunching_O0
for k in 1 2 3 4
do
    N=$((k * 10000))
    ./number_crunching_O3 $N
    gprof number_crunching_O3 gmon.out > gprof_txt/gprof_O3_N${N}.txt
    rm -f gmon.out
    ./number_crunching_O0 $N
    gprof number_crunching_O0 gmon.out > gprof_txt/gprof_O0_N${N}.txt
    rm -f gmon.out
done

rm -f number_crunching_O3
rm -f number_crunching_O0