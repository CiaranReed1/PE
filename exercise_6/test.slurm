#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=2G #up to 250G on shared queue
#SBATCH --job-name="profile-test"
#SBATCH -o /dev/null #can use /dev/null to suppress output
#SBATCH -e /dev/null #can use /dev/null to suppress error output
#SBATCH -t 00:15:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module purge 
module load gcc/12.2 likwid/5.2.0
#gcc simple_code.c -O0 -pg -march=native  -o simple_code_O0
gcc simple_code.c -O3 -fno-inline -fno-reorder-functions -pg -march=native  -o simple_code_O3
#{ time ./simple_code_O0 100000 > program_out/simple_O0.out; } 2> program_time/simple_O0.time
#mv gmon.out gmon_O0.out 
{ time ./simple_code_O3 100000 > program_out/simple_O3.out; } 2> program_time/simple_O3.time

mv gmon.out gmon_O3.out
#gprof simple_code_O0 gmon_O0.out > gprof_txt/gprof_O0.txt
gprof simple_code_O3 gmon_O3.out > gprof_txt/gprof_O3.txt
#rm simple_code_O0 gmon_O0.out
rm simple_code_O3 gmon_O3.out


