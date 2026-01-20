#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=10G #up to 250G on shared queue
#SBATCH --job-name="triad"
#SBATCH -o triad.out #can use /dev/null to suppress output
#SBATCH -e triad.err #can use /dev/null to suppress error output
#SBATCH -t 00:15:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test

module purge
module load gcc
module load likwid/5.2.0

likwid-bench -t triad -w S0:5GB:1
