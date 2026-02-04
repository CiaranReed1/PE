#!/bin/bash

# 1 node
#SBATCH -c 1  #Number of CPU cores, 1 per thread by default
#SBATCH --mem=4G #up to 250G on shared queue
#SBATCH --job-name="miniMd-test"
#SBATCH -o miniMD.out #use /dev/null to suppress output
#SBATCH -e miniMD.err #can use /dev/null to suppress error output
#SBATCH -t 00:15:00 #allowed time
#SBATCH -p shared  #Queue: shared, multi, long, bigmem, test
module purge 
module load intel/2022.2
module load intelmpi/2021.6
make intel 
make test
