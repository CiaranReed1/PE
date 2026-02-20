#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p tpg-gpu-small
#SBATCH --qos=debug
#SBATCH --mem=5G
#SBATCH -t 00-00:10:00
#SBATCH --gres=gpu:pascal:1

module load cuda/12.0

# Compile
nvcc timing-blocks.cu -o timing

# Header
echo "blocks task3 task4-a task4-b" > results-blocks.dat

# Loop over problem sizes
for blocks in 1 2 4 8 16 32 64 128 
do
    ./timing "$blocks" >> results-blocks.dat
done
rm timing
