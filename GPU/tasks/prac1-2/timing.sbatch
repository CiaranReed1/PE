#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p tpg-gpu-small
#SBATCH --qos=debug
#SBATCH --mem=5G
#SBATCH -t 02-00:00:00
#SBATCH --gres=gpu:pascal:1

module load cuda/12.0

# Compile
nvcc timing.cu -o timing

# Header
echo "N task3 task4a task4b" > results.dat

# Loop over problem sizes
for size in 64 128 256 512 1024 2048 4096 8192 16384 32768
do
    ./timing "$size" >> results.dat
done