#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p tpg-gpu-small
#SBATCH --qos=debug
#SBATCH --mem=5G
#SBATCH -t 00-00:10:00
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --job-name="numcrunch_cuda"
#SBATCH -e stderr-"cuda.err"
#SBATHC -o stdout-"cuda.out"

module purge
module load cuda/12.0

# Compile
nvcc number_crunching_.cu -o numcrunch_cuda
mkdir -p cuda_ref
for size in 10000 20000 30000 40000 50000  
do
    ./numcrunch_cuda "$size" > cuda_ref/cuda_"$size".dat
done
rm numcrunch_cuda
