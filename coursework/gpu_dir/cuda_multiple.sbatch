#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p tpg-gpu-small
#SBATCH --qos=debug
#SBATCH --mem=10G
#SBATCH -t 00-00:20:00
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --job-name="numcrunch_cuda"
#SBATCH -e "cuda.err"
#SBATCH -o /dev/null

module purge
module load cuda/12.0

# Compile
nvcc number_crunching_loop.cu -O3 -o numcrunch_cuda
nvcc number_crunching_kernel.cu -O3 -o numcrunch_kernel
nvcc number_crunching_task.cu -O3 -o numcrunch_task
nvcc number_crunching_serial.cu -O3 -o serial
mkdir -p cuda_ref

# Run each configuration 5 times
for run in {1..5}
do
    for size in 1000 5000 10000 15000 20000 25000 30000
    do
        ./numcrunch_cuda "$size" > cuda_ref/cuda_"$size".dat
        ./numcrunch_kernel "$size" > /dev/null
        ./numcrunch_task "$size" > cuda_ref/task_"$size".dat
        ./serial "$size" > /dev/null
    done
done

rm numcrunch_cuda
rm numcrunch_kernel
rm numcrunch_task