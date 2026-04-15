#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p tpg-gpu-small
#SBATCH --qos=debug
#SBATCH --mem=15G
#SBATCH -t 00-00:10:00
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --job-name="serial_reference"
#SBATCH -e "serial_reference.err"
#SBATHC -o "serial_reference.out"

module purge
module load cuda/12.0

# Compile
nvcc number_crunching_serial.cu -o serial
mkdir -p serial_ref
for size in 10000 20000 30000 40000 50000  
do
    ./serial "$size" > serial_ref/serial_"$size".dat
done
rm serial
