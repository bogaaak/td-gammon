#!/bin/bash

#SBATCH --job-name=tdGammonGPU
#SBATCH --account=free
#SBATCH --partition=batch-acc
# #SBATCH --nodes=2
# #SBATCH --ntasks-per-node=16
#SBATCH --constraint=k20x
#SBATCH --time=02:00:00
#SBATCH --error=hello_2-%j.err
#SBATCH --output=hello_2-%j.out
# #SBATCH --reservation=training



module purge
module load slurm
module load gcc
module load untested
module load tensorflow-gpu/1.3.0
# module load anaconda3
# source activate tensorflow
python main.py

