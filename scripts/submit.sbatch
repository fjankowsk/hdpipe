#!/bin/bash
#
#SBATCH --job-name=convert
#SBATCH --output=/fred/oz005/users/fjankows/sgr1935/stdout.txt
#SBATCH --chdir=/fred/oz005/users/fjankows/sgr1935
#
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --tmp=20G

srun hostname
srun python /fred/oz005/users/fjankows/sgr1935/preprocess.py
srun echo 'Convert done.'
