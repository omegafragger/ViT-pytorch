#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

source activate pytorch
cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=${1:-8888} # use your desired port 
