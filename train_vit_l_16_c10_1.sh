#!/bin/bash
#SBATCH --job-name=ViT
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=a100

source activate pytorch

python train.py \
--name cifar10_vit_l_16_seed_1 \
--dataset cifar10 \
--model_type ViT-L_16 \
--pretrained_dir checkpoint/ViT-L_16.npz \
--output_dir output_models/cifar10/ViT-L_16/Run1 \
--seed 1 \
--gradient_accumulation_steps 4

