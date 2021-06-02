#!/bin/bash

#SBATCH --job-name=sclr_c10_finetune                                 # sets the job name
#SBATCH --output=simclr_cifar10_finetune.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=simclr_cifar10_finetune.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=48:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --mem=64G
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=2

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0 python simclr-distill.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10_finetune.yml"
