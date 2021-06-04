#!/bin/bash

#SBATCH --job-name=sclr_c10                                 # sets the job name
#SBATCH --output=sclr_c10.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=sclr_c10.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=72:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0 python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml"
