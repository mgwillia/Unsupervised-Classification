#!/bin/bash

#SBATCH --job-name=simclr_imagenet_distilled                                # sets the job name
#SBATCH --output=simclr_imagenet_distilled.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=simclr_imagenet_distilled.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=36:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=high
#SBATCH --gres=gpu:p6000:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1,2,3 python simclr-distill.py --config_env configs/env.yml --config_exp configs/pretext/simclr_imagenet_distilled.yml"
