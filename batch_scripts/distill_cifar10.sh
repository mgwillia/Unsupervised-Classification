#!/bin/bash

#SBATCH --job-name=d_c10                                 # sets the job name
#SBATCH --output=outfiles/d_c10.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/d_c10.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=72:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1 python simclr-distill.py --config_env configs/env.yml --config_exp configs/pretext/distill_cifar10.yml --mode train"
