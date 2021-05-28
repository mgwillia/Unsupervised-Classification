#!/bin/bash

#SBATCH --job-name=distill_simclr_cifar10                                 # sets the job name
#SBATCH --output=distill_simclr_cifar10.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=distill_simclr_cifar10.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=36:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=high                                           # set QOS, this will determine what resources can be requested
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; python simclr-distill.py --config_env configs/env.yml --config_exp configs/pretext/simclr_distill_cifar10.yml"
