#!/bin/bash

#SBATCH --job-name=selflabel_cifar10_distilled                                 # sets the job name
#SBATCH --output=outfiles/selflabel_cifar10_distilled.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/selflabel_cifar10_distilled.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=24:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10_distilled.yml"
