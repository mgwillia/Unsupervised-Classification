#!/bin/bash

#SBATCH --job-name=slbl_c20                                 # sets the job name
#SBATCH --output=slbl_c20.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=slbl_c20.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=72:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --mem=16G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar20.yml"