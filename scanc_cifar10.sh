#!/bin/bash

#SBATCH --job-name=scanc_cifar10                                 # sets the job name
#SBATCH --output=scanc_cifar10.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=scanc_cifar10.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=24:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=medium                                           # set QOS, this will determine what resources can be requested
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; python scanc.py --config_env configs/env.yml --config_exp configs/scanc/scanc_cifar10.yml --mode train"
