#!/bin/bash

#SBATCH --job-name=lin_c20                                 # sets the job name
#SBATCH --output=outfiles/lin_c20.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/lin_c20.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=72:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1 python linearprobe.py --config_env configs/env.yml --config_exp configs/linearprobe/linearprobe_cifar20.yml --mode train"
