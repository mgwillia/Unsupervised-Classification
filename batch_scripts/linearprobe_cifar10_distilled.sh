#!/bin/bash

#SBATCH --job-name=linearprobe_c10_d                                 # sets the job name
#SBATCH --output=outfiles/distilled_linearprobe_cifar10.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/distilled_linearprobe_cifar10.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=36:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=medium                                           # set QOS, this will determine what resources can be requested
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1,2,3 python linearprobe.py --config_env configs/env.yml --config_exp configs/linearprobe/linearprobe_cifar10_distilled.yml --mode train"
