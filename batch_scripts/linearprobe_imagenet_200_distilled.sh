#!/bin/bash

#SBATCH --job-name=lin_i200_d                               # sets the job name
#SBATCH --output=outfiles/linearprobe_imagenet_200_distilled.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/linearprobe_imagenet_200_distilled.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=36:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1,2,3 python linearprobe.py --config_env configs/env.yml --config_exp configs/linearprobe/linearprobe_imagenet_200_distilled.yml --mode train"
