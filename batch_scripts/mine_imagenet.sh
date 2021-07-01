#!/bin/bash

#SBATCH --job-name=mine_inet                              # sets the job name
#SBATCH --output=outfiles/mine_inet.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/mine_inet.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=72:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "mkdir -p /scratch0/mgwillia"
srun bash -c "rsync -r /fs/vulcan-datasets/imagenet /scratch0/mgwillia/"

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1,2,3 python mine.py --config_env configs/env.yml --config_exp configs/pretext/mine_imagenet.yml"
