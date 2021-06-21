#!/bin/bash

#SBATCH --job-name=scn_i50-1                                # sets the job name
#SBATCH --output=scn_i50-1.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=scn_i50-1.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=72:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --mem=32G
#SBATCH --gres=gpu:p6000:4
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "mkdir -p /scratch0/mgwillia"
srun bash -c "rsync -r /fs/vulcan-datasets/imagenet /scratch0/mgwillia/"

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1,2,3 python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_imagenet50_1.yml --mode train"
