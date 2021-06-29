#!/bin/bash

#SBATCH --job-name=sclr_cub                               # sets the job name
#SBATCH --output=outfiles/sclr_cub.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/sclr_cub.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=72:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres gpu:8
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cub.yml"
