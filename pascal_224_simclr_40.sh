#!/bin/bash

#SBATCH --job-name=pascal_224_simclr_40                                 # sets the job name
#SBATCH --output=pascal_224_simclr_40.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=pascal_224_simclr_40.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=12:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=high                                           # set QOS, this will determine what resources can be requested
#SBATCH --gres gpu:p6000:4

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1,2,3 python simclr.py --config_env configs/env.yml --config_exp configs/pretext/pascal_224_40.yml"
