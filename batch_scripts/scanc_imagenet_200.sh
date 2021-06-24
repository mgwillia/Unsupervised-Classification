#!/bin/bash

#SBATCH --job-name=scnc_i200                                # sets the job name
#SBATCH --output=outfiles/scnc_i200.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/scnc_i200.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=72:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scanc.py --config_env configs/env.yml --config_exp configs/scanc/scanc_imagenet_200.yml --mode train"
