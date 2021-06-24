#!/bin/bash

#SBATCH --job-name=scan_cub_256                                 # sets the job name
#SBATCH --output=outfiles/scan_cub_256.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/scan_cub_256.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=24:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=high                                           # set QOS, this will determine what resources can be requested
#SBATCH --mem=128G
#SBATCH --gres gpu:p6000:4
#SBATCH --cpus-per-task=16

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1,2,3 python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cub_256.yml --mode train"
