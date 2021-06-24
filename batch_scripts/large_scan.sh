#!/bin/bash

#SBATCH --job-name=scan_pascal-voc                                 # sets the job name
#SBATCH --output=outfiles/scan_pascal-voc.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/scan_pascal-voc.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=24:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=medium                                           # set QOS, this will determine what resources can be requested
#SBATCH --gres=gpu:1

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0 python scan.py --config_env configs/env.yml --config_exp configs/scan/large_scan.yml"
