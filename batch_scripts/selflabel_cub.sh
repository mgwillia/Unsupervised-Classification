#!/bin/bash

#SBATCH --job-name=selflabel_cub                                 # sets the job name
#SBATCH --output=outfiles/selflabel_cub.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/selflabel_cub.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=24:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=high                                           # set QOS, this will determine what resources can be requested
#SBATCH --gres=gpu:p6000:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cub.yml"