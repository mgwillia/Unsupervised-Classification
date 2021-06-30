#!/bin/bash

#SBATCH --job-name=lin_stl10                                 # sets the job name
#SBATCH --output=outfiles/lin_stl10.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=outfiles/lin_stl10.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=72:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "mkdir -p /scratch0/mgwillia;"
srun bash -c "rsync -r /vulcanscratch/mgwillia/stl10_binary.tar.gz /scratch0/mgwillia/;"
srun bash -c "tar -zxf /scratch0/mgwillia/stl10_binary.tar.gz -C /scratch0/mgwillia/;"
srun bash -c "ls /scratch0/mgwillia/*;"

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1,2,3 python linearprobe.py --config_env configs/env.yml --config_exp configs/linearprobe/linearprobe_stl10.yml --mode train"
