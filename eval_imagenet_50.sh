#!/bin/bash

#SBATCH --job-name=eval_i50                                # sets the job name
#SBATCH --output=eval_imagenet_50.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=eval_imagenet_50.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=36:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=high
#SBATCH --gres=gpu:p6000:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config_exp configs/selflabel/selflabel_imagenet_50.yml --model selflabel_imagenet_50.pth.tar"