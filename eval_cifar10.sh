#!/bin/bash

#SBATCH --job-name=eval_cifar10                                # sets the job name
#SBATCH --output=eval_cifar10.out.%j                            # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=eval_cifar10.out.%j                             # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=36:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=medium
#SBATCH --gres=gpu:p6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load cuda/10.0.130                                    # run any commands necessary to setup your environment

srun bash -c "hostname; CUDA_VISIBLE_DEVICES=0 python eval.py --config_exp configs/selflabel/cifar10_eval.yml --model tutorial/cifar-10/selflabel/model.pth.tar"
