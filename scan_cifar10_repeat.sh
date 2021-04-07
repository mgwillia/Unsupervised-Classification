#!/bin/bash

#SBATCH --job-name=scan_cifar10_repeat                          # sets the job name
#SBATCH --output=scan_cifar10_repeat.out.%j                     # indicates a file to redirect STDOUT to; %j is the jobid 
#SBATCH --error=scan_cifar10_repeat.out.%j                      # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=24:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=medium                                            # set QOS, this will determine what resources can be requested
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load cuda/10.0.130                                       # run any commands necessary to setup your environment

srun bash -c "python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --mode train; rm tutorial/cifar-10/scan/*"
srun bash -c "python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --mode train; rm tutorial/cifar-10/scan/*"
srun bash -c "python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --mode train; rm tutorial/cifar-10/scan/*"
srun bash -c "python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --mode train; rm tutorial/cifar-10/scan/*"
srun bash -c "python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --mode train; rm tutorial/cifar-10/scan/*"
srun bash -c "python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --mode train; rm tutorial/cifar-10/scan/*"
srun bash -c "python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --mode train; rm tutorial/cifar-10/scan/*"
srun bash -c "python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --mode train; rm tutorial/cifar-10/scan/*"
srun bash -c "python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --mode train; rm tutorial/cifar-10/scan/*"
srun bash -c "python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --mode train"
