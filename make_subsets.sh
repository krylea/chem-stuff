#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --qos=nopreemption
#SBATCH -c 2
#SBATCH --mem=50GB

ele=$1

python split_subset.py /scratch/hdd001/home/kaselby/ocp/element_data/1/$ele --train_size 50
python split_subset.py /scratch/hdd001/home/kaselby/ocp/element_data/1/$ele --train_size 200
python split_subset.py /scratch/hdd001/home/kaselby/ocp/element_data/1/$ele --train_size 1000
python split_subset.py /scratch/hdd001/home/kaselby/ocp/element_data/1/$ele --train_size 5000