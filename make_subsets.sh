#!/bin/bash

ele=$1

python split_subset.py /scratch/hdd001/home/kaselby/ocp/element_data/1/$ele --prefix 50 --train_size 50 --val_size 1000
python split_subset.py /scratch/hdd001/home/kaselby/ocp/element_data/1/$ele --prefix 200 --train_size 200 --val_size 1000
python split_subset.py /scratch/hdd001/home/kaselby/ocp/element_data/1/$ele --prefix 1000 --train_size 1000 --val_size 1000
python split_subset.py /scratch/hdd001/home/kaselby/ocp/element_data/1/$ele --prefix 5000 --train_size 5000 --val_size 1000