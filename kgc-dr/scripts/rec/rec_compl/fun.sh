#!/bin/bash

EXPERIMENT_FORDER="/home/jupyter/jointly_rec_and_search/experiments/rec_compl/cl-drd"
FILES=($(ls -d "${EXPERIMENT_FORDER}"/*))

for fn in "${FILES[@]}"
do 
    if [ -d "$fn" ]; then
    echo $fn
    fi
done