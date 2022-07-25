#!/bin/bash 


TRAIN_PATHS=("/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec_compl/train/train_5compl_5sim.json")

EPOCHS=(32)

for TRAINING_PATH in "${TRAIN_PATHS[@]}"
do
    for EPOCH in "${EPOCHS[@]}"
    do
python -m torch.distributed.launch --nproc_per_node=4 trainer/graph_train.py \
                                    --num_train_epochs=$EPOCH \
                                    --queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec_compl/anchors_title_catalog.train.tsv" \
                                    --collection_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec_compl/collection_title_catalog.tsv" \
                                    --training_path=$TRAINING_PATH \
                                    --experiment_folder="/home/jupyter/jointly_rec_and_search/experiments/rec_compl/cl-drd/" \
                                    --train_batch_size=256 \
                                    --evaluate_steps=8000 \
                                    --independent_encoders \
                                    --query_max_len=256 \
                                    --logging_steps=50
    done
done