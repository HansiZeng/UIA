#!/bin/bash 

python -m torch.distributed.launch --nproc_per_node=4 trainer/train.py \
                                    --num_train_epochs=4 \
                                    --queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.train.tsv" \
                                    --collection_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/collection.tsv" \
                                    --training_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/1pos_1neg.train.tsv" \
                                    --experiment_folder="/home/jupyter/jointly_rec_and_search/experiments/search/cl-drd/" \
                                    --train_batch_size=128
