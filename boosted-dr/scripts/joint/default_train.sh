#!/bin/bash 

python -m torch.distributed.launch --nproc_per_node=4 trainer/joint_train_de.py \
                                    --num_train_epochs=4 \
                                    --queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.train.tsv" \
                                    --collection_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/collection.tsv" \
                                    --title_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/collection_title.tsv" \
                                    --training_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/joint_1pos_1neg.train.tsv" \
                                    --experiment_folder="/home/jupyter/jointly_rec_and_search/experiments/joint/cl-drd/" \
                                    --train_batch_size=128