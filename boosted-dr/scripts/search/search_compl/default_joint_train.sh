#!/bin/bash 


TRAIN_PATHS=("/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/joint/1pos_1neg.similar.train.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/joint/1pos_1neg.compl.train.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/joint/1pos_1neg.similar_compl.train.tsv")

for TRAINING_PATH in "${TRAIN_PATHS[@]}"
do
python -m torch.distributed.launch --nproc_per_node=4 trainer/joint_train_de.py \
                                    --num_train_epochs=4 \
                                    --queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/queries.train.tsv" \
                                    --collection_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/collection.tsv" \
                                    --title_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/collection_title.tsv" \
                                    --training_path=$TRAINING_PATH \
                                    --experiment_folder="/home/jupyter/jointly_rec_and_search/experiments/search_compl/cl-drd/" \
                                    --train_batch_size=128
done
