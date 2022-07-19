#!/bin/bash

PRETRAINED_PATH="/home/jupyter/jointly_rec_and_search/pretrained_models/boosted_dr_model/"
INDEX_DIR="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/cl-drd_index/"

python -m torch.distributed.launch --nproc_per_node=4 retriever/parallel_index_text_1.py \
                                    --pretrained_path=$PRETRAINED_PATH \
                                    --passages_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/collection.tsv" \
                                    --index_dir=$INDEX_DIR \
                                    --batch_size=256 \

python retriever/parallel_index_text_2.py --pretrained_path=$PRETRAINED_PATH \
                                --passages_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/collection.tsv" \
                                --index_dir=$INDEX_DIR \
                                --batch_size=256 \

