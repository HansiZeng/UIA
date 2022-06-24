#!/bin/bash

PRETRAINED_PATH="/home/jupyter/jointly_rec_and_search/experiments/search/cl-drd/experiment_06-23_232423/models/checkpoint_5000"
INDEX_DIR="/home/jupyter/jointly_rec_and_search/experiments/search/cl-drd/experiment_06-23_232423/index/"
INDEX_PATH="/home/jupyter/jointly_rec_and_search/experiments/search/cl-drd/experiment_06-23_232423/index/checkpoint_5000.index"
OUTPUT_PATH="/home/jupyter/jointly_rec_and_search/experiments/search/cl-drd/experiment_06-23_232423/runs/checkpoint_5000"

# 1, index
python -m torch.distributed.launch --nproc_per_node=4 retriever/parallel_index_text_1.py \
                                    --pretrained_path=$PRETRAINED_PATH \
                                    --passages_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/collection.tsv" \
                                    --index_dir=$INDEX_DIR \
                                    --batch_size=256 \

python retriever/parallel_index_text_2.py --pretrained_path=$PRETRAINED_PATH \
                                --passages_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/collection.tsv" \
                                --index_dir=$INDEX_DIR \
                                --batch_size=256 \


# 2, retrieval
# queries.test.tsv
python retriever/retrieve_top_passages.py \
--queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.tsv" \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.run \
--index_path=$INDEX_PATH \
--batch_size=512

# queries.test.exclude.tsv
python retriever/retrieve_top_passages.py \
--queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.exclude.tsv" \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.exclude.run \
--index_path=$INDEX_PATH \
--batch_size=512
