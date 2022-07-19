#!/bin/bash

PRETRAINED_PATH="/home/jupyter/jointly_rec_and_search/experiments/rec/cl-drd/experiment_06-24_202803/models/checkpoint_latest"
INDEX_DIR="/home/jupyter/jointly_rec_and_search/experiments/rec/cl-drd/experiment_06-24_202803/index/"
INDEX_PATH="/home/jupyter/jointly_rec_and_search/experiments/rec/cl-drd/experiment_06-24_202803/index/checkpoint_latest.index"
OUTPUT_PATH="/home/jupyter/jointly_rec_and_search/experiments/rec/cl-drd/experiment_06-24_202803/runs/checkpoint_latest"

QRELS_PATH_1="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/arels.test.tsv"
RANKING_PATH_1="/home/jupyter/jointly_rec_and_search/experiments/rec/cl-drd/experiment_06-24_202803/runs/checkpoint_latest.test.run"
QRELS_PATH_2="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/arels.test.exclude.tsv"
RANKING_PATH_2="/home/jupyter/jointly_rec_and_search/experiments/rec/cl-drd/experiment_06-24_202803/runs/checkpoint_latest.test.exclude.run"

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
--queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/anchors.test.tsv" \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.run \
--index_path=$INDEX_PATH \
--batch_size=512

# queries.test.exclude.tsv
python retriever/retrieve_top_passages.py \
--queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/anchors.test.exclude.tsv" \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.exclude.run \
--index_path=$INDEX_PATH \
--batch_size=512

# 3, evaluation
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_1 --ranking_path=$RANKING_PATH_1

echo "================================================ new evaluation ================================================"
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_2 --ranking_path=$RANKING_PATH_2