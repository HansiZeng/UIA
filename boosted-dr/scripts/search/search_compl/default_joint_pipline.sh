#!/bin/bash

PRETRAINED_PATH="/home/jupyter/jointly_rec_and_search/experiments/search/joint_cl-drd/experiment_06-30_145954/models/checkpoint_latest"
INDEX_DIR="/home/jupyter/jointly_rec_and_search/experiments/search/joint_cl-drd/experiment_06-30_145954/index/"
INDEX_PATH="/home/jupyter/jointly_rec_and_search/experiments/search/joint_cl-drd/experiment_06-30_145954/index/checkpoint_latest.index"
OUTPUT_PATH="/home/jupyter/jointly_rec_and_search/experiments/search/joint_cl-drd/experiment_06-30_145954/runs/checkpoint_latest"

QRELS_PATH_1="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.tsv"
QRELS_PATH_2="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.head.tsv"
QRELS_PATH_3="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.torso.tsv"
QRELS_PATH_4="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.tail.tsv"

RANKING_PATH_1="/home/jupyter/jointly_rec_and_search/experiments/search/joint_cl-drd/experiment_06-30_145954/runs/checkpoint_latest.test.run"
RANKING_PATH_2="/home/jupyter/jointly_rec_and_search/experiments/search/joint_cl-drd/experiment_06-30_145954/runs/checkpoint_latest.test.head.run"
RANKING_PATH_3="/home/jupyter/jointly_rec_and_search/experiments/search/joint_cl-drd/experiment_06-30_145954/runs/checkpoint_latest.test.torso.run"
RANKING_PATH_4="/home/jupyter/jointly_rec_and_search/experiments/search/joint_cl-drd/experiment_06-30_145954/runs/checkpoint_latest.test.tail.run"

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

# queries.test.head.tsv
python retriever/retrieve_top_passages.py \
--queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.head.tsv" \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.head.run \
--index_path=$INDEX_PATH \
--batch_size=512

# queries.test.torso.tsv
python retriever/retrieve_top_passages.py \
--queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.torso.tsv" \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.torso.run \
--index_path=$INDEX_PATH \
--batch_size=512

# queries.test.tail.tsv
python retriever/retrieve_top_passages.py \
--queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.tail.tsv" \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.tail.run \
--index_path=$INDEX_PATH \
--batch_size=512

# 3, evaluation
echo "================================================ standard qrel ================================================"
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_1 --ranking_path=$RANKING_PATH_1
echo "================================================ standard qrel exclude ================================================"
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_2 --ranking_path=$RANKING_PATH_2
echo "================================================ ext_qrel ================================================"
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_3 --ranking_path=$RANKING_PATH_3
echo "================================================ ext_qrel exclude ================================================"
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_4 --ranking_path=$RANKING_PATH_4
