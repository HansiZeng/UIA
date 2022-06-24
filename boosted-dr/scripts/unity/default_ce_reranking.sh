#!/bin/bash

RANKING_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/runs/checkpoint_250000.dev.json"
OUTPUT_DIR="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/reranks/"
#PRETRAINED_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-06_150750/models/checkpoint_250000"
QUERIES_PATH="/work/hzeng_umass_edu/datasets/msmarco-passage/queries.dev.tsv"
#QUERIES_PATH="/work/hzeng_umass_edu/datasets/msmarco-passage/trec-19/msmarco-test2019-queries.tsv"
SUFFIX="dev"

python -m torch.distributed.launch --nproc_per_node=4 reranker/ce_reranking.py \
                                --ranking_path=$RANKING_PATH \
                                --output_dir=$OUTPUT_DIR \
                                --queries_path=$QUERIES_PATH \
                                --suffix=$SUFFIX