#!/bin/bash

RANKING_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/runs/checkpoint_250000.train.json"
OUTPUT_DIR="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/runs/"

python -m torch.distributed.launch --nproc_per_node=4 reranker/reranking.py \
                                --ranking_path=$RANKING_PATH \
                                --output_dir=$OUTPUT_DIR
