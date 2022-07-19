#!/bin/bash

EXAMPLE_PATH="/work/hzeng_umass_edu/datasets/msmarco-passage/corase_to_fine_grained/10relT_20neg.train.json"
PRETRAINED_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-06_214728/models/checkpoint_250000/"
OUTPUT_DIR="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-06_214728/boosting_train/"
BOOSTING_ROUND=2

python -m torch.distributed.launch --nproc_per_node=4 reranker/de_reranking.py \
                                --example_path=$EXAMPLE_PATH \
                                --output_dir=$OUTPUT_DIR \
                                --boosting_round=$BOOSTING_ROUND \
                                --pretrained_path=$PRETRAINED_PATH
