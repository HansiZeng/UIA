#!/bin/bash

PRETRAINED_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_144149/models/checkpoint_250000/"
INDEX_DIR="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_144149/index/"

python -m torch.distributed.launch --nproc_per_node=4 retriever/parallel_index_text_1.py \
                                    --pretrained_path=$PRETRAINED_PATH \
                                    --passages_path="/work/hzeng_umass_edu/datasets/msmarco-passage/collection.tsv" \
                                    --index_dir=$INDEX_DIR \
                                    --batch_size=128 \

python retriever/parallel_index_text_2.py --pretrained_path=$PRETRAINED_PATH \
                                --passages_path="/work/hzeng_umass_edu/datasets/msmarco-passage/collection.tsv" \
                                --index_dir=$INDEX_DIR \
                                --batch_size=128 \

