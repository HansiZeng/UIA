#!/bin/bash

PRETRAINED_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-16_225219/models/checkpoint_250000/"
INDEX_DIR="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-16_225219/index/"
INDEX_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-16_225219/index/checkpoint_250000.index"
OUTPUT_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-16_225219/runs/checkpoint_250000"
RANKING_PREFIX="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-16_225219/runs/checkpoint_250000"
METRIC_PATH=$RANKING_PREFIX"_metric.log"

## step 1: index
python -m torch.distributed.launch --nproc_per_node=4 retriever/parallel_index_text_1.py \
                                    --pretrained_path=$PRETRAINED_PATH \
                                    --passages_path="/work/hzeng_umass_edu/datasets/msmarco-passage/collection.tsv" \
                                    --index_dir=$INDEX_DIR \
                                    --batch_size=128 \

python retriever/parallel_index_text_2.py --pretrained_path=$PRETRAINED_PATH \
                                --passages_path="/work/hzeng_umass_edu/datasets/msmarco-passage/collection.tsv" \
                                --index_dir=$INDEX_DIR \
                                --batch_size=128 \


## step 2: retrieval
# msmarco-dev
python retriever/retrieve_top_passages.py \
--queries_path=/work/hzeng_umass_edu/datasets/msmarco-passage/queries.dev.tsv \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.dev.run \
--index_path=$INDEX_PATH \
--batch_size=128
# trec-19
python retriever/retrieve_top_passages.py \
--queries_path=/work/hzeng_umass_edu/datasets/msmarco-passage/trec-19/msmarco-test2019-queries.tsv \
--pretrained_path=$PRETRAINED_PATH \
--index_path=$INDEX_PATH \
--output_path=${OUTPUT_PATH}.trec19.run \
--batch_size=128
# trec-20
python retriever/retrieve_top_passages.py \
--queries_path=/work/hzeng_umass_edu/datasets/msmarco-passage/trec-20/msmarco-test2020-queries.tsv \
--pretrained_path=$PRETRAINED_PATH \
--index_path=/$INDEX_PATH \
--output_path=${OUTPUT_PATH}.trec20.run \
--batch_size=128

## step 3: evaluation
python evaluation/retrieval_evaluator.py --ranking_prefix=$RANKING_PREFIX >> $METRIC_PATH