#!/bin/bash


PRETRAINED_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/models/checkpoint_250000/"
INDEX_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/index/checkpoint_250000.index"
OUTPUT_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/valid_runs/checkpoint_250000"

# msmarco-train
#python retriever/retrieve_top_passages.py \
#--queries_path=/work/hzeng_umass_edu/datasets/msmarco-passage/qrel-passages.train.tsv \
#--pretrained_path=$PRETRAINED_PATH \
#--output_path=${OUTPUT_PATH}.train.run \
#--index_path=$INDEX_PATH \
#--batch_size=64 \
#--top_k=200 \
#--query_max_len=256


# msmarco-dev
python retriever/retrieve_top_passages.py \
--queries_path=/work/hzeng_umass_edu/datasets/msmarco-passage/queries.dev.small.tsv \
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

