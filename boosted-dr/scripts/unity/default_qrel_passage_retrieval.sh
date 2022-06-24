#!/bin/bash


PRETRAINED_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/models/checkpoint_250000/"
INDEX_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/index/checkpoint_250000.index"
OUTPUT_PATH="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/qrel_passage_runs/checkpoint_250000_p2p"

# trec-19
python retriever/retrieve_top_passages.py \
--queries_path=/work/hzeng_umass_edu/datasets/msmarco-passage/qrel-passages/qrel-passages.trec19.tsv \
--pretrained_path=$PRETRAINED_PATH \
--index_path=$INDEX_PATH \
--output_path=${OUTPUT_PATH}.trec19.run \
--batch_size=64 \
--query_max_len=256
# trec-20
python retriever/retrieve_top_passages.py \
--queries_path=/work/hzeng_umass_edu/datasets/msmarco-passage/qrel-passages/qrel-passages.trec20.tsv \
--pretrained_path=$PRETRAINED_PATH \
--index_path=/$INDEX_PATH \
--output_path=${OUTPUT_PATH}.trec20.run \
--batch_size=64 \
--query_max_len=256