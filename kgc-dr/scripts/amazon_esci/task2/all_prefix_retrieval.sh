#!/bin/bash

HOME_DIR="/home/jupyter/unity_jointly_rec_and_search/datasets/amazon_esci_dataset/task_2_multiclass_product_classification"
EXPERIMENT_FORDER="/home/jupyter/unity_jointly_rec_and_search/experiments/amazon_esci/task2"

ENTITES_PATH="${HOME_DIR}/entity_prefix/prefix_all_entities.tsv"
PASSAGE_PATH="${HOME_DIR}/entity_prefix/prefix_collection_title.tsv"

DATE="experiment_09-22_100600"
PRETRAINED_PATH="${EXPERIMENT_FORDER}/${DATE}/models/checkpoint_latest"
INDEX_DIR="${EXPERIMENT_FORDER}/${DATE}/index/"
INDEX_PATH="${EXPERIMENT_FORDER}/${DATE}/index/checkpoint_latest.index"
OUTPUT_PATH="${EXPERIMENT_FORDER}/${DATE}/runs/checkpoint_latest"



# 2, retrieval
# queries.test.tsv
python retriever/retrieve_top_passages.py \
--queries_path=$ENTITES_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.all.run \
--index_path=$INDEX_PATH \
--batch_size=512 \
--query_max_len=128 \
--top_k=200

