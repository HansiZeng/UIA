#!/bin/bash


PRETRAINED_PATH="/home/jupyter/jointly_rec_and_search/pretrained_models/boosted_dr_model/"
INDEX_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/cl-drd_index/boosted_dr_model.index"
OUTPUT_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/boosted_dr_model"

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


