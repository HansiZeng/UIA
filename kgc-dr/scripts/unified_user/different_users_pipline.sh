#!/bin/bash

HOME_PREFIX="/home/jupyter/unity_jointly_rec_and_search"
MODEL_NAME="user_seq_merge_encoder"

EXPERIMENT_FORDER="${HOME_PREFIX}/experiments/unified_user/${MODEL_NAME}"
TMP_RECORD="${EXPERIMENT_FORDER}/result.log"

DATASET_PREFIX="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/users_divided_by_group"
QRELS_PATH="${DATASET_PREFIX}/urels.search.test.tsv"
SIM_ARELS_PATH="${DATASET_PREFIX}/urels.sim.test.tsv"
COMPL_ARELS_PATH="${DATASET_PREFIX}/urels.compl.test.tsv"


PASSAGE_PATH="${HOME_PREFIX}/datasets/unified_user/collection_title_catalog.tsv"
EID_PATH="${HOME_PREFIX}/datasets/unified_user/all_entities.tsv"

echo "collection_path: ${PASSAGE_PATH}" > $TMP_RECORD
echo "queries path: ${QUERIES_PATH}" >> $TMP_RECORD

DATES=($(ls "${EXPERIMENT_FORDER}"))

for i in {1..4}
do
echo "current group: $i"

PRETRAINED_PATH="${EXPERIMENT_FORDER}/experiment_10-02_155143/models/checkpoint_latest"
INDEX_PATH="${EXPERIMENT_FORDER}/experiment_10-02_155143/index/checkpoint_latest.index"
OUTPUT_PATH="${EXPERIMENT_FORDER}/experiment_10-02_155143/diff_user_runs/checkpoint_latest_group_${i}"
QUERIES_PATH="${DATASET_PREFIX}/search_sequential_group${i}.test.json"
SIM_ANCHORS_PATH="${DATASET_PREFIX}/sim_rec_sequential_group${i}.test.json"
COMPL_ANCHORS_PATH="${DATASET_PREFIX}/compl_rec_sequential_group${i}.test.json"

# 2, retrieval
# queries.test.tsv
python retriever/user_retrieve_top_passages.py \
--eid_path=$EID_PATH \
--query_examples_path=$QUERIES_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.search.small.test.run \
--index_path=$INDEX_PATH \
--batch_size=128 \
--max_length=128

python retriever/user_retrieve_top_passages.py \
--eid_path=$EID_PATH \
--query_examples_path=$SIM_ANCHORS_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.sim.small.test.run \
--index_path=$INDEX_PATH \
--batch_size=128 \
--max_length=128

python retriever/user_retrieve_top_passages.py \
--eid_path=$EID_PATH \
--query_examples_path=$COMPL_ANCHORS_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.compl.small.test.run \
--index_path=$INDEX_PATH \
--batch_size=128 \
--max_length=128


# 3, evaluation
echo "================================================ search ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH --ranking_path=${OUTPUT_PATH}.search.small.test.run >> $TMP_RECORD

echo "================================================ sim_rec ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$SIM_ARELS_PATH --ranking_path=${OUTPUT_PATH}.sim.small.test.run >> $TMP_RECORD

echo "================================================ compl_rec ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$COMPL_ARELS_PATH --ranking_path=${OUTPUT_PATH}.compl.small.test.run >> $TMP_RECORD

echo " " >> $TMP_RECORD
done
