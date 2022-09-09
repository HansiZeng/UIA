#!/bin/bash

EXPERIMENT_FORDER="/home/jupyter/unity_jointly_rec_and_search/experiments/unified_kgc"
TMP_RECORD="${EXPERIMENT_FORDER}/static_user_result.log"

SIM_ARELS_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/sequential_train_test/urels.sim.test.tsv"
SIM_ANCHORS_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/sequential_train_test/without_context/uid_anchors.test.sim.tsv"
COMPL_ARELS_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/sequential_train_test/urels.compl.test.tsv"
COMPL_ANCHORS_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/sequential_train_test/without_context/uid_anchors.test.compl.tsv"
QRELS_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/sequential_train_test/urels.search.test.tsv"
QUERIES_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/sequential_train_test/without_context/uid_queries.test.search.tsv"

PASSAGE_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/collection_title_catalog.tsv"

echo "arels_path: ${ARELS_PATH}" > $TMP_RECORD
echo "anchors_path: ${ANCHORS_PATH}" >> $TMP_RECORD
echo "collection_path: ${PASSAGE_PATH}" >> $TMP_RECORD

DATES=($(ls "${EXPERIMENT_FORDER}"))

for DATE in "${DATES[@]}"
do
if [ -d "${EXPERIMENT_FORDER}/${DATE}" ]; then
echo "current experiment folder: ${EXPERIMENT_FORDER}/${DATE}"

PRETRAINED_PATH="${EXPERIMENT_FORDER}/${DATE}/models/checkpoint_latest"
INDEX_DIR="${EXPERIMENT_FORDER}/${DATE}/index/"
INDEX_PATH="${EXPERIMENT_FORDER}/${DATE}/index/checkpoint_latest.index"
OUTPUT_PATH="${EXPERIMENT_FORDER}/${DATE}/static_user_runs/checkpoint_latest"


# 1, not need to index. since we assume the batch_pipline.sh has been runned before.


# 2, retrieval
# queries.test.tsv
python retriever/retrieve_top_passages.py \
--queries_path=$SIM_ANCHORS_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.sim.small.run \
--index_path=$INDEX_PATH \
--batch_size=512 \
--query_max_len=256

python retriever/retrieve_top_passages.py \
--queries_path=$COMPL_ANCHORS_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.compl.run \
--index_path=$INDEX_PATH \
--batch_size=512 \
--query_max_len=256

python retriever/retrieve_top_passages.py \
--queries_path=$QUERIES_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.query.small.run \
--index_path=$INDEX_PATH \
--batch_size=512 \
--query_max_len=256

# 3, evaluation
echo "================================================ similar rec ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$SIM_ARELS_PATH --ranking_path=${OUTPUT_PATH}.test.sim.small.run >> $TMP_RECORD

echo "================================================ complementary rec ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$COMPL_ARELS_PATH --ranking_path=${OUTPUT_PATH}.test.compl.run >> $TMP_RECORD

echo "================================================ search ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH --ranking_path=${OUTPUT_PATH}.test.query.small.run >> $TMP_RECORD

echo " " >> $TMP_RECORD
fi
done