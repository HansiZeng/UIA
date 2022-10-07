#!/bin/bash

HOME_PREFIX="/home/jupyter/unity_jointly_rec_and_search"
MODEL_NAME="user_seq_merge_encoder"

EXPERIMENT_FORDER="${HOME_PREFIX}/experiments/unified_user/${MODEL_NAME}"
TMP_RECORD="${EXPERIMENT_FORDER}/result.log"

DATASET_PREFIX="${HOME_PREFIX}/datasets/unified_user/mixture_sequential_train_test/"
QRELS_PATH="${DATASET_PREFIX}/urels.search.test.tsv"
QUERIES_PATH="${DATASET_PREFIX}/hlen_4_bm25/search_sequential.test.json"
SIM_ARELS_PATH="${DATASET_PREFIX}/urels.sim.test.tsv"
SIM_ANCHORS_PATH="${DATASET_PREFIX}/hlen_4_bm25/sim_rec_sequential.test.json"
COMPL_ARELS_PATH="${DATASET_PREFIX}/urels.compl.test.tsv"
COMPL_ANCHORS_PATH="${DATASET_PREFIX}/hlen_4_bm25/compl_rec_sequential.test.json"


PASSAGE_PATH="${HOME_PREFIX}/datasets/unified_user/collection_title_catalog.tsv"
EID_PATH="${HOME_PREFIX}/datasets/unified_user/all_entities.tsv"

echo "collection_path: ${PASSAGE_PATH}" > $TMP_RECORD
echo "queries path: ${QUERIES_PATH}" >> $TMP_RECORD

DATES=($(ls "${EXPERIMENT_FORDER}"))

for DATE in "${DATES[@]}"
do
if [ -d "${EXPERIMENT_FORDER}/${DATE}" ]; then
echo "current experiment folder: ${EXPERIMENT_FORDER}/${DATE}"

PRETRAINED_PATH="${EXPERIMENT_FORDER}/${DATE}/models/checkpoint_latest"
INDEX_DIR="${EXPERIMENT_FORDER}/${DATE}/index/"
INDEX_PATH="${EXPERIMENT_FORDER}/${DATE}/index/checkpoint_latest.index"
OUTPUT_PATH="${EXPERIMENT_FORDER}/${DATE}/runs/checkpoint_latest"

# 1, index
python -m torch.distributed.launch --nproc_per_node=4 retriever/user_index_1.py \
                                    --pretrained_path=$PRETRAINED_PATH \
                                    --passages_path=$PASSAGE_PATH \
                                    --index_dir=$INDEX_DIR \
                                    --batch_size=128 \
                                    --max_length=128

python retriever/user_index_2.py --pretrained_path=$PRETRAINED_PATH \
                                --passages_path=$PASSAGE_PATH \
                                --index_dir=$INDEX_DIR \
                                --batch_size=128 \


# 2, retrieval
# queries.test.tsv
python retriever/user_retrieve_top_passages.py \
--eid_path=$EID_PATH \
--query_examples_path=$QUERIES_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.search.small.test.run \
--index_path=$INDEX_PATH \
--batch_size=32 \
--max_length=128

python retriever/user_retrieve_top_passages.py \
--eid_path=$EID_PATH \
--query_examples_path=$SIM_ANCHORS_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.sim.small.test.run \
--index_path=$INDEX_PATH \
--batch_size=32 \
--max_length=128

python retriever/user_retrieve_top_passages.py \
--eid_path=$EID_PATH \
--query_examples_path=$COMPL_ANCHORS_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.compl.small.test.run \
--index_path=$INDEX_PATH \
--batch_size=32 \
--max_length=128

# 3, evaluation
echo "================================================ search ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH --ranking_path=${OUTPUT_PATH}.search.small.test.run >> $TMP_RECORD

echo "================================================ sim_rec ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$SIM_ARELS_PATH --ranking_path=${OUTPUT_PATH}.sim.small.test.run >> $TMP_RECORD

echo "================================================ compl_rec ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$COMPL_ARELS_PATH --ranking_path=${OUTPUT_PATH}.compl.small.test.run >> $TMP_RECORD

echo " " >> $TMP_RECORD
fi
done
