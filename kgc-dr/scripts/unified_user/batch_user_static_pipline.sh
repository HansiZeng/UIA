#!/bin/bash

HOME_PREFIX="/work/hzeng_umass_edu/ir-research/joint_modeling_search_and_rec"

EXPERIMENT_FORDER="${HOME_PREFIX}/experiments/unified_kgc"
TMP_RECORD="${EXPERIMENT_FORDER}/user_static_result.log"

DATASET_PREFIX="${HOME_PREFIX}/datasets/unified_kgc/unified_user/sequential_train_test"
SIM_ARELS_PATH="${DATASET_PREFIX}/urels.sim.test.tsv"
SIM_ANCHORS_PATH="${DATASET_PREFIX}/without_context/uid_anchors.test.sim.small.tsv"
COMPL_ARELS_PATH="${DATASET_PREFIX}/urels.compl.test.tsv"
COMPL_ANCHORS_PATH="${DATASET_PREFIX}/without_context/uid_anchors.test.compl.tsv"
QRELS_PATH="${DATASET_PREFIX}/urels.search.test.tsv"
QUERIES_PATH="${DATASET_PREFIX}/without_context/uid_queries.test.search.small.tsv"

PASSAGE_PATH="${HOME_PREFIX}/datasets/unified_kgc/collection_title_catalog.tsv"

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
OUTPUT_PATH="${EXPERIMENT_FORDER}/${DATE}/runs/checkpoint_latest"


# 1, index
python -m torch.distributed.launch --nproc_per_node=4 retriever/parallel_index_text_1.py \
                                    --pretrained_path=$PRETRAINED_PATH \
                                    --passages_path=$PASSAGE_PATH \
                                    --index_dir=$INDEX_DIR \
                                    --batch_size=128 \
                                    --max_length=256

python retriever/parallel_index_text_2.py --pretrained_path=$PRETRAINED_PATH \
                                --passages_path=$PASSAGE_PATH \
                                --index_dir=$INDEX_DIR \
                                --batch_size=512 \


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

echo " "
fi
done
