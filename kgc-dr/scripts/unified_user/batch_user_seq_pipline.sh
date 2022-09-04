#!/bin/bash

HOME_PREFIX="/work/hzeng_umass_edu/ir-research/joint_modeling_search_and_rec"

EXPERIMENT_FORDER="${HOME_PREFIX}/experiments/unified_user"
TMP_RECORD="${EXPERIMENT_FORDER}/unified_user_result.log"

DATASET_PREFIX="${HOME_PREFIX}/datasets/unified_kgc/unified_user/sequential_train_test/"
QRELS_PATH="${DATASET_PREFIX}/urels.search.test.tsv"
QUERIES_PATH="${DATASET_PREFIX}/hlen_4_randneg/search_sequential.small.test.json"

PASSAGE_PATH="${HOME_PREFIX}/datasets/unified_kgc/collection_title_catalog.tsv"
EID_PATH="${HOME_PREFIX}/datasets/unified_kgc/all_entities.tsv"

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

# 3, evaluation
echo "================================================ search ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH --ranking_path=${OUTPUT_PATH}.search.small.test.run >> $TMP_RECORD

echo " "
fi
done
