#!/bin/bash

HOME_DIR="/home/jupyter/unity_jointly_rec_and_search/datasets/amazon_esci_dataset/task_2_multiclass_product_classification"
EXPERIMENT_FORDER="/home/jupyter/unity_jointly_rec_and_search/experiments/amazon_esci/task2"
TMP_RECORD="${EXPERIMENT_FORDER}/results.log"

SIM_ARELS_PATH="${HOME_DIR}/unified_test/arels.test.sim.tsv"
SIM_ANCHORS_PATH="${HOME_DIR}/unified_test/anchors.test.sim.small.tsv"
COMPL_ARELS_PATH="${HOME_DIR}/unified_test/arels.test.compl.tsv"
COMPL_ANCHORS_PATH="${HOME_DIR}/unified_test/anchors.test.compl.tsv"
QRELS_PATH="${HOME_DIR}/unified_test/qrels.test.tsv"
QUERIES_PATH="${HOME_DIR}/unified_test/queries.test.tsv"

PASSAGE_PATH="${HOME_DIR}/collection_title.tsv"

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
                                    --max_length=256 \
                                    --tokenizer_name_or_path="sentence-transformers/msmarco-bert-base-dot-v5" 

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
--query_max_len=128 \
--tokenizer_name_or_path="sentence-transformers/msmarco-bert-base-dot-v5" 

python retriever/retrieve_top_passages.py \
--queries_path=$COMPL_ANCHORS_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.compl.run \
--index_path=$INDEX_PATH \
--batch_size=512 \
--query_max_len=128 \
--tokenizer_name_or_path="sentence-transformers/msmarco-bert-base-dot-v5"

python retriever/retrieve_top_passages.py \
--queries_path=$QUERIES_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.query.small.run \
--index_path=$INDEX_PATH \
--batch_size=512 \
--query_max_len=128 \
--tokenizer_name_or_path="sentence-transformers/msmarco-bert-base-dot-v5"

# 3, evaluation
echo "================================================ similar rec ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$SIM_ARELS_PATH --ranking_path=${OUTPUT_PATH}.test.sim.small.run >> $TMP_RECORD

echo "================================================ complementary rec ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$COMPL_ARELS_PATH --ranking_path=${OUTPUT_PATH}.test.compl.run >> $TMP_RECORD

echo "================================================ search ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH --ranking_path=${OUTPUT_PATH}.test.query.small.run >> $TMP_RECORD

echo " " >>  $TMP_RECORD
fi
done
