#!/bin/bash

EXPERIMENT_FORDER="/home/jupyter/jointly_rec_and_search/experiments/kgc"
TMP_RECORD="${EXPERIMENT_FORDER}/temp_record.log"

ARELS_PATH="/home/jupyter/jointly_rec_and_search/datasets/kgc/arels.compl.test.tsv"
ANCHORS_PATH="/home/jupyter/jointly_rec_and_search/datasets/kgc/test/anchors.test.tsv"
PASSAGE_PATH="/home/jupyter/jointly_rec_and_search/datasets/kgc/collection_title_catalog.tsv"

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

RANKING_PATH="${EXPERIMENT_FORDER}/${DATE}/runs/checkpoint_latest.test.run"


# 1, index
python -m torch.distributed.launch --nproc_per_node=2 retriever/parallel_index_text_1.py \
                                    --pretrained_path=$PRETRAINED_PATH \
                                    --passages_path=$PASSAGE_PATH \
                                    --index_dir=$INDEX_DIR \
                                    --batch_size=512 \
                                    --max_length=256

python retriever/parallel_index_text_2.py --pretrained_path=$PRETRAINED_PATH \
                                --passages_path=$PASSAGE_PATH \
                                --index_dir=$INDEX_DIR \
                                --batch_size=512 \


# 2, retrieval
# queries.test.tsv
python retriever/retrieve_top_passages.py \
--queries_path=$ANCHORS_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.run \
--index_path=$INDEX_PATH \
--batch_size=512 \
--query_max_len=256

# 3, evaluation
echo "================================================ standard qrel ================================================" >> $TMP_RECORD
python evaluation/retrieval_evaluator.py --qrels_path=$ARELS_PATH --ranking_path=$RANKING_PATH >> $TMP_RECORD

fi
done
