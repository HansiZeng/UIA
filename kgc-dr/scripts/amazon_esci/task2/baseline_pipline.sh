#!/bin/bash

HOME_DIR="/home/jupyter/unity_jointly_rec_and_search/datasets/amazon_esci_dataset/task_2_multiclass_product_classification"

SIM_ARELS_PATH="${HOME_DIR}/unified_test/arels.test.sim.tsv"
SIM_ANCHORS_PATH="${HOME_DIR}/unified_test/anchors.test.sim.small.tsv"
COMPL_ARELS_PATH="${HOME_DIR}/unified_test/arels.test.compl.tsv"
COMPL_ANCHORS_PATH="${HOME_DIR}/unified_test/anchors.test.compl.tsv"
QRELS_PATH="${HOME_DIR}/unified_test/qrels.test.tsv"
QUERIES_PATH="${HOME_DIR}/unified_test/queries.test.tsv"

PASSAGE_PATH="${HOME_DIR}/collection_title.tsv"
EXPERIMENT_FORDER="/home/jupyter/unity_jointly_rec_and_search/experiments/tasb/amazon_esci/"
OUTPUT_PATH="${EXPERIMENT_FORDER}/runs/checkpoint_latest"
INDEX_DIR="/home/jupyter/unity_jointly_rec_and_search/experiments/tasb/amazon_esci"
INDEX_PATH="${INDEX_DIR}/tasb.index"



# 2, retrieval
# queries.test.tsv
python baseline_retriever/retrieve_top_passages.py \
--queries_path=$SIM_ANCHORS_PATH \
--model_name_or_path="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco" \
--output_path=${OUTPUT_PATH}.test.sim.small.run \
--index_path=$INDEX_PATH \
--batch_size=512 \
--query_max_len=128

python baseline_retriever/retrieve_top_passages.py \
--queries_path=$COMPL_ANCHORS_PATH \
--model_name_or_path="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco" \
--output_path=${OUTPUT_PATH}.test.compl.run \
--index_path=$INDEX_PATH \
--batch_size=512 \
--query_max_len=128

python baseline_retriever/retrieve_top_passages.py \
--queries_path=$QUERIES_PATH \
--model_name_or_path="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco" \
--output_path=${OUTPUT_PATH}.test.query.small.run \
--index_path=$INDEX_PATH \
--batch_size=512 \
--query_max_len=128

# 3, evaluation
echo "================================================ similar rec ================================================" 
python evaluation/retrieval_evaluator.py --qrels_path=$SIM_ARELS_PATH --ranking_path=${OUTPUT_PATH}.test.sim.small.run 

echo "================================================ complementary rec ================================================" 
python evaluation/retrieval_evaluator.py --qrels_path=$COMPL_ARELS_PATH --ranking_path=${OUTPUT_PATH}.test.compl.run 

echo "================================================ search ================================================" 
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH --ranking_path=${OUTPUT_PATH}.test.query.small.run

echo " " >>  $TMP_RECORD
fi
done
