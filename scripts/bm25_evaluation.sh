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

INDEX_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/amazon_esci_dataset/task_2_multiclass_product_classification/bm25_index/"


# for retrieval 
python -m pyserini.search --topics $QUERIES_PATH  \
        --index $INDEX_PATH \
        --output "/home/jupyter/jointly_rec_and_search/experiments/bm25/amazon_esci/usearch.test.run" \
        --threads 16 --batch-size 16 \
        --bm25
        
# for sim_rec
python -m pyserini.search --topics $SIM_ANCHORS_PATH \
        --index $INDEX_PATH \
        --output "/home/jupyter/jointly_rec_and_search/experiments/bm25/amazon_esci/usim.test.run" \
        --threads 16 --batch-size 16 \
        --bm25
        
# for retrieval
python -m pyserini.search --topics $COMPL_ANCHORS_PATH  \
        --index $INDEX_PATH \
        --output "/home/jupyter/jointly_rec_and_search/experiments/bm25/amazon_esci/ucompl.test.run" \
        --threads 16 --batch-size 16 \
        --bm25

# evaluation
echo "Search Result: "

#python -m pyserini.eval.trec_eval -c -m recall.50 -m ndcg_cut.10 $QRELS_PATH "/home/jupyter/jointly_rec_and_search/experiments/bm25/amazon_esci/usearch.test.run"
python kgc-dr/evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH --ranking_path="/home/jupyter/jointly_rec_and_search/experiments/bm25/amazon_esci/usearch.test.dr.run"

echo "Sim_Rec Result: "
python -m pyserini.eval.trec_eval -c -m recall.50 -m ndcg_cut.10 $SIM_ARELS_PATH "/home/jupyter/jointly_rec_and_search/experiments/bm25/amazon_esci/usim.test.run"

echo "Compl_Rec Result: "
python -m pyserini.eval.trec_eval -c -m recall.50 -m ndcg_cut.10 $COMPL_ARELS_PATH "/home/jupyter/jointly_rec_and_search/experiments/bm25/amazon_esci/ucompl.test.run"