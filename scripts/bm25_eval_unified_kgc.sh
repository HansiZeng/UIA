#!/bin/bash



INDEX_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/bm25_index"

SIM_ARELS_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_kgc/unified_test/arels.test.sim.tsv"
SIM_ANCHORS_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_kgc/unified_test/anchors.test.sim.tsv"
COMPL_ARELS_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_kgc/unified_test/arels.test.compl.tsv"
COMPL_ANCHORS_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_kgc/unified_test/anchors.test.compl.tsv"
QRELS_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_kgc/unified_test/qrels.test.tsv"
QUERIES_PATH="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_kgc/unified_test/queries.test.tsv"

# for retrieval 
python -m pyserini.search --topics $QUERIES_PATH  \
        --index $INDEX_PATH \
        --output "/home/jupyter/jointly_rec_and_search/experiments/bm25/lowes/search.test.run" \
        --threads 16 --batch-size 16 \
        --bm25

# for sim_rec
python -m pyserini.search --topics $SIM_ANCHORS_PATH \
        --index $INDEX_PATH \
        --output "/home/jupyter/jointly_rec_and_search/experiments/bm25/lowes/sim.test.run" \
        --threads 16 --batch-size 16 \
        --bm25

# for retrieval
python -m pyserini.search --topics $COMPL_ANCHORS_PATH  \
        --index $INDEX_PATH \
        --output "/home/jupyter/jointly_rec_and_search/experiments/bm25/lowes/compl.test.run" \
        --threads 16 --batch-size 16 \
        --bm25


# evaluation
echo "Search Result: "

python -m pyserini.eval.trec_eval -c -m recall.50 -m ndcg_cut.10 $QRELS_PATH "/home/jupyter/jointly_rec_and_search/experiments/bm25/lowes/search.test.run"
#python kgc-dr/evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH --ranking_path="/home/jupyter/jointly_rec_and_search/experiments/bm25/lowes/search.test.run"

echo "Sim_Rec Result: "
python -m pyserini.eval.trec_eval -c -m recall.50 -m ndcg_cut.10 $SIM_ARELS_PATH "/home/jupyter/jointly_rec_and_search/experiments/bm25/lowes/sim.test.run"

echo "Compl_Rec Result: "
python -m pyserini.eval.trec_eval -c -m recall.50 -m ndcg_cut.10 $COMPL_ARELS_PATH "/home/jupyter/jointly_rec_and_search/experiments/bm25/lowes/compl.test.run"
