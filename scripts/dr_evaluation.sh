#!/bin/bash

QREL_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/ext_qrels.test.tsv"
RUN_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/pys_boosted_dr_model.run"

python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 $QREL_PATH $RUN_PATH

QREL_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/ext_qrels.test.exclude.tsv" 
RUN_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/pys_boosted_dr_model.exclude.run"

python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 $QREL_PATH $RUN_PATH