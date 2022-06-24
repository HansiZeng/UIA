#!/bin/bash

QREL_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/arels.test.tsv"
RUN_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/runs/pys_boosted_dr_model.test.run"

python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 $QREL_PATH $RUN_PATH

QREL_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/arels.test.exclude.tsv"
RUN_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/runs/pys_boosted_dr_model.test.exclude.run"

python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 $QREL_PATH $RUN_PATH
