#!/bin/bash

QREL_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.tsv"
RUN_PATH="/home/jupyter/jointly_rec_and_search/experiments/search/cl-drd/experiment_06-23_232423/runs/pys_checkpoint_5000.run"

python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 $QREL_PATH $RUN_PATH

QREL_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.exclude.tsv"
RUN_PATH="/home/jupyter/jointly_rec_and_search/experiments/search/cl-drd/experiment_06-23_232423/runs/pys_checkpoint_5000.exclude.run"

python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 $QREL_PATH $RUN_PATH