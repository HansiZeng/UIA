#!/bin/bash

QRELS_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/arels.test.tsv"
RANKING_PATH="/home/jupyter/jointly_rec_and_search/experiments/rec/cl-drd/experiment_06-24_202803/runs/checkpoint_latest.test.run"
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH --ranking_path=$RANKING_PATH

echo "================================================ new evaluation ================================================"

QRELS_PATH="/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/arels.test.exclude.tsv"
RANKING_PATH="/home/jupyter/jointly_rec_and_search/experiments/rec/cl-drd/experiment_06-24_202803/runs/checkpoint_latest.test.exclude.run"
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH --ranking_path=$RANKING_PATH