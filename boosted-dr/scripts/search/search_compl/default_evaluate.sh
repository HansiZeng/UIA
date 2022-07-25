#!/bin/bash

QRELS_PATH_1="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/qrels.test.tsv"
QRELS_PATH_2="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/qrels.test.head.tsv"
QRELS_PATH_3="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/qrels.test.torso.tsv"
QRELS_PATH_4="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/qrels.test.tail.tsv"

RANKING_PATH_1="/home/jupyter/jointly_rec_and_search/experiments/search_compl/cl-drd/experiment_07-04_193509/runs/checkpoint_latest.test.run"
RANKING_PATH_2="/home/jupyter/jointly_rec_and_search/experiments/search_compl/cl-drd/experiment_07-04_193509/runs/checkpoint_latest.test.head.run"
RANKING_PATH_3="/home/jupyter/jointly_rec_and_search/experiments/search_compl/cl-drd/experiment_07-04_193509/runs/checkpoint_latest.test.torso.run"
RANKING_PATH_4="/home/jupyter/jointly_rec_and_search/experiments/search_compl/cl-drd/experiment_07-04_193509/runs/checkpoint_latest.test.tail.run"


# 3, evaluation
echo "================================================ standard qrel ================================================"
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_1 --ranking_path=$RANKING_PATH_1
echo "================================================ standard qrel exclude ================================================"
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_2 --ranking_path=$RANKING_PATH_2
echo "================================================ ext_qrel ================================================"
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_3 --ranking_path=$RANKING_PATH_3
echo "================================================ ext_qrel exclude ================================================"
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_4 --ranking_path=$RANKING_PATH_4
