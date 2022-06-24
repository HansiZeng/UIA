#!/bin/bash

RANKING_PREFIX="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/valid_runs/checkpoint_250000"
#RANKING_PREFIX="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-13_032123/merge_2_runs/top-1000.sum"
METRIC_PATH=$RANKING_PREFIX"_metric.log"

python evaluation/retrieval_evaluator.py --ranking_prefix=$RANKING_PREFIX >> $METRIC_PATH
