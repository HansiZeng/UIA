#!/bin/bash

exp_paths=(experiment_06-27_024103 experiment_06-27_025239 experiment_06-27_031532 experiment_06-27_032658)
metric_path="/home/jupyter/jointly_rec_and_search/experiments/joint/cl-drd/metric.txt"

for exp_path in "${exp_paths[@]}"
do
PRETRAINED_PATH="/home/jupyter/jointly_rec_and_search/experiments/joint/cl-drd/${exp_path}/models/checkpoint_latest"
INDEX_DIR="/home/jupyter/jointly_rec_and_search/experiments/joint/cl-drd/${exp_path}/index/"
INDEX_PATH="/home/jupyter/jointly_rec_and_search/experiments/joint/cl-drd/${exp_path}/index/checkpoint_latest.index"
OUTPUT_PATH="/home/jupyter/jointly_rec_and_search/experiments/joint/cl-drd/${exp_path}/runs/checkpoint_latest"

QRELS_PATH_1="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.tsv"
RANKING_PATH_1="/home/jupyter/jointly_rec_and_search/experiments/joint/cl-drd/${exp_path}/runs/checkpoint_latest.test.run"
QRELS_PATH_2="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.exclude.tsv"
RANKING_PATH_2="/home/jupyter/jointly_rec_and_search/experiments/joint/cl-drd/${exp_path}/runs/checkpoint_latest.test.exclude.run"

QRELS_PATH_3="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/ext_qrels.test.tsv"
RANKING_PATH_3="/home/jupyter/jointly_rec_and_search/experiments/joint/cl-drd/${exp_path}/runs/checkpoint_latest.test.ext.run"
QRELS_PATH_4="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/ext_qrels.test.exclude.tsv"
RANKING_PATH_4="/home/jupyter/jointly_rec_and_search/experiments/joint/cl-drd/${exp_path}/runs/checkpoint_latest.test.exclude.ext.run"

# 1, index
python -m torch.distributed.launch --nproc_per_node=4 retriever/parallel_index_text_1.py \
                                    --pretrained_path=$PRETRAINED_PATH \
                                    --passages_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/collection.tsv" \
                                    --index_dir=$INDEX_DIR \
                                    --batch_size=256 \

python retriever/parallel_index_text_2.py --pretrained_path=$PRETRAINED_PATH \
                                --passages_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/collection.tsv" \
                                --index_dir=$INDEX_DIR \
                                --batch_size=256 \


# 2, retrieval
# queries.test.tsv
python retriever/retrieve_top_passages.py \
--queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.tsv" \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.run \
--index_path=$INDEX_PATH \
--batch_size=512

# queries.test.exclude.tsv
python retriever/retrieve_top_passages.py \
--queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.exclude.tsv" \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.exclude.run \
--index_path=$INDEX_PATH \
--batch_size=512

python retriever/retrieve_top_passages.py \
--queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.tsv" \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.ext.run \
--index_path=$INDEX_PATH \
--batch_size=512

# queries.test.exclude.tsv
python retriever/retrieve_top_passages.py \
--queries_path="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.exclude.tsv" \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.test.exclude.ext.run \
--index_path=$INDEX_PATH \
--batch_size=512

# 3, evaluation
echo "evaluation metrics on ${exp_path}" >> $metric_path
echo "================================================ standard qrel ================================================" >> $metric_path
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_1 --ranking_path=$RANKING_PATH_1 >> $metric_path
echo "================================================ standard qrel exclude ================================================" >> $metric_path
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_2 --ranking_path=$RANKING_PATH_2 >> $metric_path
echo "================================================ ext_qrel ================================================" >> $metric_path
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_3 --ranking_path=$RANKING_PATH_3 >> $metric_path
echo "================================================ ext_qrel exclude ================================================" >> $metric_path
python evaluation/retrieval_evaluator.py --qrels_path=$QRELS_PATH_4 --ranking_path=$RANKING_PATH_4 >> $metric_path
echo "================================================                ================================================" >> $metric_path
echo "================================================                ================================================" >> $metric_path
done