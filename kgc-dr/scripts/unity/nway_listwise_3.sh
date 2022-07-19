#!/bin/bash 
cd ../..

 python -m torch.distributed.launch --nproc_per_node=4 trainer/ctof_grained/nway_listwise_2.py \
                                    --num_train_epochs=2 \
                                    --learning_rate=3e-6 \
                                    --queries_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/queries.train.tsv \
                                    --collection_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/collection.tsv \
                                    --training_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/corase_to_fine_grained/30relT.train.json \
                                    --dev_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/runs/dev/tas256-dev-top200.dev.1000.tsv \
                                    --dev_queries_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/queries.dev.tsv \
                                    --dev_qrels_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/qrels.dev.small.tsv \
                                    --experiment_folder=/home/hzeng_umass_edu/scratch/my-msmarco-passage/experiments/multistep-curriculum \
                                    --model_checkpoint=/gypsum/scratch1/hzeng/my-msmarco-passage/experiments/multistep-curriculum/experiment_01-22_235703/models/checkpoint_250000.pth.tar \
                                    --label_mode=6