#!/bin/bash 

python -m torch.distributed.launch --nproc_per_node=4 trainer/train_ce.py \
                                    --num_train_epochs=4 \
                                    --queries_path=/work/hzeng_umass_edu/datasets/msmarco-passage/queries.train.tsv \
                                    --collection_path=/work/hzeng_umass_edu/datasets/msmarco-passage/collection.tsv \
                                    --training_path=/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/train_examples/10relT_20neg_ce0_de250000.train.json \
                                    --experiment_folder=/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/ \
                                    --label_mode=9 