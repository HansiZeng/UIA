#!/bin/bash 

python -m torch.distributed.launch --nproc_per_node=4 trainer/boosted_train.py \
                                    --num_train_epochs=4 \
                                    --queries_path=/work/hzeng_umass_edu/datasets/msmarco-passage/queries.train.tsv \
                                    --collection_path=/work/hzeng_umass_edu/datasets/msmarco-passage/collection.tsv \
                                    --training_path=/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-09_171925/boosting_train/checkpoint_250000_10_20_123.train.json \
                                    --experiment_folder=/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/ \
                                    --label_mode=9 \
                                    --boosted_k=2 \
                                    --boosted_b=0.5