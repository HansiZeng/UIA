#!/bin/bash 

python -m torch.distributed.launch --nproc_per_node=4 trainer/train.py \
                                    --num_train_epochs=4 \
                                    --queries_path=/work/hzeng_umass_edu/datasets/msmarco-passage/queries.train.tsv \
                                    --collection_path=/work/hzeng_umass_edu/datasets/msmarco-passage/collection.tsv \
                                    --training_path=/work/hzeng_umass_edu/datasets/msmarco-passage/corase_to_fine_grained/10relT_20neg.train.json \
                                    --experiment_folder=/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/ \
                                    --label_mode=9 