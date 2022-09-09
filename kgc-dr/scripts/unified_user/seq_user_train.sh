#!/bin/bash

home_prefix="/home/jupyter/unity_jointly_rec_and_search"
#examples_path="${home_prefix}/datasets/unified_user/sequential_train_test/hlen_4_randneg/search_sequential.train.json"
examples_path="/home/jupyter/unity_data/hlen_4_randneg/search_sequential.train.json"
eid_path="${home_prefix}/datasets/unified_user/all_entities.tsv"
experiment_folder="${home_prefix}/experiments/unified_user/user_seq_encoder/"

python -m torch.distributed.launch --nproc_per_node=4 trainer/user_seq_train.py \
                                    $bool_arg1 $bool_arg2 $bool_arg3 \
                                    --examples_path=$examples_path \
                                    --eid_path=$eid_path \
                                    --experiment_folder=$experiment_folder \
                                    --train_batch_size=384 \
                                    --apply_value_layer 
    