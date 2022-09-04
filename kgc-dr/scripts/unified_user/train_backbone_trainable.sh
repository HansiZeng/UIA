#!/bin/bash

bool_value_from_gru=("--value_from_gru" "--no_value_from_gru")
bool_apply_value_layer=("--apply_value_layer" "--no_apply_value_layer")
bool_apply_zero_attention=("--apply_zero_attention" "--no_apply_zero_attention")

experiment_folder="/work/hzeng_umass_edu/ir-research/joint_modeling_search_and_rec/experiments/unified_user/backbone_trainable/"

for bool_arg1 in "${bool_value_from_gru[@]}"
do 
    for bool_arg2 in "${bool_apply_value_layer[@]}"
    do
        for bool_arg3 in "${bool_apply_zero_attention[@]}"
        do
        python -m torch.distributed.launch --nproc_per_node=8 trainer/user_seq_train.py \
                                            $bool_arg1 $bool_arg2 $bool_arg3 \
                                            --train_batch_size=64 \
                                            --backbone_trainable \
                                            --experiment_folder=$experiment_folder \
                                            --learning_rate=1e-5 \
                                            --max_text_len=64
        done
    done 
done 