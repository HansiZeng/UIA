#!/bin/bash
#experiment_folder="/work/hzeng_umass_edu/ir-research/joint_modeling_search_and_rec/experiments/unified_user/backbone_trainable/"

home_prefix="/home/jupyter/unity_jointly_rec_and_search"
examples_path="${home_prefix}/datasets/unified_user/sequential_train_test/hlen_4_bm25/search_sequential.train.json"
eid_path="${home_prefix}/datasets/unified_user/all_entities.tsv"
experiment_folder="${home_prefix}/experiments/unified_user/user_seq_merge_encoder/"

bool_apply_position_embedding=("--apply_position_embedding" "")
bool_apply_value_layer_for_passage_emb=("--apply_value_layer_for_passage_emb" "")
lrs=(7e-4 7e-5)
neps=(4 8 12)

for bool_arg1 in "${bool_apply_position_embedding[@]}"
do 
    for bool_arg2 in "${bool_apply_value_layer_for_passage_emb[@]}"
    do
        for lr in "${lrs[@]}"
        do
            for ep in "${neps[@]}"
            do
            python -m torch.distributed.launch --nproc_per_node=4 trainer/user_seq_merge_train.py \
            --examples_path=$examples_path \
            --eid_path=$eid_path \
            --experiment_folder=$experiment_folder \
            --train_batch_size=384 \
            --learning_rate=$lr \
            --num_train_epochs=$ep \
            $bool_arg1 \
            $bool_arg2
            done
        done
    done 
done 