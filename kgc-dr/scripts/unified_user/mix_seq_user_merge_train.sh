#!/bin/bash

home_prefix="/home/jupyter/unity_jointly_rec_and_search"
eid_path="${home_prefix}/datasets/unified_user/all_entities.tsv"
experiment_folder="${home_prefix}/experiments/unified_user/user_seq_merge_encoder/"
backbone_path="/home/jupyter/unity_jointly_rec_and_search/experiments/unified_kgc/phase_1/experiment_10-02_005604/models/checkpoint_latest/"

#exp_paths=("${home_prefix}/datasets/unified_user/mixture_sequential_train_test/hlen_4_bm25/search_sequential.train.json" "${home_prefix}/datasets/unified_user/mixture_sequential_train_test/hlen_4_bm25/sim_rec_sequential.train.json" "${home_prefix}/datasets/unified_user/mixture_sequential_train_test/hlen_4_bm25/compl_rec_sequential.train.json")
exp_paths=("${home_prefix}/datasets/unified_user/mixture_sequential_train_test/hlen_4_bm25/search_sim_compl_sequential.train.json")
#exp_paths=("${home_prefix}/datasets/unified_user/mixture_sequential_train_test/hlen_4_bm25/no_intent_search_compl_seqential.train.json")
lrs=(7e-5)
neps=(8)

for examples_path in "${exp_paths[@]}"
do 
    for lr in "${lrs[@]}"
    do
        for ep in "${neps[@]}"
        do
        python -m torch.distributed.launch --nproc_per_node=4 trainer/user_seq_merge_train.py \
        --examples_path=$examples_path \
        --eid_path=$eid_path \
        --experiment_folder=$experiment_folder \
        --backbone_path=$backbone_path \
        --train_batch_size=96 \
        --learning_rate=$lr \
        --num_train_epochs=$ep \
        --backbone_trainable \
        --apply_value_layer_for_passage_emb \
        --apply_user_item_ids \
        --apply_relation_ids 
        done
    done 
done 
