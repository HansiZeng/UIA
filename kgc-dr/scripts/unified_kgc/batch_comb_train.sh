#!/bin/bash 

t_a2sp_path="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/zero_shot/unified_kgc_train/a2sp.train.tsv"
t_a2cp_path="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/zero_shot/unified_kgc_train/a2cp.train.tsv"
t_q2a_path="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/zero_shot/unified_kgc_train/max2_qorient_q2p.train.tsv"

entites_path="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/all_entities.tsv"
experiment_folder="/home/jupyter/unity_jointly_rec_and_search/experiments/zero_shot/unified_kgc/"

#all_paths=("${t_a2sp_path}-None-None" "None-${t_a2cp_path}-None" "None-None-${t_q2a_path}" "${t_a2sp_path}-${t_a2cp_path}-None" "${t_a2sp_path}-None-${t_q2a_path}" "None-${t_a2cp_path}-${t_q2a_path}" "${t_a2sp_path}-${t_a2cp_path}-${t_q2a_path}")
all_paths=("${t_a2sp_path}-${t_a2cp_path}-None" "${t_a2sp_path}-None-${t_q2a_path}" "None-${t_a2cp_path}-${t_q2a_path}" "${t_a2sp_path}-${t_a2cp_path}-${t_q2a_path}")
#all_paths=("${t_a2sp_path}-${t_a2cp_path}-${t_q2a_path}")

EPOCHS=48

for paths in "${all_paths[@]}"
do
    IFS=- read -r a2sp_path a2cp_path q2a_path <<< $paths
    python -m torch.distributed.launch --nproc_per_node=4 trainer/unified_train.py \
                        --a2sp_path=$a2sp_path \
                        --a2cp_path=$a2cp_path \
                        --q2a_path=$q2a_path \
                        --s2sp_path="None" \
                        --s2cp_path="None" \
                        --q2s_path="None" \
                        --task="JOINT" \
                        --entites_path=$entites_path \
                        --num_train_epochs=$EPOCHS \
                        --experiment_folder=$experiment_folder \
                        --train_batch_size=384 \
                        --evaluate_steps=1000000 \
                        --independent_encoders \
                        --max_head_text_len=256 \
                        --max_tail_text_len=256 \
                        --max_global_steps=100000000
done
