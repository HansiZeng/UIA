#!/bin/bash 

home_dir="/home/jupyter/unity_jointly_rec_and_search/datasets/spotify_mpd/processed"
t_a2sp_path="${home_dir}/unified_train/a2sp.train.tsv"
t_q2p_path="${home_dir}/unified_train/q2p.train.tsv"

# check files
wc -l $t_q2p_path
wc -l $t_a2sp_path

entites_path="${home_dir}/all_entities.tsv"
experiment_folder="/home/jupyter/unity_jointly_rec_and_search/experiments/spotify_mpd"

all_paths=("${t_a2sp_path}-None" "None-${t_q2p_path}" "${t_a2sp_path}-${t_q2p_path}")

EPOCHS=(48)

for paths in "${all_paths[@]}"
do
    for epoch in "${EPOCHS[@]}"
    do
    IFS=- read -r a2sp_path q2p_path <<< $paths
    echo $a2sp_path
    echo $q2p_path
    python -m torch.distributed.launch --nproc_per_node=4 trainer/unified_train.py \
                        --a2sp_path=$a2sp_path \
                        --a2cp_path="None" \
                        --q2a_path=$q2p_path \
                        --s2sp_path="None" \
                        --s2cp_path="None" \
                        --q2s_path="None" \
                        --task="JOINT" \
                        --entites_path=$entites_path \
                        --num_train_epochs=$epoch \
                        --experiment_folder=$experiment_folder \
                        --train_batch_size=384 \
                        --evaluate_steps=1000000 \
                        --independent_encoders \
                        --max_head_text_len=128 \
                        --max_tail_text_len=128 \
                        --max_global_steps=100000000
    done
done
