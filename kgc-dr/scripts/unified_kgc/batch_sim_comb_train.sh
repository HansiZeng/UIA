#!/bin/bash 

a2sp_path="/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/sim_train/a2sp.train.tsv"
t_a2cp_path="/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/sim_train/a2cp.train.tsv"
t_q2a_path="/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/sim_train/q2a.train.tsv"
t_s2sp_path="/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/sim_train/s2sp.train.tsv"
t_s2cp_path="/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/sim_train/s2cp.train.tsv"
t_q2s_path="/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/sim_train/q2s.train.tsv"

all_paths=("${t_a2cp_path}-${t_q2a_path}-${t_s2sp_path}-${t_s2cp_path}-None" "${t_a2cp_path}-${t_q2a_path}-${t_s2sp_path}-None-${t_q2s_path}" "${t_a2cp_path}-${t_q2a_path}-None-${t_s2cp_path}-${t_q2s_path}" "${t_a2cp_path}-None-${t_s2sp_path}-${t_s2cp_path}-${t_q2s_path}" "None-${t_q2a_path}-${t_s2sp_path}-${t_s2cp_path}-${t_q2s_path}" "${t_a2cp_path}-${t_q2a_path}-${t_s2sp_path}-${t_s2cp_path}-${t_q2s_path}")

EPOCHS=24

for paths in "${all_paths[@]}"
do
    IFS=- read -r a2cp_path q2a_path s2sp_path s2cp_path q2s_path <<< $paths
    python -m torch.distributed.launch --nproc_per_node=4 trainer/unified_train.py \
                        --a2sp_path=$a2sp_path \
                        --a2cp_path=$a2cp_path \
                        --q2a_path=$q2a_path \
                        --s2sp_path=$s2sp_path \
                        --s2cp_path=$s2cp_path \
                        --q2s_path=$q2s_path \
                        --task="SIMILAR_REC" \
                        --entites_path="/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/all_entities.tsv" \
                        --num_train_epochs=$EPOCHS \
                        --experiment_folder="/home/jupyter/jointly_rec_and_search/experiments/unified_kgc/" \
                        --train_batch_size=384 \
                        --evaluate_steps=1000000 \
                        --independent_encoders \
                        --max_head_text_len=256 \
                        --max_tail_text_len=256 \
                        --max_global_steps=160000
done
