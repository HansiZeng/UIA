#!/bin/bash 

home_dir="/home/jupyter/unity_jointly_rec_and_search/datasets/amazon_esci_dataset/task_2_multiclass_product_classification"
t_a2sp_path="${home_dir}/self_train/a2s.50.train.tsv"
t_a2cp_path="${home_dir}/self_train/a2c.train.tsv"
t_q2a_path="${home_dir}/self_train/q2p.train.tsv"

# check files
wc -l $t_a2sp_path
wc -l  $t_a2cp_path
wc -l  $t_q2a_path

entites_path="${home_dir}/entity_prefix/prefix_all_entities.tsv"
experiment_folder="/home/jupyter/unity_jointly_rec_and_search/experiments/amazon_esci/task2"
pretrained_path="${experiment_folder}/phase_3/experiment_09-22_100600/models/checkpoint_latest"

#all_paths=("${t_a2sp_path}-None-None" "None-${t_a2cp_path}-None" "None-None-${t_q2a_path}" "${t_a2sp_path}-${t_a2cp_path}-None" "${t_a2sp_path}-None-${t_q2a_path}" "None-${t_a2cp_path}-${t_q2a_path}" "${t_a2sp_path}-${t_a2cp_path}-${t_q2a_path}")
all_paths=("${t_a2sp_path}-${t_a2cp_path}-${t_q2a_path}")

EPOCHS=(72)

for paths in "${all_paths[@]}"
do
    for epoch in "${EPOCHS[@]}"
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
                        --num_train_epochs=$epoch \
                        --experiment_folder=$experiment_folder \
                        --train_batch_size=384 \
                        --evaluate_steps=1000000 \
                        --independent_encoders \
                        --max_head_text_len=128 \
                        --max_tail_text_len=128 \
                        --max_global_steps=100000000 \
                        --model_pretrained_path=$pretrained_path
    done
done
