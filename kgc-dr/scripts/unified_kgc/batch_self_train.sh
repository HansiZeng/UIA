#!/bin/bash 

experiment_folder="/home/jupyter/unity_jointly_rec_and_search/experiments/unified_kgc"


in_dir="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/"
entites_path="${in_dir}/all_entities.tsv"

declare -a elems=(
"phase_1/experiment_09-07_005742 ${experiment_folder}/phase_1/experiment_09-07_005742/self_train/a2sp.train.tsv:${experiment_folder}/phase_1/experiment_09-07_005742/self_train/a2cp.train.tsv:${experiment_folder}/phase_1/experiment_09-07_005742/self_train/q2p.train.tsv" 
)


epoch=48
for elem in "${elems[@]}"
do
    read -a strarr <<< "$elem"
    echo ${strarr[0]} ${strarr[1]}
    model_name=${strarr[0]}
    IFS=: read -r a2sp_path a2cp_path q2a_path <<< ${strarr[1]}
    
    pretrained_path="${experiment_folder}/${model_name}/models/checkpoint_latest"

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