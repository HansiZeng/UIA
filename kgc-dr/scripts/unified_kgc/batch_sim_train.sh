#!/bin/bash 

a2sp_path="/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/sim_train/a2sp.train.tsv"
a2cp_paths=("None" "/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/sim_train/a2cp.train.tsv")
q2a_paths=("None" "/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/sim_train/q2a.train.tsv")
s2sp_paths=("None" "/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/sim_train/s2sp.train.tsv")
s2cp_paths=("None")
q2s_paths=("None")
#s2cp_paths=("None" "/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/sim_train/s2cp.train.tsv")
#q2s_paths=("None" "/home/jupyter/jointly_rec_and_search/datasets/unified_kgc/sim_train/q2s.train.tsv")

EPOCHS=16

for a2cp_path in "${a2cp_paths[@]}"
do
    for q2a_path in "${q2a_paths[@]}"
    do
        for s2sp_path in "${s2sp_paths[@]}"
        do
            for s2cp_path in "${s2cp_paths[@]}"
            do 
                for q2s_path in "${q2s_paths[@]}"
                do
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
                                    --max_global_steps=1000
                done
            done
        done
    done
done
