#!/bin/bash 


train_q2p_paths=("/home/jupyter/jointly_rec_and_search/datasets/kgc_search/train/bm25_neg/max2_triples.train_q2p.tsv")
q2p_paths=("None" "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/train/bm25_neg/max5_triples.q2p.tsv")
p2sp_paths=("None" "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/train/bm25_neg/max5_triples.p2sp.tsv")
p2cp_paths=("/home/jupyter/jointly_rec_and_search/datasets/kgc_search/train/bm25_neg/max5_triples.p2cp.tsv")


for train_q2p_path in "${train_q2p_paths[@]}"
do
    for q2p_path in "${q2p_paths[@]}"
    do
        for p2sp_path in "${p2sp_paths[@]}"
        do
            for p2cp_path in "${p2cp_paths[@]}"
            do 
                python -m torch.distributed.launch --nproc_per_node=4 trainer/search_train.py \
                                    --train_q2p_path=$train_q2p_path \
                                    --q2p_path=$q2p_path \
                                    --p2sp_path=$p2sp_path \
                                    --p2cp_path=$p2cp_path \
                                    --entites_path="/home/jupyter/jointly_rec_and_search/datasets/kgc_search/all_entites.tsv" \
                                    --num_train_epochs=24 \
                                    --experiment_folder="/home/jupyter/jointly_rec_and_search/experiments/kgc_search/" \
                                    --train_batch_size=384 \
                                    --evaluate_steps=1000000 \
                                    --independent_encoders \
                                    --max_head_text_len=256 \
                                    --max_tail_text_len=256
                done
            done
        done
    done
done
