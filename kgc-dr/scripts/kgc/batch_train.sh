#!/bin/bash 

#a2sp_paths=("None" "/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.a2sp.rnd.tsv")
a2sp_paths=("/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.a2sp.rnd.tsv")
q2a_paths=("/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.q2a.tsv")
c2cp_paths=("/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.c2cp.tsv")
c2sp_paths=("/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.c2sp.rnd.tsv")
q2c_paths=("/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.q2c.tsv")


EPOCHS=32

for a2sp_path in "${a2sp_paths[@]}"
do
    for q2a_path in "${q2a_paths[@]}"
    do
        for c2cp_path in "${c2cp_paths[@]}"
        do
            for c2sp_path in "${c2sp_paths[@]}"
            do 
                for q2c_path in "${q2c_paths[@]}"
                do
                python -m torch.distributed.launch --nproc_per_node=4 trainer/train.py \
                                    --a2sp_path=$a2sp_path \
                                    --q2a_path=$q2a_path \
                                    --c2cp_path=$c2cp_path \
                                    --c2sp_path=$c2sp_path \
                                    --q2c_path=$q2c_path \
                                    --entites_path="/home/jupyter/jointly_rec_and_search/datasets/kgc/all_entites.tsv" \
                                    --num_train_epochs=32 \
                                    --experiment_folder="/home/jupyter/jointly_rec_and_search/experiments/kgc/" \
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
