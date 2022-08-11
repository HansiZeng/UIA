#!/bin/bash 

all_paths=("/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.a2sp.rnd.tsv-None-/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.c2cp.tsv-/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.c2sp.rnd.tsv-/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.q2c.tsv"  )


EPOCHS=32

for paths in "${all_paths[@]}"
do
    IFS=- read -r a2sp_path q2a_path c2cp_path c2sp_path q2c_path <<< $paths
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
                                    --evaluate_steps=10000 \
                                    --independent_encoders \
                                    --max_head_text_len=256 \
                                    --max_tail_text_len=256
done
    
