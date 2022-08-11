#!/bin/bash 

EPOCHS=32
python -m torch.distributed.launch --nproc_per_node=4 trainer/train.py \
                    --a2cp_path="/home/jupyter/jointly_rec_and_search/datasets/kgc/train/control_train/max5_triples.all.tsv" \
                    --a2sp_path="None" \
                    --q2a_path="None" \
                    --c2cp_path="None" \
                    --c2sp_path="None" \
                    --q2c_path="None" \
                    --entites_path="/home/jupyter/jointly_rec_and_search/datasets/kgc/all_entites.tsv" \
                    --num_train_epochs=32 \
                    --experiment_folder="/home/jupyter/jointly_rec_and_search/experiments/kgc/" \
                    --train_batch_size=384 \
                    --evaluate_steps=1000000 \
                    --independent_encoders \
                    --max_head_text_len=256 \
                    --max_tail_text_len=256
      