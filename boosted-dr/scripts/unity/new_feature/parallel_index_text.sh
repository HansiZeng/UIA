#!/bin/bash

RESUME_PATH="/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/multistep-curriculum/experiment_01-29_230126/models/checkpoint_120000.pth.tar"
INDEX_PATH="/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/multistep-curriculum/experiment_01-29_230126/parallel_index/"

python -m torch.distributed.launch --nproc_per_node=4 retriever/parallel_index_text.py \
                                    --resume=$RESUME_PATH \
                                    --index_dir=$INDEX_PATH \
                                    --batch_size=128