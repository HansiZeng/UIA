#!/bin/bash

HOME_DIR="/home/jupyter/unity_jointly_rec_and_search/datasets/amazon_esci_dataset/task_2_multiclass_product_classification"
EXPERIMENT_FORDER="/home/jupyter/unity_jointly_rec_and_search/experiments/amazon_esci/task2"

ENTITES_PATH="${HOME_DIR}/all_entities.tsv"
PASSAGE_PATH="${HOME_DIR}/collection_title.tsv"

DATES=("experiment_10-05_032456")

for DATE in "${DATES[@]}"
do
    PRETRAINED_PATH="${EXPERIMENT_FORDER}/${DATE}/models/checkpoint_latest"
    INDEX_DIR="${EXPERIMENT_FORDER}/${DATE}/index/"
    INDEX_PATH="${EXPERIMENT_FORDER}/${DATE}/index/checkpoint_latest.index"
    OUTPUT_PATH="${EXPERIMENT_FORDER}/${DATE}/runs/checkpoint_latest"

    # 1, index
#    python -m torch.distributed.launch --nproc_per_node=4 retriever/parallel_index_text_1.py \
#                                        --pretrained_path=$PRETRAINED_PATH \
#                                        --passages_path=$PASSAGE_PATH \
#                                        --index_dir=$INDEX_DIR \
#                                        --batch_size=128 \
#                                        --max_length=256

#    python retriever/parallel_index_text_2.py --pretrained_path=$PRETRAINED_PATH \
#                                    --passages_path=$PASSAGE_PATH \
#                                    --index_dir=$INDEX_DIR \
#                                    --batch_size=512 \

    # 2, retrieval
    # queries.test.tsv
    python retriever/retrieve_top_passages.py \
    --queries_path=$ENTITES_PATH \
    --pretrained_path=$PRETRAINED_PATH \
    --output_path=${OUTPUT_PATH}.all.run \
    --index_path=$INDEX_PATH \
    --batch_size=512 \
    --query_max_len=128 \
    --top_k=200 \
    --tokenizer_name_or_path="sentence-transformers/msmarco-bert-base-dot-v5" 
done

