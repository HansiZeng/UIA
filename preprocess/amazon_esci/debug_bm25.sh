#!/bin/bash

# collection index
root_dir="/home/jupyter/unity_jointly_rec_and_search/datasets/amazon_esci_dataset/data/processed/public/task_1_query_product_ranking"
mkdir -p "${root_dir}/product_v3_debug/"
cp -u "${root_dir}/product.jsonl" "${root_dir}/product_v3_debug/product.jsonl"

python -m pyserini.index -collection JsonCollection \
 -generator DefaultLuceneDocumentGenerator \
 -threads 1 -input "${root_dir}/product_v3_debug/" \
 -index "${root_dir}/bm25_index/" \
 -storePositions -storeDocvectors -storeRaw


# for TRAIN query, similar_ivm, compl_ivm retrieval
python -m pyserini.search --topics "${root_dir}/all_entities.tsv"  \
        --index "${root_dir}/bm25_index/" \
        --output "${root_dir}/runs/debug_bm25.all.run" \
        --threads 16 --batch-size 16 \
        --hits 200 \
        --bm25

# for TEST query retrieval and evaluation 
# python -m pyserini.search --topics "${root_dir}/queries.test.tsv"  \
#       --index "${root_dir}/bm25_index/" \
#        --output "${root_dir}/runs/bm25.test.run" \
#        --threads 16 --batch-size 16 \
#        --hits 1000 \
#        --bm25

# echo "===========================================================  all queries ==========================================================="#
# python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "${root_dir}/qrels.test.tsv" #"${root_dir}/runs/bm25.test.run"
