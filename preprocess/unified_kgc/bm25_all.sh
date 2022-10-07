#!/bin/bash

# collection index
mkdir -p "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/product_v3/"
cp -u "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/product.jsonl" "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/product_v3/product.jsonl"

python -m pyserini.index -collection JsonCollection \
 -generator DefaultLuceneDocumentGenerator \
 -threads 1 -input "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/product_v3/" \
 -index "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/bm25_index/" \
 -storePositions -storeDocvectors -storeRaw


# for TRAIN query, similar_ivm, compl_ivm retrieval
#python -m pyserini.search --topics "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/all_entities.tsv"  \
#        --index "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/bm25_index/" \
#        --output "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/runs/bm25.all.run" \
#        --threads 16 --batch-size 16 \
#        --hits 200 \
#        --bm25

# for TEST query retrieval and evaluation 
#python -m pyserini.search --topics "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/queries.test.tsv"  \
#       --index "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/bm25_index/" \
#        --output "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/runs/bm25.test.run" \
#        --threads 16 --batch-size 16 \
#        --hits 1000 \
#        --bm25

#echo "===========================================================  all queries ==========================================================="#
#python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/qrels.test.tsv" #"/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/runs/bm25.test.run"
