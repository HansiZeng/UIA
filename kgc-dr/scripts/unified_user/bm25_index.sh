#!/bin/bash

# collection index
mkdir -p "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/product_v3/"
cp -u "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/product.jsonl" "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/product_v3/product.jsonl"

python -m pyserini.index -collection JsonCollection \
 -generator DefaultLuceneDocumentGenerator \
 -threads 1 -input "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/product_v3/" \
 -index "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/bm25_index/" \
 -storePositions -storeDocvectors -storeRaw