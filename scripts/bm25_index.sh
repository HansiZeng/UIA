#!/bin/bash

mkdir -p "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/product_v3/"
cp -u "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/product.jsonl" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/product_v3/product.jsonl"

python -m pyserini.index -collection JsonCollection \
 -generator DefaultLuceneDocumentGenerator \
 -threads 1 -input "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/product_v3/" \
 -index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/bm25_index/" \
 -storePositions -storeDocvectors -storeRaw