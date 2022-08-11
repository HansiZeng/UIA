#!/bin/bash

# collection index
mkdir -p "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/product_v3/"
cp -u "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/product.jsonl" "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/product_v3/product.jsonl"

python -m pyserini.index -collection JsonCollection \
 -generator DefaultLuceneDocumentGenerator \
 -threads 1 -input "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/product_v3/" \
 -index "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/bm25_index/" \
 -storePositions -storeDocvectors -storeRaw


# for TRAIN query, similar_ivm, compl_ivm retrieval
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/entities.train.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/runs/bm25.train.run" \
        --threads 16 --batch-size 16 \
        --hits 200 \
        --bm25

# for TEST query retrieval and evaluation 
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/queries.test.tsv"  \
       --index "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/runs/bm25.test.run" \
        --threads 16 --batch-size 16 \
        --hits 1000 \
        --bm25

echo "===========================================================  all queries ==========================================================="#
python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/qrels.test.tsv" "/home/jupyter/jointly_rec_and_search/datasets/kgc_search/runs/bm25.test.run"
