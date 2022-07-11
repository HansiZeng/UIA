#!/bin/bash

# collection index
mkdir -p "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/product_v3/"
cp -u "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/product.jsonl" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/product_v3/product.jsonl"

python -m pyserini.index -collection JsonCollection \
 -generator DefaultLuceneDocumentGenerator \
 -threads 1 -input "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/product_v3/" \
 -index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/bm25_index/" \
 -storePositions -storeDocvectors -storeRaw
 
 
# for TRAIN query, similar_ivm, compl_ivm retrieval
SUFFIX="train"
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/queries.${SUFFIX}.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/runs/bm25.${SUFFIX}.run" \
        --threads 16 --batch-size 16 \
        --bm25
        
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/anchors.similar.train.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/runs/bm25.similar.train.run" \
        --threads 16 --batch-size 16 \
        --bm25
        
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/anchors.compl.train.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/runs/bm25.compl.train.run" \
        --threads 16 --batch-size 16 \
        --bm25
        
        
# for TEST query retrieval and evaluation 
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/queries.test.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/runs/bm25.test.run" \
        --threads 16 --batch-size 16 \
        --bm25
        
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/queries.test.head.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/runs/bm25.test.head.run" \
        --threads 16 --batch-size 16 \
        --bm25
        
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/queries.test.torso.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/runs/bm25.test.torso.run" \
        --threads 16 --batch-size 16 \
        --bm25
        
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/queries.test.tail.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/runs/bm25.test.tail.run" \
        --threads 16 --batch-size 16 \
        --bm25

echo "===========================================================  all queries ==========================================================="
python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/qrels.test.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/runs/bm25.test.run"

echo "===========================================================  head queries ==========================================================="
python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/qrels.test.head.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/runs/bm25.test.head.run"

echo "===========================================================  torso queries ==========================================================="
python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/qrels.test.torso.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/runs/bm25.test.torso.run"

echo "===========================================================  tail queries ==========================================================="
python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/qrels.test.tail.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/runs/bm25.test.tail.run"