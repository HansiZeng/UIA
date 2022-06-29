#!/bin/bash

# for retrieval
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/bm25.test.run" \
        --threads 16 --batch-size 16 \
        --bm25
        
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.head.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/bm25.test.head.run" \
        --threads 16 --batch-size 16 \
        --bm25
        
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.torso.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/bm25.test.torso.run" \
        --threads 16 --batch-size 16 \
        --bm25
        
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.test.tail.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/bm25.test.tail.run" \
        --threads 16 --batch-size 16 \
        --bm25
        
# for evaluation
echo "===========================================================  all queries ==========================================================="
python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/bm25.test.run"

echo "===========================================================  head queries ==========================================================="
python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.head.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/bm25.test.head.run"

echo "===========================================================  torso queries ==========================================================="
python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.torso.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/bm25.test.torso.run"

echo "===========================================================  tail queries ==========================================================="
python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.tail.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/bm25.test.tail.run"