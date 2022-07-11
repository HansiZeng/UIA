#!/bin/bash

python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/anchors.train.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/anchor/bm25.similar.train.run" \
        --threads 16 --batch-size 16 \
        --bm25