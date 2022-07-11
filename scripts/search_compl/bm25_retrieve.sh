#!/bin/bash

SUFFIX="train"
python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.${SUFFIX}.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/bm25.${SUFFIX}.run" \
        --threads 16 --batch-size 16 \
        --bm25