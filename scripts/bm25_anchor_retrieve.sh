#!/bin/bash

python -m pyserini.search --topics "/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/anchors.test.exclude.tsv"  \
        --index "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/bm25_index/" \
        --output "/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/runs/bm25.test.exclude.run" \
        --threads 16 --batch-size 16 \
        --bm25