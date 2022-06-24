#!/bin/bash

python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/arels.test.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec/runs/bm25.test.run"