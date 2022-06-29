#!/bin/bash

python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/qrels.test.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/bm25.test.run"

#echo "================================================================================================================================"

#python -m pyserini.eval.trec_eval -c -m recall.1000 -m ndcg_cut.10 "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/ext_qrels.test.exclude.tsv" "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/runs/bm25.test.exclude.run"