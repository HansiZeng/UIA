import sys 
sys.path += ["./"]
import os 
import pickle 
import argparse
from typing import Optional, Union, List, Dict
import glob
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
import time
import random
import yaml
from pathlib import Path
import shutil
from collections import OrderedDict

import faiss
import torch
import numpy as np 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, HfArgumentParser

from modeling.dual_encoder import GraphDualEncoder
from retriever.retrieval_utils import get_embeddings_from_scratch, convert_index_to_gpu, index_retrieve
from dataset import SequenceDataset
from arguments import RetrievalArguments

def get_args():
    parser = HfArgumentParser(RetrievalArguments)
    args = parser.parse_args_into_dataclasses()
    assert len(args) == 1
    args = args[0]

    return args

def main(args):
    if "train" in args.queries_path:
        print("retrieve train queries")
        assert "train" in args.output_path
    if "dev" in args.queries_path:
        print("retrieve dev queries")
        assert "dev" in args.output_path
    if "2019" in args.queries_path:
        print("retrieve trec-19 queries")
        assert "trec19" in args.output_path
    if "2020" in args.queries_path:
        print("retrieve trec-20 queries")
        assert "trec20" in args.output_path

    # for model 
    assert os.path.isdir(args.pretrained_path), args.pretrained_path
    model = GraphDualEncoder.from_pretrained(args.pretrained_path)
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # for dataset 
    print("query_max_len: ", args.query_max_len)
    dataset = SequenceDataset.create_from_seqs_file(args.queries_path, tokenizer, args.query_max_len, is_query=True)
    query_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    query_embs, query_ids = get_embeddings_from_scratch(model, query_loader, use_fp16=True, is_query=True, show_progress_bar=True)

    # clean memory
    print("gpu memory allocated: {} MB".format(torch.cuda.memory_allocated() / 1024**2))
    del model 
    torch.cuda.empty_cache()
    print("gpu memory allocated: {} MB".format(torch.cuda.memory_allocated() / 1024**2))

    index = faiss.read_index(args.index_path)
    if torch.cuda.device_count() > 1:
        index = convert_index_to_gpu(index, list(range(torch.cuda.device_count())), False)
    else:
        index = convert_index_to_gpu(index, 0, False)
    
    query_embs = query_embs.astype(np.float32) if query_embs.dtype==np.float16 else query_embs
    nn_scores, nn_doc_ids = index_retrieve(index, query_embs, args.top_k, batch=128)

    qid_to_ranks = {}
    for qid, docids, scores in zip(query_ids, nn_doc_ids, nn_scores):
        for docid, s in zip(docids, scores):
            if qid not in qid_to_ranks:
                qid_to_ranks[qid] = [(docid, s)]
            else:
                qid_to_ranks[qid] += [(docid, s)]
    print(f"# unique query = {len(qid_to_ranks)}")

    if not os.path.exists(Path(args.output_path).parent):
        os.mkdir(Path(args.output_path).parent)
    total_rank = 0
    with open(args.output_path, "w") as f:
        for qid in qid_to_ranks:
            ranks = qid_to_ranks[qid]
            for i, (docid, s) in enumerate(ranks):
                f.write(f"{qid}\t{docid}\t{i+1}\t{s}\n")
            total_rank += len(ranks)
    
    print(f"average ranks per query = {total_rank/len(qid_to_ranks)}")


if __name__ == "__main__":
    args = get_args()
    main(args)
