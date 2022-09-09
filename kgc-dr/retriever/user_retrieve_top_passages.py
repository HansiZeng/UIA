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

from modeling import UserSeqEncoder, UserSeqMergeEncoder
from retriever.retrieval_utils import get_user_side_embedding_from_scratch, convert_index_to_gpu, index_retrieve
from user_argument import RetrievalArguments
from mn_to_model import NAME_TO_MODEL
from dataset import UserSequentialDataset

def get_model_args(model_path):
    assert os.path.isdir(model_path)
    model_args_path = os.path.join(model_path, "model_args.pt")
    model_args = torch.load(model_args_path)
    
    return model_args

def get_args():
    parser = HfArgumentParser(RetrievalArguments)
    args = parser.parse_args_into_dataclasses()
    assert len(args) == 1
    args = args[0]

    return args

def main(args):
    # for model 
    assert os.path.isdir(args.pretrained_path), args.pretrained_path
    model_args = get_model_args(args.pretrained_path)
    if args.local_rank <= 0:
        print(f"MODEL NAME: {model_args.model_name}")
    model = NAME_TO_MODEL[model_args.model_name].from_pretrained(args.pretrained_path)
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # for dataset 
    print("query_max_len: ", args.max_length)
    dataset = UserSequentialDataset.create_from_json_file(
                            eid_path=args.eid_path,
                            examples_path=args.query_examples_path,
                            tokenizer=tokenizer,
                            max_text_len=args.max_length,
                            is_train=False,
                            apply_zero_attention=model_args.apply_zero_attention)
    query_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    query_embs, query_ids = get_user_side_embedding_from_scratch(
                                    model, 
                                    query_loader, 
                                    use_fp16=True, 
                                    is_query=True, 
                                    show_progress_bar=True)

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