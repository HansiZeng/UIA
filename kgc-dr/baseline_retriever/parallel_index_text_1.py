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
import torch.distributed as dist
import ujson 

from modeling.baseline_dual_encoder import BaseDualEncoder as DualEncoder
from retriever.retrieval_utils import write_embeddings_to_disk
from dataset import KGCSequenceDataset
from arguments import RetrievalArguments, ModelArguments


BLOCK_SIZE = 50_000

def set_env(args):
    args.distributed = False
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group("nccl")
        args.device = device
        args.distributed = dist.get_world_size() > 1
        args.nranks = dist.get_world_size()
        args.ngpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cuda")

def get_args():
    parser = HfArgumentParser((RetrievalArguments, ModelArguments))
    args, model_args = parser.parse_args_into_dataclasses()
    #assert len(args) == 1
    #args = args[0]
    
    if args.local_rank < 1:
        if not os.path.exists(args.index_dir):
            os.mkdir(args.index_dir)

    return args, model_args

def main(args, model_args):
    # for model
    #assert os.path.isdir(args.pretrained_path), args.pretrained_path
    model = DualEncoder(model_args)
    model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.local_rank],
                                                            output_device=args.local_rank)

    # index path
    index_path = os.path.join(args.index_dir, "tasb.index")

    # for dataset
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    print("===============================batch_size:{} ==================================".format(args.batch_size))
    if args.distributed:
        dataset = KGCSequenceDataset.create_from_seqs_file(args.passages_path, tokenizer, args.max_length,
                                                        rank=args.local_rank, nranks=args.nranks)
        text_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)
    else:
        dataset = KGCSequenceDataset.create_from_seqs_file(args.passages_path, tokenizer, args.max_length)
        text_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    write_freq = args.chunk_size // args.batch_size 
    num_chunks = write_embeddings_to_disk(model, text_loader, use_fp16=True, rank_idx=args.local_rank, write_freq=write_freq, 
                                   index_dir=args.index_dir, unified_kgc=True)
    print("num chunks for each rank = {}".format(num_chunks))

    # save plan 
    plan = {"nranks": args.nranks, "num_chunks": num_chunks, "index_path": index_path}
    with open(os.path.join(args.index_dir, "plan.json"), "w") as fout:
        ujson.dump(plan, fout)


if __name__ == "__main__":
    args, model_args = get_args()
    set_env(args)
    main(args, model_args)
