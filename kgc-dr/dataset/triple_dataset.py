import os 
import sys 
sys.path += ["./"]
from pathlib import Path 
from typing import Dict, List, Tuple
import logging
logger = logging.getLogger(__name__)
import time
import ujson

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset 

from .utils import load_queries, load_passages

class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, qid_to_query, pid_to_passage, train_examples, 
                        tokenizer, max_query_len, max_passage_len):
        super(TripleDataset, self).__init__()
        self.qid_to_query = qid_to_query
        self.pid_to_passage = pid_to_passage
        self.train_examples = train_examples
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len 
        
    
    def __getitem__(self, idx):
        example = self.train_examples[idx]
        qid, relpid, negpid = example

        query = self.qid_to_query[qid]
        rel_passage = self.pid_to_passage[relpid] 
        neg_passage = self.pid_to_passage[negpid] 

        return {
            "qid": qid,
            "relpid": relpid,
            "negpid": negpid,
            "query": query,
            "rel_passage": rel_passage,
            "neg_passage": neg_passage
        }

    def __len__(self):
        return len(self.train_examples)

    def collate_fn(self, batch):
        qids, rel_pids, neg_pids, queries, passages = [], [], [], [], []
        for idx, elem in enumerate(batch):
            qids.append(elem["qid"])
            rel_pids.append(elem["relpid"])
            neg_pids.append(elem["negpid"])
            queries.append(elem["query"]) 
            assert type(elem["rel_passage"]) == str and type(elem["neg_passage"]) == str
            passages +=  [elem["rel_passage"], elem["neg_passage"]]

        
        qids =  np.array(qids, dtype=np.int64)
        rel_pids = np.array(rel_pids, dtype=np.int64)
        neg_pids = np.array(neg_pids, dtype=np.int64)

        queries = self.tokenizer(queries, padding=True, truncation='longest_first', return_token_type_ids=True,
                                        return_tensors="pt", max_length=self.max_query_len) # [bz]
        passages = self.tokenizer(passages, padding=True, truncation='longest_first', return_token_type_ids=True,
                                        return_tensors="pt", max_length=self.max_passage_len) # [2*bz]
        
        assert len(passages["input_ids"]) == 2 * len(queries["input_ids"])
        bz = len(queries["input_ids"])
        passages = {k:v.view(bz, 2, -1) for k, v in passages.items()}
        
        return {
            "qid": qids,
            "relpid": rel_pids,
            "negpid": neg_pids,
            "query": queries,
            "passages": passages,
        }            
             
    @classmethod
    def create_from_triple_file(cls, queries_path, passages_path, training_path, tokenizer, max_query_len, max_passage_len, 
                                    rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1

        qid_to_query = {}
        with open(queries_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                qid, query = int(array[0]), array[1]
                qid_to_query[qid] = query 
        
        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                if len(array) == 2:
                    pid, passage = int(array[0]), array[1]
                    pid_to_passage[pid] = passage 
                elif len(array) == 3:
                    pid, title, para = int(array[0]), array[1], array[2]
                    pid_to_passage[pid] = {"title": title, "para": para}
                else:
                    raise ValueError("array {}, with illegal length".format(array))

        if rank == -1:
            train_examples = []
            with open(training_path, "r") as fin:
                for line in fin:
                    qid, relpid, negpid = line.rstrip().split("\t")
                    qid, relpid, negpid = int(qid), int(relpid), int(negpid)
                    
                    train_examples.append((qid, relpid, negpid))
        else:
            train_examples = []
            with open(training_path, "r") as fin:
                for line_idx, line in enumerate(fin):
                    if line_idx % nranks == rank:
                        qid, relpid, negpid = line.rstrip().split("\t")
                        qid, relpid, negpid = int(qid), int(relpid), int(negpid)

                        train_examples.append((qid, relpid, negpid))
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_query_len, max_passage_len)



if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch

    queries_path = "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/queries.train.tsv"
    #queries_path = "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/collection_title.tsv"
    passages_path = "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/collection.tsv"
    training_path = "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/1pos_1neg.train.tsv"
    #training_path = "/home/jupyter/jointly_rec_and_search/datasets/rec_search/search_compl/anchor/1pos_1neg.compl.train.tsv"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   
    dataset = TripleDataset.create_from_triple_file(queries_path, passages_path, training_path, tokenizer, max_query_len=30, max_passage_len=64)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for b_idx, batch in enumerate(dataloader):
        for loop_idx, (q_token_ids, bi_token_ids) in enumerate(zip(batch["query"]["input_ids"], 
                                                            batch["passages"]["input_ids"])):
            print("-----query-----")
            print("qid: ", batch["qid"][loop_idx])
            print(tokenizer.decode(q_token_ids, skip_special_tokens=True))
            print("relpid: ", batch["relpid"][loop_idx], "negpid: ", batch["negpid"][loop_idx])
            print("bi_token_ids shape = {}".format(bi_token_ids.shape))
            print("rel passage: ", tokenizer.decode(bi_token_ids[0]))
            print("neg passage: ", tokenizer.decode(bi_token_ids[1]))
        if b_idx == 5:
            break
