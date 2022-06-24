import os 
import sys 
sys.path += ["./"]
from pathlib import Path 
from typing import Dict, List, Tuple
import logging
logger = logging.getLogger(__name__)
import time
import ujson
import random 
random.seed(4680)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset 

from .utils import load_queries, load_passages

class BoostedNDCGNwayDataset(torch.utils.data.Dataset):
    def __init__(self, qid_to_query, pid_to_passage, train_examples, 
                        tokenizer, max_query_len, max_passage_len, label_mode):
        super().__init__()
        self.qid_to_query = qid_to_query
        self.pid_to_passage = pid_to_passage
        self.train_examples = train_examples
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len 
        self.label_mode = label_mode 

        assert self.label_mode in ["9"]

    def __getitem__(self, idx):
        example = self.train_examples[idx]
        qid = example["qid"]
        relT_pids = example["relT_pids"]
        neg_pids = example["neg_pids"] 
        top_scores = example["top_scores"]
        neg_scores = example["neg_scores"]

        query = self.qid_to_query[qid]
        relT_passages = [self.pid_to_passage[pid] for pid in relT_pids]
        neg_passages = [self.pid_to_passage[pid] for pid in neg_pids]
        
        if self.label_mode == "9":
            assert len(top_scores) == 10 and len(neg_scores) == 20
            labels = top_scores + neg_scores
        elif self.label_mode == "2":
            assert len(top_scores) == 20 and len(neg_scores) == 10
            labels = top_scores + neg_scores
        elif self.label_mode == "3":
            assert len(top_scores) == 30 and len(neg_scores) == 0
            labels = top_scores + neg_scores            
        else:
            raise ValueError(f"label_mode: {self.label_mode} not defined.")
        

        all_pids = relT_pids + neg_pids
        nway_passages = relT_passages + neg_passages
        return {
            "qid": qid,
            "pids": all_pids,
            "query": query,
            "nway_passages": nway_passages,
            "labels": labels,
        }

    def __len__(self):
        return len(self.train_examples)

    def collate_fn(self, batch):
        qids, nway_pids, queries, nway_passages, labels = [], [], [], [], []
        for idx, elem in enumerate(batch):
            qids.append(elem["qid"])
            nway_pids.append(elem["pids"])
            queries.append(elem["query"])
            nway_passages += elem["nway_passages"]
            labels.append(elem["labels"])

        qids =  np.array(qids, dtype=np.int64)
        nway_pids = np.array(nway_pids, dtype=np.int64)
        bz, nway = nway_pids.shape 
        assert nway == 30

        queries = self.tokenizer(queries, padding=True, truncation='longest_first', 
                                        return_tensors="pt", max_length=self.max_query_len)
        nway_passages = self.tokenizer(nway_passages, padding=True, truncation='longest_first', 
                                        return_tensors="pt", max_length=self.max_passage_len)
        nway_passages = {k:v.view(bz, nway, -1) for k, v in nway_passages.items()}
        labels = torch.FloatTensor(labels)
        # make all labels >= 0.
        labels = torch.relu(labels)

        # mean the neg_scores
        labels[:,10:20] = labels[:,10:20].mean(dim=1,keepdim=True).repeat(1,10) #[bz, 10]
        labels[:,20:30] = labels[:,20:30].mean(dim=1,keepdim=True).repeat(1,10) #[bz, 10]


        return {
            "qid": qids,
            "nway_pids": nway_pids,
            "query": queries,
            "nway_passages": nway_passages,
            "labels": labels,
        } 
    
    @classmethod
    def create_from_10relT_20neg_file(cls, queries_path, passages_path, training_path, tokenizer, max_query_len, max_passage_len, 
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
                    example = ujson.loads(line)
                    train_examples.append({
                            "qid": example["qid"],
                            "relT_pids": example["relT_pids"],
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"],
                            "top_scores": example["top_scores"],
                            "neg_scores": example["most_hard_scores"] + example["semi_hard_scores"]
                        })
        else:
            train_examples = []
            with open(training_path, "r") as fin:
                for line_idx, line in enumerate(fin):
                    if line_idx % nranks == rank:
                        example = ujson.loads(line)
                        train_examples.append({
                            "qid": example["qid"],
                            "relT_pids": example["relT_pids"],
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"],
                            "top_scores": example["top_scores"],
                            "neg_scores": example["most_hard_scores"] + example["semi_hard_scores"]
                        })
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_query_len, max_passage_len, label_mode="9")
        
if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch

    queries_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/queries.train.tsv"
    passages_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/collection.tsv"
    training_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/corase_to_fine_grained/teacher_scores/30T_score.train.json"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = NDCGNwayDataset.create_from_30relT_file(queries_path, passages_path, training_path, tokenizer, 
                                                        max_query_len=30, max_passage_len=128)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for b_idx, batch in enumerate(dataloader):
        for loop_idx, (q_token_ids, nway_token_ids) in enumerate(zip(batch["query"]["input_ids"], 
                                                            batch["nway_passages"]["input_ids"])):
            #print("-----query-----")
            #print("qid: ", batch["qid"][loop_idx])
            #print(tokenizer.decode(q_token_ids, skip_special_tokens=True))
            #print("nway_pids: \n", batch["nway_pids"][loop_idx])
            print("labels:", batch["labels"][loop_idx])
            for idx in range(len(nway_token_ids)):
                token_ids = nway_token_ids[idx]
                #print("nway_passage: \n", tokenizer.decode(token_ids, skip_special_tokens=True))
                if idx == 3:
                    break
        if b_idx == 2:
            break