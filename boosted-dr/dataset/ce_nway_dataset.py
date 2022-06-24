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

class CENwayDataset(torch.utils.data.Dataset):
    def __init__(self, qid_to_query, pid_to_passage, train_examples, 
                        tokenizer, max_length, label_mode):
        super().__init__()
        self.qid_to_query = qid_to_query
        self.pid_to_passage = pid_to_passage
        self.train_examples = train_examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_mode = label_mode

        assert self.label_mode in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        logger.info(f"label mode: {self.label_mode}")
    
    def __getitem__(self, idx):
        example = self.train_examples[idx]
        qid = example["qid"]
        relT_pids = example["relT_pids"]
        neg_pids = example["neg_pids"] 

        query = self.qid_to_query[qid]
        relT_passages = [self.pid_to_passage[pid] for pid in relT_pids]
        neg_passages = [self.pid_to_passage[pid] for pid in neg_pids]
        if self.label_mode == "1":
            assert len(relT_pids) == 1 and len(neg_pids) == 5
            labels = [1.] + [0.] * len(neg_pids)
        elif self.label_mode == "2":
            assert len(relT_pids) == 10 and len(neg_pids) == 20
            labels = [1.] * len(relT_pids) + [1/2.] * 10 + [0.] * 10
        elif self.label_mode == "3":
            assert len(relT_pids) == 10 and len(neg_pids) == 20
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [0.] * len(neg_pids)
        elif self.label_mode == "4":
            assert len(relT_pids) == 10 and len(neg_pids) == 20
            labels = [1.] + [0.9] * 9 + [1/2.] * 10 + [0.] * 10
        elif self.label_mode == "5":
            assert len(relT_pids) == 20 and len(neg_pids) == 10
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [0.] * len(neg_pids)
        elif self.label_mode == "6":
            assert len(relT_pids) == 30 and len(neg_pids) == 0
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [0.] * len(neg_pids)
        elif self.label_mode == "7":
            assert len(relT_pids) == 5 and len(neg_pids) == 25
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [0.] * len(neg_pids) 
        elif self.label_mode == "8":
            assert len(relT_pids) == 5 and len(neg_pids) == 25
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [-0.25] * 12 + [-0.5] * 13
        elif self.label_mode == "9":
            assert len(relT_pids) == 10 and len(neg_pids) == 20
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [-0.25] * 10 + [-0.5] * 10
        elif self.label_mode == "10":
            assert len(relT_pids) == 20 and len(neg_pids) == 10
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [-0.25] * 5 + [-0.5] * 5
        else:
            raise ValueError(f"{self.label_mode} do not defined")

        return {
            "qid": qid,
            "relT_pids": relT_pids,
            "neg_pids": neg_pids,
            "query": query,
            "relT_passages": relT_passages,
            "neg_passages": neg_passages,
            "labels": labels,
        }

    def __len__(self):
        return len(self.train_examples)

    def collate_fn(self, batch):
        qids, relT_pids, neg_pids, queries, nway_passages, labels = [], [], [], [], [], []
        for idx, elem in enumerate(batch):
            relT_pids.append(elem["relT_pids"])
            neg_pids.append(elem["neg_pids"])
            nway_passages +=  elem["relT_passages"] + elem["neg_passages"]

            nway = len(elem["relT_passages"]) + len(elem["neg_passages"])
            qids.append(elem["qid"])
            queries += [elem["query"]]*nway
            labels.append(elem["labels"])

        qids =  np.array(qids, dtype=np.int64)
        relT_pids = np.array(relT_pids, dtype=np.int64)
        neg_pids = np.array(neg_pids, dtype=np.int64)
        bz, relT_len, neg_len = len(batch), relT_pids.shape[1], neg_pids.shape[1]
        nway = relT_len + neg_len

        query_passages = self.tokenizer(queries, nway_passages, padding=True, truncation='longest_first', 
                                        return_tensors="pt", max_length=self.max_length)
        query_passages = {k:v.view(bz, nway, -1) for k, v in query_passages.items()}
        labels = torch.FloatTensor(labels)
        
        return {
            "qid": qids,
            "relT_pids": relT_pids,
            "neg_pids": neg_pids,
            "nway_pids": np.concatenate((relT_pids, neg_pids), axis=-1),
            "query_passages": query_passages,
            "labels": labels
        }            


    @classmethod
    def create_from_10relT_20neg_file(cls, queries_path, passages_path, training_path, tokenizer, max_length, 
                                    label_mode, rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1
        assert label_mode in ["3", "9"]

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
                    new_example = {}
                    example = ujson.loads(line)
                    train_examples.append({
                        "qid": example["qid"],
                        "relT_pids": example["relT_pids"],
                        "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"],
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
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
                        })
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_length, label_mode)
    
    @classmethod
    def create_from_20relT_10neg_file(cls, queries_path, passages_path, training_path, tokenizer, max_length, 
                                    label_mode, rank=-1, nranks=None):
        # canbe 20relT_10neg or 20T_10neg
        if rank != -1:
            assert rank in range(nranks) and nranks > 1
        assert label_mode in ["5", "10"]

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
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
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
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
                        })
            
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_length, label_mode)

    @classmethod
    def create_from_30relT_file(cls, queries_path, passages_path, training_path, tokenizer, max_length, 
                                    label_mode, rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1
        assert label_mode == "6"

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
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
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
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
                        })
            
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_length, label_mode)

    @classmethod
    def create_from_5relT_25neg_file(cls, queries_path, passages_path, training_path, tokenizer, max_length, 
                                    label_mode, rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1
        assert label_mode in ["7", "8"]

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
                    new_example = {}
                    example = ujson.loads(line)
                    train_examples.append({
                        "qid": example["qid"],
                        "relT_pids": example["relT_pids"],
                        "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"],
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
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
                        })
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_length, label_mode)
    
if __name__ == "__main__":
    import sys 
    sys.path += ["./"]
    
    from transformers import HfArgumentParser, AutoTokenizer
    from arguments import CETrainArguments, CEModelArguments
    import ujson

    parser = HfArgumentParser((CETrainArguments, CEModelArguments))
    args, model_args = parser.parse_args_into_dataclasses()

    queries_path="/work/hzeng_umass_edu/datasets/msmarco-passage/queries.train.tsv"
    collection_path="/work/hzeng_umass_edu/datasets/msmarco-passage/collection.tsv"
    training_path="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/train_examples/10relT_20neg_ce0_de250000.train.json"
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    dataset = CENwayDataset.create_from_10relT_20neg_file(queries_path, collection_path, training_path, tokenizer, max_length=300, label_mode="9")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
    
    for b_idx, batch in enumerate(dataloader):
        for loop_idx, qp_token_ids in enumerate(batch["query_passages"]["input_ids"]):
            print("-----query-----")
            print("qid: ", batch["qid"][loop_idx])
            print("nway_pids: \n", batch["nway_pids"][loop_idx])
            print("labels: \n", batch["labels"][loop_idx])
            for idx in range(len(qp_token_ids)):
                token_ids = qp_token_ids[idx]
                print("query_passages: \n", tokenizer.decode(token_ids, skip_special_tokens=True))
                if idx == 3:
                    break
        if b_idx == 2:
            break