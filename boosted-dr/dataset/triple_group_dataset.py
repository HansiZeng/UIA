import os 
import sys 
sys.path += ["./"]
from pathlib import Path 
from typing import Dict, List, Tuple
import logging
logger = logging.getLogger(__name__)
import time
import ujson
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset 

class TripleGroupDataset(torch.utils.data.Dataset):
    def __init__(self, aid_to_anchor, pid_to_passage, train_examples, 
                        tokenizer, max_anchor_len, max_passage_len):
        super(TripleGroupDataset, self).__init__()
        self.aid_to_anchor = aid_to_anchor
        self.pid_to_passage = pid_to_passage
        self.train_examples = train_examples
        self.tokenizer = tokenizer
        self.max_anchor_len = max_anchor_len
        self.max_passage_len = max_passage_len 
        
    
    def __getitem__(self, idx):
        aid, compl_pids, sim_pids = self.train_examples[idx]
        
        anchor = self.aid_to_anchor[aid]
        compl_passages = [self.pid_to_passage[cpid] for cpid in compl_pids] 
        sim_passages = [self.pid_to_passage[spid] for spid in sim_pids] 
        passage_num = len(compl_passages)

        return {
            "aid": aid,
            "compl_pids": compl_pids,
            "sim_pids": sim_pids,
            "anchor": anchor,
            "compl_passages": compl_passages,
            "sim_passages": sim_passages,
            "passage_num": passage_num
        }

    def __len__(self):
        return len(self.train_examples)

    def collate_fn(self, batch):
        aids, compl_pids, sim_pids, anchors, compl_passages, sim_passages, passage_nums = [], [], [], [], [], [], []
        for idx, elem in enumerate(batch):
            aids.append(elem["aid"])
            compl_pids += elem["compl_pids"]
            sim_pids += elem["sim_pids"]
            anchors.append(elem["anchor"])
            compl_passages += elem["compl_passages"]
            sim_passages += elem["sim_passages"]
            passage_nums.append(elem["passage_num"])

        
        aids =  np.array(aids, dtype=np.int64)
        compl_pids = np.array(compl_pids, dtype=np.int64)
        sim_pids = np.array(sim_pids, dtype=np.int64)
        assert len(compl_passages) == len(sim_passages) == sum(passage_nums), (len(compl_passages), len(sim_passages))
        
        anchor_indices = list(itertools.chain.from_iterable([[idx]*passage_nums[idx] for idx in range(len(passage_nums))]))
        anchor_indices = torch.LongTensor(anchor_indices)
        anchors = self.tokenizer(anchors, padding=True, truncation='longest_first', 
                                        return_tensors="pt", max_length=self.max_anchor_len) # [bz]
        compl_passages = self.tokenizer(compl_passages, padding=True, truncation='longest_first', 
                                        return_tensors="pt", max_length=self.max_passage_len) 
        sim_passages = self.tokenizer(sim_passages, padding=True, truncation='longest_first', 
                                        return_tensors="pt", max_length=self.max_passage_len)
        passage_nums = torch.LongTensor(passage_nums)
                
        return {
            "aids": aids,
            "compl_pids": compl_pids,
            "sim_pids": sim_pids,
            "anchors": anchors,
            "compl_passages": compl_passages,
            "sim_passages": sim_passages,
            "passage_nums": passage_nums,
            "anchor_indices": anchor_indices
        }            
             
    @classmethod
    def create_from_json_file(cls, anchors_path, passages_path, training_path, tokenizer, max_anchor_len, max_passage_len, 
                                    rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1

        aid_to_anchor = {}
        with open(anchors_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                aid, anchor = int(array[0]), array[1]
                aid_to_anchor[aid] = anchor 
        
        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                assert len(array) == 2, array
                pid, passage = int(array[0]), array[1]
                pid_to_passage[pid] = passage 

        if rank == -1:
            train_examples = []
            with open(training_path, "r") as fin:
                for line in fin:
                    example = ujson.loads(line.rstrip())
                    aid = example["aid"]
                    compl_pids = example["compl_pids"]
                    sim_pids = example["sim_pids"]
                    assert len(compl_pids) == len(sim_pids), (len(compl_pids), len(sim_pids))
                    
                    train_examples.append((aid, compl_pids, sim_pids))
        else:
            train_examples = []
            with open(training_path, "r") as fin:
                for line_idx, line in enumerate(fin):
                    if line_idx % nranks == rank:
                        example = ujson.loads(line.rstrip())
                        aid = example["aid"]
                        compl_pids = example["compl_pids"]
                        sim_pids = example["sim_pids"]
                        assert len(compl_pids) == len(sim_pids), (len(compl_pids), len(sim_pids))

                        train_examples.append((aid, compl_pids, sim_pids))
                    
        return cls(aid_to_anchor, pid_to_passage, train_examples, tokenizer, max_anchor_len, max_passage_len)



if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch

    anchors_path = "/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec_compl/anchors_title_catalog.train.tsv"
    passages_path = "/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec_compl/collection_title_catalog.tsv"
    training_path = "/home/jupyter/jointly_rec_and_search/datasets/rec_search/rec_compl/train/train_5compl_5sim.json"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   
    dataset = TripleGroupDataset.create_from_json_file(anchors_path, passages_path, training_path, tokenizer, max_anchor_len=64, max_passage_len=64)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for b_idx, batch in enumerate(dataloader):
        print("-----anchor-----")
        print("aids: ", batch["aids"])
        print("compl_pids: ", batch["compl_pids"])
        print("sim_pids: ", batch["sim_pids"])
        print("passage_nums: ", batch["passage_nums"])
        print("anchor_indices: ", batch["anchor_indices"])
        for loop_idx, (compl_token_ids, sim_token_ids) in enumerate(zip(batch["compl_passages"]["input_ids"], 
                                                                                     batch["sim_passages"]["input_ids"])):
            print("anchor: ", tokenizer.decode(batch["anchors"]["input_ids"][batch["anchor_indices"][loop_idx]], skip_special_tokens=True))
            print("compl passage: ", tokenizer.decode(compl_token_ids, skip_special_tokens=True))
            print("sim passage: ", tokenizer.decode(sim_token_ids, skip_special_tokens=True))
            print("-"*50)
            if loop_idx >= 4:
                break
        if b_idx == 5:
            break
