import os 
import sys 
sys.path += ["./"]
from pathlib import Path 
from typing import Dict, List, Tuple
import logging
logger = logging.getLogger(__name__)
import time
import ujson
from itertools import chain

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset 


class KGCTripleDataset(torch.utils.data.Dataset):
    def __init__(self, hid_to_head, tid_to_tail, train_examples, 
                        tokenizer, max_head_text_len, max_tail_text_len):
        super(KGCTripleDataset, self).__init__()
        self.hid_to_head = hid_to_head
        self.tid_to_tail = tid_to_tail
        self.train_examples = train_examples
        self.tokenizer = tokenizer
        self.max_head_text_len = max_head_text_len
        self.max_tail_text_len = max_tail_text_len 
        
    
    def __getitem__(self, idx):
        hid, reltid, negtid, relation = self.train_examples[idx]

        head_text = self.hid_to_head[hid]
        rel_tail_text = self.tid_to_tail[reltid] 
        neg_tail_text = self.tid_to_tail[negtid] 

        return {
            "hid": hid,
            "reltid": reltid,
            "negtid": negtid,
            "head_text": head_text,
            "rel_tail_text": rel_tail_text,
            "neg_tail_text": neg_tail_text,
            "relation": relation
        }

    def __len__(self):
        return len(self.train_examples)

    def collate_fn(self, batch):
        hids, rel_tids, neg_tids, head_texts, tail_texts, relations = [], [], [], [], [], []
        for idx, elem in enumerate(batch):
            hids.append(elem["hid"])
            rel_tids.append(elem["reltid"])
            neg_tids.append(elem["negtid"])
            head_texts.append(elem["head_text"]) 
            assert type(elem["rel_tail_text"]) == str and type(elem["neg_tail_text"]) == str
            tail_texts +=  [elem["rel_tail_text"], elem["neg_tail_text"]]
            relations.append(elem["relation"])

        
        hids =  np.array(hids, dtype=np.int64)
        rel_tids = np.array(rel_tids, dtype=np.int64)
        neg_tids = np.array(neg_tids, dtype=np.int64)

        hr_texts = self.tokenizer(head_texts, text_pair=relations, padding=True, truncation='longest_first', return_token_type_ids=True,
                                        return_tensors="pt", max_length=self.max_head_text_len) # [bz]
        tail_texts = self.tokenizer(tail_texts, padding=True, truncation='longest_first', return_token_type_ids=True,
                                        return_tensors="pt", max_length=self.max_tail_text_len) # [2*bz]
        
        assert len(tail_texts["input_ids"]) == 2 * len(hr_texts["input_ids"])
        bz = len(hr_texts["input_ids"])
        tail_texts = {k:v.view(bz, 2, -1) for k, v in tail_texts.items()}
        
        return {
            "hid": hids,
            "reltid": rel_tids,
            "negtid": neg_tids,
            "hr_texts": hr_texts,
            "tail_texts": tail_texts,
        }            
             
    @classmethod
    def create_from_triple_file(cls, heads_path, tails_path, training_path, tokenizer, max_head_text_len, max_tail_text_len, 
                                    rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1

        hid_to_head = {}
        with open(heads_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                assert len(array) == 3
                hid, head_text,  = int(array[0]), array[1]
                hid_to_head[hid] = head_text
        
        tid_to_tail = {}
        with open(tails_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                assert len(array) == 2
                
                tid, tail_text = int(array[0]), array[1]
                tid_to_tail[tid] = tail_text 
                
        if rank == -1:
            train_examples = []
            with open(training_path, "r") as fin:
                for line in fin:
                    hid, reltid, negtid, relation = line.rstrip().split("\t")
                    hid, reltid, negtid, relation = int(hid), int(reltid), int(negtid), relation
                    
                    train_examples.append((hid, reltid, negtid, relation))
        else:
            train_examples = []
            with open(training_path, "r") as fin:
                for line_idx, line in enumerate(fin):
                    if line_idx % nranks == rank:
                        hid, reltid, negtid, relation = line.rstrip().split("\t")
                        hid, reltid, negtid, relation = int(hid), int(reltid), int(negtid), relation

                        train_examples.append((hid, reltid, negtid, relation))
        return cls(hid_to_head, tid_to_tail, train_examples, tokenizer, max_head_text_len, max_tail_text_len)
    
    @classmethod
    def create_from_six_triple_files(cls, entites_path, tokenizer, max_head_text_len, max_tail_text_len, 
                                     a2cp_path, a2sp_path=None, q2a_path=None, c2cp_path=None, c2sp_path=None, q2c_path=None,
                                     rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1
            
        eid_to_entity = {}
        with open(entites_path) as fin:
            for line in fin:
                array = line.rstrip().split("\t")
                assert len(array) == 2
                eid_to_entity[int(array[0])] = array[1]
                
        all_paths = [path for path in [a2cp_path, a2sp_path, q2a_path, c2cp_path, c2sp_path, q2c_path] if path is not None]
        if rank <= 0:
            print("all_paths: ", all_paths, len(all_paths))
        
        if rank == -1:
            train_examples = []
            for training_path in all_paths:
                with open(training_path) as fin:
                    for line_idx, line in enumerate(fin):
                        hid, reltid, negtid, relation = line.rstrip().split("\t")
                        hid, reltid, negtid, relation = int(hid), int(reltid), int(negtid), relation
                        train_examples.append((hid, reltid, negtid, relation))
        else:
            train_examples = []
            for training_path in all_paths:
                with open(training_path) as fin:
                    for line_idx, line in enumerate(fin):
                        if line_idx % nranks == rank:
                            hid, reltid, negtid, relation = line.rstrip().split("\t")
                            hid, reltid, negtid, relation = int(hid), int(reltid), int(negtid), relation
                            train_examples.append((hid, reltid, negtid, relation))
                            
        return cls(eid_to_entity, eid_to_entity, train_examples, tokenizer, max_head_text_len, max_tail_text_len)



if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch

    entites_path = "/home/jupyter/jointly_rec_and_search/datasets/kgc/all_entites.tsv"
    a2cp_path = "/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.a2cp.tsv"
    a2sp_path = "/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.a2sp.rnd.tsv"
    q2a_path = "/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.q2a.tsv"
    c2cp_path = "/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.c2cp.tsv"
    c2sp_path = "/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.c2sp.rnd.tsv"
    q2c_path = "/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.q2c.tsv"
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
   
    dataset = KGCTripleDataset.create_from_six_triple_files(entites_path, tokenizer, 64, 64,
                                                           a2cp_path, a2sp_path, q2a_path, c2cp_path, c2sp_path, q2c_path)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for b_idx, batch in enumerate(dataloader):
        for loop_idx, (q_token_ids, bi_token_ids) in enumerate(zip(batch["hr_texts"]["input_ids"], 
                                                            batch["tail_texts"]["input_ids"])):
            print("-----head-----")
            print("hid: ", batch["hid"][loop_idx])
            print("hr_text: ", tokenizer.decode(q_token_ids))
            print("reltid: ", batch["reltid"][loop_idx], "negtid: ", batch["negtid"][loop_idx])
            print("bi_token_ids shape = {}".format(bi_token_ids.shape))
            print("rel tail_text: ", tokenizer.decode(bi_token_ids[0]))
            print("neg tail_text: ", tokenizer.decode(bi_token_ids[1]))
        if b_idx == 5:
            break
            
    print("="*150)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn)

    for b_idx, batch in enumerate(dataloader):
        for loop_idx, (q_token_ids, bi_token_ids, hr_token_type_ids, tail_token_type_ids) in enumerate(zip(batch["hr_texts"]["input_ids"], 
                                                            batch["tail_texts"]["input_ids"], batch["hr_texts"]["token_type_ids"],
                                                            batch["tail_texts"]["token_type_ids"])):
            print("-----head-----")
            print("hid: ", batch["hid"][loop_idx])
            print(tokenizer.decode(q_token_ids, skip_special_tokens=True))
            print("reltid: ", batch["reltid"][loop_idx], "negtid: ", batch["negtid"][loop_idx])
            print("bi_token_ids shape = {}".format(bi_token_ids.shape))
            print("rel tail_text: ", tokenizer.decode(bi_token_ids[0]))
            print("neg tail_text: ", tokenizer.decode(bi_token_ids[1]))
            print("token_type_ids: ", hr_token_type_ids, tail_token_type_ids)
        if b_idx == 5:
            break
