from torch.utils.data import DataLoader, Dataset
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import torch

import transformers

class DualEncoderRerankDataset(Dataset):
    def __init__(self, eid_to_text, train_examples, tokenizer,
                 query_max_length, passage_max_length):
        super(DualEncoderRerankDataset, self).__init__()
        
        self.eid_to_text = eid_to_text
        self.train_examples = train_examples
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        
    def __len__(self):
        return len(self.train_examples)
    
    def __getitem__(self, idx):
        example = self.train_examples[idx]
        
        qid, pid, label, relation = example
        query = self.eid_to_text[qid]
        passage = self.eid_to_text[pid]
        
        return {
            "query": query,
            "passage": passage,
            "label": label,
            
            "qid": qid,
            "pid": pid,
            "relation": relation
        } 
    
    def collate_fn(self, batch):
        queries, passages, labels, qids, pids, relations = [], [], [], [], [], []
        for elem in batch:
            queries.append(elem["query"])
            passages.append(elem["passage"])
            labels.append(elem["label"])
            qids.append(elem["qid"])
            pids.append(elem["pid"])
            relations.append(elem["relation"])
            
        tokenized_queries = self.tokenizer(queries, text_pair=relations,padding=True, 
                                   truncation='longest_first', return_tensors="pt", 
                                   max_length=self.query_max_length)
        tokenized_passages = self.tokenizer(passages, padding=True, truncation='longest_first', return_token_type_ids=True,
                                        return_tensors="pt", max_length=self.passage_max_length)
        labels = torch.tensor(labels, dtype=torch.float)
        
        return tokenized_queries, tokenized_passages, labels, {"qids": qids, "pids": pids, "relations": relations}
    
    
    @classmethod
    def create_from_pair_file(cls, entities_path, train_examples_path, tokenizer, query_max_length,
                             passage_max_length):
        eid_to_text = {}
        with open(entities_path) as fin:
            for line in fin:
                eid, text = line.strip().split("\t")
                eid_to_text[int(eid)] = text
                
        train_examples = []
        with open(train_examples_path) as fin:
            for line in fin:
                qid, pid, label, relation = line.strip().split("\t")
                example = (int(qid), int(pid), float(label), relation)
                train_examples.append(example)
                
        return cls(eid_to_text, train_examples, tokenizer, query_max_length, passage_max_length)