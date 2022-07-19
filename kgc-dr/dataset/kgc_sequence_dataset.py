import os 
import sys 
from pathlib import Path 
from typing import Union, List, Dict

from torch.utils.data import DataLoader, Dataset 

class KGCSequenceDataset(Dataset):
    def __init__(self, id_seqs, tokenizer, max_length):
        super(KGCSequenceDataset, self).__init__()
        self.tokenizer = tokenizer 
        self.max_length = max_length

        self.id_seqs = id_seqs

    def __getitem__(self, idx):
        array_len = len(self.id_seqs[0])
        if array_len == 2:
            sid, seq = self.id_seqs[idx]
        elif array_len == 3:
            sid, seq, relation = self.id_seqs[idx]
        else:
            raise ValueError(f"array_len = {array_len} is not legal.")
        
        if array_len == 2:
            return {
                "seq": seq,
                "id": sid 
            }
        else:
            return {
                "seq": seq,
                "relation": relation,
                "id": sid 
            }
                
    
    def __len__(self):
        return len(self.id_seqs)

    @classmethod
    def create_from_seqs_file(cls, seqs_file, tokenizer, max_length, rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1
        
        # make sure not duplicate id 
        set_ids = set()
        if rank == -1:
            id_seqs = []
            with open(seqs_file, "r") as f:
                for line in f:
                    array = line.strip().split("\t")
                    if len(array) == 2:
                        sid, seq = line.strip().split("\t")
                        id_seqs.append((int(sid), seq))
                    elif len(array) == 3:
                        sid, seq, relation = line.strip().split("\t")
                        id_seqs.append((int(sid), seq, relation))
                    else:
                        raise ValueError(f"{array} doesn't have valid length. It should be 2 or 3.")
                    set_ids.add(sid)
        else:
            id_seqs = []
            with open(seqs_file, "r") as f:
                for line_idx, line in enumerate(f):
                    if line_idx % nranks == rank:
                        array = line.strip().split("\t")
                        if len(array) == 2:
                            sid, seq = line.strip().split("\t")
                            id_seqs.append((int(sid), seq))
                        elif len(array) == 3:
                            sid, seq, relation = line.strip().split("\t")
                            id_seqs.append((int(sid), seq, relation))
                        else:
                            raise ValueError(f"{array} doesn't have valid length. It should be 2 or 3.")
                        set_ids.add(sid)
                        
        assert len(set_ids) == len(id_seqs), (len(set_ids), len(id_seqs))

        return cls(id_seqs, tokenizer, max_length)

    def collate_fn(self, batch):
        ids, seqs, relations = [], [], []
        for elem in batch:
            seqs.append(elem["seq"])
            ids.append(elem["id"])
            if "relation" in elem:
                relations.append(elem["relation"])
        
        if len(relations) == 0:
            seqs = self.tokenizer(seqs, padding=True, truncation='longest_first', 
                                return_tensors="pt", max_length=self.max_length, return_token_type_ids=True)
            return {
                "seq": seqs,
                "id": ids
            }
        else:
            assert len(relations) == len(ids), (len(relations), len(ids))
            seqs = self.tokenizer(seqs, text_pair=relations, padding=True, truncation='longest_first', return_token_type_ids=True,
                                        return_tensors="pt", max_length=self.max_length)
            return {
                "seq": seqs,
                "id": ids
            }
        
        
if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch
    
    #seqs_path = "/home/jupyter/jointly_rec_and_search/datasets/kgc/test/anchors.test.tsv"
    seqs_path = "/home/jupyter/jointly_rec_and_search/datasets/kgc/collection_title_catalog.tsv"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = KGCSequenceDataset.create_from_seqs_file(seqs_path, tokenizer, max_length=64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
    
    for b_idx, batch in enumerate(dataloader):
        for i, (token_ids, token_type_ids) in enumerate(zip(batch["seq"]["input_ids"], batch["seq"]["token_type_ids"])):
            print(tokenizer.decode(token_ids))
            print(token_type_ids)
            print("-"*75)
        if b_idx == 5:
            break
            