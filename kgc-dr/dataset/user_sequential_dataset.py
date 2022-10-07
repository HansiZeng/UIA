import os
from tracemalloc import is_tracing

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import ujson 
import numpy as np

SIM_RELATION = "is_similar_to"
COMPL_RELATION = "is_complementary_to"
REL_RELATION = "is_relevant_to"

RELATION_TO_RELID = {
    SIM_RELATION: 0,
    COMPL_RELATION: 1,
    REL_RELATION: 2
}


class UserSequentialDataset(Dataset):
    def __init__(self, eid_to_text, data_examples, tokenizer, max_text_len, apply_zero_attention, is_train) :
        super().__init__()
        self.eid_to_text = eid_to_text
        self.data_examples = data_examples
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.apply_zero_attention = apply_zero_attention
        self.is_train = is_train

    def __len__(self):
        return len(self.data_examples)

    def __getitem__(self, idx):
        example = self.data_examples[idx]
        query_texts = [self.eid_to_text[eid] for eid in example["query_ids"]]
        context_key_texts = [self.eid_to_text[eid] for eid in example["context_key_ids"]]
        context_value_texts = [self.eid_to_text[eid] for eid in example["context_value_ids"]]

        if self.is_train:
            target_value_texts = [self.eid_to_text[eid] for eid in example["target_value_ids"]]
            neg_value_texts = [self.eid_to_text[eid] for eid in example["neg_value_ids"]]
            
            return {
                "uid": example["uid"],
                "query_ids": example["query_ids"],
                "context_key_ids": example["context_key_ids"],
                "context_value_ids": example["context_value_ids"],
                "query_texts": query_texts,
                "context_key_texts": context_key_texts,
                "context_value_texts": context_value_texts,
                
                "relation": example["relation"] if "relation" in example else example["relations"],

                "target_value_ids": example["target_value_ids"],
                "neg_value_ids": example["neg_value_ids"],
                "target_value_texts": target_value_texts,
                "neg_value_texts": neg_value_texts,
            }
        else:
            return {
                "uid": example["uid"],
                "query_ids": example["query_ids"],
                "context_key_ids": example["context_key_ids"],
                "context_value_ids": example["context_value_ids"],
                "query_texts": query_texts,
                "context_key_texts": context_key_texts,
                "context_value_texts": context_value_texts,
                
                "relation": example["relation"] if "relation" in example else example["relations"],
            }


    @staticmethod
    def get_in_batch_masked_indices(seq_lengths):
        """
        @concrete_example can be seq_lengths = [2,3,4,1] 
        @concrete_example return: ([0,1,2,2,3,3,4,4,5,5,5,6,6,6,7,7,7,8,8,8], [1,0,3,4,2,4,2,3,6,7,8,5,7,8,5,6,8,5,8,7])
        """
        group_end_idxes = np.cumsum(seq_lengths).tolist()
        group_start_idxes = [0] + group_end_idxes[:-1]
        repeat_times = [seq_lengths[i]-1 for i in range(len(seq_lengths))]

        cur_group = 0
        x_indices = []
        grouped_idxes = [[] for _ in range(len(seq_lengths))]
        for i in range(sum(seq_lengths)):
            if i >= group_end_idxes[cur_group]:
                cur_group += 1
            x_indices += [i] * repeat_times[cur_group]
            grouped_idxes[cur_group].append(i)

        cur_group = 0
        y_indices = []
        for i in range(sum(seq_lengths)):
            if i >= group_end_idxes[cur_group]:
                cur_group += 1
            cur_group_idxes = grouped_idxes[cur_group]
            ignore_idx = i - group_start_idxes[cur_group]
            y_indices += [cur_group_idxes[j] for j in range(len(cur_group_idxes)) if j != ignore_idx]

        x_indices = torch.LongTensor(x_indices)
        y_indices = torch.LongTensor(y_indices)
        return (x_indices, y_indices)

    def collate_fn(self, batch):
        uids, query_ids, context_key_ids, context_value_ids = [], [], [], []
        query_texts, context_key_texts, context_value_texts = [], [], []
        relations, seq_lengths, user_ids, relation_ids = [], [], [], []
        if self.is_train:
            target_value_ids, target_value_texts = [], []
            neg_value_ids, neg_value_texts = [], []

        for elem in batch:
            uids.append(elem["uid"])
            user_ids.extend([elem["uid"]]*len(elem["query_ids"]))
            query_ids += elem["query_ids"]
            context_key_ids += elem["context_key_ids"]
            context_value_ids += elem["context_value_ids"]
            
            assert len(elem["query_ids"]) == len(elem["context_key_ids"]) == len(elem["context_value_ids"])
            if type(elem["relation"]) != list:
                relation_ids.extend([RELATION_TO_RELID[elem["relation"]]]*len(elem["query_ids"]))
                relations += [elem["relation"]] * len(elem["query_ids"]) 
            else:
                assert len(elem["relation"]) == len(elem["query_ids"])
                relation_ids.extend([RELATION_TO_RELID[rel] for rel in elem["relation"]])
                relations += elem["relation"]
            seq_lengths.append(len(elem["query_ids"]) )

            query_texts += elem["query_texts"]
            context_key_texts += elem["context_key_texts"]
            context_value_texts += elem["context_value_texts"]
            if self.is_train:
                target_value_ids += elem["target_value_ids"]
                neg_value_ids += elem["neg_value_ids"]
                target_value_texts += elem["target_value_texts"]
                neg_value_texts += elem["neg_value_texts"]
        
        batch_size, max_length = len(batch), max(seq_lengths)
        id_masks = torch.zeros(batch_size, max_length) #[bz, max_length]
        for length, xs in zip(seq_lengths, id_masks):
            xs[:length] = 1. 
        id_masks = id_masks.unsqueeze(-1) #[bz, max_length, 1]
        casual_masks = torch.tril(torch.ones(max_length, max_length)) #[max_length, max_length]
        if self.apply_zero_attention:
            casual_masks = torch.cat((torch.ones(max_length).view(-1,1), casual_masks), dim=1) # [max_length, max_length+1]
        id_masks = id_masks * casual_masks.unsqueeze(0)

        in_batch_mask = torch.ones(sum(seq_lengths), sum(seq_lengths))
        masked_indices = self.get_in_batch_masked_indices(seq_lengths=seq_lengths)
        in_batch_mask[masked_indices] = 0. 

        query_relation_texts = self.tokenizer(query_texts, text_pair=relations, padding=True, truncation='longest_first', 
                                            return_token_type_ids=True, return_tensors="pt", max_length=self.max_text_len) 
        context_key_relation_texts = self.tokenizer(context_key_texts, text_pair=relations, padding=True, truncation='longest_first', 
                                            return_token_type_ids=True, return_tensors="pt", max_length=self.max_text_len)
        context_value_texts = self.tokenizer(context_value_texts, padding=True, truncation='longest_first', 
                                    return_token_type_ids=True, return_tensors="pt", max_length=self.max_text_len)
        if self.is_train:
            target_value_texts = self.tokenizer(target_value_texts, padding=True, truncation='longest_first', 
                                        return_token_type_ids=True, return_tensors="pt", max_length=self.max_text_len)
            neg_value_texts = self.tokenizer(neg_value_texts, padding=True, truncation='longest_first', 
                                        return_token_type_ids=True, return_tensors="pt", max_length=self.max_text_len)

            return {
                "uids": torch.LongTensor(uids),
                "user_ids": torch.LongTensor(user_ids),
                "relation_ids": torch.LongTensor(relation_ids),
                "query_ids": torch.LongTensor(query_ids),
                "context_key_ids": torch.LongTensor(context_key_ids),
                "context_value_ids": torch.LongTensor(context_value_ids),
                "query_relation": query_relation_texts,
                "context_key_relation": context_key_relation_texts,
                "context_value": context_value_texts,
                
                "seq_lengths": torch.LongTensor(seq_lengths),
                "id_attention_masks": id_masks,
                "in_batch_mask": in_batch_mask,

                "target_value_ids": torch.LongTensor(target_value_ids),
                "neg_value_ids": torch.LongTensor(neg_value_ids),
                "target_value": target_value_texts,
                "neg_value": neg_value_texts,
            }
        else:
            return {
                "uids": torch.LongTensor(uids),
                "user_ids": torch.LongTensor(user_ids),
                "relation_ids": torch.LongTensor(relation_ids),
                "query_ids": torch.LongTensor(query_ids),
                "context_key_ids": torch.LongTensor(context_key_ids),
                "context_value_ids": torch.LongTensor(context_value_ids),
                "query_relation": query_relation_texts,
                "context_key_relation": context_key_relation_texts,
                "context_value": context_value_texts,
                
                "seq_lengths": torch.LongTensor(seq_lengths),
                "id_attention_masks": id_masks,
                "in_batch_mask": in_batch_mask,
            }

    @classmethod
    def create_from_json_file(cls, eid_path, examples_path, tokenizer, max_text_len, 
                            apply_zero_attention, is_train=True,
                            rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1
        
        eid_to_text = {}
        with open(eid_path) as fin:
            for line in fin:
                eid, text = line.strip().split("\t")
                eid_to_text[int(eid)] = text

        data_examples = []
        if rank == -1:
            with open(examples_path) as fin:
                for line in fin:
                    data_examples.append(ujson.loads(line))
        else:
            with open(examples_path) as fin:
                for line_idx, line in enumerate(fin):
                    if line_idx % nranks == rank:
                        data_examples.append(ujson.loads(line))

        return cls(eid_to_text, data_examples, tokenizer, max_text_len, apply_zero_attention, is_train)


       

if __name__ == "__main__":
    from transformers import AutoTokenizer

    eid_path = "/work/hzeng_umass_edu/ir-research/joint_modeling_search_and_rec/datasets/unified_kgc/all_entities.tsv"
    examples_path = "/work/hzeng_umass_edu/ir-research/joint_modeling_search_and_rec/datasets/unified_kgc/unified_user/sequential_train_test/hlen_4_randneg/search_sequential.train.json"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = UserSequentialDataset.create_from_json_file(eid_path, examples_path, tokenizer, 50, apply_zero_attention=True, 
                                                        is_train=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    # special case examination
    seq_lengths = [2,3,1,1,2]
    print(dataset.get_in_batch_masked_indices(seq_lengths))

    for b_idx, batch in enumerate(dataloader):
        """
        #print(batch["seq_lengths"] < 12)
        if any(batch["seq_lengths"] < 12):
            print("q, k, v ids: {}, {}, {}".format(batch["query_ids"].tolist(), batch["context_key_ids"].tolist(), batch["context_value_ids"].tolist()))
            print("id masks: {}".format(batch["id_attention_masks"]))
            print("seq lengths: {}".format(batch["seq_lengths"]))
            print("in_batch_mask: {}, {}".format(batch["in_batch_mask"], torch.sum(batch["in_batch_mask"], dim=-1)))
            print(75*"=")

            break
        """
        query_inputs = batch["query_relation"]["input_ids"]
        target_value_inputs = batch["target_value"]["input_ids"]
        context_key_inputs = batch["context_key_relation"]["input_ids"]
        context_value_inputs = batch["context_value"]["input_ids"]

        for query_input, target_input, context_key, context_value  in zip(query_inputs, target_value_inputs,
                                                                        context_key_inputs, context_value_inputs):
            print("query: ", tokenizer.decode(query_input, skip_special_tokens=True))
            print("target value: ", tokenizer.decode(target_input, skip_special_tokens=True))
            print("most recent context key: ", tokenizer.decode(context_key, skip_special_tokens=True))
            print("most recent context value: ", tokenizer.decode(context_value, skip_special_tokens=True))
            print(75*"=")

        break

    
