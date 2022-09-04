import os 
import math

import torch
import torch.nn as nn 

from transformers import (AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup)
from losses import ContrastiveLoss

class DualEncoder(nn.Module):
    def __init__(self, model_args, add_pooling_layer = False):
        super().__init__()
        self.model_args = model_args

        self.model_name_or_path = self.model_args.model_name_or_path
        self.apply_tanh = self.model_args.apply_tanh
        self.independent_encoders = self.model_args.independent_encoders

        self.bert = AutoModel.from_pretrained(self.model_name_or_path, add_pooling_layer=add_pooling_layer)
        if self.independent_encoders:
            self.query_bert = AutoModel.from_pretrained(self.model_name_or_path, add_pooling_layer=add_pooling_layer)
        else:
            self.query_bert = self.bert
        

    def forward(self, queries, nway_passages, is_nway=True):
        """
        queries: each value shape: [bz, seq_len]
        nway_passage : Dict. each value shape: [bz, nway, seq_len]
        """
        query_reps = self.query_embs(queries) # [bz, D]
        if is_nway:
            nway_passage_reps = self.nway_passage_embs(nway_passages) #[bz, nway, D]
            assert query_reps.dim() == 2 and nway_passage_reps.dim() == 3
            logits = torch.sum(query_reps.unsqueeze(1) * nway_passage_reps, dim=-1) #[bz, nway]
        else:
            passage_reps = self.passage_embs(nway_passages)
            assert query_reps.dim() == 2 and passage_reps.dim() == 2
            logits = torch.sum(query_reps * passage_reps, dim=-1).view(-1) #[bz]
        
        return logits

    def query_embs(self, queries):
        if self.independent_encoders:
            query_reps = self.query_bert(**queries).last_hidden_state[:, 0]
        else:
            query_reps = self.bert(**queries).last_hidden_state[:, 0]

        return query_reps

    def passage_embs(self, passages):
        passage_reps = self.bert(**passages).last_hidden_state[:,0]

        return passage_reps

    def nway_passage_embs(self, nway_passages):
        input_ids, attention_mask, token_type_ids = nway_passages["input_ids"], nway_passages["attention_mask"], nway_passages["token_type_ids"]
        bz, nway, seq_len = input_ids.shape 

        input_ids, attention_mask, token_type_ids = input_ids.view(bz*nway, seq_len), attention_mask.view(bz*nway, seq_len), token_type_ids.view(bz*nway, seq_len)
        passage_reps = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        passage_reps = passage_reps.view(bz, nway, -1)

        return passage_reps

    @classmethod
    def from_pretrained(cls, model_path, model_args=None):
        if os.path.isdir(model_path):
            ckpt_path = os.path.join(model_path, "model.pt")
            model_args_path = os.path.join(model_path, "model_args.pt")
            model_args = torch.load(model_args_path)
        elif os.path.isfile(model_path):
            assert model_args != None 
            ckpt_path = model_path
        else:
            raise ValueError("model_path: {} is not expected".format(model_path))
        
        model = cls(model_args)
        print("load pretrained model from local path {}".format(model_path))

        model_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(model_dict)

        return model 

    def save_pretrained(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        model_to_save = self
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, "model.pt"))
        torch.save(self.model_args, os.path.join(save_dir, "model_args.pt"))

    def get_output_dim(self):
        return self.bert.config.hidden_size
    
    
        
class ContrastiveDualEncoder(DualEncoder):
    def __init__(self, model_args):
        super().__init__(model_args)
        self.loss_fn = ContrastiveLoss()
        
    def forward(self, queries, passages):
        """
        Args:
            queries: [bz, seq_len]
            passages: [bz, nway, seq_len]
        """
        query_reps = self.query_embs(queries) 
        passage_reps = self.nway_passage_embs(passages)
        
        bz, nway, dim = passage_reps.size()
        passage_reps = passage_reps.view(bz*nway, dim)
        
        return self.loss_fn(query_reps, passage_reps)

          
    