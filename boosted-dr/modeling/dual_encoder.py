import os 
import math

import torch
import torch.nn as nn 
import torch_geometric
from torch_scatter import scatter_add

from transformers import (AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup)
from losses import ContrastiveLoss

class DualEncoder(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args

        self.model_name_or_path = self.model_args.model_name_or_path
        self.use_compress = self.model_args.use_compress 
        self.apply_tanh = self.model_args.apply_tanh
        self.independent_encoders = self.model_args.independent_encoders

        self.bert = AutoModel.from_pretrained(self.model_name_or_path)
        if self.independent_encoders:
            self.query_bert = AutoModel.from_pretrained(self.model_name_or_path)
        if self.use_compress:
            self.compress_dim = self.model_args.compress_dim
            self.compressor = nn.Linear(self.bert.config.hidden_size, self.compress_dim)
        

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
            query_reps = self.query_bert(**queries)[0][:, 0, :] 
        else:
            query_reps = self.bert(**queries)[0][:, 0, :] 
        
        if self.use_compress:
            query_reps = self.compressor(query_reps)
        if self.apply_tanh:
            query_reps = torch.tanh(query_reps)

        return query_reps

    def passage_embs(self, passages):
        passage_reps = self.bert(**passages)[0][:,0,:]

        if self.use_compress:
            passage_reps = self.compressor(passage_reps)
        if self.apply_tanh:
            passage_reps = torch.tanh(passage_reps)

        return passage_reps

    def nway_passage_embs(self, nway_passages):
        input_ids, attention_mask = nway_passages["input_ids"], nway_passages["attention_mask"]
        bz, nway, seq_len = input_ids.shape 

        input_ids, attention_mask = input_ids.view(bz*nway, seq_len), attention_mask.view(bz*nway, seq_len)
        passage_reps = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :]
        passage_reps = passage_reps.view(bz, nway, -1)

        if self.use_compress:
            passage_reps = self.compressor(passage_reps)
        if self.apply_tanh:
            passage_reps = torch.tanh(passage_reps)

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
        return self.bert.config.hidden_size if not self.use_compress else self.compress_dim
    
    
        
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

    
class _GraphDualEncoder(DualEncoder):
    def __init__(self, model_args):
        super().__init__(model_args)
        self.loss_fn = ContrastiveLoss()
        
        
    def _detach_record(self, anchor_reps, pos_reps, neg_reps):
        # for recording
        detach_anchor_reps = anchor_reps.detach()
        detach_pos_reps = pos_reps.detach() #[bz, hdim]
        detach_neg_reps = neg_reps.detach()
        
        pos_scores = torch.sum(detach_anchor_reps*detach_pos_reps, dim=-1)
        neg_scores = torch.sum(detach_anchor_reps*detach_neg_reps, dim=-1)
        all_scores = detach_anchor_reps @ torch.cat([detach_pos_reps, detach_neg_reps]).t()
        
        row = len(pos_scores)
        col = 2*row
        indice_pairs = [(i, j) for i in range(row) for j in range(col) if i != j]
        indices = torch.LongTensor([i*col + j for (i, j) in indice_pairs]).to(device=pos_scores.device)
        all_neg_score = torch.mean(torch.index_select(all_scores.view(-1), 0, indices))
        
        score_diffs = (pos_scores - neg_scores).detach()
        
        records = {"train/pos_score": pos_scores.cpu().mean().item(), "train/hard_neg_score": neg_scores.cpu().mean().item(),
                  "train/all_neg_score": all_neg_score.cpu().item(), "train/score_diff": score_diffs.cpu().mean().item()}
        
        return records
        
    
    def forward(self, anchors, pos_passages, neg_passages, anchor_indices):
        """
        Args:
            anchors: BatchEncoding with [bz, seq_len]
            pos_passages: BatchEncoding with [total_passage_num, seq_len]
            neg_passages: BatchEncoding with [total_passage_num, seq_len]
            anchor_indices: [total_passage_num]
            
        Returns:
            loss: 
            records:
        """
        anchor_reps = self.query_embs(anchors) #[bz, hdim]
        pos_passage_reps = self.passage_embs(pos_passages)
        neg_passage_reps = self.passage_embs(neg_passages)
        
        expand_anchor_reps = torch.index_select(anchor_reps, 0, anchor_indices) #[total_passage_num, hdim]
        pos_logits = torch.sum(expand_anchor_reps*pos_passage_reps, dim=-1)
        neg_logits = torch.sum(expand_anchor_reps*neg_passage_reps, dim=-1)
        
        pos_probs = torch_geometric.utils.softmax(pos_logits, anchor_indices).view(-1,1)
        neg_probs = torch_geometric.utils.softmax(neg_logits, anchor_indices).view(-1,1)
        
        weighted_pos_reps = pos_passage_reps * pos_probs
        weighted_neg_reps = neg_passage_reps * neg_probs
        
        pos_reps = scatter_add(weighted_pos_reps, anchor_indices, dim=0) #[bz, hdim]
        neg_reps = scatter_add(weighted_neg_reps, anchor_indices, dim=0) #[bz, hdim]
        
        assert pos_reps.shape[0] == neg_reps.shape[0]
        dest_indices = torch.LongTensor([[idx, idx+len(pos_reps)] for idx in range(len(pos_reps))]).view(-1).to(device=pos_reps.device)
        tmp_all_reps = torch.cat([pos_reps, neg_reps])
        all_reps = torch.index_select(tmp_all_reps, 0, dest_indices) #[2*bz, hdim]
        loss = self.loss_fn(anchor_reps, all_reps)
        
        records = self._detach_record(anchor_reps, pos_reps, neg_reps)
        
        return loss, records
    
class GraphDualEncoder(DualEncoder):
    def __init__(self, model_args):
        super().__init__(model_args)
        self.loss_fn = ContrastiveLoss()
        self.W = nn.Linear(768, 768, bias=False)
        self.register_parameter("a", nn.Parameter(torch.empty(1, 768*2)))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        
    def _detach_record(self, anchor_reps, pos_reps, neg_reps):
        # for recording
        detach_anchor_reps = anchor_reps.detach()
        detach_pos_reps = pos_reps.detach() #[bz, hdim]
        detach_neg_reps = neg_reps.detach()
        
        pos_scores = torch.sum(detach_anchor_reps*detach_pos_reps, dim=-1)
        neg_scores = torch.sum(detach_anchor_reps*detach_neg_reps, dim=-1)
        all_scores = detach_anchor_reps @ torch.cat([detach_pos_reps, detach_neg_reps]).t()
        
        row = len(pos_scores)
        col = 2*row
        indice_pairs = [(i, j) for i in range(row) for j in range(col) if i != j]
        indices = torch.LongTensor([i*col + j for (i, j) in indice_pairs]).to(device=pos_scores.device)
        all_neg_score = torch.mean(torch.index_select(all_scores.view(-1), 0, indices))
        
        score_diffs = (pos_scores - neg_scores).detach()
        
        records = {"train/pos_score": pos_scores.cpu().mean().item(), "train/hard_neg_score": neg_scores.cpu().mean().item(),
                  "train/all_neg_score": all_neg_score.cpu().item(), "train/score_diff": score_diffs.cpu().mean().item()}
        
        return records
        
    
    def forward(self, anchors, pos_passages, neg_passages, anchor_indices):
        """
        Args:
            anchors: BatchEncoding with [bz, seq_len]
            pos_passages: BatchEncoding with [total_passage_num, seq_len]
            neg_passages: BatchEncoding with [total_passage_num, seq_len]
            anchor_indices: [total_passage_num]
            
        Returns:
            loss: 
            records:
        """
        anchor_reps = self.query_embs(anchors) #[bz, hdim]
        pos_passage_reps = self.passage_embs(pos_passages)
        neg_passage_reps = self.passage_embs(neg_passages)
        
        expand_anchor_reps = torch.index_select(anchor_reps, 0, anchor_indices) #[total_passage_num, hdim]
        pos_logits = torch.sum(torch.cat([self.W(expand_anchor_reps), self.W(pos_passage_reps)], dim=-1) * self.a, dim=-1)
        neg_logits = torch.sum(torch.cat([self.W(expand_anchor_reps), self.W(neg_passage_reps)], dim=-1) * self.a, dim=-1)
        pos_logits = nn.functional.leaky_relu(pos_logits)
        neg_logits = nn.functional.leaky_relu(neg_logits)
        
        pos_probs = torch_geometric.utils.softmax(pos_logits, anchor_indices).view(-1,1)
        neg_probs = torch_geometric.utils.softmax(neg_logits, anchor_indices).view(-1,1)
        
        weighted_pos_reps = pos_passage_reps * pos_probs
        weighted_neg_reps = neg_passage_reps * neg_probs
        
        pos_reps = scatter_add(weighted_pos_reps, anchor_indices, dim=0) #[bz, hdim]
        neg_reps = scatter_add(weighted_neg_reps, anchor_indices, dim=0) #[bz, hdim]
        
        assert pos_reps.shape[0] == neg_reps.shape[0]
        dest_indices = torch.LongTensor([[idx, idx+len(pos_reps)] for idx in range(len(pos_reps))]).view(-1).to(device=pos_reps.device)
        tmp_all_reps = torch.cat([pos_reps, neg_reps])
        all_reps = torch.index_select(tmp_all_reps, 0, dest_indices) #[2*bz, hdim]
        loss = self.loss_fn(anchor_reps, all_reps)
        
        records = self._detach_record(anchor_reps, pos_reps, neg_reps)
        
        return loss, records
          
    