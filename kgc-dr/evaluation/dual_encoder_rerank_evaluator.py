import os 
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict

from transformers.tokenization_utils_base import BatchEncoding

def batch_to_cuda(batch):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda()
        if isinstance(batch[key], dict) or isinstance(batch[key], BatchEncoding):
            for sub_key in batch[key]:
                if isinstance(batch[key][sub_key], torch.Tensor):
                    batch[key][sub_key] = batch[key][sub_key].cuda()

    return batch

class RerankingEvaluator():
    def __init__(self, eval_dataloader, topk=10):
        self.eval_dataloader = eval_dataloader 
        self.topk = topk
        
        self.rel_to_qrels = {}
        for _, _,  labels, meta_data in self.eval_dataloader:
            qids, pids, relations = meta_data["qids"], meta_data["pids"], meta_data["relations"]
            labels = labels.tolist()
            
            for qid, pid, label, rel in zip(qids, pids, labels, relations):
                if rel not in self.rel_to_qrels:
                    self.rel_to_qrels[rel] = defaultdict(list)
                
                if label == 1.:
                    self.rel_to_qrels[rel][qid].append(pid)
        
    def load_qrels_path(self, qrels_path):
        qid_to_relpids = defaultdict(list)
        with open(qrels_path) as fin:
            for line in fin:
                qid, _, pid, _ = line.strip().split("\t")
                qid_to_relpids[int(qid)].append(int(pid))
        
        return qid_to_relpids
    
    def compute_mrr_and_recall(self, qid_to_rankdata, qid_to_relpids, topk):
        num_queries = len(qid_to_rankdata)
        total_mrr, total_recall = 0., 0.
        for qid, rankdata in qid_to_rankdata.items():
            sorted_pids = [pid for pid, score in sorted(
                            rankdata, key=lambda item: item[1], reverse=True)][:topk]
            
            mrr = 0.
            for idx, pid in enumerate(sorted_pids):
                if pid in set(qid_to_relpids[qid]):
                    mrr = 1. / (idx+1)
                    break
            total_mrr += mrr 
            total_recall += len(set(sorted_pids).intersection(set(qid_to_relpids[qid]))) / len(set(qid_to_relpids[qid]))
            
        return total_mrr / num_queries, total_recall / num_queries
            
                
    def __call__(self, model):
        model.eval()
        topk = self.topk
        rel_qid_to_rankdata = {
            rel: defaultdict(list) for rel in self.rel_to_qrels.keys()
        }
        for tokenized_queries, tokenized_passages, _, meta_data in tqdm(self.eval_dataloader, total=len(self.eval_dataloader),
                                                desc="Reranking"):
            tokenized_queries = batch_to_cuda(tokenized_queries)
            tokenized_passages = batch_to_cuda(tokenized_passages)
            with torch.no_grad():
                scores = torch.sum(model.query_embs(tokenized_queries) * model.passage_embs(tokenized_passages), dim=-1)
                scores = scores.cpu().tolist()
            
            for qid, pid, score, rel in zip(meta_data["qids"], meta_data["pids"], scores, meta_data["relations"]):
                rel_qid_to_rankdata[rel][qid].append((pid, score))
        
        rel_to_scores = {}
        for rel, qid_to_rankdata in rel_qid_to_rankdata.items():
            mrr, recall = self.compute_mrr_and_recall(qid_to_rankdata,
                                                self.rel_to_qrels[rel],
                                                self.topk)
            rel_to_scores[rel] = {}
            rel_to_scores[rel][f"MRR@{topk}"] = mrr 
            rel_to_scores[rel][f"R@{topk}"] = recall
        model.train()
        return rel_to_scores

if __name__ == "__main__":
    import sys 
    sys.path.append("/home/jupyter/unity_jointly_rec_and_search/kgc-dr")
    from modeling.dual_encoder import DualEncoder
    from dataset.dual_encoder_rerank_dataset import DualEncoderRerankDataset
    import torch
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, Dataset
    
    device = torch.device("cuda")
    #pretrained_path="/home/jupyter/unity_jointly_rec_and_search/experiments/unified_kgc/phase_1/experiment_10-02_005604/models/checkpoint_latest"
    
    #entities_path="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/all_entities.tsv"
    #eval_examples_path="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_kgc/unified_rerank_evaluate/pairs.all.rerank"
    pretrained_path="/home/jupyter/unity_jointly_rec_and_search/experiments/amazon_esci/task2/phase_dot-v5/experiment_10-05_195239/models/checkpoint_latest"
    entities_path="/home/jupyter/unity_jointly_rec_and_search/datasets/amazon_esci_dataset/task_2_multiclass_product_classification/all_entities.tsv" 
    
    eval_examples_path="/home/jupyter/unity_jointly_rec_and_search/datasets/amazon_esci_dataset/task_2_multiclass_product_classification/unified_rerank_evaluate/pairs.all.rerank"
    query_max_length = passage_max_length = 256
    
    model = DualEncoder.from_pretrained(pretrained_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    eval_dataset = DualEncoderRerankDataset.create_from_pair_file(entities_path,
                                                             eval_examples_path,
                                                             tokenizer,
                                                             query_max_length,
                                                            passage_max_length)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=128, collate_fn=eval_dataset.collate_fn)
    
    reranking_evaluator = RerankingEvaluator(eval_dataloader)
    rel_to_scores = reranking_evaluator(model)
    print(rel_to_scores)
    """
    Lowes:
    {'is_similar_to': {'MRR@10': 0.6326011904761905, 'R@10': 0.44426292593547}, 'is_complementary_to': {'MRR@10': 0.6502480158730158, 'R@10': 0.33999831797529056}, 'is_relevant_to': {'MRR@10': 0.8550833333333333, 'R@10': 0.3930045701918066}}
    
    amazon esci:
    {'is_similar_to': {'MRR@10': 0.3340158730158733, 'R@10': 0.32694071870158814}, 'is_complementary_to': {'MRR@10': 0.524172619047619, 'R@10': 0.6593710374862454}, 'is_relevant_to': {'MRR@10': 0.5671091269841271, 'R@10': 0.32695344533546516}, }
    """