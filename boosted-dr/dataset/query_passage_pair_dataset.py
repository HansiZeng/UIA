import os 
import logging
logger = logging.getLogger(__name__)

import torch
import ujson 

class QueryPassagePairDataset(torch.utils.data.Dataset):
    def __init__(self, qid_pid_pairs, qid_to_query, pid_to_passage, tokenizer, max_length=None, 
                    query_max_len=None, passage_max_len=None):
        super(QueryPassagePairDataset, self).__init__()
        self.qid_pid_pairs = qid_pid_pairs
        self.qid_to_query = qid_to_query
        self.pid_to_passage = pid_to_passage
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

    def __getitem__(self, idx):
        qid, pid = self.qid_pid_pairs[idx]
        query = self.qid_to_query[qid]
        passage = self.pid_to_passage[pid] 

        return {
            "query": query,
            "passage": passage,
            "qid": qid,
            "pid": pid
        }

    def collate_fn(self, batch):
        elem = batch[0]
        assert len(elem) == 4 and isinstance(elem, dict)

        queries = []
        passages = []
        query_ids = []
        passsage_ids = []

        for elem in batch:
            queries.append(elem["query"])
            passages.append(elem["passage"])
            query_ids.append(elem["qid"])
            passsage_ids.append(elem["pid"])
        
        encoded_QueryPassages = self.tokenizer(queries, passages, padding=True, truncation='longest_first', 
                                        return_tensors="pt", max_length=self.max_length)

        return {
            "encoded_QueryPassage": encoded_QueryPassages,
            "query_id": query_ids,
            "passage_id": passsage_ids,
        }

    def de_collate_fn(self, batch):
        elem = batch[0]
        assert len(elem) == 4 and isinstance(elem, dict)

        queries = []
        passages = []
        qids = []
        pids = []

        for elem in batch:
            queries.append(elem["query"])
            passages.append(elem["passage"])
            qids.append(elem["qid"])
            pids.append(elem["pid"])
        
        queries = self.tokenizer(queries, padding=True, truncation='longest_first', 
                                        return_tensors="pt", max_length=self.query_max_len)
        passages = self.tokenizer(passages, padding=True, truncation='longest_first', 
                                        return_tensors="pt", max_length=self.passage_max_len)

        return {
            "qid": qids,
            "pid": pids,
            "query": queries,
            "passage": passages
        }

    def __len__(self):
        return len(self.qid_pid_pairs)

    @classmethod
    def create_from_file(cls, rankfile, queries_path, passages_path, rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1
        
        start_time = time.time()
        qid_to_passageids = {}
        with open(rankfile, "r") as f:
            for line in f:
                qid, pid = line.strip().split("\t")
                qid, pid = int(qid), int(pid)
                if qid not in qid_to_passageids:
                    qid_to_passageids[qid] = [pid]
                else:
                    qid_to_passageids[qid] += [pid]

        qid_pid_pairs = []
        for qid in qid_to_passageids:
            for pid in qid_to_passageids[qid]:
                qid_pid_pairs.append((qid, pid))
        
        logger.info(f"Read rankfile from {rankfile}")
        print("Spent {:.2f} seconds".format(time.time() - start_time))
        
        qid_to_query = {}
        with open(queries_path, "r") as f:
            for line in f:
                qid, query = line.strip().split("\t")
                qid_to_query[int(qid)] = query
        logger.info(f"Read queries from {queries_path}")

        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                pid, passage = line.strip().split("\t")
                pid_to_passage[int(pid)] = passage
        logger.info(f"Read passages from {passages_path}") 

        return cls(qid_pid_pairs=qid_pid_pairs, qid_to_query=qid_to_query, pid_to_passage=pid_to_passage)

    @classmethod
    def create_from_ujson_file(cls, ranking_path, queries_path, passages_path, tokenizer, max_length, 
                                rank=-1, nranks=None):
        if rank == -1:
            raise ValueError("Require multiple gpus training")

        qid_to_query = {}
        with open(queries_path, "r") as f:
            for line in f:
                qid, query = line.strip().split("\t")
                qid_to_query[int(qid)] = query
        logger.info(f"Read queries from {queries_path}")

        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                pid, passage = line.strip().split("\t")
                pid_to_passage[int(pid)] = passage
        logger.info(f"Read passages from {passages_path}")

        qid_pid_pairs = []
        with open(ranking_path, "r") as fin:
            for line_idx, line in enumerate(fin):
                if line_idx % nranks == rank:
                    example = ujson.loads(line)
                    qid = example["qid"]
                    pids = example["pids"]
                    for pid in pids:
                        qid_pid_pairs.append((qid, pid))
        
        return cls(qid_pid_pairs, qid_to_query, pid_to_passage, tokenizer, max_length=max_length)

    @classmethod
    def create_from_10relT_20neg_file(cls, example_path, queries_path, passages_path, tokenizer, 
                                query_max_len, passage_max_len, rank=-1, nranks=None):
        if rank == -1:
            raise ValueError("Require multiple gpus training")

        qid_to_query = {}
        with open(queries_path, "r") as f:
            for line in f:
                qid, query = line.strip().split("\t")
                qid_to_query[int(qid)] = query
        logger.info(f"Read queries from {queries_path}")

        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                pid, passage = line.strip().split("\t")
                pid_to_passage[int(pid)] = passage
        logger.info(f"Read passages from {passages_path}")

        qid_pid_pairs = []
        with open(example_path, "r") as fin:
            for line_idx, line in enumerate(fin):
                if line_idx % nranks == rank:
                    example = ujson.loads(line)
                    qid = example["qid"]
                    pids = example["relT_pids"] + example["most_hard_pids"] + example["semi_hard_pids"]
                    for pid in pids:
                        qid_pid_pairs.append((qid, pid))
        
        return cls(qid_pid_pairs, qid_to_query, pid_to_passage, tokenizer, 
                    query_max_len=query_max_len, passage_max_len=passage_max_len)

