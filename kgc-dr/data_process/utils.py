import os 
from tqdm import tqdm 

def load_queries(path):
    qid_to_query = {}
    with open(path, "r") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            qid_to_query[int(qid)] = query
    return qid_to_query

def load_passages(path):
    pid_to_passage = {}
    with open(path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        pid, passage = line.strip().split("\t")
        pid_to_passage[int(pid)] = passage 

    return pid_to_passage

def load_rankilist(path):
    qid_to_rankdata = {}
    with open(path, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        array = line.strip().split("\t")
        assert len(array) == 4
        
        qid, pid, _, score = array 
        qid, pid, score = int(qid), int(pid), float(score)

        if qid not in qid_to_rankdata:
            qid_to_rankdata[qid] = {}
            qid_to_rankdata[qid][pid] = score
        else:
            qid_to_rankdata[qid][pid] = score
    
    return qid_to_rankdata

def load_qrels(qrel_path, is_trec=False):
    qid_to_relevant_data = {}
    with open(qrel_path, "r") as f:
        for line in f:
            if is_trec:
                qid, _, pid, grade = line.strip().split(" ")
            else:
                qid, _, pid, grade = line.strip().split("\t")
            
            qid, pid = int(qid), int(pid)
            if float(grade) <= 0.00001:
                continue

            if qid not in qid_to_relevant_data:
                qid_to_relevant_data[qid] = {}
                qid_to_relevant_data[qid][pid] = float(grade)
            else:
                qid_to_relevant_data[qid][pid] = float(grade)
    
    return qid_to_relevant_data
    

if __name__ == "__main__":
    doc_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/collection.tsv"

    pid_to_passage = load_passages(doc_path)
    print(pid_to_passage[7130348])