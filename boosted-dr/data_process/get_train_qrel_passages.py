def load_passages(path):
    pid_to_passage = {}
    with open(path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        pid, passage = line.strip().split("\t")
        pid_to_passage[int(pid)] = passage 

    return pid_to_passage

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

def get_from_msmarco():
    qrel_path = "/work/hzeng_umass_edu/datasets/msmarco-passage/qrels.train.tsv"
    passage_path = "/work/hzeng_umass_edu/datasets/msmarco-passage/collection.tsv"
    out_path = "/work/hzeng_umass_edu/datasets/msmarco-passage/qrel-passages/qrel-passages.train.tsv"
    pid2qid_path = "/work/hzeng_umass_edu/datasets/msmarco-passage/qrel-passages/qrel_pid_to_qid.train.tsv"

    qid_to_rel_data = load_qrels(qrel_path)

    num_qrel = 0
    qid_to_rel_pids = {}
    pid_to_qid = []
    for qid, rel_data in qid_to_rel_data.items():
        for pid, score in rel_data.items():
            if qid not in qid_to_rel_pids:
                qid_to_rel_pids[qid] = [pid]
            else:
                qid_to_rel_pids[qid] += [pid]
            num_qrel += 1
            
            pid_to_qid.append((pid, qid))


    num_qrel_passage = 0
    pid_to_passage = load_passages(passage_path)
                
    with open(out_path, "w") as fout:
        for _, pids in qid_to_rel_pids.items():
            for pid in pids:
                passage = pid_to_passage[pid]
                fout.write(f"{pid}\t{passage}\n")
                num_qrel_passage += 1
    with open(pid2qid_path, "w") as fout:
        for (pid, qid) in pid_to_qid:
            fout.write(f"{pid}\t{qid}\n")
    
    print("number of qrel = {}, number of qrel_passage = {}".format(num_qrel, num_qrel_passage))

def get_from_trec():
    qrel_path = "/work/hzeng_umass_edu/datasets/msmarco-passage/trec-20/2020qrels-pass.txt"
    passage_path = "/work/hzeng_umass_edu/datasets/msmarco-passage/collection.tsv"
    out_path = "/work/hzeng_umass_edu/datasets/msmarco-passage/qrel-passages/qrel-passages.trec20.tsv"
    pid2qid_path = "/work/hzeng_umass_edu/datasets/msmarco-passage/qrel-passages/qrel_pid_to_qid.trec20.tsv"
    qid_to_rel_data = load_qrels(qrel_path, is_trec=True)

    num_qrel = 0
    qid_to_rel_pids = {}
    pid_to_qid = []
    max_score_dist = {0.:0, 1.:0, 2.:0, 3.:0}
    for qid, rel_data in qid_to_rel_data.items():
        max_score = max([s for k, s in rel_data.items()])
        max_score_dist[max_score] += 1
        for pid, score in rel_data.items():
            if score != max_score:
                continue
            if qid not in qid_to_rel_pids:
                qid_to_rel_pids[qid] = [pid]
                pid_to_qid.append((pid, qid))
                num_qrel += 1

    num_qrel_passage = 0
    pid_to_passage = load_passages(passage_path)
                
    with open(out_path, "w") as fout:
        for _, pids in qid_to_rel_pids.items():
            for pid in pids:
                assert len(pids) == 1
                passage = pid_to_passage[pid]
                fout.write(f"{pid}\t{passage}\n")
                num_qrel_passage += 1
    with open(pid2qid_path, "w") as fout:
        for (pid, qid) in pid_to_qid:
            fout.write(f"{pid}\t{qid}\n")
    
    print("number of qrel = {}, number of qrel_passage = {}".format(num_qrel, num_qrel_passage))
    print("max score distribution = {}".format(max_score_dist))


if __name__ == "__main__":
    #get_from_msmarco()
    get_from_trec()
