import os 

p2p_run_path = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/qrel_passage_runs/checkpoint_250000_p2p.trec20.run"
pid_to_qid_path = "/work/hzeng_umass_edu/datasets/msmarco-passage/qrel-passages/qrel_pid_to_qid.trec20.tsv"
output_path = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/qrel_passage_runs/checkpoint_250000.trec20.run"
topk = 1000

relpid_to_rankdata = {}
with open(p2p_run_path) as fin:
    for line in fin:
        relpid, pid, rank, score = line.strip().split("\t")
        relpid, pid, rank, score = int(relpid), int(pid), int(rank), float(score)
        if relpid not in relpid_to_rankdata:
            relpid_to_rankdata[relpid] = [(pid, rank, score)]
        else:
            if rank > topk:
                continue
            relpid_to_rankdata[relpid].append((pid, rank, score)) 

pid_to_qid = {}
with open(pid_to_qid_path) as fin:
    for line in fin:
        pid, qid = line.strip().split("\t")
        pid, qid = int(pid), int(qid)
        assert pid not in pid_to_qid
        pid_to_qid[pid] = qid 

with open(output_path, "w") as fout:
    for rel_pid, rankdata in relpid_to_rankdata.items():
        qid = pid_to_qid[rel_pid]
        for pid, rank, score in rankdata:
            fout.write(f"{qid}\t{pid}\t{rank}\t{score}\n") 


