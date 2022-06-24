import os 

import ujson 

run_file = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/runs/checkpoint_250000.trec20.run"
json_file = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/runs/checkpoint_250000.trec20.json"
topk = 200

qid_to_rankdata = {}
with open(run_file, "r") as fin:
    for line in fin:
        qid, pid, rank, score = line.strip().split("\t")
        qid, pid, rank, score = int(qid), int(pid), int(rank), float(score)
        if qid not in qid_to_rankdata:
            qid_to_rankdata[qid] = [(pid, rank, score)]
        else:
            if rank > topk:
                continue
            qid_to_rankdata[qid].append((pid, rank, score))

example = {}
with open(json_file, "w") as fout:
    for qid, rankdata in qid_to_rankdata.items():
        example["qid"] = qid
        pids, ranks, scores = zip(*rankdata)
        example["pids"] = pids
        example["ranks"] = ranks 
        example["scores"] = scores
        assert len(pids) == len(ranks) == len(scores) == topk, len(pids)

        fout.write(ujson.dumps(example) + "\n")
        example = {} 
