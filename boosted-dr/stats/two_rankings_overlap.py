import os 

import numpy as np

topk = 10
ranking_path_1 = "/gypsum/scratch1/hzeng/datasets/msmarco-passage/runs/tas256-train-ce-rerank-top200.run"
ranking_path_2 = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/runs/MiniLM-L-6-v2.train.rerank"

qid_to_pids_1 = {}
qid_to_pids_2 = {}

with open(ranking_path_1, "r") as fin:
    for line in fin:
        array = line.strip().split("\t")
        assert len(array) == 4, array
        qid, pid, rank, _ = array
        qid, pid, rank = int(qid), int(pid), int(rank)
        if qid not in qid_to_pids_1:
            qid_to_pids_1[qid] = [pid]
        else:
            if len(qid_to_pids_1[qid]) >= topk:
                continue
            qid_to_pids_1[qid].append(pid)

with open(ranking_path_2, "r") as fin:
    for line in fin:
        array = line.strip().split("\t")
        assert len(array) == 4, array
        qid, pid, rank, _ = array
        qid, pid, rank = int(qid), int(pid), int(rank)
        if qid not in qid_to_pids_2:
            qid_to_pids_2[qid] = [pid]
        else:
            if len(qid_to_pids_2[qid]) >= topk:
                continue
            qid_to_pids_2[qid].append(pid)

print("number of queries in two rankings = {}, {}".format(len(qid_to_pids_1), len(qid_to_pids_2)))
assert set(qid_to_pids_2.keys()).issubset(set(qid_to_pids_1.keys()))

qid_to_overlap = {}
for idx, qid in enumerate(qid_to_pids_2):
    pids_1 = set(qid_to_pids_1[qid])
    pids_2 = set(qid_to_pids_2[qid])

    qid_to_overlap[qid] = len(pids_1&pids_2) / len(pids_1)

    assert len(pids_1) == len(pids_2) == topk, (len(pids_1), len(pids_2))

print("number of queries in stats = {}".format(len(qid_to_overlap)))
overlap = np.array(list(qid_to_overlap.values()))

print("min overlap = {}, max overlap = {}, mean overlap = {}".format(min(overlap), max(overlap), np.mean(overlap)))
print("0.25, 0.5, 0.75, 0.9 quantiles = {}".format(np.quantile(overlap, [0.25, 0.5, 0.75, 0.9])))

#topk = 10
#min overlap = 0.0, max overlap = 1.0, mean overlap = 0.984
#0.25, 0.5, 0.75, 0.9 quantiles = [1. 1. 1. 1.]

#topk = 30
#min overlap = 0.0, max overlap = 1.0, mean overlap = 0.955
#0.25, 0.5, 0.75, 0.9 quantiles = [0.93333333 1.         1.         1.        ]

#topk = 50
#min overlap = 0.02, max overlap = 1.0, mean overlap = 0.928
#0.25, 0.5, 0.75, 0.9 quantiles = [0.9  0.96 1.   1.  ]
