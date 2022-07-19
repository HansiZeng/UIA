import random 
import copy

import numpy as np 
from tqdm import tqdm 
import ujson 

random.seed(4680)

qrels_file = "/work/hzeng_umass_edu/datasets/msmarco-passage/qrels.train.tsv"
qid_to_relevant_pids = {}

with open(qrels_file, "r") as f:
    for line in f:
        qid, _, pid, _ = line.strip().split("\t")         
        if qid not in qid_to_relevant_pids:
            qid_to_relevant_pids[int(qid)] = [int(pid)]
        else:
            qid_to_relevant_pids[int(qid)] += [int(pid)]

topk_passage_file = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/runs/MiniLM-L-6-v2.train.rerank"

qid_to_T_top_pids = {}
with open(topk_passage_file, "r") as f:
    for line in f:
        qid, pid, _, _ = line.strip().split("\t") 
        qid, pid = int(qid), int(pid)

        if qid not in qid_to_T_top_pids:
            qid_to_T_top_pids[qid] = [pid]
        else:
            qid_to_T_top_pids[qid] += [pid]

assert set(qid_to_relevant_pids.keys()).issubset(set(qid_to_T_top_pids.keys()))

train_examples = []

remove_a_count = 0
remove_b_count = 0
for qid in tqdm(qid_to_relevant_pids):
    relevant_pids = np.array(qid_to_relevant_pids[qid])
    T_top_pids = qid_to_T_top_pids[qid]
    assert len(T_top_pids) == 200

    # construct relevant T
    teacher_top10_pids = np.array(T_top_pids[:10])
    rel_T_pids = np.hstack(
        (relevant_pids, teacher_top10_pids[~np.in1d(teacher_top10_pids, relevant_pids)])
    )
    assert len(rel_T_pids) >= 10
    rel_T_pids = [x.item() for x in rel_T_pids[:10]]

    # construct most hard 
    teacher_top50_pids = copy.deepcopy(T_top_pids[10:50])
    for rel_pid in relevant_pids:
        if rel_pid in teacher_top50_pids:
            teacher_top50_pids.remove(rel_pid)
            remove_a_count += 1
    rnd_idxs = random.sample(range(len(teacher_top50_pids)), 10)
    rnd_idxs.sort()
    most_hard_pids = np.array(teacher_top50_pids)[rnd_idxs]
    most_hard_pids = [x.item() for x in most_hard_pids]

    # construct semi hard 
    teacher_bottom150_pids = copy.deepcopy(T_top_pids[50:])
    for rel_pid in relevant_pids:
        if rel_pid in teacher_bottom150_pids:
            teacher_bottom150_pids.remove(rel_pid)
            remove_b_count += 1
    rnd_idxs = random.sample(range(len(teacher_bottom150_pids)), 10)
    rnd_idxs.sort() 
    semi_hard_pids = np.array(teacher_bottom150_pids)[rnd_idxs]
    semi_hard_pids = [x.item() for x in semi_hard_pids]

    assert len(set(rel_T_pids) & set(most_hard_pids)) == 0 and len(set(rel_T_pids) & set(semi_hard_pids)) == 0
    assert len(set(most_hard_pids) & set(semi_hard_pids)) == 0
    assert len(rel_T_pids) == 10 and len(semi_hard_pids) == 10 and len(most_hard_pids) == 10

    train_examples.append({
        "qid": qid,
        "relT_pids": rel_T_pids,
        "most_hard_pids": most_hard_pids,
        "semi_hard_pids": semi_hard_pids
    })


print("number of train examples = {}".format(len(train_examples)))
print("number of relevant pids in most hard negatives = {}".format(remove_a_count))
print("number of relevant pids in semi hard negatives = {}".format(remove_b_count))
output_path = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/train_examples/10relT_20neg_ce0_de250000.train.json"
with open(output_path, "w") as fout:
    for example in train_examples:
        fout.write(ujson.dumps(example))
        fout.write("\n")