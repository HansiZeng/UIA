import os 

import numpy as np
import ujson 

original_example_path = "/work/hzeng_umass_edu/datasets/msmarco-passage/corase_to_fine_grained/10relT_20neg.train.json"
boosted_example_path = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/boosting_train/checkpoint_250000_10_20_1.train.json"

org_qid_to_data = {}
with open(original_example_path, "r") as fin:
    for line in fin:
        example = ujson.loads(line.rstrip())
        qid = example["qid"]
        org_qid_to_data[qid] = {"relT_pids": example["relT_pids"], "most_hard_pids": example["most_hard_pids"], 
                                    "semi_hard_pids": example["semi_hard_pids"]}

bst_qid_to_data = {}
with open(boosted_example_path, "r") as fin:
    for line in fin:
        example = ujson.loads(line.rstrip())
        qid = example["qid"]
        relT_pids, _ = zip(*example["relT_pids"])
        most_hard_pids, _ = zip(*example["most_hard_pids"])
        semi_hard_pids, _ = zip(*example["semi_hard_pids"])
        bst_qid_to_data[qid] = {"relT_pids": relT_pids, "most_hard_pids": most_hard_pids, "semi_hard_pids": semi_hard_pids}

assert set(org_qid_to_data.keys()) == set(bst_qid_to_data.keys())

for idx, qid in enumerate(org_qid_to_data):
    org_example = org_qid_to_data[qid]
    bst_example = bst_qid_to_data[qid]
    for key, org_val in org_example.items():
        org_val = np.array(org_val)
        bst_val = np.array(bst_example[key])
        sum_val = np.sum(org_val-bst_val)
        if idx in [10, 100, 1000, 1e4, 1e5, 2e5]:
            print(sum_val)
        if sum_val != 0:
            print("key; ", key, "org_val: ", org_val, "bst_val: ", bst_val)
            raise ValueError("Two data examples are not not same.")

print("Congrats, the {} and {} contain some training examples".format(original_example_path, boosted_example_path))

