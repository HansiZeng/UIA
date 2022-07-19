import os 
import sys 
sys.path.append("./")

import ujson 


in_path = "/work/hzeng_umass_edu/datasets/msmarco-passage/corase_to_fine_grained/10relT_20neg.train.json"
low_perf_queries_path = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/stats/low_performance_queries.train.json"
out_path = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/low_perf_query_train/10relT_20neg.train.json"

low_perf_queries = set()
with open(low_perf_queries_path) as fin:
    for line in fin:
        exp = ujson.loads(line)
        try:
            qid = int(exp["qid"])
            low_perf_queries.add(int(exp["qid"]))
        except:
            print(f"{exp} is not the training example")

examples = []
with open(out_path, "w") as fout:
    with open(in_path) as fin:
        for line in fin:
            exp = ujson.loads(line)
            if int(exp["qid"]) not in low_perf_queries:
                continue
            fout.write(line)