import os 

import ujson 

json_file = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/runs/checkpoint_250000.train.json"
run_file = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/runs/checkpoint_250000.train.run"

examples = []
with open(json_file) as fin:
    for line in fin:
        example = ujson.loads(line)
        examples.append(example)

with open(run_file, "w") as fout:
    for exp in examples:
        qid = exp["qid"]
        pids, ranks, scores = exp["pids"], exp["ranks"], exp["scores"]
        for pid, rank, score in zip(pids, ranks, scores):
            fout.write(f"{qid}\t{pid}\t{rank}\t{score}\n")
