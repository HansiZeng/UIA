import os 
import sys 

import ujson 

base_path = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/"
boosted_train_path = os.path.join(base_path, "experiment_05-07_152150/boosting_train/checkpoint_250000_10_20_12.train.json")

new_train_path = os.path.join(base_path, "experiment_05-09_171925/boosting_train/checkpoint_250000_10_20_3.train.json")
output_path = os.path.join(base_path, "experiment_05-09_171925/boosting_train/checkpoint_250000_10_20_123.train.json")

boosted_hist = len(os.path.basename(boosted_train_path).split(".")[0].split("_")[-1])

boosted_examples = []
with open(boosted_train_path) as fin:
    for line in fin:
        example = ujson.loads(line)
        boosted_examples.append(example)

new_examples = []
with open(new_train_path) as fin:
    for line in fin:
        example = ujson.loads(line)
        new_examples.append(example)

print("="*50, " boosted_hist = {} ".format(boosted_hist), "="*50)
print("output_path = {}".format(output_path))

with open(output_path, "w") as fout:
    for (boosted_exp, new_exp) in zip(boosted_examples, new_examples):
        example = {}
        assert boosted_exp["qid"] == new_exp["qid"]
        example["qid"] = boosted_exp["qid"]
        for key in boosted_exp:
            if key == "qid":
                continue
            example[key] = []
            boosted_pairs = boosted_exp[key]
            new_pairs = new_exp[key]
            for (b_pid, b_score), (n_pid, n_score) in zip(boosted_pairs, new_pairs):
                score = (b_score * boosted_hist + n_score) / float(boosted_hist+1)
                assert b_pid == n_pid
                example[key].append((b_pid, score))
        
        fout.write(ujson.dumps(example) + "\n")




