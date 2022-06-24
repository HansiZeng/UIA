import os 
import ujson 
import argparse

import torch 
import torch.nn as nn
import numpy as np 

def merge_and_aggregate(pid2score_list, aggr_mode="sum", return_sorted=True):
    assert aggr_mode in ["sum", "max"], aggregate_mode
    global_pid2fs = {}
    field2mins = {}
    fields = []

    # merge 
    for field, pid2score in enumerate(pid2score_list):
        for pid, score in pid2score.items():
            if pid not in global_pid2fs:
                global_pid2fs[pid] = {field: score}
            else:
                global_pid2fs[pid][field] = score

        field2mins[field] =  min(pid2score.values()) 
        fields.append(field)

    # aggregate
    for pid, field2score in global_pid2fs.items():
        for field in fields:
            if field not in field2score:
                global_pid2fs[pid][field] = field2mins[field]
    
    for pid, field2score in global_pid2fs.items():
        assert len(field2score) == len(fields), field2score
        if aggr_mode == "sum":
            field2score["aggr"] = sum(field2score.values())
        elif aggr_mode == "max":
            field2score["aggr"] = max(field2score.values())
        else:
            raise ValueError("not aggr_mode = {}".format(aggr_mode))
    
    if return_sorted:
        sorted_tuples = sorted(global_pid2fs.items(), key=lambda item: item[1]["aggr"], reverse=True)
        global_pid2fs = {k:v for k, v in sorted_tuples}

    return global_pid2fs, field2mins
    
def check_mono_decrease_tuples(tuples, field):
    old_val = 1e5
    for _tuple in tuples:
        val = _tuple[field]
        if old_val < val:
            raise ValueError("the tuples = {} is not monotone decreasing".format(tuples))
        old_val = val 
    

def read_ranklist(ranklist_path):
    qid_to_pid_to_score = {}
    with open(ranklist_path) as fin:
        for line in fin:
            array = line.strip().split("\t")
            assert len(array) == 4, array 
            qid, pid, _, score =  array 
            qid, pid, score = int(qid), int(pid), float(score)

            if qid not in qid_to_pid_to_score:
                qid_to_pid_to_score[qid] = {pid: score}
            else:
                qid_to_pid_to_score[qid][pid] = score 
    
    return qid_to_pid_to_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranklist_1", default="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/runs/checkpoint_250000")
    parser.add_argument("--ranklist_2", default="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-13_032123/runs/checkpoint_250000")
    #parser.add_argument("--ranklist_3", default="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-09_171925/runs/checkpoint_250000")
    
    parser.add_argument("--cutoff", default=1000)
    parser.add_argument("--eval_type", default="trec20")
    parser.add_argument("--aggr_type", default="sum")

    parser.add_argument("--output_dir", default="/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_05-13_032123/merge_2_runs/")

    args = parser.parse_args()

    return args 


def main(args):
    if "train" in args.eval_type:
        print("retrieve train queries")
        raise NotImplementedError
    if "dev" in args.eval_type:
        print("retrieve dev queries")
        args.ranklist_1 = args.ranklist_1 + ".dev.run"
        args.ranklist_2 = args.ranklist_2 + ".dev.run"
        #args.ranklist_3 = args.ranklist_3 + ".dev.run"
    if "trec19" in args.eval_type:
        print("retrieve trec-19 queries")
        args.ranklist_1 = args.ranklist_1 + ".trec19.run"
        args.ranklist_2 = args.ranklist_2 + ".trec19.run"
        #args.ranklist_3 = args.ranklist_3 + ".trec19.run"
    if "trec20" in args.eval_type:
        print("retrieve trec-20 queries")
        args.ranklist_1 = args.ranklist_1 + ".trec20.run"
        args.ranklist_2 = args.ranklist_2 + ".trec20.run"
        #args.ranklist_3 = args.ranklist_3 + ".trec20.run"

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    qid2pid2score_1 = read_ranklist(args.ranklist_1)
    qid2pid2score_2 = read_ranklist(args.ranklist_2)
    #qid2pid2score_3 = read_ranklist(args.ranklist_3)

    assert set(qid2pid2score_1.keys()) == set(qid2pid2score_2.keys())# == set(qid2pid2score_3.keys())

    ujson_f = open(os.path.join(args.output_dir, "score_field.{}.{}.json".format(args.aggr_type, args.eval_type)), "w")
    rank_f = open(os.path.join(args.output_dir, "top-{}.{}.{}.run".format(args.cutoff, args.aggr_type, args.eval_type)), "w")
    for qid in qid2pid2score_1.keys():
        pid2score_list = [qid2pid2score_1[qid], qid2pid2score_2[qid]] #, qid2pid2score_3[qid]]
        pid2fs, _ = merge_and_aggregate(pid2score_list=pid2score_list, aggr_mode=args.aggr_type)
        pid2fs = list(pid2fs.items())[:args.cutoff]
        pid2fs = {k:v for k, v in pid2fs}
 
        ujson_f.write(ujson.dumps({qid: pid2fs}) + "\n")

        pid2score = [(k,v["aggr"]) for k,v in pid2fs.items()] # assume already sorted 
        check_mono_decrease_tuples(pid2score, field=1)
        for idx, (pid, score) in enumerate(pid2score):
            rank_f.write(f"{qid}\t{pid}\t{idx+1}\t{score}\n")

    
    ujson_f.close()
    rank_f.close()





if __name__ == "__main__":
    #p2s_1 = {k: v for k,v in zip([1,2,3,4,5], [2.4,2.3, 2.1,1.7, 1.5])}
    #p2s_2 = {k: v for k,v in zip([1,3,2,5,6], [2.3, 2.2, 2.0, 1.8, 1.6])}

    #print(merge_and_aggregate([p2s_1, p2s_2]))

    args = get_args()
    main(args)


