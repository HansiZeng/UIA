import os 
import pathlib
import random 
random.seed(4680)

import ujson 

SIM_RELATION = "is_similar_to"
COMPL_RELATION = "is_complementary_to"
REL_RELATION = "is_relevant_to"

in_dir = "/work/hzeng_umass_edu/ir-research/joint_modeling_search_and_rec/datasets/unified_kgc/unified_user/"
in_dir += "sequential_train_test/hlen_4_randneg"

out_dir = os.path.join(in_dir, "without_context")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

uid_to_sim_aid = {}
uid_to_compl_aid = {}
uid_to_search_qid = {}
small_uid_to_sim_aid = {}
small_uid_to_search_qid = {}

uid_to_sim_pid = {}
uid_to_compl_pid = {}
uid_to_rel_pid = {}
small_uid_to_sim_pid = {}
small_uid_to_rel_pid = {}

eid_path = "/work/hzeng_umass_edu/ir-research/joint_modeling_search_and_rec/datasets/unified_kgc/all_entities.tsv"
eid_to_text = {}
with open(eid_path) as fin:
    for line in fin:
        eid, text = line.strip().split("\t")
        eid_to_text[int(eid)] = text

# create small test for sim_rec and search
if not os.path.exists(os.path.join(in_dir, "sim_rec_sequential.test.small.json")):
    with open(os.path.join(in_dir, "sim_rec_sequential.test.json")) as fin:
        tmp_data = []
        for line in fin:
            tmp_data.append(ujson.loads(line))
    assert len(tmp_data) >= 10_000
    with open(os.path.join(in_dir, "sim_rec_sequential.test.small.json"), "w") as fout:
        sampled_data = random.sample(tmp_data, k=10_000)
        for example in sampled_data:
            fout.write(ujson.dumps(example) + "\n")

if not os.path.exists(os.path.join(in_dir, "search_sequential.test.small.json")):
    with open(os.path.join(in_dir, "search_sequential.test.json")) as fin:
        tmp_data = []
        for line in fin:
            tmp_data.append(ujson.loads(line))
    assert len(tmp_data) >= 10_000
    with open(os.path.join(in_dir, "search_sequential.test.small.json"), "w") as fout:
        sampled_data = random.sample(tmp_data, k=10_000)
        for example in sampled_data:
            fout.write(ujson.dumps(example) + "\n")


fn_to_data = {
    os.path.join(in_dir, "sim_rec_sequential.test.json"): (uid_to_sim_aid, uid_to_sim_pid),
    os.path.join(in_dir, "compl_rec_sequential.test.json"): (uid_to_compl_aid, uid_to_compl_pid),
    os.path.join(in_dir, "search_sequential.test.json"): (uid_to_search_qid, uid_to_rel_pid),

    os.path.join(in_dir, "sim_rec_sequential.test.small.json"): (small_uid_to_sim_aid, small_uid_to_sim_pid),
    os.path.join(in_dir, "search_sequential.test.small.json"): (small_uid_to_search_qid, small_uid_to_rel_pid),
}
        

for fn, (uid_to_key, uid_to_value) in fn_to_data.items():
    with open(fn) as fin:
        for line in fin:
            example = ujson.loads(line)
            uid, qid, pid = example["uid"], example["query_ids"][-1], example["value_ids"][-1]
            uid_to_key[uid] = qid 
            uid_to_value[uid] = pid 

fn_to_query = {
    os.path.join(out_dir, "uid_anchors.test.sim.tsv"): (uid_to_sim_aid, SIM_RELATION),
    os.path.join(out_dir, "uid_anchors.test.compl.tsv"): (uid_to_compl_aid, COMPL_RELATION),
    os.path.join(out_dir, "uid_queries.test.search.tsv"): (uid_to_search_qid, REL_RELATION),

    os.path.join(out_dir, "uid_anchors.test.sim.small.tsv"): (small_uid_to_sim_aid, SIM_RELATION),
    os.path.join(out_dir, "uid_queries.test.search.small.tsv"): (small_uid_to_search_qid, REL_RELATION),
}
for fn, (uid_to_eid, relation) in fn_to_query.items():
    with open(fn, "w") as fout:
        for uid, eid in uid_to_eid.items():
            fout.write(f"{uid}\t{eid_to_text[eid]}\t{relation}\n")

fn_to_reldata = {
    os.path.join(out_dir, "urels.test.sim.tsv"): uid_to_sim_pid,
    os.path.join(out_dir, "urels.test.compl.tsv"): uid_to_compl_pid,
    os.path.join(out_dir, "urels.test.search.tsv"): uid_to_rel_pid,
}
for fn, reldata in fn_to_reldata.items():
    with open(fn, "w") as fout:
        for uid, relpid in reldata.items():
            fout.write(f"{uid}\tQ0\t{relpid}\t{1}\n")
        




            

