import os 
import pickle 

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import ujson 


SIM_RELATION = "is_similar_to"
COMPL_RELATION = "is_complementary_to"
REL_RELATION = "is_relevant_to"

in_dir = "/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/"
    
train_sim_data, train_compl_data, train_search_data = None, None, None
data_fns = [
    os.path.join(in_dir, "train_sim_recs.csv"),
    os.path.join(in_dir, "train_compl_recs.csv"),
    os.path.join(in_dir, "train_searchs.csv"),
]
datas = []
for fn in data_fns:
    datas.append(pd.read_csv(fn, index_col=0))
train_sim_data, train_compl_data, train_search_data = datas


datas = []
test_sim_data, test_compl_data, test_search_data = None, None, None
data_fns = [
    os.path.join(in_dir, "test_sim_recs.csv"),
    os.path.join(in_dir, "test_compl_recs.csv"),
    os.path.join(in_dir, "test_searchs.csv"),
]
datas = []
for fn in data_fns:
    datas.append(pd.read_csv(fn, index_col=0))
test_sim_data, test_compl_data, test_search_data = datas
datas = None

root_dir="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/"
eid_to_text = {}
with open(os.path.join(root_dir, "all_entities.tsv")) as fin:
    for line in fin:
        eid, text = line.strip().split("\t")
        eid_to_text[int(eid)] = text

train_sim_data["relation"] = SIM_RELATION
test_sim_data["relation"] = SIM_RELATION
train_compl_data["relation"] = COMPL_RELATION
test_compl_data["relation"] = COMPL_RELATION
train_search_data["relation"] = REL_RELATION
test_search_data["relation"] = REL_RELATION

train_sim_data.rename({"aid": "hid", "sim_pids": "tids"}, axis=1, inplace=True)
test_sim_data.rename({"aid": "hid", "sim_pids": "tids"}, axis=1, inplace=True)
train_compl_data.rename({"aid": "hid", "compl_pids": "tids"}, axis=1, inplace=True)
test_compl_data.rename({"aid": "hid", "compl_pids": "tids"}, axis=1, inplace=True)
train_search_data.rename({"qid": "hid", "rel_pids": "tids"}, axis=1, inplace=True)
test_search_data.rename({"qid": "hid", "rel_pids": "tids"}, axis=1, inplace=True)

train_merge_data = pd.concat([train_sim_data, train_compl_data, train_search_data])
train_merge_data["date_time"] = pd.to_datetime(train_merge_data["date_time"])
train_merge_data = train_merge_data.sort_values(by=["uid", "date_time"])

print("length of sim_rec train and test = {:,}, {:,}".format(len(train_sim_data), len(test_sim_data)))
print("length of compl_rec train and test = {:,}, {:,}".format(len(train_compl_data), len(test_compl_data)))
print("length of search train and test = {:,}, {:,}".format(len(train_search_data), len(test_search_data)))
print("length of train_merge_data = {:,}".format(len(train_merge_data)))
print("number of entites = {:,}".format(len(eid_to_text)))

assert set(test_sim_data.uid).issubset(set(train_sim_data.uid)) \
and set(test_compl_data.uid).issubset(set(train_compl_data.uid)) \
and set(test_search_data.uid).issubset(set(train_search_data.uid))
assert len(train_merge_data) == len(train_sim_data) + len(train_compl_data) + len(train_search_data)
print("test users for each data are subset of their corresponding train users.")

SIM_RELATION = "is_similar_to"
COMPL_RELATION = "is_complementary_to"
REL_RELATION = "is_relevant_to"

MAX_LEN=20
out_dir = os.path.join(in_dir, "users_divided_by_group")
if not os.path.exists(out_dir):
    os.mkdir(out_dir) 

seq_examples_list = []
prefixes_to_datas= {
    os.path.join(out_dir, "search_sequential"): (train_search_data, test_search_data, search_uid_groups, "urels.search.test.tsv"),
    os.path.join(out_dir, "sim_rec_sequential"): (train_sim_data, test_sim_data, sim_uid_groups, "urels.sim.test.tsv"),
    os.path.join(out_dir, "compl_rec_sequential"): (train_compl_data, test_compl_data, compl_uid_groups, "urels.compl.test.tsv"),
}
for prefix, (train_data, test_data, uid_groups, urel_path) in prefixes_to_datas.items():
    test_uid_to_pospids = {}
    for uid in tqdm(train_data.uid.unique(), desc=prefix.split("/")[-1]):
        # for test
        test_row = test_data[test_data.uid == uid]
        if len(test_row) == 0:
            continue
        assert len(test_row) == 1, test_row
            
        if "search_sequential" in prefix:
            test_uid_to_pospids[uid] = eval(test_row.iloc[0].tids)
        elif "sim_rec_sequential" in prefix:
            test_uid_to_pospids[uid] = eval(test_row.iloc[0].tids)
        elif "compl_rec_sequential" in prefix:
            test_uid_to_pospids[uid] = eval(test_row.iloc[0].tids)
        else:
            raise ValueError(f"{prefix} not valid.")
            
        with open(os.path.join(out_dir, urel_path), "w") as fout:
            for uid, pos_pids in test_uid_to_pospids.items():
                for pos_pid in pos_pids:
                    fout.write(f"{uid}\tQ0\t{pos_pid}\t{1}\n")
                    
                    
# 2, retrieval
# queries.test.tsv
python retriever/user_retrieve_top_passages.py \
--eid_path=$EID_PATH \
--query_examples_path=$QUERIES_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.search.small.test.run \
--index_path=$INDEX_PATH \
--batch_size=128 \
--max_length=128

python retriever/user_retrieve_top_passages.py \
--eid_path=$EID_PATH \
--query_examples_path=$SIM_ANCHORS_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.sim.small.test.run \
--index_path=$INDEX_PATH \
--batch_size=128 \
--max_length=128

python retriever/user_retrieve_top_passages.py \
--eid_path=$EID_PATH \
--query_examples_path=$COMPL_ANCHORS_PATH \
--pretrained_path=$PRETRAINED_PATH \
--output_path=${OUTPUT_PATH}.compl.small.test.run \
--index_path=$INDEX_PATH \
--batch_size=128 \
--max_length=128