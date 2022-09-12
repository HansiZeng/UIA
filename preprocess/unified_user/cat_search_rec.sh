#!/bin/bash


home_dir="/home/jupyter/unity_jointly_rec_and_search/datasets/unified_user/sequential_train_test/hlen_4_bm25"
train_search_path="${home_dir}/search_sequential.train.json"
test_search_path="${home_dir}/search_sequential.test.json"

train_sim_path="${home_dir}/sim_rec_sequential.train.json"
test_sim_path="${home_dir}/sim_rec_sequential.test.json"

train_compl_path="${home_dir}/compl_rec_sequential.train.json"
test_compl_path="${home_dir}/compl_rec_sequential.test.json"

if [ ! -d "${home_dir}/rec_search/" ]
then mkdir "${home_dir}/rec_search/"
fi 


dest_1="${home_dir}/rec_search/search_sim_sequential.train.json"
cat $train_search_path $train_sim_path > $dest_1
wc -l $dest_1

dest_3="${home_dir}/rec_search/search_compl_sequential.train.json"
cat $train_search_path $train_compl_path > $dest_3
wc -l $dest_3

dest_5="${home_dir}/rec_search/sim_compl_sequential.train.json"
cat $train_sim_path $train_compl_path > $dest_5
wc -l $dest_5

dest_7="${home_dir}/rec_search/search_sim_compl_sequential.train.json"
cat $train_search_path $train_sim_path $train_compl_path > $dest_7
wc -l $dest_7