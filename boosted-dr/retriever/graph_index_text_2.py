import os 
import sys 
sys.path += ["./"]

from transformers import  HfArgumentParser
import ujson 
import pickle
import numpy as np
import faiss 

from arguments import RetrievalArguments

def get_args():
    parser = HfArgumentParser(RetrievalArguments)
    args = parser.parse_args_into_dataclasses()
    assert len(args) == 1
    args = args[0]

    return args

def main(args):
    # check all file 
    plan_path = os.path.join(args.index_dir, "plan.json")
    with open(plan_path, "r") as fin:
        plan = ujson.load(fin)
    
    print(plan)
    nranks = plan["nranks"]
    num_chunks = plan["num_chunks"]
    index_path = plan["index_path"]
    
    # start index
    text_embs, text_ids = [], []
    for i in range(nranks):
        for chunk_idx in range(num_chunks):
            text_embs.append(np.load(os.path.join(args.index_dir,"embs_{}_{}.npy".format(i, chunk_idx))))
            text_ids.append(np.load(os.path.join(args.index_dir,"ids_{}_{}.npy".format(i, chunk_idx))))
    
    text_embs = np.concatenate(text_embs)
    text_embs = text_embs.astype(np.float32) if text_embs.dtype == np.float16 else text_embs
    text_ids = np.concatenate(text_ids)
    
    assert len(text_embs) == len(text_ids), (len(text_embs), len(text_ids))
    assert text_ids.ndim == 1, text_ids.shape
    print("embs dtype: ", text_embs.dtype, "embs size: ", text_embs.shape)
    print("ids dtype: ", text_ids.dtype)

    index = faiss.IndexFlatIP(text_embs.shape[1])
    index = faiss.IndexIDMap(index)

    #assert isinstance(text_ids, list)
    #text_ids = np.array(text_ids)

    index.add_with_ids(text_embs, text_ids)
    faiss.write_index(index, index_path)
    
    meta = {"text_ids": text_ids, "num_embeddings": len(text_ids)}
    with open(os.path.join(args.index_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
        
    # remove embs, ids
    for i in range(nranks):
        for chunk_idx in range(num_chunks):
            os.remove(os.path.join(args.index_dir,"embs_{}_{}.npy".format(i, chunk_idx)))
            os.remove(os.path.join(args.index_dir,"ids_{}_{}.npy".format(i, chunk_idx)))

if __name__ == "__main__":
    args = get_args()
    main(args)
