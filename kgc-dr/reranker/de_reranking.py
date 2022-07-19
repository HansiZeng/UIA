import sys 
sys.path += ["./"]
import os 
from collections import OrderedDict
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import HfArgumentParser, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import BatchEncoding
import torch.distributed as dist
from tqdm import tqdm 
import ujson

from modeling.dual_encoder import DualEncoder
from dataset import QueryPassagePairDataset
from arguments import ModelArguments, RerankingArguments

def batch_to_cuda(batch):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda()
        if isinstance(batch[key], dict) or isinstance(batch[key], BatchEncoding):
            for sub_key in batch[key]:
                if isinstance(batch[key][sub_key], torch.Tensor):
                    batch[key][sub_key] = batch[key][sub_key].cuda()

    return batch

def set_env(args):
    args.distributed = False
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group("nccl")
        args.device = device
        args.distributed = dist.get_world_size() > 1
        args.nranks = dist.get_world_size()
        args.ngpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cuda")

def get_args():
    parser = HfArgumentParser((RerankingArguments, ModelArguments))
    args, model_args = parser.parse_args_into_dataclasses()

    if args.local_rank < 1:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

    return args, model_args 


def main(args, model_args):
    # rerank or give score for training_example?
    task = "rerank"
    if args.example_path:
        assert args.ranking_path == None, args.ranking_path
        task = "assign_score"
        print(f"The task is = {task} for training example: {args.example_path}!")
    
    if task != "assign_score":
        raise NotImplementedError

    # for model 
    model = DualEncoder.from_pretrained(args.pretrained_path, model_args)
    model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.local_rank],
                                                            output_device=args.local_rank)
    if args.distributed:
        dist.barrier()
    
    # for tokenizer. DIRTY implementation !!!!!
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # for dataset 
    dataset = QueryPassagePairDataset.create_from_10relT_20neg_file(args.example_path, args.queries_path, args.passages_path, tokenizer, 
                                                            query_max_len=args.query_max_len, passage_max_len=args.passage_max_len,
                                                            rank=args.local_rank, nranks=args.nranks)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.de_collate_fn, shuffle=False, num_workers=1)

    # start reranking (inference)
    qid_to_rankdata = OrderedDict()
    model.eval()
    for idx, batch in enumerate(tqdm(dataloader, desc="reranking", disable=args.local_rank>1, total=len(dataloader))):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.use_fp16):
                batch = batch_to_cuda(batch)
                qids = batch["qid"]
                pids = batch["pid"]
                assert isinstance(qids, list) and isinstance(pids, list)

                scores = model(batch["query"], batch["passage"], is_nway=False)
                scores = scores.cpu().tolist()

        for qid, pid, score in zip(qids, pids, scores):
            if qid not in qid_to_rankdata:
                qid_to_rankdata[qid] = OrderedDict()
                qid_to_rankdata[qid][pid] = score 
            else:
                qid_to_rankdata[qid][pid] = score

    # start writing 
    with open(args.example_path, "r") as fin:
        for line in fin:
            example = ujson.loads(line)
            len_relT_pids = len(example["relT_pids"])
            len_most_hard_pids = len(example["most_hard_pids"])
            len_semi_hard_pids = len(example["semi_hard_pids"])
            break
    print("len of three pids fields = {}, {}, {}".format(len_relT_pids, len_most_hard_pids, len_semi_hard_pids))

    ckpt_name = args.pretrained_path.strip("/").split("/")[-1]
    train_example_name = f"{len_relT_pids}_{len_most_hard_pids+len_semi_hard_pids}"
    output_path = os.path.join(args.output_dir, f"{args.local_rank}_{ckpt_name}_{train_example_name}_{args.boosting_round}.train.json")
    example = {}
    with open(output_path, "w") as fout:
        for qid, rankdata in qid_to_rankdata.items():
            example["qid"] = qid
            example["relT_pids"] = []
            example["most_hard_pids"] = []
            example["semi_hard_pids"] = []
            for idx, (pid, score) in enumerate(rankdata.items()):
                if idx < len_relT_pids:
                    example["relT_pids"].append((pid, score))
                elif idx >= len_relT_pids and idx < (len_relT_pids + len_most_hard_pids):
                    example["most_hard_pids"].append((pid, score))
                else:
                    example["semi_hard_pids"].append((pid,score)) 
            
            assert len(example["relT_pids"]) == len_relT_pids, len(example["relT_pids"])
            assert len(example["most_hard_pids"]) == len_most_hard_pids, len(example["most_hard_pids"])
            assert len(example["semi_hard_pids"]) == len_semi_hard_pids, len(example["semi_hard_pids"])

            fout.write(ujson.dumps(example) + "\n")
            example = {}



    

if __name__ == "__main__":
    args, model_args = get_args()
    set_env(args)
    main(args, model_args)
