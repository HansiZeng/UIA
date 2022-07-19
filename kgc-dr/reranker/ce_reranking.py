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

from modeling.cross_encoder import CrossEncoder
from dataset import QueryPassagePairDataset
from arguments import CEModelArguments, RerankingArguments

def batch_to_cuda(batch):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda()
        if isinstance(batch[key], dict):
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
    parser = HfArgumentParser((RerankingArguments, CEModelArguments))
    args, model_args = parser.parse_args_into_dataclasses()

    if args.local_rank < 1:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        assert os.path.isdir(args.output_dir), args.output_dir

    return args, model_args 


def main(args, model_args):
    # for model
    if args.pretrained_path:
        model = CrossEncoder.from_pretrained(args.pretrained_path)
        print("Initialize model from pretrained path from = {}".format(args.pretrained_path))
    else:
        model = CrossEncoder(model_args)
        print("Initialize model from model name = {}".format(model_args.model_name_or_path))
    model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.local_rank],
                                                            output_device=args.local_rank)
    if args.distributed:
        dist.barrier()
    
    # for tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # for dataset 
    dataset = QueryPassagePairDataset.create_from_ujson_file(args.ranking_path, args.queries_path, args.passages_path, tokenizer, 
                                                            max_length=args.max_length, rank=args.local_rank, nranks=args.nranks)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=False, num_workers=1)

    # start reranking (inference)
    qid_to_rankdata = OrderedDict()
    model.eval()
    for idx, batch in enumerate(tqdm(dataloader, desc="reranking", disable=args.local_rank>1, total=len(dataloader))):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.use_fp16):
                batch_encoded_text = batch_to_cuda(batch["encoded_QueryPassage"])
                query_ids = batch["query_id"]
                passage_ids = batch["passage_id"]
                assert isinstance(query_ids, list) and isinstance(passage_ids, list)

                scores = model(batch_encoded_text)
                scores = scores.cpu().tolist()

        for qid, pid, score in zip(query_ids, passage_ids, scores):
            if qid not in qid_to_rankdata:
                qid_to_rankdata[qid] = OrderedDict()
                qid_to_rankdata[qid][pid] = score 
            else:
                qid_to_rankdata[qid][pid] = score


    # output
    suffix = args.queries_path.split(".")[-2]
    if suffix not in ["dev", "train"]:
        suffix = args.suffix
    print("suffix for output = {}".format(suffix))

    if args.pretrained_path:
        output_path = os.path.join(args.output_dir, f"checkpoint250000_{args.local_rank}.{suffix}.run")
    else:
        output_path = os.path.join(args.output_dir, f"MiniLM-L-6-v2_{args.local_rank}.{suffix}.run")

    with open(output_path, "w") as fout:
        for qid in qid_to_rankdata:
            rank_data = qid_to_rankdata[qid]
            sorted_data = {k: v for k, v in sorted(rank_data.items(), key=lambda item: item[1], reverse=True)}
            for rank, (pid, score) in enumerate(sorted_data.items()):
                fout.write(f"{qid}\t{pid}\t{rank+1}\t{score}\n")

if __name__ == "__main__":
    args, model_args = get_args()
    set_env(args)
    main(args, model_args)


