import sys 
sys.path += ["./"]
import os 
import pickle 
import argparse
from typing import Optional, Union, List, Dict
import glob
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
import time
import random
import yaml
from pathlib import Path
import shutil

import torch 
import torch.nn as nn 
from transformers.tokenization_utils_base import BatchEncoding
import torch.distributed as dist
from transformers import (AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup, HfArgumentParser)
import numpy as np 
from tqdm import tqdm, trange


from dataset import KGCTripleDataset
from utils import MetricMonitor, AverageMeter, is_first_worker
from modeling.dual_encoder import ContrastiveDualEncoder
from arguments import ModelArguments, UnifiedDataTrainArguments

import torch.cuda.amp as amp 

BLOCK_SIZE = 50_000
HIDDEN_SIZE = 768
PAD_REL_PID = -1
target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def save_checkpoint(state, is_best, filename="checkpoint.pt.tar"):
    torch.save(state, filename)
    if is_best:
        p = Path(filename)
        shutil.copyfile(filename, os.path.join(p.parent, "model_best.pth.tar"))

def batch_to_device(batch, target_device: torch.device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
        if isinstance(batch[key], dict) or isinstance(batch[key], BatchEncoding):
            for sub_key in batch[key]:
                if isinstance(batch[key][sub_key], torch.Tensor):
                    batch[key][sub_key] = batch[key][sub_key].to(target_device)

    return batch

def write_train_logs(epoch, step, loss_val, lr, filename, cutoff=10, **kwargs):
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write(f"epoch\tstep\tloss_val\tlr")
            for k in kwargs:
                f.write(f"\t{k}")
            f.write("\n")
    else:
        with open(filename, "a") as f:
            f.write(f"{epoch}\t{step}\t{loss_val:.3f}\t{lr:.10f}")
            for k,v in kwargs.items():
                f.write(f"\t{v:.3f}")
            f.write("\n")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_args():
    parser = HfArgumentParser((UnifiedDataTrainArguments, ModelArguments))
    args, model_args = parser.parse_args_into_dataclasses()
    
    if args.local_rank < 1:
        args.run_folder = os.path.join(args.experiment_folder, args.run_folder+"_"+time.strftime("%m-%d_%H%M%S", time.localtime()))
        
        if not os.path.exists(args.run_folder):
            os.mkdir(args.run_folder)

        args.log_dir = os.path.join(args.run_folder, args.log_dir)
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        
        args.model_save_dir = os.path.join(args.run_folder, args.model_save_dir)
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)

        config_path = os.path.join(args.run_folder, "config.yaml")
        with open(config_path, "w") as f:
            yaml_dict = {"args": args.__dict__, "model_args": model_args.__dict__}
            yaml.dump(yaml_dict, f)

        fh = logging.FileHandler(filename=os.path.join(args.run_folder, "train_logs.log"))
        logger.addHandler(fh)

    return args, model_args

def train(args, model_args):
    loss_avg_meter = AverageMeter()
    test_monitor = MetricMonitor()
    
    # assign task
    assert args.task in ["SIMILAR_REC", "COMPL_REC", "RELEVANT_SEARCH", "JOINT"], args.task
    if "REC" in args.task:
        if args.a2sp_path == "None":
            args.a2sp_path = None
        if args.a2cp_path == "None":
            args.a2cp_path = None
        if args.q2a_path == "None":
            args.q2a_path = None
        
        if args.task == "SIMILAR_REC":
            assert args.a2sp_path != None
            if args.s2sp_path == "None":
                args.s2sp_path = None
            if args.s2cp_path == "None":
                args.s2cp_path = None
            if args.q2s_path == "None":
                args.q2s_path = None
        elif args.task == "COMPL_REC":
            assert args.a2cp_path != None
            if args.c2sp_path == "None":
                args.c2sp_path = None
            if args.c2cp_path == "None":
                args.c2cp_path = None
            if args.q2c_path == "None":
                args.q2c_path = None
        else:
            raise ValueError(f"task: {args.task} is not legal.")
    elif "SEARCH" in args.task:
        raise NotImplementedError()
    elif "JOINT" in args.task:
        if args.a2sp_path == "None":
            args.a2sp_path = None
        if args.a2cp_path == "None":
            args.a2cp_path = None
        if args.q2a_path == "None":
            args.q2a_path = None
    else:
        raise ValueError(f"task: {args.task} is not legal.")
    
    # dataset
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if args.distributed:
        assert args.train_batch_size % args.nranks == 0
        if args.task == "SIMILAR_REC":
            train_dataset = KGCTripleDataset.create_from_simrec_task_files(args.entites_path, tokenizer, 
                                                                           args.max_head_text_len, args.max_tail_text_len, 
                                                                        a2sp_path=args.a2sp_path, a2cp_path=args.a2cp_path, 
                                                                           q2a_path=args.q2a_path, s2sp_path=args.s2sp_path, 
                                                                           s2cp_path=args.s2cp_path, q2s_path=args.q2s_path,
                                                                           rank=args.local_rank, nranks=args.nranks)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size//args.nranks, shuffle=True, 
                                                    num_workers=4,collate_fn=train_dataset.collate_fn, drop_last=True)
        elif args.task == "JOINT":
            train_dataset = KGCTripleDataset.create_from_joint_task_files(args.entites_path, tokenizer, 
                                                                           args.max_head_text_len, args.max_tail_text_len, 
                                                                        a2sp_path=args.a2sp_path, a2cp_path=args.a2cp_path, 
                                                                           q2a_path=args.q2a_path, 
                                                                           rank=args.local_rank, nranks=args.nranks)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size//args.nranks, shuffle=True, 
                                                    num_workers=4,collate_fn=train_dataset.collate_fn, drop_last=True)
        else:
            raise NotImplementedError
        
    else:
        raise NotImplementedError

    # for model 
    if args.model_pretrained_path: 
        assert not args.resume
        model = ContrastiveDualEncoder.from_pretrained(args.model_pretrained_path, model_args)
    else:
        model = ContrastiveDualEncoder(model_args)
    model.cuda()
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.local_rank],
                                                            output_device=args.local_rank)
    if args.distributed:
        dist.barrier()

    # for optimizer
    t_total = len(train_dataloader) * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    # start training 
    if args.local_rank < 1:
        logger.info("***** training info *****")
        logger.info(f"Total number of samples in train_dataset = {len(train_dataset)}")
        logger.info(f"Total training steps = {t_total}\nnum epochs = {args.num_train_epochs}\nlr = {args.learning_rate}\nusef_fp16 = {args.use_fp16}")
        if args.distributed:
            logger.info(f"process rank: {args.local_rank}\nngpu: {args.ngpu}\nworld size: {args.nranks}")
        logger.info("***** model_args *****")
        logger.info(str(model_args))
        
    set_seed(args)

    model.zero_grad()
    global_step = 0
    best_metric = 0. # record mrr@10
    start_epoch = 0

    # resume training
    if args.resume:
        raise NotImplementedError

    # start training 
    if args.use_fp16:
        scaler = amp.GradScaler()
    for epoch_idx, epoch in enumerate(trange(start_epoch, args.num_train_epochs, desc="Epoch")):
        for step, batch in enumerate(tqdm(train_dataloader, desc="query_epoch", disable=args.local_rank >=1, total=len(train_dataloader))):
            model.train()
            batch = batch_to_device(batch, target_device)
            with amp.autocast(enabled=args.use_fp16):
                loss = model(batch["hr_texts"], batch["tail_texts"])
            if args.use_fp16:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1 
            
            if args.local_rank < 1:
                train_loss = loss.item() 
                loss_avg_meter.update(train_loss)
                
                if global_step % args.logging_steps == 0:
                    cur_loss = loss_avg_meter.avg 
                    write_train_logs(epoch+1, global_step, cur_loss, scheduler.get_lr()[0], 
                                    filename=os.path.join(args.log_dir, "train_logs.log"))
                    
                    loss_avg_meter.reset()

                if (global_step) % args.evaluate_steps == 0:
                    save_dir = os.path.join(args.model_save_dir, f"checkpoint_{global_step}")
                    if args.distributed:
                        model.module.save_pretrained(save_dir)
                    else:
                        model.save_pretrained(save_dir)

                    traininig_state_dict = {
                            'epoch': epoch+1,
                            'global_step': global_step,
                            'optimizer' : optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()
                            }
                    torch.save(traininig_state_dict, os.path.join(save_dir, "training_state.pt"))
        if global_step >= args.max_global_steps:
            break
    
    if args.local_rank < 1:
        save_dir = os.path.join(args.model_save_dir, f"checkpoint_latest")
        if args.distributed:
            model.module.save_pretrained(save_dir)
        else:
            model.save_pretrained(save_dir)



if __name__ == "__main__":
    args, model_args = get_args()
    set_env(args)
    train(args, model_args)
