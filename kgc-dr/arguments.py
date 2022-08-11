from dataclasses import dataclass, field
from typing import Union, Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="bert-base-uncased")
    use_compress: bool = field(default=False)
    compress_dim: int = field(default=768)
    apply_tanh: bool = field(default=False, metadata={"help": "Apply the tanh activiation on the output vector."})
    independent_encoders: bool = field(default=False)

@dataclass
class DataTrainArguments:
    entites_path: str = field(default="/home/jupyter/jointly_rec_and_search/datasets/kgc/all_entites.tsv")
    a2cp_path: str  = field(default="/home/jupyter/jointly_rec_and_search/datasets/kgc/train/bm25_neg/max5_triples.a2cp.tsv")
    a2sp_path: str  = field(default="")
    q2a_path: str  = field(default="")
    c2cp_path: str  = field(default="")
    c2sp_path: str  = field(default="")
    q2c_path: str  = field(default="")

    experiment_folder: str  = field(default="")
    run_folder: str = field(default="experiment")
    log_dir: str = field(default="log/")
    model_save_dir: str = field(default="models")
    seed: int =  field(default=4680) 
    show_progress: bool = field(default=True)
    logging_steps: int = field(default=50)
    evaluate_steps: int = field(default=1000)
    model_pretrained_path: Optional[str] = field(default=None)
    resume: bool = field(default=False)
    resume_path: Optional[str] = field(default=None)

    learning_rate: float = field(default=7e-6)
    weight_decay: float =  field(default=0.01)
    adam_epsilon: float = field(default=1e-8)
    max_grad_norm: float =  field(default=1.0)
    num_train_epochs: int =  field(default=4)
    warmup_steps: int = field(default=4000)

    tokenizer_name_or_path: str = field(default="bert-base-uncased")
    max_head_text_len: int =  field(default=128)
    max_tail_text_len: int =  field(default=128)
    use_fp16: bool = field(default=True)
    train_batch_size: int = field(default=128)

    

    n_gpu: int = field(default=1)
    local_rank: int = field(default=-1)
    
@dataclass
class SearchDataTrainArguments:
    entites_path: str = field(default="/home/jupyter/jointly_rec_and_search/datasets/kgc/all_entites.tsv")
    train_q2p_path: str  = field(default="/home/jupyter/jointly_rec_and_search/datasets/kgc_search/train/bm25_neg/max2_triples.train_q2p.tsv")
    q2p_path: str  = field(default="")
    p2sp_path: str  = field(default="")
    p2cp_path: str  = field(default="")

    experiment_folder: str  = field(default="")
    run_folder: str = field(default="experiment")
    log_dir: str = field(default="log/")
    model_save_dir: str = field(default="models")
    seed: int =  field(default=4680) 
    show_progress: bool = field(default=True)
    logging_steps: int = field(default=50)
    evaluate_steps: int = field(default=1000)
    model_pretrained_path: Optional[str] = field(default=None)
    resume: bool = field(default=False)
    resume_path: Optional[str] = field(default=None)

    learning_rate: float = field(default=7e-6)
    weight_decay: float =  field(default=0.01)
    adam_epsilon: float = field(default=1e-8)
    max_grad_norm: float =  field(default=1.0)
    num_train_epochs: int =  field(default=4)
    warmup_steps: int = field(default=4000)

    tokenizer_name_or_path: str = field(default="bert-base-uncased")
    max_head_text_len: int =  field(default=128)
    max_tail_text_len: int =  field(default=128)
    use_fp16: bool = field(default=True)
    train_batch_size: int = field(default=128)

    

    n_gpu: int = field(default=1)
    local_rank: int = field(default=-1)
    
@dataclass
class UnifiedDataTrainArguments:
    entites_path: str = field(default="")
    task: str = field(default="")
    
    # common paths for recommendation
    a2sp_path: str = field(default="")
    a2cp_path: str = field(default="")
    q2a_path: str = field(default="")
    
    # similar recommendation
    s2sp_path: str = field(default="")
    s2cp_path: str = field(default="")
    q2s_path: str = field(default="")
    
    # complementary recommendation
    c2sp_path: str = field(default="")
    c2cp_path: str = field(default="")
    q2c_path: str = field(default="")
    
    # paths for search
    train_q2p_path: str  = field(default="")
    q2p_path: str  = field(default="")
    p2sp_path: str  = field(default="")
    p2cp_path: str  = field(default="")

    experiment_folder: str  = field(default="")
    run_folder: str = field(default="experiment")
    log_dir: str = field(default="log/")
    model_save_dir: str = field(default="models")
    seed: int =  field(default=4680) 
    show_progress: bool = field(default=True)
    logging_steps: int = field(default=50)
    evaluate_steps: int = field(default=1000)
    model_pretrained_path: Optional[str] = field(default=None)
    resume: bool = field(default=False)
    resume_path: Optional[str] = field(default=None)

    learning_rate: float = field(default=7e-6)
    weight_decay: float =  field(default=0.01)
    adam_epsilon: float = field(default=1e-8)
    max_grad_norm: float =  field(default=1.0)
    num_train_epochs: int =  field(default=4)
    warmup_steps: int = field(default=4000)
    max_global_steps: int = field(default=10000)

    tokenizer_name_or_path: str = field(default="bert-base-uncased")
    max_head_text_len: int =  field(default=128)
    max_tail_text_len: int =  field(default=128)
    use_fp16: bool = field(default=True)
    train_batch_size: int = field(default=128)

    

    n_gpu: int = field(default=1)
    local_rank: int = field(default=-1)
    
@dataclass
class RetrievalArguments:
    passages_path: str = field(default="")
    index_dir: str =  field(default="")
    tokenizer_name_or_path: str = field(default="bert-base-uncased")
    max_length: int = field(default=256)
    pretrained_path: str = field(default="")
    batch_size: int = field(default=128)
    chunk_size: int = field(default=20_000)

    top_k: int = field(default=1000)
    output_path: Optional[str] = field(default=None)
    queries_path: Optional[str] = field(default=None)
    query_max_len: int = field(default=256)
    index_path: Optional[str] = field(default=None)

    n_gpu: int = field(default=1)
    local_rank: int = field(default=-1)



if __name__ == "__main__":
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ModelArguments, DataTrainArguments))
    model_args, args = parser.parse_args_into_dataclasses()
    print("use_compress  = {}, n_gpu = {}, local_rank = {}".format(model_args.use_compress, args.n_gpu, args.local_rank))
    print("show_progress = {}, use_fp16 = {}".format(args.show_progress, args.use_fp16))
    print(model_args.__dict__)
