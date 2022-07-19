from dataclasses import dataclass, field
from typing import Union, Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="distilbert-base-uncased")
    use_compress: bool = field(default=False)
    compress_dim: int = field(default=768)
    apply_tanh: bool = field(default=False, metadata={"help": "Apply the tanh activiation on the output vector."})
    independent_encoders: bool = field(default=False)

@dataclass
class DataTrainArguments:
    queries_path: str = field(default="")
    collection_path: str = field(default="")
    training_path: str = field(default="")

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

    tokenizer_name_or_path: str = field(default="distilbert-base-uncased")
    query_max_len: int =  field(default=30)
    passage_max_len: int =  field(default=256)
    use_fp16: bool = field(default=True)
    train_batch_size: int = field(default=8)

    distill_teacher_score: bool = field(default=False)

    n_gpu: int = field(default=1)
    local_rank: int = field(default=-1)
    
@dataclass
class JointDataTrainArguments(DataTrainArguments):
    title_path: str = field(default="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/collection_title.tsv")
    training_path: str = field(default="/home/jupyter/jointly_rec_and_search/datasets/rec_search/joint_1pos_1neg.train.tsv")

@dataclass 
class CETrainArguments(DataTrainArguments):
    max_length: int = field(default=300, metadata={"help": "hyperparameter for cross_encoder"}) 
    tokenizer_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

@dataclass
class BoostedTrainArguments(DataTrainArguments):
    apply_boosted_weight: bool = field(default=True)
    boosted_k: float = field(default=0.5)
    boosted_b: float = field(default=0.5)
    
@dataclass
class RetrievalArguments:
    passages_path: str = field(default="")
    index_dir: str =  field(default="")
    tokenizer_name_or_path: str = field(default="distilbert-base-uncased")
    max_length: int = field(default=256)
    is_query: bool = field(default=False)
    pretrained_path: str = field(default="")
    batch_size: int = field(default=128)
    chunk_size: int = field(default=20_000)

    top_k: int = field(default=1000)
    output_path: Optional[str] = field(default=None)
    queries_path: Optional[str] = field(default=None)
    query_max_len: int = field(default=30)
    index_path: Optional[str] = field(default=None)

    n_gpu: int = field(default=1)
    local_rank: int = field(default=-1)

@dataclass
class CEModelArguments:
    model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

@dataclass
class RerankingArguments:
    queries_path: str = field(default="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/queries.train.tsv")
    passages_path: str = field(default="/home/jupyter/jointly_rec_and_search/datasets/rec_search/search/collection.tsv")
    ranking_path: Optional[str] = field(default=None)
    example_path: Optional[str] = field(default=None)  # for traininig_example get score 
    output_dir: str = field(default="")
    suffix: str = field(default="")

    max_length: int = field(default=512, metadata={"help": "hyperparameter for cross_encoder"})
    batch_size: int = field(default=128)
    use_fp16: bool =  field(default=True)

    boosting_round: int = field(default=1)
    query_max_len: int =  field(default=30, metadata={"help": "hyperparameter for dual_encoder"})
    passage_max_len: int =  field(default=256, metadata={"help": "hyperparameter for dual_encoder"})
    pretrained_path: Optional[str] = field(default=None)

    n_gpu: int = field(default=1)
    local_rank: int = field(default=-1)


if __name__ == "__main__":
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ModelArguments, DataTrainArguments))
    model_args, args = parser.parse_args_into_dataclasses()
    print("use_compress  = {}, n_gpu = {}, local_rank = {}".format(model_args.use_compress, args.n_gpu, args.local_rank))
    print("show_progress = {}, use_fp16 = {}".format(args.show_progress, args.use_fp16))
    print(model_args.__dict__)
