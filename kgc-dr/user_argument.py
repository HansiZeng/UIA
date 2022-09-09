from dataclasses import dataclass, field
from typing import Union, Optional

@dataclass 
class ModelArguments: 
    # tunable parameters
    backbone_trainable: bool = field(default=False)
    apply_position_embedding: bool = field(default=False)
    
    # should be true when training
    value_from_gru: bool = field(default=False)
    apply_value_layer: bool = field(default=False)
    apply_zero_attention: bool = field(default=False)
    
    seq_output_act_fn: str = field(default="gelu")
    backbone_path: str = field(
        default="/home/jupyter/unity_jointly_rec_and_search/experiments/unified_kgc/experiment_09-07_005742/models/checkpoint_latest")
    output_id_attentions: bool = field(default=False)
    hidden_size: int = field(default=768)
    num_attention_heads: int = field(default=12)
    hidden_dropout_prob: float = field(default=0.1)
    attention_probs_dropout_prob: float = field(default=0.1)

    layer_norm_eps: float = field(default=1e-12)
    max_position_embeddeings: int = field(default=14)
    
@dataclass
class MergerModelArguments:
    # tunable parameters
    backbone_trainable: bool = field(default=False)
    apply_position_embedding: bool = field(default=False)
    apply_value_layer_for_passage_emb: bool = field(default=False)
    apply_zero_attention: bool = field(default=False)
    
    seq_output_act_fn: str = field(default="gelu")
    backbone_path: str = field(
        default="/home/jupyter/unity_jointly_rec_and_search/experiments/unified_kgc/experiment_09-07_005742/models/checkpoint_latest")
    output_id_attentions: bool = field(default=False)
    hidden_size: int = field(default=768)
    num_attention_heads: int = field(default=12)
    hidden_dropout_prob: float = field(default=0.1)
    attention_probs_dropout_prob: float = field(default=0.1)

    layer_norm_eps: float = field(default=1e-12)
    max_position_embeddeings: int = field(default=14)

@dataclass
class DataTrainArguments:
    # tunnable parameters
    examples_path: str = field(default="")
    learning_rate: float = field(default=7e-4)
    train_batch_size: int = field(default=128)
    num_train_epochs: int =  field(default=4)
    
    eid_path: str = field(default="")
    experiment_folder: str  = field(default="")
    run_folder: str = field(default="experiment")
    log_dir: str = field(default="log/")
    model_save_dir: str = field(default="models")
    seed: int =  field(default=4680) 
    show_progress: bool = field(default=True)
    logging_steps: int = field(default=50)
    evaluate_steps: int = field(default=100000000000)
    model_pretrained_path: Optional[str] = field(default=None)
    resume: bool = field(default=False)
    resume_path: Optional[str] = field(default=None)

    weight_decay: float =  field(default=0.01)
    adam_epsilon: float = field(default=1e-8)
    max_grad_norm: float =  field(default=1.0)
    warmup_steps: int = field(default=4000)
    max_global_steps: int = field(default=100000000000)

    tokenizer_name_or_path: str = field(default="bert-base-uncased")
    max_text_len: int =  field(default=128)
    use_fp16: bool = field(default=True)    

    n_gpu: int = field(default=1)
    local_rank: int = field(default=-1)
    
@dataclass
class RetrievalArguments:
    tokenizer_name_or_path: str = field(default="bert-base-uncased")
    pretrained_path: str = field(default="")

    passages_path: str = field(default="")
    index_dir: str =  field(default="")
    eid_path: str = field(default="")
    query_examples_path: str = field(default="")

    max_length: int = field(default=128)
    batch_size: int = field(default=128)
    chunk_size: int = field(default=20_000)

    top_k: int = field(default=1000)
    output_path: Optional[str] = field(default=None)
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
