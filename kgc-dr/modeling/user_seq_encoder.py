from curses.ascii import isdigit
import math
from multiprocessing.sharedctypes import Value
import os
from typing import Optional

import torch 
import torch.nn as nn 

from .dual_encoder import DualEncoder
from .modeling_utils import pad_catted_tensor, get_extended_attention_mask, cat_padded_tensor, get_seq_last_output
from losses import SeqContrastiveLoss
from .activations import ACT2FN

class GRUOutput(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gru = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)
    
    def forward(self, input_tensor: torch.Tensor, seq_lengths: torch.LongTensor):
        """
        Args:
            input_tensor: [B, L, D]

        Returns:
            hidden_states: [B, L, D]
        """
        packed_tensor = torch.nn.utils.rnn.pack_padded_sequence(input_tensor, seq_lengths.cpu().numpy(), 
                                                                batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_tensor)
        hidden_states, _ =  torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return hidden_states


# highly copied from transformers.models.bert.modeling_bert.BertSelfAttention with some modifications
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        if config.value_from_gru and not config.apply_value_layer:
            self.value = nn.Identity()
        else:
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        #if not config.value_from_gru:
            #assert config.apply_value_layer, f"apply_value_layer must be true but got {config.apply_value_layer}"
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.apply_zero_attention = config.apply_zero_attention

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def cat_zero_vector_to_seq_first(self, input_tensor: torch.Tensor):
        # for zero attention, assume input_tensor with shape of BxLxD
        batch_size, _, hidden_size = input_tensor.size()
        zero_vector = torch.zeros(batch_size, 1, hidden_size).to(input_tensor.dtype).to(input_tensor.device)
        output_tensor = torch.cat([zero_vector, input_tensor], dim=1)
        return output_tensor

    def forward(
        self, 
        query_hidden_states: torch.Tensor,
        key_hidden_states: torch.Tensor,
        value_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor, 
        output_attentions: Optional[bool] = False
    ):
        query_layer = self.transpose_for_scores(self.query(query_hidden_states))
        
        if self.apply_zero_attention:
            key_layer = self.transpose_for_scores(self.cat_zero_vector_to_seq_first(self.key(key_hidden_states)))
            value_layer = self.transpose_for_scores(self.cat_zero_vector_to_seq_first(self.value(value_hidden_states)))
        else:
            key_layer = self.transpose_for_scores(self.key(key_hidden_states))
            value_layer = self.transpose_for_scores(self.value(value_hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #print("attention_scores shape, attention_mask shape: ",attention_scores.shape, attention_mask.shape)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        #print("context_layer before: ", context_layer.shape)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        #print("context_layer after: ", context_layer.shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)
    
    def forward(
        self,
        query_hidden_states: torch.Tensor,
        key_hidden_states: torch.Tensor,
        value_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor, 
        output_attentions: Optional[bool] = False
        ):
        self_outputs = self.self(query_hidden_states, key_hidden_states, value_hidden_states,
                                attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], query_hidden_states)

        if output_attentions:
            return attention_output + (self_outputs[1],)
        else:
            return (attention_output,)

class SeqOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = ACT2FN[config.seq_output_act_fn]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_tensor: torch.Tensor):
        hidden_states = self.act(self.dense(input_tensor))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class UserSeqEncoder(nn.Module):
    def __init__(self, model_args, backbone_args=None):
        super().__init__()
        self.model_args = model_args 

        # backbone
        assert os.path.isdir(model_args.backbone_path), model_args.backbone_path
        backbone_args = torch.load(os.path.join(model_args.backbone_path, "model_args.pt"))
        self.backbone = DualEncoder.from_pretrained(model_args.backbone_path, backbone_args)
        for param in self.backbone.parameters():
            param.requires_grad = model_args.backbone_trainable

        # attention module 
        self.attention = Attention(model_args)

        # gru 
        if model_args.value_from_gru:
            self.gru = GRUOutput(model_args)
        
        # finaly module
        self.out = SeqOutput(model_args)

        # loss_fn
        self.loss_fn = SeqContrastiveLoss()
        
    def forward(
            self,
            query_relation,
            context_key_relation,
            context_value,
            target_value,
            neg_value,
            seq_lengths,
            id_attention_masks,
            in_batch_mask,
            seq_last_output = False,
        ):
        query_outputs = self.query_embs(
                            query_relation, 
                            context_key_relation, 
                            context_value, 
                            seq_lengths, 
                            id_attention_masks, 
                            seq_last_output,
                        )
        query_emb = query_outputs[0] # either be L1xD for training or BxD for inference.
        passage_emb = self.passage_embs(target_value)
        if neg_value != None:
            neg_passage_emb = self.passage_embs(neg_value)
        else:
            neg_passage_emb = None

        # compute loss 
        loss = self.loss_fn(query_emb, passage_emb, in_batch_mask, neg_passage_emb)

        return loss

    def query_embs(
        self,
        query_relation,
        context_key_relation,
        context_value,
        seq_lengths,
        id_attention_masks,
        seq_last_output,
    ):
        query_hidden_states = pad_catted_tensor(self.backbone.query_embs(query_relation), seq_lengths) # BxLxD
        key_hidden_states = pad_catted_tensor(self.backbone.query_embs(context_key_relation), seq_lengths)
        value_hidden_states = pad_catted_tensor(self.backbone.passage_embs(context_value), seq_lengths)
    
        if self.model_args.value_from_gru:
            value_hidden_states = self.gru(value_hidden_states, seq_lengths)
        else:
            value_hidden_states = value_hidden_states

        extended_id_attention_masks = get_extended_attention_mask(id_attention_masks) # Bxnum_headsxLxD/num_heads
        
        attention_outputs = self.attention(
            query_hidden_states,
            key_hidden_states,
            value_hidden_states,
            extended_id_attention_masks,
            output_attentions = self.model_args.output_id_attentions
        )

        attention_output = cat_padded_tensor(attention_outputs[0], seq_lengths) # BxLxD --> L1xD where L1 = sum(seq_lengths)
        layer_output = self.out(attention_output) # L1xD
        
        if seq_last_output:
            layer_output = get_seq_last_output(layer_output, seq_lengths)  # BxD. for inference

        outputs = (layer_output, ) + attention_outputs[1:]

        return outputs
    
    def passage_embs(
        self,
        value,
    ):
        value_hidden_states = self.backbone.passage_embs(value) # L1xD or BxD
        return value_hidden_states

    @classmethod
    def from_pretrained(cls, model_path, model_args=None):
        if os.path.isdir(model_path):
            ckpt_path = os.path.join(model_path, "model.pt")
            model_args_path = os.path.join(model_path, "model_args.pt")
            model_args = torch.load(model_args_path)
        elif os.path.isfile(model_path):
            assert model_args != None 
            ckpt_path = model_path
        else:
            raise ValueError("model_path: {} is not expected".format(model_path))

        model = cls(model_args)
        print("load pretrained model from local path {}".format(model_path))

        model_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(model_dict)

        return model 

    def save_pretrained(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        model_to_save = self
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, "model.pt"))
        torch.save(self.model_args, os.path.join(save_dir, "model_args.pt"))

    def get_output_dim(self):
        return self.bert.config.hidden_size



