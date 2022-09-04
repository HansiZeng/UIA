import torch 
import torch.nn as nn
from torch.types import Number

def get_extended_attention_mask( 
    attention_mask: torch.Tensor):
    # create mask for multi-head attention [bz, num_heads, seq_length, seq_length]
    assert attention_mask.dim() == 3 
    extended_attention_mask = attention_mask[:, None, :, :]

    dtype = attention_mask.dtype 
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    
    return extended_attention_mask

# Heavly copied from torchrua.core.major_sizes_to_ptr
@torch.no_grad()
def major_sizes_to_ptr(sizes: torch.Tensor):
    batch_indices = torch.repeat_interleave(repeats=sizes)
    # [4, 7, 3] --> [0, 4, 11]
    start_indices = sizes.cumsum(dim=0).roll(shifts=1, dims=0)
    start_indices[0] = 0

    # [0, 4, 11] --> [0,0,0,0,4,4,4,4,4,4,4,11,11,11] --> [0,1,2,3,0,1,2,3,4,5,6, 0,1,2]
    token_indices = torch.repeat_interleave(start_indices, repeats=sizes)
    token_indices = torch.arange(token_indices.size()[0]).to(device=sizes.device) - token_indices

    return batch_indices, token_indices

def pad_catted_tensor(in_tensor: torch.Tensor, sizes: torch.LongTensor, padding_value: Number = 0):
    assert sum(sizes) == in_tensor.size(0) and sizes.dim() == 1, (sizes, in_tensor)
    device = in_tensor.device
    dtype = in_tensor.dtype

    batch_size = sizes.size()[0]
    max_token_size = sizes.max().item()

    batch_indices, token_indices = major_sizes_to_ptr(sizes)
    batch_indices = batch_indices.to(device=device)
    token_indices = token_indices.to(device=device)
    indices = (batch_indices, token_indices)

    out_tensor = torch.full((batch_size, max_token_size, *in_tensor.size()[1:]), padding_value,
                            dtype=dtype, device=device, requires_grad=False)
    out_tensor[indices] = in_tensor

    return out_tensor

def cat_padded_tensor(in_tensor: torch.Tensor, sizes: torch.LongTensor):
    assert len(sizes) == len(in_tensor) and sizes.dim() == 1, (sizes, len(in_tensor))
    device = in_tensor.device

    batch_indices, token_indices = major_sizes_to_ptr(sizes)
    batch_indices = batch_indices.to(device=device)
    token_indices = token_indices.to(device=device)
    indices = (batch_indices, token_indices)

    out_tensor = in_tensor[indices]
    return out_tensor

@torch.no_grad()
def get_sizes_last_indice(sizes: torch.LongTensor):
    last_indices = sizes.cumsum(dim=0) - torch.ones(sizes.size(0)).to(dtype=sizes.dtype, device=sizes.device)
    return last_indices

def get_seq_last_output(catted_tensor: torch.Tensor, sizes: torch.LongTensor):
    last_indices = get_sizes_last_indice(sizes)
    last_indices = last_indices.to(device=catted_tensor.device)
    
    out_tensor = catted_tensor[last_indices]
    return out_tensor

