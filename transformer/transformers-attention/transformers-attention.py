import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    raw = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[-1])
    attn_w = F.softmax(raw, dim=-1)
    output =  torch.matmul(attn_w, V)
    return output
    pass