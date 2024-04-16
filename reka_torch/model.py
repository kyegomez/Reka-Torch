
import torch
from torch import nn
from zeta.nn.modules.swiglu import SwiGLU
from zeta.nn.attention import Attention
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm

class RekaTransformerBlock(nn.Module):
    """
    RekaTransformerBlock is a module that represents a single transformer block in the RekaTransformer model.

    Args:
        dim (int): The input dimension of the transformer block.
        depth (int): The number of layers in the transformer block.
        dim_head (int, optional): The dimension of each head in the multi-head attention mechanism. Defaults to 64.
        heads (int, optional): The number of attention heads. Defaults to 8.
        ff_mult (int, optional): The multiplier for the hidden dimension in the feed-forward network. Defaults to 4.
        ff_dropout (float, optional): The dropout probability for the feed-forward network. Defaults to 0.0.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 2,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.ff_mult = ff_mult
        self.ff_dropout = ff_dropout
        
        
        # Attention Layers
        self.attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            causal=True,
            qk_norm=True,
            kv_heads=4,
        )
        
        # Feed Forward Layers
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult, bias=False),
            SwiGLU(),
            SimpleRMSNorm(dim * ff_mult),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * ff_mult, dim, bias=False),
        )
        
    def forward(self, x):
        """
        Forward pass of the RekaTransformerBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        skip = x
        x, _ = self.attn(x)
        # print(x.shape, skip.shape)
        x += skip
        
        # x = self.ffn(x)
        # x += skip
        return x


x = torch.randn(1, 10, 512)

block = RekaTransformerBlock(dim=512)
print(block(x))


