from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers.ops

    IS_XFORMERS_AVAILABLE = True
except ImportError:
    IS_XFORMERS_AVAILABLE = False
    print("xFormers not installed. Using withouth it.")


ATTN_BACKENDS = ["xformers", "pytorch"]
ATTN_BACKENDS_TYPE = Literal["xformers", "pytorch"]

if IS_XFORMERS_AVAILABLE:
    ATTN_BACKEND = "xformers"
else:
    ATTN_BACKEND = "pytorch"


def set_attn_backend(backend: ATTN_BACKENDS_TYPE):
    """
    Set the attention backend to use for the SelfAttention module.

    :param backend: The backend to use for attention calculation.
    """
    global ATTN_BACKEND
    if backend not in ATTN_BACKENDS:
        raise ValueError(
            f"Invalid backend. Must be one of {ATTN_BACKENDS}. Got {backend}"
        )
    ATTN_BACKEND = backend
    return ATTN_BACKEND


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        :param embed_dim: The embedding dimension (must be divisible by num_heads)
        :param num_heads: Number of attention heads
        :param dropout: Dropout rate for output projection
        """
        super(SelfAttention, self).__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim should be divisible by num_heads. Got embed_dim={embed_dim} and num_heads={num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Query, Key, and Value projections (same tensor used for query, key, and value in self-attention)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)

        # Output projection layer
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=out_proj_bias),
            nn.Dropout(dropout),
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, causal_mask=False):
        """
        :param x torch.Tensor: Input tensor of shape (batch_size, seq_len, embed_dim)
        :param causal_mask bool: Whether to apply a causal mask to the attention mechanism
        :return torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        b = x.size(0)
        h = self.num_heads  # Number of heads
        q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, embed_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, embed_dim)

        # Reshaping for multi-head attention
        q, k, v = map(
            lambda t: t.view(b, -1, h, self.head_dim).contiguous(),
            (q, k, v),
        )  # (batch_size, seq_len, num_heads, head_dim)

        if ATTN_BACKEND == "xformers":
            # Use xFormers' memory efficient attention
            out = xformers.ops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=(
                    xformers.ops.LowerTriangularMask() if causal_mask else None
                ),
            )  # (batch_size, seq_len, num_heads, head_dim)
        else:
            # Standard PyTorch attention mechanism
            q, k, v = map(
                lambda t: t.transpose(1, 2),
                (q, k, v),
            )  # (batch_size, num_heads, seq_len, head_dim)

            # Scale dot-product attention
            scale = self.head_dim**-0.5
            q = q * scale  # Scale queries to prevent overflow in large values
            attn = q @ k.transpose(
                -2, -1
            )  # (batch_size, num_heads, seq_len, seq_len)
            if causal_mask:
                mask = torch.triu(
                    torch.ones(attn.size(-1), attn.size(-1), dtype=torch.bool),
                    diagonal=1,
                ).to(attn.device)
                attn = attn.masked_fill(mask, -torch.inf)
            attn = F.softmax(
                attn, dim=-1
            )  # Softmax on the last dimension (context length)
            out = attn @ v  # (batch_size, num_heads, seq_len, head_dim)
            out = out.transpose(
                1, 2
            ).contiguous()  # (batch_size, seq_len, num_heads, head_dim)

        # Combine heads and project back to the original embedding dimension
        out = out.view(
            b, -1, self.embed_dim
        )  # (batch_size, seq_len, embed_dim)
        out = self.out_proj(out)  # (batch_size, seq_len, embed_dim)
        return out


class CrossAttention(nn.Module):
    def __init__(
        self,
        context_dim,
        latent_dim,
        embed_dim,
        num_heads=8,
        in_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
    ):
        """
        :param context_dim: The context dimension
        :param latent_dim: The latent dimension
        :param embed_dim: The embedding dimension (must be divisible by num_heads)
        :param num_heads: Number of attention heads
        :param dropout: Dropout rate for output projection
        """
        super(CrossAttention, self).__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim should be divisible by num_heads. Got embed_dim={embed_dim} and num_heads={num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Query, Key, and Value projections
        self.q_proj = nn.Linear(
            latent_dim, embed_dim, bias=in_proj_bias
        )  # or decoder
        self.k_proj = nn.Linear(
            context_dim, embed_dim, bias=in_proj_bias
        )  # or encoder
        self.v_proj = nn.Linear(
            context_dim, embed_dim, bias=in_proj_bias
        )  # or encoder

        # Output projection layer
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, latent_dim, bias=out_proj_bias),
            nn.Dropout(dropout),
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, latent: torch.Tensor, context: torch.Tensor):
        """
        :param latent torch.Tensor: (batch_size, latent_seq_len, latent_dim)
        :param context torch.Tensor: (batch_size, context_seq_len, context_dim)

        :return torch.Tensor: (batch_size, latent_seq_len, latent_dim)
        """
        b = latent.size(0)
        h = self.num_heads  # scalar
        q = self.q_proj(latent)  # (batch_size, latent_seq_len, embed_dim)
        k = self.k_proj(context)  # (batch_size, context_seq_len, embed_dim)
        v = self.v_proj(context)  # (batch_size, context_seq_len, embed_dim)

        q, k, v = map(
            lambda t: t.view(b, -1, h, self.head_dim).contiguous(),
            (q, k, v),
        )  # (batch_size, seq_len, num_heads, head_dim)

        if ATTN_BACKEND == "xformers":
            out = xformers.ops.memory_efficient_attention(
                q, k, v
            )  # (batch_size, latent_seq_len, num_heads, head_dim)
        else:
            q, k, v = map(
                lambda t: t.transpose(1, 2),
                (q, k, v),
            )  # (batch_size, num_heads, seq_len, head_dim)

            # Scale dot product attention
            scale = self.head_dim**-0.5
            q = q * scale
            attn = q @ k.transpose(
                -2, -1
            )  # (batch_size, num_heads, latent_seq_len, context_seq_len)
            attn = F.softmax(
                attn, dim=-1
            )  # (batch_size, num_heads, latent_seq_len, context_seq_len)
            out = attn @ v  # (batch_size, num_heads, latent_seq_len, head_dim)
            out = out.transpose(
                1, 2
            ).contiguous()  # (batch_size, latent_seq_len, num_heads, head_dim)
        out = out.view(
            b, -1, self.embed_dim
        )  # (batch_size, latent_seq_len, embed_dim)
        out = self.out_proj(out)  # (batch_size, latent_seq_len, latent_dim)
        return out
