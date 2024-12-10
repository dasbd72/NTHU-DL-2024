import os
from hashlib import sha256
from typing import Literal

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(
        self,
        num_vocab: int,
        embed_dim: int,
        max_seq_len: int,
    ):
        super(CLIPEmbedding, self).__init__()

        self.token_embedding = nn.Embedding(num_vocab, embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros((max_seq_len, embed_dim))
        )

    def forward(self, x: torch.Tensor):
        """
        :param x torch.Tensor: The input tensor of shape (batch_size, seq_len)
        :return torch.Tensor: The output tensor of shape (batch_size, seq_len, embed_dim)
        """
        x = self.token_embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x + self.position_embedding  # (batch_size, seq_len, embed_dim)
        return x


class CLIPLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super(CLIPLayer, self).__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(
            embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        :param x torch.Tensor: The input tensor of shape (batch_size, seq_len, embed_dim)
        :return torch.Tensor: The output tensor of shape (batch_size, seq_len, embed_dim)
        """

        # Self-attention
        residual = x
        x = self.ln1(x)
        x = self.attn(x, causal_mask=True)  # (batch_size, seq_len, embed_dim)
        x = x + residual  # (batch_size, seq_len, embed_dim)

        # FFN
        residual = x
        x = self.ln2(x)
        x = self.fc1(x)  # (batch_size, seq_len, 4 * embed_dim)
        x = x * F.sigmoid(1.702 * x)  # Quick gelu
        x = self.fc2(x)  # (batch_size, seq_len, embed_dim)
        x = x + residual  # (batch_size, seq_len, embed_dim)
        return x


class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()

        self.embedding = CLIPEmbedding(
            num_vocab=49408, embed_dim=768, max_seq_len=77
        )
        self.layers = nn.ModuleList(
            [
                CLIPLayer(embed_dim=768, num_heads=12, dropout=0.0)
                for _ in range(12)
            ]
        )
        self.ln = nn.LayerNorm(768)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        :param x torch.LongTensor: The input tensor of shape (batch_size, seq_len)
        :return torch.Tensor: The output tensor of shape (batch_size, seq_len, embed_dim)
        """
        x = x.type(torch.long)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)  # (batch_size, seq_len, embed_dim)
        return x


class FrozenOpenCLIPEmbedder(nn.Module):
    """
    OpenCLIP Embedder with frozen weights.

    Tested on version `open_clip_torch==2.24.0`.

    Referenced from:
    https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/encoders/modules.py
    """

    LAYERS = [
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="ViT-H-14-378-quickgelu",
        version="dfn5b",
        max_length=77,
        freeze=True,
        precision: Literal["pure_fp16", "fp32"] = "fp32",
        layer: Literal["last", "penultimate"] = "last",
        device="cuda",
        cache_dir=None,
    ):
        super().__init__()

        if layer not in self.LAYERS:
            raise ValueError(f"Invalid layer: {layer}")

        if cache_dir is None:
            cache_dir = "./.cache/frozen_open_clip_embedder"
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        self.max_length = max_length
        self.layer_idx = self.LAYERS.index(layer)
        self.precision = precision
        self.device = torch.device(device)

        # Load model from cache or download
        model_args = {
            "arch": arch,
            "version": version,
            "precision": precision,
        }
        cache_path = os.path.join(
            cache_dir,
            sha256(str(model_args).encode()).hexdigest()[0:10] + ".pth",
        )
        self.model: open_clip.CLIP
        if os.path.exists(cache_path):
            self.model = torch.load(
                cache_path, map_location=self.device, weights_only=False
            )
            self.model = self.model.to(self.device)
        else:
            self.model = open_clip.create_model(
                arch,
                device=self.device,
                pretrained=version,
                precision=precision,
            )
            del self.model.visual
            torch.save(self.model, cache_path)
        self.tokenizer = open_clip.get_tokenizer(
            arch, context_length=self.max_length
        )

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, text: str):
        """
        :param text str: The input text
        :return torch.Tensor: The output tensor of shape (batch_size, seq_len, embed_dim)
            ViT-H-14-378-quickgelu: (batch_size, 77, 1024)
        """
        tokens = self.tokenizer(text).to(self.device)
        z = self.encode_with_transformer(tokens)
        z = z.to(torch.float32)
        return z

    def encode_with_transformer(self, text: torch.LongTensor):
        x: torch.Tensor
        x = self.model.token_embedding(text)  # [batch_size, seq_len, emb_dim]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        num_layers = len(self.model.transformer.resblocks)
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == num_layers - self.layer_idx:
                break
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x
