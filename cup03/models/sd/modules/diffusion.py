from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .initialize import UniformInitLinear
from .unet import UNet, UNetOutputBlock


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super(TimeEmbedding, self).__init__()

        self.fc1 = UniformInitLinear(embed_dim, 4 * embed_dim)
        self.fc2 = UniformInitLinear(4 * embed_dim, 4 * embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)  # (1, 4 * embed_dim)
        x = F.silu(x)  # (1, 4 * embed_dim)
        x = self.fc2(x)  # (1, 4 * embed_dim)
        return x


class Diffusion(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        time_dim: int,
        hidden_context_dim: Optional[int] = None,
        num_heads=8,
        unet_scale=4,
    ):
        """
        :param latent_dim int: The latent dimension
        :param context_dim int: The context dimension
        :param hidden_context_dim int: The hidden context dimension
        :param time_dim int: The time dimension
        :param num_heads int: The number of heads in the attention mechanism
        :param unet_scale int: An extra parameter for the UNet to reduce the model size
        """
        super(Diffusion, self).__init__()

        self.time_embedding = TimeEmbedding(time_dim)
        if (
            hidden_context_dim is not None
            and context_dim != hidden_context_dim
        ):
            self.proj_context = UniformInitLinear(
                context_dim, hidden_context_dim
            )
        else:
            self.proj_context = nn.Identity()
        self.unet = UNet(
            latent_dim,
            time_dim,
            hidden_context_dim,
            num_heads=num_heads,
            scale=unet_scale,
        )
        self.unet_out = UNetOutputBlock(
            latent_dim,
            scale=unet_scale,
        )

    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ):
        """
        :param latent torch.Tensor: The latent tensor of shape (batch_size, latent_dim, h, w)
        :param context torch.Tensor: The context tensor of shape (batch_size, seq_len, context_dim)
        :param time torch.Tensor: The time tensor of shape (1 or batch_size, time_dim)
        :return torch.Tensor: The output tensor of shape (batch_size, latent_dim, h, w)
        """
        time = self.time_embedding(time)  # (1 or batch_size, 4 * time_dim)
        context = self.proj_context(
            context
        )  # (batch_size, seq_len, hidden_context_dim)
        out = self.unet(latent, context, time)  # (batch_size, 320, h, w)
        out = self.unet_out(out)  # (batch_size, latent_dim, h, w)
        return out
